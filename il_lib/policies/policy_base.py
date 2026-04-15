import logging
import os
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F
from abc import ABC, abstractmethod
from collections import deque
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from il_lib.utils.array_tensor_utils import any_concat
from il_lib.utils.convert_utils import any_to_torch
from il_lib.utils.config_utils import register_omegaconf_resolvers
from il_lib.utils.training_utils import load_state_dict, load_torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from omnigibson.learning.utils.eval_utils import (
    ACTION_QPOS_INDICES,
    PROPRIOCEPTION_INDICES,
    PROPRIO_QPOS_INDICES,
    JOINT_RANGE,
    ROBOT_CAMERA_NAMES,
    CAMERA_INTRINSICS,
    EEF_POSITION_RANGE,
)
from omnigibson.learning.utils.obs_utils import (
    create_video_writer, 
    process_fused_point_cloud,
    MIN_DEPTH,
    MAX_DEPTH,
)
from omnigibson.macros import gm
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from typing import Any, Dict, List, Optional


logger = logging.getLogger("BasePolicy")


def _collect_base_policy_overrides() -> List[str]:
    """
    Recover top-level CLI overrides that should also apply to the nested base policy config.
    """
    if GlobalHydra.instance().is_initialized():
        try:
            hydra_cfg = HydraConfig.get()
            return [
                override
                for override in hydra_cfg.overrides.task
                if not override.startswith(("arch=", "+arch=", "++arch="))
            ]
        except Exception:
            pass

    passthrough_overrides = []
    excluded_prefixes = (
        "arch=",
        "+arch=",
        "++arch=",
        "ckpt_path=",
        "+ckpt_path=",
        "++ckpt_path=",
        "module.",
        "+module.",
        "++module.",
        "resume.",
        "+resume.",
        "++resume.",
        "hydra.",
        "+hydra.",
        "++hydra.",
    )
    for override in sys.argv[1:]:
        if override.startswith("-"):
            continue
        if override.startswith(excluded_prefixes):
            continue
        passthrough_overrides.append(override)
    return passthrough_overrides


class BasePolicy(LightningModule, ABC):
    """
    Base class for policies that is used for training and rollout
    """

    def __init__(
        self, 
        *args,
        online_eval: Optional[DictConfig] = None, 
        policy_wrapper: Optional[DictConfig] = None, 
        robot_type: str = "R1Pro",
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # require evaluator for online testing
        self.online_eval_config = online_eval
        self.policy_wrapper_config = policy_wrapper
        if self.online_eval_config is not None:
            OmegaConf.resolve(self.online_eval_config)
            assert self.policy_wrapper_config is not None, "policy_wrapper config must be provided for online evaluation!"
            OmegaConf.resolve(self.policy_wrapper_config)
        else:
            logger.info("No evaluation config provided, online evaluation will not be performed during training.")
        self.evaluator = None
        self.test_id = 0
        self.robot_type = robot_type

    @abstractmethod
    def forward(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of the policy.
        This is used for inference and should return the action.
        """
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def act(self, obs, policy_state, deterministic=None) -> torch.Tensor:
        """
        Args:
            obs: dict of (B, L=1, ...)
            policy_state: (h_0, c_0) or h_0
            deterministic: whether to use deterministic action or not
        Returns:
            action: (B, L=1, A) where A is the action dimension
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the policy
        """
        raise NotImplementedError

    @abstractmethod
    def policy_training_step(self, batch, batch_idx) -> Any:
        raise NotImplementedError

    @abstractmethod
    def policy_evaluation_step(self, batch, batch_idx) -> Any:
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Get optimizers, which are subsequently used to train.
        """
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        loss, log_dict, batch_size = self.policy_training_step(*args, **kwargs)
        log_dict = {f"train/{k}": v for k, v in log_dict.items()}
        log_dict["train/loss"] = loss
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def validation_step(self, *args, **kwargs):
        loss, log_dict, real_batch_size = self.policy_evaluation_step(*args, **kwargs)
        log_dict = {f"val/{k}": v for k, v in log_dict.items()}
        log_dict["val/loss"] = loss
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=real_batch_size,
            sync_dist=True,
        )
        return log_dict

    def test_step(self, *args, **kwargs):
        logger.info("Skipping test step.")

    def on_validation_epoch_end(self):
        # only run test for global zero rank
        if self.trainer.is_global_zero:
            if self.online_eval_config is not None:
                # evaluator for online evaluation should only be created once
                if self.evaluator is None:
                    self.evaluator = self.create_evaluator()
                if not self.trainer.sanity_checking:
                    self.log_dict(self.run_online_evaluation())
        # Synchronize all processes to prevent timeout
        if dist.is_initialized():
            dist.barrier()

    def create_evaluator(self):
        """
        Create a evaluator parameter config containing vectorized distributed envs.
        This will be used to spawn the OmniGibson environments for online evaluation
        """
        # For performance optimization
        gm.DEFAULT_VIEWER_WIDTH = 128
        gm.DEFAULT_VIEWER_HEIGHT = 128
        gm.HEADLESS = self.online_eval_config.cfg.headless

        # update parameters with policy cfg file
        assert self.online_eval_config is not None, "online_eval_config must be provided to create evaluator!"
        evaluator = instantiate(self.online_eval_config, _recursive_=False)
        # instantiate policy wrapper and set the policy 
        policy_wrapper = instantiate(self.policy_wrapper_config)
        policy_wrapper.policy = self
        evaluator.policy.policy = policy_wrapper
        return evaluator

    def run_online_evaluation(self):
        """
        Run online evaluation using the evaluator.
        """
        assert self.evaluator is not None, "evaluator is not created!"
        self.evaluator.reset()
        self.evaluator.env._current_episode = 0
        if self.online_eval_config.cfg.write_video:
            video_name = f"videos/test_{self.test_id}.mp4"
            os.makedirs("videos", exist_ok=True)
            self.evaluator.video_writer = create_video_writer(
                fpath=video_name,
                resolution=(224, 448),
            )
        done = False
        while not done:
            terminated, truncated = self.evaluator.step()
            if self.online_eval_config.cfg.write_video:
                self.evaluator._write_video()
            if terminated:
                self.evaluator.env.reset()
            if truncated:
                done = True
        if self.online_eval_config.cfg.write_video:
            self.evaluator.video_writer = None
        self.test_id += 1
        results = {"eval/success_rate": self.evaluator.n_success_trials / self.evaluator.n_trials}
        return results
    
    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the action from [-1, 1] to [min, max] range.
        Also, rectify gripper actions to either -1 or 1.
        Args:
            action: (B, L, A) where A is the action dimension
        Returns:
            unnormalized_action: (B, L, A)
        """
        # rectify gripper actions
        for k, v in ACTION_QPOS_INDICES[self.robot_type].items():
            if "gripper" in k:
                action[..., v] = torch.where(action[..., v] > 0, 1.0, -1.0)
            else:
                action[..., v] = (action[..., v] + 1) / 2 * (
                    JOINT_RANGE[self.robot_type][k][1] - JOINT_RANGE[self.robot_type][k][0]
                ) + JOINT_RANGE[self.robot_type][k][0]
        return action
    

class PolicyWrapper:
    """
    A Wrapper for handling policy observations and actions
    """

    def __init__(
        self,
        *args,
        # ====== policy model ======
        deployed_action_steps: int,
        obs_window_size: int = 1,
        multi_view_cameras: Dict[str, Any],
        visual_obs_types: List[str],
        use_task_info: bool = False,
        task_info_range: Optional[ListConfig] = None,
        pcd_range: Optional[List[float]] = None,
        robot_type: str = "R1Pro",
        # ====== other args for base class ======
        **kwargs,
    ) -> None:
        self.policy = None # to be filled
        self.robot_type = robot_type
        # move all tensor to self.device
        self._post_processing_fn = lambda x: x.to(self.policy.device)
        assert set(visual_obs_types).issubset(
            {"rgb", "depth_linear", "seg_instance_id", "pcd"}
        ), "visual_obs_types must be a subset of {'rgb', 'depth_linear', 'seg_instance_id', 'pcd'}!"
        self.visual_obs_types = visual_obs_types
        self._use_task_info = use_task_info
        self._task_info_range = (
            torch.tensor(OmegaConf.to_container(task_info_range)) if task_info_range is not None else None
        )
        if "pcd" in visual_obs_types:
            # store camera intrinsics
            self.camera_intrinsics = dict()
            for camera_id, camera_name in ROBOT_CAMERA_NAMES[self.robot_type].items():
                scale_factor = 3.0 if camera_id == "head" else 2.0
                camera_intrinsics = torch.from_numpy(CAMERA_INTRINSICS[self.robot_type][camera_id]) / scale_factor
                camera_intrinsics[-1, -1] = 1.0  # make it homogeneous
                self.camera_intrinsics[camera_name] = camera_intrinsics
        self._pcd_range = tuple(pcd_range) if pcd_range is not None else None
        # action steps for deployed policy
        self.deployed_action_steps = deployed_action_steps
        self.obs_window_size = obs_window_size
        self.obs_output_size = {k: tuple(v["resolution"]) for k, v in multi_view_cameras.items()}
        self._obs_history = deque(maxlen=obs_window_size)
        self._action_traj_pred = None
        self._action_idx = 0
        self._robot_name = None
        self.joint_range = JOINT_RANGE[self.robot_type]

    def act(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        obs = any_to_torch(obs, device="cpu")
        obs = self.process_obs(obs=obs)
        obs = self._stack_obs_history(obs)

        need_inference = self._action_idx % self.deployed_action_steps == 0
        if need_inference:
            self._action_traj_pred = self.policy.act({"obs": obs}).squeeze(0)  # (T_A, A)
            self._action_idx = 0
        action = self._action_traj_pred[self._action_idx]
        self._action_idx += 1
        return action

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()
        self._obs_history = deque(maxlen=self.obs_window_size)
        self._action_traj_pred = None
        self._action_idx = 0

    def _stack_obs_history(self, obs: dict, history: Optional[deque] = None) -> torch.Tensor:
        history = self._obs_history if history is None else history
        if len(history) == 0:
            for _ in range(history.maxlen):
                history.append(obs)
        else:
            history.append(obs)
        return any_concat(history, dim=1)

    def process_obs(self, obs: dict) -> dict:
        # Expand twice to get B and T_A dimensions
        processed_obs = {"qpos": dict()}
        if self._robot_name is None:
            for key in obs:
                if "proprio" in key:
                    self._robot_name = key.split("::")[0]
                    break
        proprio = obs[f"{self._robot_name}::proprio"].unsqueeze(0).unsqueeze(0)
        if "base_qvel" in PROPRIOCEPTION_INDICES[self.robot_type]:
            processed_obs["odom"] = {
                "base_velocity": self._post_processing_fn(
                    2
                    * (proprio[..., PROPRIOCEPTION_INDICES[self.robot_type]["base_qvel"]] - self.joint_range["base"][0])
                    / (self.joint_range["base"][1] - self.joint_range["base"][0])
                    - 1
                ),
            }
        for key in PROPRIO_QPOS_INDICES[self.robot_type]:
            if "gripper" in key:
                # rectify gripper actions to {-1, 1}
                processed_obs["qpos"][key] = torch.mean(
                    proprio[..., PROPRIO_QPOS_INDICES[self.robot_type][key]], dim=-1, keepdim=True
                )
                processed_obs["qpos"][key] = self._post_processing_fn(
                    torch.where(
                        processed_obs["qpos"][key]
                        > (JOINT_RANGE[self.robot_type][key][0] + JOINT_RANGE[self.robot_type][key][1]) * 0.8,
                        1.0,
                        -1.0,
                    )
                )
            else:
                # normalize the qpos to [-1, 1]
                processed_obs["qpos"][key] = self._post_processing_fn(
                    2
                    * (proprio[..., PROPRIO_QPOS_INDICES[self.robot_type][key]] - JOINT_RANGE[self.robot_type][key][0])
                    / (JOINT_RANGE[self.robot_type][key][1] - JOINT_RANGE[self.robot_type][key][0])
                    - 1.0
                )
        if self.robot_type in EEF_POSITION_RANGE:
            processed_obs["eef"] = dict()
            for key in EEF_POSITION_RANGE[self.robot_type]:
                processed_obs["eef"][f"{key}_pos"] = self._post_processing_fn(
                    2
                    * (
                        proprio[..., PROPRIOCEPTION_INDICES[self.robot_type][f"eef_{key}_pos"]]
                        - EEF_POSITION_RANGE[self.robot_type][key][0]
                    )
                    / (EEF_POSITION_RANGE[self.robot_type][key][1] - EEF_POSITION_RANGE[self.robot_type][key][0])
                    - 1.0
                )
                # don't normalize the eef orientation
                processed_obs["eef"][f"{key}_quat"] = self._post_processing_fn(
                    proprio[..., PROPRIOCEPTION_INDICES[self.robot_type][f"eef_{key}_quat"]]
                )
        if "pcd" in self.visual_obs_types:
            pcd_obs = dict()
        for camera_id, camera in ROBOT_CAMERA_NAMES[self.robot_type].items():
            if "rgb" in self.visual_obs_types or "pcd" in self.visual_obs_types:
                rgb_obs = F.interpolate(
                    obs[f"{camera}::rgb"][..., :3].unsqueeze(0).movedim(-1, -3).to(torch.float32),
                    self.obs_output_size[camera_id],
                    mode="nearest-exact",
                ).unsqueeze(0)
                if "pcd" in self.visual_obs_types:
                    # move rgb dim back
                    pcd_obs[f"{camera}::rgb"] = rgb_obs.movedim(-3, -1).to(self.policy.device)
                else:
                    processed_obs[f"{camera}::rgb"] = self._post_processing_fn(rgb_obs)
            if "depth_linear" in self.visual_obs_types or "pcd" in self.visual_obs_types:
                depth_obs = F.interpolate(
                    obs[f"{camera}::depth_linear"].unsqueeze(0).unsqueeze(0).to(torch.float32),
                    self.obs_output_size[camera_id],
                    mode="nearest-exact",
                )
                # clamp depth to [MIN_DEPTH, MAX_DEPTH]
                depth_obs = torch.clamp(depth_obs, MIN_DEPTH, MAX_DEPTH)
                if "pcd" in self.visual_obs_types:
                    pcd_obs[f"{camera}::depth_linear"] = depth_obs.to(self.policy.device)
                else:
                    processed_obs[f"{camera}::depth_linear"] = self._post_processing_fn(depth_obs)
            if "seg_instance_id" in self.visual_obs_types:
                processed_obs[f"{camera}::seg_instance_id"] = self._post_processing_fn(
                    F.interpolate(
                        obs[f"{camera}::seg_instance_id"].unsqueeze(0).unsqueeze(0).to(torch.float32),
                        self.obs_output_size[camera_id],
                        mode="nearest-exact",
                    )
                )
        if "pcd" in self.visual_obs_types:
            pcd_obs["cam_rel_poses"] = (
                obs["robot_r1::cam_rel_poses"].unsqueeze(0).unsqueeze(0).to(torch.float32).to(self.policy.device)
            )
            processed_obs["pcd"] = self._post_processing_fn(
                process_fused_point_cloud(
                    obs=pcd_obs,
                    camera_intrinsics=self.camera_intrinsics,
                    pcd_range=self._pcd_range,
                    pcd_num_points=4096,
                    use_fps=True,
                )
            )
        if self._use_task_info:
            for key in obs:
                if key.startswith("task::"):
                    if self._task_info_range is not None:
                        # Normalize task info to [-1, 1]
                        processed_obs["task"] = (
                            self._post_processing_fn(
                                2
                                * (obs[key] - self._task_info_range[0])
                                / (self._task_info_range[1] - self._task_info_range[0])
                                - 1.0
                            )
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .to(torch.float32)
                        )
                    else:
                        # If no range is provided, just use the raw data
                        processed_obs["task"] = self._post_processing_fn(
                            obs[key].unsqueeze(0).unsqueeze(0).to(torch.float32)
                        )
                    break
        return processed_obs


class ResidualPolicyWrapper(PolicyWrapper):
    """
    A specialized wrapper for ResidualPolicy that manages both base policy and residual policy
    with different action execution frequencies.
    
    Base policy: Predicts action chunks (e.g., 16 actions every 16 steps)
    Residual policy: Predicts per-step corrections (1 correction every step)
    """

    def __init__(
        self,
        *args,
        base_deployed_action_steps: int,  # Base policy's action chunk size
        base_policy: str,
        base_policy_ckpt_path: str,
        base_policy_overrides: Optional[List[str]] = None,
        residual_deployed_action_steps: int = 1,  # Residual policy's action step (usually 1)
        **kwargs,
    ) -> None:
        # Initialize parent with residual policy's deployment frequency
        super().__init__(*args, deployed_action_steps=residual_deployed_action_steps, **kwargs)
        
        # Base policy specific attributes
        self.base_deployed_action_steps = base_deployed_action_steps
        self._base_action_buffer = None  # Will store (T_A, A) from base policy
        self._base_action_idx = 0
        self.base_policy = None  # Will be set to the base policy from residual_policy.base_policy
        self._base_obs_history = None  # Created lazily once the base policy is attached
        self._base_policy_name = base_policy
        self._base_policy_ckpt_path = base_policy_ckpt_path
        self._base_policy_overrides = base_policy_overrides
        self._base_policy_device = None

    def _get_base_policy(self):
        if self.base_policy is None:
            assert self._base_policy_ckpt_path is not None, "base_policy_ckpt_path must be provided for residual inference!"
            overrides = [f"arch={self._base_policy_name}"]
            overrides.extend(_collect_base_policy_overrides())
            if self._base_policy_overrides is not None:
                overrides.extend(self._base_policy_overrides)

            if GlobalHydra.instance().is_initialized():
                base_policy_cfg = compose(config_name="base_config", overrides=overrides).module
            else:
                config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
                config_dir = os.path.abspath(config_dir)
                with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
                    base_policy_cfg = compose(config_name="base_config", overrides=overrides).module

            register_omegaconf_resolvers()
            OmegaConf.resolve(base_policy_cfg)
            self.base_policy = instantiate(base_policy_cfg, _recursive_=False)

            ckpt = load_torch(self._base_policy_ckpt_path, map_location="cpu")
            load_state_dict(self.base_policy, ckpt["state_dict"], strict=True)
            self._base_policy_device = self.policy.device
            self.base_policy = self.base_policy.to(self._base_policy_device)
            self.base_policy.eval()
        return self.base_policy

    def _get_base_obs_history(self) -> deque:
        base_policy = self._get_base_policy()
        if base_policy is None:
            raise ValueError("base_policy not found in residual policy!")
        if self._base_obs_history is None:
            base_obs_window_size = getattr(base_policy, "num_latest_obs", self.obs_window_size)
            self._base_obs_history = deque(maxlen=base_obs_window_size)
        return self._base_obs_history
    
    def act(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        """
        Coordinated action generation:
        1. Get base action from buffer (refresh every base_deployed_action_steps)
        2. Get residual correction from residual policy (every step)
        3. Combine: final_action = base_action + residual_correction
        """
        obs = any_to_torch(obs, device="cpu")
        obs = self.process_obs(obs=obs)
        residual_obs = {"obs": self._stack_obs_history(obs)}

        # ===== Base Policy: Action Chunking =====
        need_base_inference = self._base_action_idx % self.base_deployed_action_steps == 0
        if need_base_inference:
            base_policy = self._get_base_policy()
            base_obs = {"obs": self._stack_obs_history(obs, history=self._get_base_obs_history())}
            self._base_action_buffer = base_policy.act(base_obs).squeeze(0)  # (T_A, A)
            self._base_action_idx = 0
        elif self._base_obs_history is not None:
            self._stack_obs_history(obs, history=self._base_obs_history)
        
        # Get current base action from buffer (raw radians, denormalized by base policy)
        base_action = self._base_action_buffer[self._base_action_idx]  # (A,)
        self._base_action_idx += 1
        base_action = self._post_processing_fn(base_action)

        # Normalize base action back to [-1, 1] so it matches the residual's training space
        base_action_normalized = self._normalize_action(base_action.clone())
        if "base_action" in getattr(self.policy, "_features", set()):
            residual_obs["obs"]["base_action"] = self._post_processing_fn(
                base_action_normalized.view(1, 1, -1)
            )

        # ===== Residual Policy: Per-step Correction =====
        # Get residual correction (intervention decision + action correction)
        residual_action, intervention = self.policy.act(residual_obs["obs"])
        residual_action = residual_action.squeeze()  # (A,)
        intervention = intervention.squeeze()  # scalar

        # ===== Combine Actions =====
        if intervention >= 0.5:  # Intervention needed
            # Add residual in normalized space, then denormalize
            combined_normalized = base_action_normalized + residual_action
            final_action = self._denormalize_action(combined_normalized)
        else:
            # No intervention, use base action as-is (already denormalized)
            final_action = base_action

        return final_action

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize action from raw joint space to [-1, 1]."""
        for k, v in ACTION_QPOS_INDICES[self.robot_type].items():
            if "gripper" not in k:
                joint_min = torch.as_tensor(
                    JOINT_RANGE[self.robot_type][k][0],
                    device=action.device,
                    dtype=action.dtype,
                )
                joint_max = torch.as_tensor(
                    JOINT_RANGE[self.robot_type][k][1],
                    device=action.device,
                    dtype=action.dtype,
                )
                action[..., v] = (
                    2 * (action[..., v] - joint_min)
                    / (joint_max - joint_min)
                    - 1.0
                )
        return action

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Denormalize action from [-1, 1] to raw joint space."""
        for k, v in ACTION_QPOS_INDICES[self.robot_type].items():
            if "gripper" in k:
                action[..., v] = torch.where(action[..., v] > 0, 1.0, -1.0)
            else:
                joint_min = torch.as_tensor(
                    JOINT_RANGE[self.robot_type][k][0],
                    device=action.device,
                    dtype=action.dtype,
                )
                joint_max = torch.as_tensor(
                    JOINT_RANGE[self.robot_type][k][1],
                    device=action.device,
                    dtype=action.dtype,
                )
                action[..., v] = (action[..., v] + 1) / 2 * (
                    joint_max - joint_min
                ) + joint_min
        return action

    def reset(self) -> None:
        """Reset both policies and their states"""
        super().reset()
        self._base_action_buffer = None
        self._base_action_idx = 0
        self._base_obs_history = None
        if self.base_policy is not None:
            self.base_policy.reset()
