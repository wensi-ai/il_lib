import atexit
import logging
import os
import sys
from pathlib import Path
from time import perf_counter

import h5py
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
    Residual policy: Predicts correction chunks at its deployment frequency
    """

    def __init__(
        self,
        *args,
        base_deployed_action_steps: int,  # Base policy's action chunk size
        base_policy: str,
        base_policy_ckpt_path: str,
        base_policy_overrides: Optional[List[str]] = None,
        residual_deployed_action_steps: int = 1,  # Residual policy's action step (usually 1)
        intervention_policy: Optional[str] = None,
        intervention_policy_ckpt_path: Optional[str] = None,
        intervention_policy_overrides: Optional[List[str]] = None,
        intervention_policy_threshold: Optional[float] = None,
        intervention_include_current_action_in_history: bool = False,
        intervention_prop_keys: Optional[List[str]] = None,
        trace_hdf5_path: Optional[str] = None,
        trace_overwrite: bool = False,
        **kwargs,
    ) -> None:
        compat_deployed_action_steps = kwargs.pop("deployed_action_steps", None)
        if compat_deployed_action_steps is not None:
            residual_deployed_action_steps = compat_deployed_action_steps

        # Initialize parent with residual policy's deployment frequency
        super().__init__(*args, deployed_action_steps=residual_deployed_action_steps, **kwargs)
        
        # Base policy specific attributes
        self.base_deployed_action_steps = base_deployed_action_steps
        self._base_action_buffer = None  # Will store (T_A, A) from base policy
        self._base_action_idx = 0
        self._residual_action_buffer = None
        self._residual_intervention_buffer = None
        self._residual_action_idx = 0
        self._intervention_steps_remaining = 0
        self.base_policy = None  # Will be set to the base policy from residual_policy.base_policy
        self._base_obs_history = None  # Created lazily once the base policy is attached
        self._base_policy_name = base_policy
        self._base_policy_ckpt_path = base_policy_ckpt_path
        self._base_policy_overrides = base_policy_overrides
        self._base_policy_device = None
        self.intervention_policy = None
        self._intervention_policy_name = intervention_policy
        self._intervention_policy_ckpt_path = intervention_policy_ckpt_path
        self._intervention_policy_overrides = intervention_policy_overrides
        self._intervention_policy_threshold = intervention_policy_threshold
        self._intervention_include_current_action_in_history = (
            intervention_include_current_action_in_history
        )
        self._intervention_prop_keys = intervention_prop_keys or [
            "qpos/arm",
            "qpos/gripper",
        ]
        self._intervention_obs_history = None
        self._executed_action_history = None

        self._trace_hdf5_path = (
            Path(trace_hdf5_path).expanduser().resolve()
            if trace_hdf5_path is not None
            else None
        )
        self._trace_overwrite = trace_overwrite
        self._trace_file = None
        self._trace_demo_idx = 0
        self._trace_step_idx = 0
        self._trace_episode_start_time = None
        self._trace_steps: List[Dict[str, torch.Tensor]] = []
        self._trace_closed = False
        if self._trace_hdf5_path is not None:
            self._init_trace_writer()
            atexit.register(self.close)

    def _init_trace_writer(self) -> None:
        assert self._trace_hdf5_path is not None
        self._trace_hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        file_mode = "w" if self._trace_overwrite else "a"
        self._trace_file = h5py.File(self._trace_hdf5_path, file_mode)
        if "data" not in self._trace_file:
            self._trace_file.create_group("data")
        existing_demo_ids = []
        for key in self._trace_file["data"].keys():
            if key.startswith("demo_"):
                try:
                    existing_demo_ids.append(int(key.split("_")[-1]))
                except ValueError:
                    continue
        self._trace_demo_idx = max(existing_demo_ids, default=-1) + 1
        logger.info("Residual trace logging enabled: %s", self._trace_hdf5_path)

    def _to_trace_tensor(self, value: torch.Tensor) -> torch.Tensor:
        return value.detach().to("cpu", copy=True).reshape(-1).to(torch.float32)

    def _record_trace_step(
        self,
        *,
        base_action: torch.Tensor,
        residual_action: torch.Tensor,
        combined_action: torch.Tensor,
        applied_action: torch.Tensor,
        intervention: torch.Tensor,
    ) -> None:
        if self._trace_hdf5_path is None:
            return
        if self._trace_episode_start_time is None:
            self._trace_episode_start_time = perf_counter()
            self._trace_step_idx = 0

        self._trace_steps.append(
            {
                "base_action": self._to_trace_tensor(base_action),
                "residual_action": self._to_trace_tensor(residual_action),
                "combined_action": self._to_trace_tensor(combined_action),
                "applied_action": self._to_trace_tensor(applied_action),
                # Keep this alias so the existing correction visualizer can render
                # residual traces without requiring a new dataset schema.
                "oracle_action": self._to_trace_tensor(combined_action),
                "intervention": self._to_trace_tensor(intervention),
                "is_oracle_active": self._to_trace_tensor(intervention >= 0.5),
                "int_state": self._to_trace_tensor(
                    torch.where(intervention >= 0.5, 2.0, 1.0)
                ),
                "timestamp_ms": torch.tensor(
                    [1000.0 * (perf_counter() - self._trace_episode_start_time)],
                    dtype=torch.float32,
                ),
            }
        )
        self._trace_step_idx += 1

    def _flush_trace_demo(self) -> None:
        if self._trace_file is None or not self._trace_steps:
            self._trace_steps = []
            self._trace_episode_start_time = None
            self._trace_step_idx = 0
            return

        data_group = self._trace_file["data"]
        demo_key = f"demo_{self._trace_demo_idx}"
        if demo_key in data_group:
            del data_group[demo_key]
        demo_group = data_group.create_group(demo_key)
        policy_group = demo_group.create_group("policy")
        time_group = demo_group.create_group("time")

        def _stack(name: str) -> torch.Tensor:
            return torch.stack([step[name] for step in self._trace_steps], dim=0)

        base_action = _stack("base_action").numpy()
        residual_action = _stack("residual_action").numpy()
        combined_action = _stack("combined_action").numpy()
        applied_action = _stack("applied_action").numpy()
        oracle_action = _stack("oracle_action").numpy()
        intervention = _stack("intervention").numpy()
        is_oracle_active = _stack("is_oracle_active").numpy()
        int_state = _stack("int_state").numpy()
        timestamp_ms = _stack("timestamp_ms").numpy()

        demo_group.create_dataset("action", data=applied_action)
        policy_group.create_dataset("base_action", data=base_action)
        policy_group.create_dataset("residual_action", data=residual_action)
        policy_group.create_dataset("combined_action", data=combined_action)
        policy_group.create_dataset("applied_action", data=applied_action)
        policy_group.create_dataset("oracle_action", data=oracle_action)
        policy_group.create_dataset("intervention", data=intervention)
        policy_group.create_dataset("is_oracle_active", data=is_oracle_active)
        policy_group.create_dataset("int_state", data=int_state)
        time_group.create_dataset("timestamp_ms", data=timestamp_ms)
        demo_group.attrs["source"] = "ResidualPolicyWrapper"

        self._trace_file.flush()
        logger.info(
            "Saved residual trace %s with %d steps to %s",
            demo_key,
            len(self._trace_steps),
            self._trace_hdf5_path,
        )
        self._trace_demo_idx += 1
        self._trace_steps = []
        self._trace_episode_start_time = None
        self._trace_step_idx = 0

    def close(self) -> None:
        if self._trace_closed:
            return
        self._trace_closed = True
        try:
            self._flush_trace_demo()
        finally:
            if self._trace_file is not None:
                self._trace_file.close()
                self._trace_file = None

    def _compose_module_cfg(self, arch_name: str, extra_overrides: Optional[List[str]] = None):
        overrides = [f"arch={arch_name}"]
        overrides.extend(_collect_base_policy_overrides())
        if extra_overrides is not None:
            overrides.extend(extra_overrides)

        if GlobalHydra.instance().is_initialized():
            module_cfg = compose(config_name="base_config", overrides=overrides).module
        else:
            config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
            config_dir = os.path.abspath(config_dir)
            with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
                module_cfg = compose(config_name="base_config", overrides=overrides).module

        register_omegaconf_resolvers()
        OmegaConf.resolve(module_cfg)
        return module_cfg

    def _instantiate_arch_module(
        self,
        *,
        arch_name: str,
        ckpt_path: str,
        extra_overrides: Optional[List[str]] = None,
    ):
        if ckpt_path is None:
            raise AssertionError(f"{arch_name} requires a checkpoint path for inference.")
        module_cfg = self._compose_module_cfg(arch_name, extra_overrides=extra_overrides)
        module = instantiate(module_cfg, _recursive_=False)
        ckpt = load_torch(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        load_state_dict(module, state_dict, strict=True)
        module = module.to(self.policy.device)
        module.eval()
        return module

    def _get_base_policy(self):
        if self.base_policy is None:
            assert self._base_policy_ckpt_path is not None, "base_policy_ckpt_path must be provided for residual inference!"
            self.base_policy = self._instantiate_arch_module(
                arch_name=self._base_policy_name,
                ckpt_path=self._base_policy_ckpt_path,
                extra_overrides=self._base_policy_overrides,
            )
            self._base_policy_device = self.policy.device
        return self.base_policy

    def _get_intervention_policy(self):
        if self._intervention_policy_name is None:
            return None
        if self.intervention_policy is None:
            self.intervention_policy = self._instantiate_arch_module(
                arch_name=self._intervention_policy_name,
                ckpt_path=self._intervention_policy_ckpt_path,
                extra_overrides=self._intervention_policy_overrides,
            )
        return self.intervention_policy

    def _get_base_obs_history(self) -> deque:
        base_policy = self._get_base_policy()
        if base_policy is None:
            raise ValueError("base_policy not found in residual policy!")
        if self._base_obs_history is None:
            base_obs_window_size = getattr(base_policy, "num_latest_obs", self.obs_window_size)
            self._base_obs_history = deque(maxlen=base_obs_window_size)
        return self._base_obs_history

    def _get_intervention_input_steps(self, key: str, default: int = 1) -> int:
        intervention_policy = self._get_intervention_policy()
        if intervention_policy is None:
            return default
        cfg = getattr(intervention_policy, "input_configs", {})
        if key not in cfg:
            return default
        return int(cfg[key]["steps"])

    def _get_intervention_obs_history(self) -> deque:
        intervention_policy = self._get_intervention_policy()
        if intervention_policy is None:
            raise ValueError("intervention_policy not found for residual inference.")
        if self._intervention_obs_history is None:
            obs_steps = max(
                self._get_intervention_input_steps("proprioception", default=1),
                self._get_intervention_input_steps("task", default=1),
            )
            self._intervention_obs_history = deque(maxlen=obs_steps)
        return self._intervention_obs_history

    def _get_executed_action_history(self) -> deque:
        if self._executed_action_history is None:
            history_steps = self._get_intervention_input_steps("action_history", default=1)
            self._executed_action_history = deque(maxlen=history_steps)
        return self._executed_action_history

    def _append_executed_action(self, normalized_action: torch.Tensor) -> None:
        if self._intervention_policy_name is None:
            return
        history = self._get_executed_action_history()
        history.append(normalized_action.detach().clone())

    def _pad_action_sequence(
        self,
        sequence: torch.Tensor,
        target_steps: int,
        *,
        pad_with_last: bool = True,
    ) -> torch.Tensor:
        if sequence.shape[0] >= target_steps:
            return sequence[:target_steps]
        if sequence.shape[0] == 0:
            raise ValueError("Cannot pad an empty action sequence.")
        pad_value = sequence[-1:] if pad_with_last else torch.zeros_like(sequence[:1])
        pad = pad_value.repeat(target_steps - sequence.shape[0], 1)
        return torch.cat([sequence, pad], dim=0)

    def _build_intervention_inputs(
        self,
        *,
        obs: dict,
        base_action_chunk: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        intervention_policy = self._get_intervention_policy()
        if intervention_policy is None:
            raise ValueError("Standalone intervention policy is not configured.")

        input_keys = getattr(intervention_policy, "input_keys", [])
        obs_window = self._stack_obs_history(obs, history=self._get_intervention_obs_history())
        inputs = {}
        if "proprioception" in input_keys:
            prop_obs = []
            for prop_key in self._intervention_prop_keys:
                group, key = prop_key.split("/", 1)
                prop_obs.append(obs_window[group][key])
            inputs["proprioception"] = torch.cat(prop_obs, dim=-1)
        if "task" in input_keys:
            if "task" not in obs_window:
                raise KeyError("Intervention policy expects task input, but task info is unavailable.")
            inputs["task"] = obs_window["task"]
        if "base_action_chunk" in input_keys:
            steps = self._get_intervention_input_steps(
                "base_action_chunk", default=base_action_chunk.shape[0]
            )
            base_chunk = self._pad_action_sequence(base_action_chunk, steps, pad_with_last=True)
            inputs["base_action_chunk"] = base_chunk.unsqueeze(0)
        if "action_history" in input_keys:
            steps = self._get_intervention_input_steps("action_history", default=1)
            history = list(self._get_executed_action_history())
            history_values = history[-steps:]
            if history_values:
                history_tensor = torch.stack(history_values, dim=0)
                if history_tensor.shape[0] < steps:
                    pad = torch.zeros(
                        (steps - history_tensor.shape[0], history_tensor.shape[-1]),
                        device=history_tensor.device,
                        dtype=history_tensor.dtype,
                    )
                    history_tensor = torch.cat([pad, history_tensor], dim=0)
            else:
                history_tensor = torch.zeros(
                    (steps, self.policy.action_dim),
                    device=base_action_chunk.device,
                    dtype=base_action_chunk.dtype,
                )
            inputs["action_history"] = history_tensor.unsqueeze(0)
        return inputs

    @torch.no_grad()
    def _predict_standalone_intervention(
        self,
        *,
        obs: dict,
        base_action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        intervention_policy = self._get_intervention_policy()
        if intervention_policy is None:
            raise ValueError("Standalone intervention policy is not configured.")
        inputs = self._build_intervention_inputs(obs=obs, base_action_chunk=base_action_chunk)
        logits = intervention_policy(inputs)
        probs = torch.sigmoid(logits)
        threshold = self._intervention_policy_threshold
        if threshold is None:
            threshold = float(getattr(intervention_policy, "decision_threshold", 0.5))
        return (probs >= threshold).to(torch.float32)
    
    def act(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        """
        Coordinated action generation:
        1. Get base action from buffer (refresh every base_deployed_action_steps)
        2. Get residual correction from residual policy
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
        base_action_idx = self._base_action_idx
        base_action = self._base_action_buffer[base_action_idx]  # (A,)
        self._base_action_idx += 1
        base_action = self._post_processing_fn(base_action)

        # Normalize base action back to [-1, 1] so it matches the residual's training space
        base_action_normalized = self._normalize_action(base_action.clone())
        residual_horizon = getattr(self.policy, "action_prediction_horizon", 1)
        intervention_chunk_steps = self._get_intervention_input_steps(
            "base_action_chunk", default=residual_horizon
        )
        base_action_chunk_steps = max(residual_horizon, intervention_chunk_steps)
        base_action_chunk = self._base_action_buffer[
            base_action_idx : base_action_idx + base_action_chunk_steps
        ]
        if base_action_chunk.shape[0] < base_action_chunk_steps:
            pad = base_action_chunk[-1:].repeat(
                base_action_chunk_steps - base_action_chunk.shape[0], 1
            )
            base_action_chunk = torch.cat([base_action_chunk, pad], dim=0)
        base_action_chunk = self._post_processing_fn(base_action_chunk)
        base_action_chunk = self._normalize_action(base_action_chunk.clone())
        if "base_action" in getattr(self.policy, "_features", set()):
            if residual_horizon > 1:
                residual_obs["obs"]["base_action"] = base_action_chunk[:residual_horizon].view(
                    1, 1, residual_horizon, -1
                )
            else:
                residual_obs["obs"]["base_action"] = base_action_normalized.view(1, 1, -1)

        # ===== Residual Policy: Correction Chunk =====
        need_residual_inference = (
            self._residual_action_buffer is None
            or self._residual_action_idx >= self._residual_action_buffer.shape[0]
            or self._residual_action_idx % self.deployed_action_steps == 0
        )
        if need_residual_inference:
            residual_action, intervention = self.policy.act(residual_obs["obs"])
            residual_action = residual_action.squeeze(0)
            if residual_action.dim() == 3:
                residual_action = residual_action[-1]  # (T_A, A)
            elif residual_action.dim() == 2:
                residual_action = residual_action[-1:].clone()  # (1, A)
            self._residual_action_buffer = residual_action

            if self._intervention_policy_name is None:
                intervention = intervention.squeeze(0)
                if intervention.dim() > 0:
                    intervention = intervention[-1]
                self._residual_intervention_buffer = intervention.reshape(1).repeat(
                    self._residual_action_buffer.shape[0]
                )
            self._residual_action_idx = 0

        residual_action = self._residual_action_buffer[self._residual_action_idx]
        if self._intervention_policy_name is not None:
            intervention = self._predict_standalone_intervention(
                obs=obs,
                base_action_chunk=base_action_chunk,
            ).reshape(-1)[0]
        else:
            intervention = self._residual_intervention_buffer[self._residual_action_idx]
        self._residual_action_idx += 1

        min_intervention_steps = max(
            1,
            int(getattr(self.policy, "intervention_min_duration_steps", 1)),
        )
        intervention_active = bool(float(intervention) >= 0.5)
        if intervention_active:
            self._intervention_steps_remaining = max(
                self._intervention_steps_remaining,
                min_intervention_steps,
            )
        if self._intervention_steps_remaining > 0:
            intervention = intervention.new_ones(intervention.shape)
            self._intervention_steps_remaining -= 1

        # ===== Combine Actions =====
        combined_normalized = base_action_normalized + residual_action
        combined_action = self._denormalize_action(combined_normalized.clone())
        if intervention >= 0.5:  # Intervention needed
            final_action = combined_action.clone()
            source_label = "\033[1m\033[92mRESIDUAL\033[0m"
        else:
            # No intervention, use base action as-is (already denormalized)
            final_action = base_action.clone()
            source_label = "\033[1m\033[94mBASE\033[0m"

        print(
            "\033[1m\033[96m[ResidualPolicyWrapper]\033[0m "
            f"executing={source_label} "
            f"intervention={float(intervention):.3f} "
            f"chunk_step={self._residual_action_idx}/{self._residual_action_buffer.shape[0]} "
            f"base_step={self._base_action_idx}/{self._base_action_buffer.shape[0]}",
            flush=True,
        )

        self._record_trace_step(
            base_action=base_action,
            residual_action=residual_action,
            combined_action=combined_action,
            applied_action=final_action,
            intervention=intervention.reshape(1),
        )
        self._append_executed_action(
            combined_normalized if intervention >= 0.5 else base_action_normalized
        )

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
        self._flush_trace_demo()
        super().reset()
        self._base_action_buffer = None
        self._base_action_idx = 0
        self._residual_action_buffer = None
        self._residual_intervention_buffer = None
        self._residual_action_idx = 0
        self._intervention_steps_remaining = 0
        self._base_obs_history = None
        self._intervention_obs_history = None
        self._executed_action_history = None
        if self.base_policy is not None:
            self.base_policy.reset()


class BaseChunkPolicyWrapper(ResidualPolicyWrapper):
    """
    A wrapper for policies that condition on a base policy chunk and directly predict
    the final action chunk in the normalized action space.
    """

    def __init__(
        self,
        *args,
        intervention_policy: Optional[str] = None,
        intervention_policy_ckpt_path: Optional[str] = None,
        intervention_policy_overrides: Optional[List[str]] = None,
        intervention_policy_threshold: Optional[float] = None,
        intervention_include_current_action_in_history: bool = False,
        intervention_prop_keys: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.intervention_policy = None
        self._intervention_policy_name = intervention_policy
        self._intervention_policy_ckpt_path = intervention_policy_ckpt_path
        self._intervention_policy_overrides = intervention_policy_overrides
        self._intervention_policy_threshold = intervention_policy_threshold
        self._intervention_include_current_action_in_history = (
            intervention_include_current_action_in_history
        )
        self._intervention_prop_keys = intervention_prop_keys or [
            "qpos/arm",
            "qpos/gripper",
        ]
        self._intervention_obs_history = None
        self._executed_action_history = None

    def _get_intervention_policy(self):
        if self._intervention_policy_name is None:
            return None
        if self.intervention_policy is None:
            self.intervention_policy = self._instantiate_arch_module(
                arch_name=self._intervention_policy_name,
                ckpt_path=self._intervention_policy_ckpt_path,
                extra_overrides=self._intervention_policy_overrides,
            )
        return self.intervention_policy

    def _get_intervention_input_steps(self, key: str, default: int = 1) -> int:
        intervention_policy = self._get_intervention_policy()
        if intervention_policy is None:
            return default
        cfg = getattr(intervention_policy, "input_configs", {})
        if key not in cfg:
            return default
        return int(cfg[key]["steps"])

    def _get_intervention_obs_history(self) -> deque:
        intervention_policy = self._get_intervention_policy()
        if intervention_policy is None:
            raise ValueError("intervention_policy not found for base chunk inference.")
        if self._intervention_obs_history is None:
            obs_steps = max(
                self._get_intervention_input_steps("proprioception", default=1),
                self._get_intervention_input_steps("task", default=1),
            )
            self._intervention_obs_history = deque(maxlen=obs_steps)
        return self._intervention_obs_history

    def _get_executed_action_history(self) -> deque:
        if self._executed_action_history is None:
            history_steps = self._get_intervention_input_steps("action_history", default=1)
            self._executed_action_history = deque(maxlen=history_steps)
        return self._executed_action_history

    def _append_executed_action(self, normalized_action: torch.Tensor) -> None:
        if self._intervention_policy_name is None:
            return
        history = self._get_executed_action_history()
        history.append(normalized_action.detach().clone())

    def _normalize_chunk(self, action_chunk: torch.Tensor) -> torch.Tensor:
        return self._normalize_action(action_chunk.clone())

    def _pad_action_sequence(
        self,
        sequence: torch.Tensor,
        target_steps: int,
        *,
        pad_with_last: bool = True,
    ) -> torch.Tensor:
        if sequence.shape[0] >= target_steps:
            return sequence[:target_steps]
        if sequence.shape[0] == 0:
            raise ValueError("Cannot pad an empty action sequence.")
        pad_value = sequence[-1:] if pad_with_last else torch.zeros_like(sequence[:1])
        pad = pad_value.repeat(target_steps - sequence.shape[0], 1)
        return torch.cat([sequence, pad], dim=0)

    def _build_intervention_inputs(
        self,
        *,
        obs: dict,
        base_action_chunk: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        intervention_policy = self._get_intervention_policy()
        if intervention_policy is None:
            raise ValueError("Standalone intervention policy is not configured.")

        input_keys = getattr(intervention_policy, "input_keys", [])
        obs_window = self._stack_obs_history(obs, history=self._get_intervention_obs_history())
        inputs = {}
        if "proprioception" in input_keys:
            prop_obs = []
            for prop_key in self._intervention_prop_keys:
                group, key = prop_key.split("/", 1)
                prop_obs.append(obs_window[group][key])
            inputs["proprioception"] = torch.cat(prop_obs, dim=-1)
        if "task" in input_keys:
            if "task" not in obs_window:
                raise KeyError("Intervention policy expects task input, but task info is unavailable.")
            inputs["task"] = obs_window["task"]
        if "base_action_chunk" in input_keys:
            steps = self._get_intervention_input_steps("base_action_chunk", default=base_action_chunk.shape[0])
            base_chunk = self._pad_action_sequence(base_action_chunk, steps, pad_with_last=True)
            inputs["base_action_chunk"] = base_chunk.unsqueeze(0)
        if "action_history" in input_keys:
            steps = self._get_intervention_input_steps("action_history", default=1)
            history = list(self._get_executed_action_history())
            history_values = history[-steps:]
            if history_values:
                history_tensor = torch.stack(history_values, dim=0)
                if history_tensor.shape[0] < steps:
                    pad = torch.zeros(
                        (steps - history_tensor.shape[0], history_tensor.shape[-1]),
                        device=history_tensor.device,
                        dtype=history_tensor.dtype,
                    )
                    history_tensor = torch.cat([pad, history_tensor], dim=0)
            else:
                history_tensor = torch.zeros(
                    (steps, self.policy.action_dim),
                    device=base_action_chunk.device,
                    dtype=base_action_chunk.dtype,
                )
            inputs["action_history"] = history_tensor.unsqueeze(0)
        return inputs

    @torch.no_grad()
    def _predict_standalone_intervention(
        self,
        *,
        obs: dict,
        base_action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        intervention_policy = self._get_intervention_policy()
        if intervention_policy is None:
            raise ValueError("Standalone intervention policy is not configured.")
        inputs = self._build_intervention_inputs(obs=obs, base_action_chunk=base_action_chunk)
        logits = intervention_policy(inputs)
        probs = torch.sigmoid(logits)
        threshold = self._intervention_policy_threshold
        if threshold is None:
            threshold = float(getattr(intervention_policy, "decision_threshold", 0.5))
        return (probs >= threshold).to(torch.float32)

    def act(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        obs = any_to_torch(obs, device="cpu")
        obs = self.process_obs(obs=obs)
        policy_obs = {"obs": self._stack_obs_history(obs)}

        need_base_inference = self._base_action_idx % self.base_deployed_action_steps == 0
        if need_base_inference:
            base_policy = self._get_base_policy()
            base_obs = {"obs": self._stack_obs_history(obs, history=self._get_base_obs_history())}
            self._base_action_buffer = base_policy.act(base_obs).squeeze(0)
            self._base_action_idx = 0
        elif self._base_obs_history is not None:
            self._stack_obs_history(obs, history=self._base_obs_history)

        base_action_idx = self._base_action_idx
        base_action = self._post_processing_fn(self._base_action_buffer[base_action_idx])
        self._base_action_idx += 1

        residual_horizon = getattr(self.policy, "action_prediction_horizon", 1)
        base_action_horizon = getattr(self.policy, "base_action_horizon", residual_horizon)
        base_action_chunk = self._base_action_buffer[
            base_action_idx : base_action_idx + base_action_horizon
        ]
        if base_action_chunk.shape[0] < base_action_horizon:
            pad = base_action_chunk[-1:].repeat(base_action_horizon - base_action_chunk.shape[0], 1)
            base_action_chunk = torch.cat([base_action_chunk, pad], dim=0)
        base_action_chunk = self._post_processing_fn(base_action_chunk)
        base_action_chunk = self._normalize_chunk(base_action_chunk)
        policy_obs["obs"]["base_action"] = base_action_chunk.view(1, 1, base_action_horizon, -1)

        need_policy_inference = (
            self._residual_action_buffer is None
            or self._residual_action_idx >= self._residual_action_buffer.shape[0]
            or self._residual_action_idx % self.deployed_action_steps == 0
        )
        if need_policy_inference:
            policy_output = self.policy.act(policy_obs["obs"])
            if isinstance(policy_output, tuple):
                pred_action, intervention = policy_output
            else:
                pred_action, intervention = policy_output, None
            pred_action = pred_action.squeeze(0)
            if pred_action.dim() == 3:
                pred_action = pred_action[-1]
            elif pred_action.dim() == 2:
                pred_action = pred_action.clone()
            self._residual_action_buffer = pred_action

            if self._intervention_policy_name is None:
                if intervention is None:
                    raise ValueError("Base chunk policy did not return intervention outputs.")
                intervention = intervention.squeeze(0)
                if intervention.dim() > 0:
                    intervention = intervention[-1]
                self._residual_intervention_buffer = intervention.reshape(1).repeat(
                    self._residual_action_buffer.shape[0]
                )
            self._residual_action_idx = 0

        pred_action = self._residual_action_buffer[self._residual_action_idx]
        if self._intervention_policy_name is not None:
            intervention = self._predict_standalone_intervention(
                obs=obs,
                base_action_chunk=base_action_chunk,
            ).reshape(-1)[0]
        else:
            intervention = self._residual_intervention_buffer[self._residual_action_idx]
        self._residual_action_idx += 1

        min_intervention_steps = max(
            1,
            int(getattr(self.policy, "intervention_min_duration_steps", 1)),
        )
        intervention_active = bool(float(intervention) >= 0.5)
        if intervention_active:
            self._intervention_steps_remaining = max(
                self._intervention_steps_remaining,
                min_intervention_steps,
            )
        if self._intervention_steps_remaining > 0:
            intervention = intervention.new_ones(intervention.shape)
            self._intervention_steps_remaining -= 1

        pred_action = pred_action.clamp(-1.0, 1.0)
        pred_action_denormalized = self._denormalize_action(pred_action.clone())
        if intervention >= 0.5:
            final_action = pred_action_denormalized
            source_label = "\033[1m\033[92mBASE-CHUNK\033[0m"
        else:
            final_action = base_action.clone()
            source_label = "\033[1m\033[94mBASE\033[0m"

        print(
            "\033[1m\033[96m[BaseChunkPolicyWrapper]\033[0m "
            f"executing={source_label} "
            f"intervention={float(intervention):.3f} "
            f"chunk_step={self._residual_action_idx}/{self._residual_action_buffer.shape[0]} "
            f"base_step={self._base_action_idx}/{self._base_action_buffer.shape[0]}",
            flush=True,
        )

        self._record_trace_step(
            base_action=base_action,
            residual_action=pred_action - self._normalize_action(base_action.clone()),
            combined_action=pred_action_denormalized,
            applied_action=final_action,
            intervention=intervention.reshape(1),
        )
        self._append_executed_action(
            pred_action if intervention >= 0.5 else self._normalize_action(base_action.clone())
        )

        return final_action

    def reset(self) -> None:
        super().reset()
        self._intervention_obs_history = None
        self._executed_action_history = None
