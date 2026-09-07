import atexit
import logging
import os
import sys
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
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


def _debug_break(label: str) -> None:
    enabled = {
        item.strip()
        for item in os.environ.get("IIIL_PDB", "").split(",")
        if item.strip()
    }
    if enabled.intersection({"1", "all", label, "policy_base"}):
        print(f"\n[IIIL_PDB:{label}] entering pdb", flush=True)
        import pdb; pdb.set_trace()


def _to_numpy_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {k: _to_numpy_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_numpy_tree(v) for v in value]
    return value


def _rgb_to_openpi_array(value) -> np.ndarray:
    image = _to_numpy_tree(value)
    image = np.asarray(image)

    if image.ndim >= 3 and image.shape[-1] not in (3, 4) and image.shape[-3] in (3, 4):
        image = np.moveaxis(image, -3, -1)

    # Drop singleton history/time axes while preserving HWC or BHWC image layout.
    while image.ndim > 4:
        squeezed = False
        for axis in range(image.ndim - 3):
            if image.shape[axis] == 1:
                image = np.squeeze(image, axis=axis)
                squeezed = True
                break
        if not squeezed:
            image = image.reshape((-1, *image.shape[-3:]))
            break

    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    if image.dtype != np.uint8:
        image = image.astype(np.float32, copy=False)
        if image.size:
            image_min = float(np.nanmin(image))
            image_max = float(np.nanmax(image))
            if image_min >= 0.0 and image_max <= 1.0:
                image = image * 255.0
            elif image_min >= -1.0 and image_max <= 1.0:
                image = (image + 1.0) * 127.5
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _to_base_policy_obs(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if isinstance(key, str) and key.endswith("::rgb"):
                result[key] = _rgb_to_openpi_array(item)
            else:
                result[key] = _to_base_policy_obs(item)
        return result
    return _to_numpy_tree(value)


def _coerce_action_chunk(value: Any) -> torch.Tensor:
    if isinstance(value, dict):
        if "actions" in value:
            value = value["actions"]
        elif "action" in value:
            value = value["action"]
        else:
            raise KeyError(
                "Websocket base policy response must include 'actions' or 'action'. "
                f"Got keys: {sorted(value)}"
            )
    action = torch.as_tensor(value, dtype=torch.float32)
    if action.dim() == 1:
        action = action.unsqueeze(0)
    elif action.dim() > 2:
        action = action.reshape(-1, action.shape[-1])
    return action


class WebsocketBasePolicyAdapter:
    """Adapts an openpi-style websocket policy to the local base-policy act API."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        api_key: Optional[str] = None,
    ) -> None:
        self.num_latest_obs = 1
        self._uses_openpi_client = True
        try:
            from openpi_client import websocket_client_policy
        except ModuleNotFoundError:
            workspace_root = Path(__file__).resolve().parents[3]
            openpi_client_src = workspace_root / "openpi" / "packages" / "openpi-client" / "src"
            if openpi_client_src.exists() and str(openpi_client_src) not in sys.path:
                sys.path.insert(0, str(openpi_client_src))
            try:
                from openpi_client import websocket_client_policy
            except ModuleNotFoundError:
                # Local il_lib servers speak OmniGibson's msgpack websocket
                # protocol, so they do not require the optional OpenPI client.
                from omnigibson.learning.utils.network_utils import WebsocketClientPolicy

                self._uses_openpi_client = False
                self._client = WebsocketClientPolicy(
                    host=host,
                    port=port,
                    api_key=api_key,
                    allow_reconnect=True,
                )

        if self._uses_openpi_client:
            self._client = websocket_client_policy.WebsocketClientPolicy(
                host=host,
                port=port,
                api_key=api_key,
            )
        self._last_response = None

    def act(self, obs: dict) -> torch.Tensor:
        _debug_break("base_ws_before_infer")
        base_obs = _to_base_policy_obs(obs)
        if self._uses_openpi_client:
            self._last_response = self._client.infer(base_obs)
        else:
            self._last_response = self._client.act(base_obs)
        _debug_break("base_ws_after_infer")
        return _coerce_action_chunk(self._last_response)

    def get_last_response(self) -> Optional[dict]:
        return self._last_response

    def reset(self) -> None:
        if not self._uses_openpi_client:
            self._client.reset()
            self._last_response = None
            return
        ws = getattr(self._client, "_ws", None)
        packer = getattr(self._client, "_packer", None)
        if ws is not None and packer is not None:
            ws.send(packer.pack({"reset": True}))
        else:
            self._client.reset()
        self._last_response = None


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
        base_policy_ckpt_path: Optional[str] = None,
        base_policy_overrides: Optional[List[str]] = None,
        base_policy_use_websocket: bool = False,
        base_policy_host: str = "127.0.0.1",
        base_policy_port: Optional[int] = None,
        base_policy_api_key: Optional[str] = None,
        base_policy_execution_horizon: Optional[int] = None,
        residual_deployed_action_steps: int = 1,  # Residual policy's action step (usually 1)
        intervention_policy: Optional[str] = None,
        intervention_policy_ckpt_path: Optional[str] = None,
        intervention_policy_overrides: Optional[List[str]] = None,
        intervention_policy_threshold: Optional[float] = None,
        intervention_include_current_action_in_history: bool = False,
        intervention_prop_keys: Optional[List[str]] = None,
        base_policy_only: bool = False,
        clamp_combined_arm_action: bool = False,
        gripper_from_base: bool = False,
        arm_from_base: bool = False,
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
        self.base_policy_execution_horizon = (
            base_deployed_action_steps
            if base_policy_execution_horizon is None
            else int(base_policy_execution_horizon)
        )
        if self.base_policy_execution_horizon < 1:
            raise ValueError("base_policy_execution_horizon must be >= 1.")
        self._base_action_buffer = None  # Will store (T_A, A) from base policy
        self._base_rl_token = None
        self._base_action_idx = 0
        self._residual_action_buffer = None
        self._residual_intervention_buffer = None
        self._residual_action_idx = 0
        self._base_action_dim = None
        self._intervention_steps_remaining = 0
        self.base_policy = None  # Will be set to the base policy from residual_policy.base_policy
        self._base_obs_history = None  # Created lazily once the base policy is attached
        self._base_policy_name = base_policy
        self._base_policy_ckpt_path = base_policy_ckpt_path
        self._base_policy_overrides = base_policy_overrides
        self._base_policy_use_websocket = base_policy_use_websocket or base_policy_ckpt_path is None
        self._base_policy_host = base_policy_host
        self._base_policy_port = base_policy_port
        self._base_policy_api_key = base_policy_api_key
        self._base_policy_device = None
        self.intervention_policy = None
        self._intervention_policy_name = intervention_policy
        self._intervention_policy_ckpt_path = intervention_policy_ckpt_path
        self._intervention_policy_overrides = intervention_policy_overrides
        self._intervention_policy_threshold = intervention_policy_threshold
        self._intervention_include_current_action_in_history = (
            intervention_include_current_action_in_history
        )
        self._base_policy_only = bool(base_policy_only)
        self._clamp_combined_arm_action = bool(clamp_combined_arm_action)
        # Channel ablations: keep the residual's output in the trace, but hand the
        # gripper and/or arm channel of the *applied* action back to the base policy.
        self._gripper_from_base = bool(gripper_from_base)
        self._arm_from_base = bool(arm_from_base)
        if self._gripper_from_base:
            logger.info("ResidualPolicyWrapper ablation: gripper channel taken from the base policy.")
        if self._arm_from_base:
            logger.info("ResidualPolicyWrapper ablation: arm channels taken from the base policy.")
        self._intervention_prop_keys = intervention_prop_keys or [
            "qpos/arm",
            "qpos/gripper",
        ]
        self._intervention_obs_history = None
        self._executed_action_history = None
        self._current_state = None

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

    def _clone_state_tensor(self, value: torch.Tensor) -> torch.Tensor:
        return value.detach().to("cpu", copy=True)

    def _action_l2(self, action: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(action.detach().reshape(-1)).cpu())

    def _chunk_intervention_min_steps(self) -> int:
        policy_min_steps = int(getattr(self.policy, "intervention_min_duration_steps", 1))
        buffer_steps = 1
        if self._residual_action_buffer is not None:
            buffer_steps = int(self._residual_action_buffer.shape[0])
        return max(1, policy_min_steps, int(self.deployed_action_steps), buffer_steps)

    def _set_current_state(
        self,
        *,
        base_action: torch.Tensor,
        predicted_action: torch.Tensor,
        applied_action: torch.Tensor,
        intervention: torch.Tensor,
        residual_action: Optional[torch.Tensor] = None,
    ) -> None:
        intervention_flag = torch.as_tensor(
            float(intervention.detach().reshape(-1)[0] >= 0.5),
            device=applied_action.device,
            dtype=applied_action.dtype,
        )
        state = {
            "base_action": self._clone_state_tensor(base_action),
            "raw_base_policy_action": self._clone_state_tensor(base_action),
            "base_policy_action": self._clone_state_tensor(base_action),
            "base_policy_action_chunk": self._clone_state_tensor(self._base_action_buffer),
            "predicted_action": self._clone_state_tensor(predicted_action),
            "applied_action": self._clone_state_tensor(applied_action),
            "oracle_action": self._clone_state_tensor(applied_action),
            "intervention": self._clone_state_tensor(intervention.reshape(1)),
            "is_oracle_active": self._clone_state_tensor(intervention_flag.reshape(())),
            "int_state": self._clone_state_tensor(
                torch.where(
                    intervention_flag >= 0.5,
                    torch.as_tensor(2.0, device=applied_action.device, dtype=applied_action.dtype),
                    torch.as_tensor(1.0, device=applied_action.device, dtype=applied_action.dtype),
                ).reshape(())
            ),
        }
        if residual_action is not None:
            state["residual_action"] = self._clone_state_tensor(residual_action)
        self._current_state = state

    def get_current_state(self) -> Optional[Dict[str, torch.Tensor]]:
        return self._current_state

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
            if self._base_policy_use_websocket:
                if self._base_policy_port is None:
                    raise ValueError(
                        "base_policy_port must be provided when using a websocket base policy."
                    )
                _debug_break("base_ws_connect")
                self.base_policy = WebsocketBasePolicyAdapter(
                    host=self._base_policy_host,
                    port=int(self._base_policy_port),
                    api_key=self._base_policy_api_key,
                )
                self._base_policy_device = torch.device("cpu")
            else:
                assert self._base_policy_ckpt_path is not None, (
                    "base_policy_ckpt_path must be provided for local residual inference."
                )
                self.base_policy = self._instantiate_arch_module(
                    arch_name=self._base_policy_name,
                    ckpt_path=self._base_policy_ckpt_path,
                    extra_overrides=self._base_policy_overrides,
                )
                self._base_policy_device = self.policy.device
        return self.base_policy

    def _refresh_base_action_buffer(self, *, raw_obs: dict, obs: dict) -> None:
        _debug_break("base_refresh")
        base_policy = self._get_base_policy()
        if self._base_policy_use_websocket:
            self._base_action_buffer = self._infer_websocket_base_action_buffer(
                base_policy=base_policy,
                raw_obs=raw_obs,
            )
            self._base_rl_token = self._extract_websocket_base_rl_token(base_policy)
        else:
            base_obs = {"obs": self._stack_obs_history(obs, history=self._get_base_obs_history())}
            self._base_action_buffer = base_policy.act(base_obs).squeeze(0)
            self._base_rl_token = None
        self._base_action_buffer = _coerce_action_chunk(self._base_action_buffer)
        self._base_action_dim = int(self._base_action_buffer.shape[-1])
        self._base_action_idx = 0

    def _get_base_action_condition_horizon(self) -> int:
        default_horizon = int(
            getattr(
                self.policy,
                "base_action_horizon",
                getattr(self.policy, "action_prediction_horizon", 1),
            )
        )
        intervention_horizon = self._get_intervention_input_steps(
            "base_action_chunk",
            default=default_horizon,
        )
        return max(1, default_horizon, intervention_horizon)

    def _get_websocket_base_buffer_target_steps(self) -> int:
        return self._get_base_action_condition_horizon()

    def _infer_websocket_base_action_buffer(self, *, base_policy, raw_obs: dict) -> torch.Tensor:
        target_steps = self._get_websocket_base_buffer_target_steps()
        chunks = []
        collected_steps = 0
        while collected_steps < target_steps:
            _debug_break("base_ws_chunk_request")
            chunk = _coerce_action_chunk(base_policy.act(raw_obs))
            _debug_break("base_ws_chunk_response")
            if chunk.shape[0] == 0:
                raise ValueError("Websocket base policy returned an empty action chunk.")
            chunks.append(chunk)
            collected_steps += int(chunk.shape[0])
        return torch.cat(chunks, dim=0)

    def _extract_websocket_base_rl_token(self, base_policy) -> Optional[torch.Tensor]:
        if not hasattr(base_policy, "get_last_response"):
            return None
        response = base_policy.get_last_response()
        if not isinstance(response, dict) or "rl_token" not in response:
            return None
        token = torch.as_tensor(response["rl_token"], dtype=torch.float32, device=self.policy.device)
        if token.dim() == 1:
            token = token.view(1, 1, -1)
        elif token.dim() == 2:
            token = token.unsqueeze(1)
        return token

    def _attach_base_rl_token(self, policy_obs: dict) -> None:
        if self._base_rl_token is None:
            return
        if hasattr(self.policy, "_features") and "rl_token" in self.policy._features:
            obs = policy_obs["obs"]
            obs_time = self._infer_obs_time_for_rl_token(obs)
            token = self._base_rl_token.to(device=self.policy.device)
            if token.shape[1] == 1 and obs_time > 1:
                token = token.expand(-1, obs_time, -1)
            elif token.shape[1] != obs_time:
                token = token[:, -1:, :].expand(-1, obs_time, -1)
            obs["rl_token"] = token

    def _infer_obs_time_for_rl_token(self, obs: dict) -> int:
        def iter_times(value, *, key: str = ""):
            if key in {"base_action", "rl_token"}:
                return
            if torch.is_tensor(value):
                if value.dim() >= 3:
                    yield int(value.shape[1])
                return
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    yield from iter_times(child_value, key=child_key)

        for obs_time in iter_times(obs):
            return obs_time
        return int(getattr(self, "obs_window_size", self._base_rl_token.shape[1]))

    def _needs_base_inference(self) -> bool:
        return (
            self._base_action_buffer is None
            or self._base_action_idx >= self._base_action_buffer.shape[0]
            or self._base_action_idx % self.base_policy_execution_horizon == 0
        )

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

    def _ensure_intervention_obs_modalities(self) -> None:
        intervention_policy = self._get_intervention_policy()
        if intervention_policy is None:
            return
        requested_inputs = set(getattr(intervention_policy, "input_keys", [])) | set(
            getattr(intervention_policy, "_features", [])
        )
        visual_obs_types = set(self.visual_obs_types)
        if "rgb" in requested_inputs or "rgbd" in requested_inputs:
            visual_obs_types.add("rgb")
        if "rgbd" in requested_inputs:
            visual_obs_types.add("depth_linear")
        if "pcd" in requested_inputs:
            visual_obs_types.add("pcd")
        self.visual_obs_types = sorted(visual_obs_types)

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
            step_cfg = getattr(intervention_policy, "input_steps", {})
            if key not in step_cfg:
                input_dim = self._get_intervention_extractor_input_dim(key)
                if input_dim is not None and key in {"action_history", "base_action_chunk"}:
                    action_dim = self._get_intervention_action_dim()
                    return max(1, int(input_dim) // int(action_dim))
                return default
            return int(step_cfg[key])
        return int(cfg[key]["steps"])

    def _get_intervention_extractor_input_dim(self, key: str) -> Optional[int]:
        intervention_policy = self._get_intervention_policy()
        feature_extractor = getattr(intervention_policy, "feature_extractor", None)
        extractors = getattr(feature_extractor, "_extractors", {})
        if hasattr(extractors, "get"):
            extractor = extractors.get(key)
        else:
            extractor = extractors[key] if key in extractors else None
        if extractor is None:
            return None

        input_dim = getattr(extractor, "input_dim", None)
        if input_dim is not None:
            return int(input_dim)

        for module in extractor.modules():
            if isinstance(module, torch.nn.Linear):
                return int(module.in_features)
        return None

    def _get_intervention_action_dim(self) -> int:
        intervention_policy = self._get_intervention_policy()
        for source in (intervention_policy, getattr(intervention_policy, "hparams", None)):
            if source is None:
                continue
            action_dim = getattr(source, "action_dim", None)
            if action_dim is None and hasattr(source, "get"):
                action_dim = source.get("action_dim")
            if action_dim is not None:
                return int(action_dim)

        if self._base_action_dim is not None:
            return int(self._base_action_dim)
        action_key_dims = getattr(self.policy, "_action_key_dims", None)
        if action_key_dims:
            return int(sum(action_key_dims.values()))
        return int(self.policy.action_dim)

    def _validate_intervention_flat_input_dim(self, key: str, value: torch.Tensor) -> None:
        input_dim = self._get_intervention_extractor_input_dim(key)
        if input_dim is None:
            return

        flat_dim = int(value.reshape(value.shape[0], -1).shape[-1])
        if flat_dim != int(input_dim):
            raise ValueError(
                f"Intervention input '{key}' has flattened dim {flat_dim}, "
                f"but the loaded probe expects {int(input_dim)}. "
                "This usually means the serving process is using the wrong action-history horizon."
            )

    def _get_intervention_obs_history(self) -> deque:
        intervention_policy = self._get_intervention_policy()
        if intervention_policy is None:
            raise ValueError("intervention_policy not found for residual inference.")
        if self._intervention_obs_history is None:
            obs_steps = max(
                self._get_intervention_input_steps("proprioception", default=1),
                self._get_intervention_input_steps("task", default=1),
                self._get_intervention_input_steps("rgb", default=1),
                self._get_intervention_input_steps("rgbd", default=1),
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

        input_keys = set(getattr(intervention_policy, "input_keys", []))
        feature_keys = set(getattr(intervention_policy, "_features", []))
        requested_inputs = input_keys | feature_keys
        obs_window = self._stack_obs_history(obs, history=self._get_intervention_obs_history())
        inputs = {}
        if "proprioception" in requested_inputs:
            prop_obs = []
            for prop_key in self._intervention_prop_keys:
                group, key = prop_key.split("/", 1)
                prop_obs.append(obs_window[group][key])
            inputs["proprioception"] = torch.cat(prop_obs, dim=-1)
        if "task" in requested_inputs:
            if "task" not in obs_window:
                raise KeyError("Intervention policy expects task input, but task info is unavailable.")
            inputs["task"] = obs_window["task"]
        if "rgb" in requested_inputs:
            rgb_inputs = self._collect_rgb_inputs(obs_window)
            if not rgb_inputs:
                raise KeyError("Intervention policy expects RGB input, but RGB observations are unavailable.")
            inputs["rgb"] = rgb_inputs
        if "rgbd" in requested_inputs:
            rgb = self._collect_rgb_inputs(obs_window)
            depth = self._collect_depth_inputs(obs_window)
            if not rgb or not depth:
                raise KeyError("Intervention policy expects RGBD input, but RGB/depth observations are unavailable.")
            inputs["rgbd"] = {key: {"rgb": rgb[key], "depth": depth[key].unsqueeze(-3)} for key in rgb if key in depth}
        if "base_action_chunk" in requested_inputs:
            steps = self._get_intervention_input_steps(
                "base_action_chunk", default=base_action_chunk.shape[0]
            )
            base_chunk = self._pad_action_sequence(base_action_chunk, steps, pad_with_last=True)
            self._validate_intervention_flat_input_dim("base_action_chunk", base_chunk.unsqueeze(0))
            inputs["base_action_chunk"] = base_chunk.unsqueeze(0)
        if "action_history" in requested_inputs:
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
                    (steps, self._get_intervention_action_dim()),
                    device=base_action_chunk.device,
                    dtype=base_action_chunk.dtype,
                )
            self._validate_intervention_flat_input_dim("action_history", history_tensor.unsqueeze(0))
            inputs["action_history"] = history_tensor.unsqueeze(0)
        return inputs

    def _collect_rgb_inputs(self, obs_window: dict) -> dict[str, torch.Tensor]:
        return {
            key.rsplit("::", 1)[0]: value.float() / 255.0
            for key, value in self._iter_obs_tensors(obs_window)
            if key.endswith("::rgb")
        }

    def _collect_depth_inputs(self, obs_window: dict) -> dict[str, torch.Tensor]:
        return {
            key.rsplit("::", 1)[0]: (value.float() - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
            for key, value in self._iter_obs_tensors(obs_window)
            if "depth" in key
        }

    def _iter_obs_tensors(self, value, prefix: str = ""):
        if torch.is_tensor(value):
            yield prefix, value
            return
        if isinstance(value, dict):
            for key, child in value.items():
                child_key = f"{prefix}/{key}" if prefix else key
                yield from self._iter_obs_tensors(child, child_key)

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

    def _combine_normalized_actions(
        self,
        base_action: torch.Tensor,
        policy_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine continuous residuals while allowing absolute gripper commands."""
        if bool(getattr(self.policy, "_predict_direct_action", False)):
            return policy_action, policy_action - base_action

        combined_action = base_action + policy_action
        if getattr(self.policy, "_gripper_action_mode", "delta") == "absolute":
            for action_key, indices in ACTION_QPOS_INDICES[self.robot_type].items():
                if "gripper" in action_key:
                    combined_action[..., indices] = policy_action[..., indices]
        if self._clamp_combined_arm_action:
            for action_key, indices in ACTION_QPOS_INDICES[self.robot_type].items():
                if "gripper" not in action_key:
                    combined_action[..., indices] = combined_action[..., indices].clamp(
                        -1.0, 1.0
                    )
        gripper_from_base = bool(getattr(self, "_gripper_from_base", False))
        arm_from_base = bool(getattr(self, "_arm_from_base", False))
        if gripper_from_base or arm_from_base:
            for action_key, indices in ACTION_QPOS_INDICES[self.robot_type].items():
                is_gripper = "gripper" in action_key
                if (is_gripper and gripper_from_base) or (not is_gripper and arm_from_base):
                    combined_action[..., indices] = base_action[..., indices]
        return combined_action, policy_action
    
    def act(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        """
        Coordinated action generation:
        1. Get base action from buffer (refresh every base_policy_execution_horizon steps)
        2. Get residual correction from residual policy
        3. Combine: final_action = base_action + residual_correction
        """
        self._ensure_intervention_obs_modalities()
        raw_obs = any_to_torch(obs, device="cpu")
        obs = self.process_obs(obs=raw_obs)
        residual_obs = {"obs": self._stack_obs_history(obs)}
        _debug_break("wrapper_act")

        # ===== Base Policy: Action Chunking =====
        base_refreshed = self._needs_base_inference()
        if base_refreshed:
            self._refresh_base_action_buffer(raw_obs=raw_obs, obs=obs)
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

        if self._base_policy_only:
            residual_action = torch.zeros_like(base_action_normalized)
            intervention = torch.zeros(1, device=base_action.device, dtype=base_action.dtype)
            print(
                "\033[1m\033[96m[ResidualPolicyWrapper]\033[0m "
                "executing=\033[1m\033[94mBASE-ONLY\033[0m "
                f"intervention={float(intervention):.3f} "
                f"residual_l2={self._action_l2(residual_action):.4f} "
                f"base_step={self._base_action_idx}/{self._base_action_buffer.shape[0]}",
                flush=True,
            )
            self._record_trace_step(
                base_action=base_action,
                residual_action=residual_action,
                combined_action=base_action,
                applied_action=base_action,
                intervention=intervention.reshape(1),
            )
            self._set_current_state(
                base_action=base_action,
                residual_action=residual_action,
                predicted_action=base_action,
                applied_action=base_action,
                intervention=intervention.reshape(1),
            )
            self._append_executed_action(base_action_normalized)
            return base_action.clone()

        # ===== Residual Policy: Correction Chunk =====
        need_residual_inference = (
            base_refreshed
            or self._residual_action_buffer is None
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

        min_intervention_steps = self._chunk_intervention_min_steps()
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
        combined_normalized, residual_for_logging = self._combine_normalized_actions(
            base_action_normalized,
            residual_action,
        )
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
            f"residual_l2={self._action_l2(residual_for_logging):.4f} "
            f"chunk_step={self._residual_action_idx}/{self._residual_action_buffer.shape[0]} "
            f"base_step={self._base_action_idx}/{self._base_action_buffer.shape[0]}",
            flush=True,
        )

        self._record_trace_step(
            base_action=base_action,
            residual_action=residual_for_logging,
            combined_action=combined_action,
            applied_action=final_action,
            intervention=intervention.reshape(1),
        )
        self._set_current_state(
            base_action=base_action,
            residual_action=residual_for_logging,
            predicted_action=combined_action,
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
        self._current_state = None
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
            step_cfg = getattr(intervention_policy, "input_steps", {})
            if key not in step_cfg:
                input_dim = self._get_intervention_extractor_input_dim(key)
                if input_dim is not None and key in {"action_history", "base_action_chunk"}:
                    action_dim = self._get_intervention_action_dim()
                    return max(1, int(input_dim) // int(action_dim))
                return default
            return int(step_cfg[key])
        return int(cfg[key]["steps"])

    def _validate_intervention_flat_input_dim(self, key: str, value: torch.Tensor) -> None:
        input_dim = self._get_intervention_extractor_input_dim(key)
        if input_dim is None:
            return

        flat_dim = int(value.reshape(value.shape[0], -1).shape[-1])
        if flat_dim != int(input_dim):
            raise ValueError(
                f"Intervention input '{key}' has flattened dim {flat_dim}, "
                f"but the loaded probe expects {int(input_dim)}. "
                "This usually means the serving process is using the wrong action-history horizon."
            )

    def _get_intervention_obs_history(self) -> deque:
        intervention_policy = self._get_intervention_policy()
        if intervention_policy is None:
            raise ValueError("intervention_policy not found for base chunk inference.")
        if self._intervention_obs_history is None:
            obs_steps = max(
                self._get_intervention_input_steps("proprioception", default=1),
                self._get_intervention_input_steps("task", default=1),
                self._get_intervention_input_steps("rgb", default=1),
                self._get_intervention_input_steps("rgbd", default=1),
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

        input_keys = set(getattr(intervention_policy, "input_keys", []))
        feature_keys = set(getattr(intervention_policy, "_features", []))
        requested_inputs = input_keys | feature_keys
        obs_window = self._stack_obs_history(obs, history=self._get_intervention_obs_history())
        inputs = {}
        if "proprioception" in requested_inputs:
            prop_obs = []
            for prop_key in self._intervention_prop_keys:
                group, key = prop_key.split("/", 1)
                prop_obs.append(obs_window[group][key])
            inputs["proprioception"] = torch.cat(prop_obs, dim=-1)
        if "task" in requested_inputs:
            if "task" not in obs_window:
                raise KeyError("Intervention policy expects task input, but task info is unavailable.")
            inputs["task"] = obs_window["task"]
        if "rgb" in requested_inputs:
            rgb_inputs = self._collect_rgb_inputs(obs_window)
            if not rgb_inputs:
                raise KeyError("Intervention policy expects RGB input, but RGB observations are unavailable.")
            inputs["rgb"] = rgb_inputs
        if "rgbd" in requested_inputs:
            rgb = self._collect_rgb_inputs(obs_window)
            depth = self._collect_depth_inputs(obs_window)
            if not rgb or not depth:
                raise KeyError("Intervention policy expects RGBD input, but RGB/depth observations are unavailable.")
            inputs["rgbd"] = {key: {"rgb": rgb[key], "depth": depth[key].unsqueeze(-3)} for key in rgb if key in depth}
        if "base_action_chunk" in requested_inputs:
            steps = self._get_intervention_input_steps("base_action_chunk", default=base_action_chunk.shape[0])
            base_chunk = self._pad_action_sequence(base_action_chunk, steps, pad_with_last=True)
            self._validate_intervention_flat_input_dim("base_action_chunk", base_chunk.unsqueeze(0))
            inputs["base_action_chunk"] = base_chunk.unsqueeze(0)
        if "action_history" in requested_inputs:
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
                    (steps, self._get_intervention_action_dim()),
                    device=base_action_chunk.device,
                    dtype=base_action_chunk.dtype,
                )
            self._validate_intervention_flat_input_dim("action_history", history_tensor.unsqueeze(0))
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
        self._ensure_intervention_obs_modalities()
        raw_obs = any_to_torch(obs, device="cpu")
        obs = self.process_obs(obs=raw_obs)
        policy_obs = {"obs": self._stack_obs_history(obs)}

        base_refreshed = self._needs_base_inference()
        if base_refreshed:
            self._refresh_base_action_buffer(raw_obs=raw_obs, obs=obs)
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
        self._attach_base_rl_token(policy_obs)

        if self._base_policy_only:
            pred_action = self._normalize_action(base_action.clone())
            pred_action_denormalized = base_action.clone()
            residual_action = torch.zeros_like(pred_action)
            intervention = torch.zeros(1, device=base_action.device, dtype=base_action.dtype)
            policy_intervention_chunk = torch.zeros(
                max(1, int(self.deployed_action_steps)),
                device=base_action.device,
                dtype=base_action.dtype,
            )
            print(
                "\033[1m\033[96m[BaseChunkPolicyWrapper]\033[0m "
                "executing=\033[1m\033[94mBASE-ONLY\033[0m "
                f"intervention={float(intervention):.3f} "
                f"residual_l2={self._action_l2(residual_action):.4f} "
                f"base_step={self._base_action_idx}/{self._base_action_buffer.shape[0]}",
                flush=True,
            )
            self._record_trace_step(
                base_action=base_action,
                residual_action=residual_action,
                combined_action=pred_action_denormalized,
                applied_action=base_action,
                intervention=intervention.reshape(1),
            )
            self._set_current_state(
                base_action=base_action,
                residual_action=residual_action,
                predicted_action=pred_action_denormalized,
                applied_action=base_action,
                intervention=intervention.reshape(1),
            )
            self._current_state["policy_action"] = self._clone_state_tensor(base_action)
            self._current_state["policy_intervention_chunk"] = self._clone_state_tensor(
                policy_intervention_chunk
            )
            self._append_executed_action(pred_action)
            return base_action.clone()

        need_policy_inference = (
            base_refreshed
            or self._residual_action_buffer is None
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
        if self._residual_intervention_buffer is not None:
            policy_intervention_chunk = self._residual_intervention_buffer
        else:
            policy_intervention_chunk = intervention.reshape(1).repeat(
                self._residual_action_buffer.shape[0]
            )

        min_intervention_steps = self._chunk_intervention_min_steps()
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
            f"residual_l2={self._action_l2(pred_action - self._normalize_action(base_action.clone())):.4f} "
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
        self._set_current_state(
            base_action=base_action,
            residual_action=pred_action - self._normalize_action(base_action.clone()),
            predicted_action=pred_action_denormalized,
            applied_action=final_action,
            intervention=intervention.reshape(1),
        )
        self._current_state["policy_action"] = self._clone_state_tensor(final_action)
        self._current_state["policy_intervention_chunk"] = self._clone_state_tensor(
            policy_intervention_chunk
        )
        self._append_executed_action(
            pred_action if intervention >= 0.5 else self._normalize_action(base_action.clone())
        )

        return final_action

    def reset(self) -> None:
        super().reset()
        self._intervention_obs_history = None
        self._executed_action_history = None


class GatedPolicyWrapper(ResidualPolicyWrapper):
    """
    Wrapper for policies that internally blend a base-policy action chunk with
    a learned diffusion action chunk. There is no intervention policy/head.
    """

    def act(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        raw_obs = any_to_torch(obs, device="cpu")
        obs = self.process_obs(obs=raw_obs)
        policy_obs = {"obs": self._stack_obs_history(obs)}

        base_refreshed = self._needs_base_inference()
        if base_refreshed:
            self._refresh_base_action_buffer(raw_obs=raw_obs, obs=obs)
        elif self._base_obs_history is not None:
            self._stack_obs_history(obs, history=self._base_obs_history)

        base_action_idx = self._base_action_idx
        base_action = self._post_processing_fn(self._base_action_buffer[base_action_idx])
        self._base_action_idx += 1

        action_horizon = getattr(self.policy, "action_prediction_horizon", 1)
        base_action_horizon = getattr(self.policy, "base_action_horizon", action_horizon)
        base_action_chunk = self._base_action_buffer[
            base_action_idx : base_action_idx + base_action_horizon
        ]
        if base_action_chunk.shape[0] < base_action_horizon:
            pad = base_action_chunk[-1:].repeat(base_action_horizon - base_action_chunk.shape[0], 1)
            base_action_chunk = torch.cat([base_action_chunk, pad], dim=0)
        base_action_chunk = self._post_processing_fn(base_action_chunk)
        normalized_base_chunk = self._normalize_action(base_action_chunk.clone())
        policy_obs["obs"]["base_action"] = normalized_base_chunk.view(1, 1, base_action_horizon, -1)
        self._attach_base_rl_token(policy_obs)

        if self._base_policy_only:
            pred_action = self._normalize_action(base_action.clone())
            residual_action = torch.zeros_like(pred_action)
            gate_proxy = torch.zeros(1, device=base_action.device, dtype=base_action.dtype)
            print(
                "\033[1m\033[96m[GatedPolicyWrapper]\033[0m "
                "executing=\033[1m\033[94mBASE-ONLY\033[0m "
                f"residual_l2={self._action_l2(residual_action):.4f} "
                f"base_step={self._base_action_idx}/{self._base_action_buffer.shape[0]}",
                flush=True,
            )
            self._record_trace_step(
                base_action=base_action,
                residual_action=residual_action,
                combined_action=base_action,
                applied_action=base_action,
                intervention=gate_proxy,
            )
            self._set_current_state(
                base_action=base_action,
                residual_action=residual_action,
                predicted_action=base_action,
                applied_action=base_action,
                intervention=gate_proxy,
            )
            return base_action.clone()

        need_policy_inference = (
            base_refreshed
            or self._residual_action_buffer is None
            or self._residual_action_idx >= self._residual_action_buffer.shape[0]
            or self._residual_action_idx % self.deployed_action_steps == 0
        )
        if need_policy_inference:
            pred_action = self.policy.act(policy_obs["obs"]).squeeze(0)
            if pred_action.dim() == 3:
                pred_action = pred_action[-1]
            elif pred_action.dim() != 2:
                raise ValueError(
                    "Gated policy must return an action chunk with shape (H, A) "
                    f"or (T, H, A), got {pred_action.shape}."
                )
            self._residual_action_buffer = pred_action
            self._residual_action_idx = 0

        pred_action = self._residual_action_buffer[self._residual_action_idx].clamp(-1.0, 1.0)
        self._residual_action_idx += 1

        final_action = self._denormalize_action(pred_action.clone())
        base_action_normalized = self._normalize_action(base_action.clone())
        gate_proxy = torch.ones(1, device=pred_action.device, dtype=pred_action.dtype)

        print(
            "\033[1m\033[96m[GatedPolicyWrapper]\033[0m "
            f"executing=\033[1m\033[92mGATED\033[0m "
            f"residual_l2={self._action_l2(pred_action - base_action_normalized):.4f} "
            f"chunk_step={self._residual_action_idx}/{self._residual_action_buffer.shape[0]} "
            f"base_step={self._base_action_idx}/{self._base_action_buffer.shape[0]}",
            flush=True,
        )

        self._record_trace_step(
            base_action=base_action,
            residual_action=pred_action - base_action_normalized,
            combined_action=final_action,
            applied_action=final_action,
            intervention=gate_proxy,
        )
        self._set_current_state(
            base_action=base_action,
            residual_action=pred_action - base_action_normalized,
            predicted_action=final_action,
            applied_action=final_action,
            intervention=gate_proxy,
        )

        return final_action
