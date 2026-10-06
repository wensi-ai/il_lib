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
        base_obs = _to_base_policy_obs(obs)
        if self._uses_openpi_client:
            self._last_response = self._client.infer(base_obs)
        else:
            self._last_response = self._client.act(base_obs)
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
    Wrapper that composes a frozen base policy with a residual correction policy.

    The base policy predicts an action chunk and is queried again every
    `base_policy_execution_horizon` steps (defaults to `base_deployed_action_steps`).
    The residual policy is re-inferred whenever the base chunk is refreshed (and
    every `residual_deployed_action_steps` steps otherwise), predicting one
    correction per executed base step, so residual step k is added to base
    chunk step k.

    The base policy is either instantiated locally from `base_policy` +
    `base_policy_ckpt_path`, or queried over websocket when
    `base_policy_use_websocket` is set (or no checkpoint path is given), which
    also covers OpenPI servers such as a pi 0.5 base.
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
        residual_deployed_action_steps: int = 1,  # Residual chunk length actually executed
        base_policy_only: bool = False,
        clamp_combined_arm_action: bool = False,
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
        self._base_action_idx = 0
        self._residual_action_buffer = None
        self._residual_intervention_buffer = None
        self._residual_action_idx = 0
        self._base_action_dim = None
        self._intervention_steps_remaining = 0
        self.base_policy = None  # Created lazily on first act()
        self._base_obs_history = None  # Created lazily once the base policy is attached
        self._base_policy_name = base_policy
        self._base_policy_ckpt_path = base_policy_ckpt_path
        self._base_policy_overrides = base_policy_overrides
        self._base_policy_use_websocket = base_policy_use_websocket or base_policy_ckpt_path is None
        self._base_policy_host = base_policy_host
        self._base_policy_port = base_policy_port
        self._base_policy_api_key = base_policy_api_key
        self._base_policy_device = None
        self._base_policy_only = bool(base_policy_only)
        self._clamp_combined_arm_action = bool(clamp_combined_arm_action)
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
        """Expose per-step policy state (served to clients that record correction data)."""
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

    # ----- Trace logging ---------------------------------------------------------

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
                # Alias so the correction visualizer can render residual traces
                # without a new dataset schema.
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

        demo_group.create_dataset("action", data=_stack("applied_action").numpy())
        for name in (
            "base_action",
            "residual_action",
            "combined_action",
            "applied_action",
            "oracle_action",
            "intervention",
            "is_oracle_active",
            "int_state",
        ):
            policy_group.create_dataset(name, data=_stack(name).numpy())
        time_group.create_dataset("timestamp_ms", data=_stack("timestamp_ms").numpy())
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

    # ----- Base policy -------------------------------------------------------------

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

    def _get_base_obs_history(self) -> deque:
        base_policy = self._get_base_policy()
        if self._base_obs_history is None:
            base_obs_window_size = getattr(base_policy, "num_latest_obs", self.obs_window_size)
            self._base_obs_history = deque(maxlen=base_obs_window_size)
        return self._base_obs_history

    def _refresh_base_action_buffer(self, *, raw_obs: dict, obs: dict) -> None:
        base_policy = self._get_base_policy()
        if self._base_policy_use_websocket:
            self._base_action_buffer = self._infer_websocket_base_action_buffer(
                base_policy=base_policy,
                raw_obs=raw_obs,
            )
        else:
            base_obs = {"obs": self._stack_obs_history(obs, history=self._get_base_obs_history())}
            self._base_action_buffer = base_policy.act(base_obs).squeeze(0)
        self._base_action_buffer = _coerce_action_chunk(self._base_action_buffer)
        self._base_action_dim = int(self._base_action_buffer.shape[-1])
        self._base_action_idx = 0

    def _get_base_action_condition_horizon(self) -> int:
        """Number of base steps the residual conditions on (its chunk length)."""
        return max(
            1,
            int(
                getattr(
                    self.policy,
                    "base_action_horizon",
                    getattr(self.policy, "action_prediction_horizon", 1),
                )
            ),
        )

    def _infer_websocket_base_action_buffer(self, *, base_policy, raw_obs: dict) -> torch.Tensor:
        # A websocket base may return fewer steps per request than the residual
        # conditions on; keep requesting until the buffer covers the horizon.
        target_steps = self._get_base_action_condition_horizon()
        chunks = []
        collected_steps = 0
        while collected_steps < target_steps:
            chunk = _coerce_action_chunk(base_policy.act(raw_obs))
            if chunk.shape[0] == 0:
                raise ValueError("Websocket base policy returned an empty action chunk.")
            chunks.append(chunk)
            collected_steps += int(chunk.shape[0])
        return torch.cat(chunks, dim=0)

    def _needs_base_inference(self) -> bool:
        return (
            self._base_action_buffer is None
            or self._base_action_idx >= self._base_action_buffer.shape[0]
            or self._base_action_idx % self.base_policy_execution_horizon == 0
        )

    # ----- Composition -------------------------------------------------------------

    def _combine_normalized_actions(
        self,
        base_action: torch.Tensor,
        policy_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add continuous residuals to the base action; the gripper may be an absolute command."""
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
        return combined_action, policy_action

    def act(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        """
        1. Get the base action from the buffer (refreshed every base_policy_execution_horizon steps)
        2. Get the residual correction chunk (re-inferred when the base chunk is refreshed)
        3. Combine: final_action = base_action + residual_correction
        """
        raw_obs = any_to_torch(obs, device="cpu")
        obs = self.process_obs(obs=raw_obs)
        residual_obs = {"obs": self._stack_obs_history(obs)}

        # ===== Base Policy: Action Chunking =====
        base_refreshed = self._needs_base_inference()
        if base_refreshed:
            self._refresh_base_action_buffer(raw_obs=raw_obs, obs=obs)
        elif self._base_obs_history is not None:
            self._stack_obs_history(obs, history=self._base_obs_history)

        # Current base action from the buffer (raw joint space)
        base_action_idx = self._base_action_idx
        base_action = self._base_action_buffer[base_action_idx]  # (A,)
        self._base_action_idx += 1
        base_action = self._post_processing_fn(base_action)

        # Normalize the base action to [-1, 1] so it matches the residual's training space
        base_action_normalized = self._normalize_action(base_action.clone())
        residual_horizon = getattr(self.policy, "action_prediction_horizon", 1)
        base_action_chunk = self._base_action_buffer[
            base_action_idx : base_action_idx + residual_horizon
        ]
        if base_action_chunk.shape[0] < residual_horizon:
            pad = base_action_chunk[-1:].repeat(residual_horizon - base_action_chunk.shape[0], 1)
            base_action_chunk = torch.cat([base_action_chunk, pad], dim=0)
        base_action_chunk = self._post_processing_fn(base_action_chunk)
        base_action_chunk = self._normalize_action(base_action_chunk.clone())
        if "base_action" in getattr(self.policy, "_features", set()):
            if residual_horizon > 1:
                residual_obs["obs"]["base_action"] = base_action_chunk.view(
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

            intervention = intervention.squeeze(0)
            if intervention.dim() > 0:
                intervention = intervention[-1]
            self._residual_intervention_buffer = intervention.reshape(1).repeat(
                self._residual_action_buffer.shape[0]
            )
            self._residual_action_idx = 0

        residual_action = self._residual_action_buffer[self._residual_action_idx]
        intervention = self._residual_intervention_buffer[self._residual_action_idx]
        self._residual_action_idx += 1

        # Once the residual decides to intervene, hold the decision for at least
        # the executed chunk so the correction is not cut mid-chunk.
        min_intervention_steps = self._chunk_intervention_min_steps()
        if bool(float(intervention) >= 0.5):
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
        if intervention >= 0.5:
            final_action = combined_action.clone()
            source_label = "\033[1m\033[92mRESIDUAL\033[0m"
        else:
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
        """Reset both policies and their buffers."""
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
        if self.base_policy is not None:
            self.base_policy.reset()
