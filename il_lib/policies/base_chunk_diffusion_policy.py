from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from il_lib.nn.distributions import CategoricalNet
from il_lib.nn.features import SimpleFeatureFusion
from il_lib.optim import CosineScheduleFunction
from il_lib.policies.base_chunk_policy import _AlwaysInterveneDistribution
from il_lib.utils.training_utils import freeze_params, unfreeze_params
from il_lib.utils.array_tensor_utils import any_concat, get_batch_size
from omegaconf import DictConfig

try:
    from il_lib.policies.policy_base import BasePolicy
except ModuleNotFoundError as exc:
    if exc.name != "omnigibson":
        raise
    from pytorch_lightning import LightningModule as _LightningModule

    class BasePolicy(_LightningModule):
        """Minimal offline fallback for base-chunk training without OmniGibson."""

        def __init__(self, *args, **kwargs):
            super().__init__()

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
            loss, log_dict, batch_size = self.policy_evaluation_step(*args, **kwargs)
            log_dict = {f"val/{k}": v for k, v in log_dict.items()}
            log_dict["val/loss"] = loss
            self.log_dict(
                log_dict,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
            return log_dict

        def test_step(self, *args, **kwargs):
            return None

try:
    from omnigibson.learning.utils.obs_utils import MAX_DEPTH, MIN_DEPTH
except ModuleNotFoundError as exc:
    if exc.name != "omnigibson":
        raise
    MIN_DEPTH = 0.0
    MAX_DEPTH = 10.0


class BaseChunkDiffusionPolicy(BasePolicy):
    """
    Diffusion policy that keeps the base-chunk policy contract:
    condition on an optional base action chunk and optionally emit intervention logits.
    """

    is_sequence_policy = True

    def __init__(
        self,
        *args,
        prop_dim: int,
        prop_keys: List[str],
        action_keys: List[str],
        action_key_dims: dict[str, int],
        feature_extractors: Dict[str, DictConfig],
        feature_fusion_hidden_depth: int = 1,
        feature_fusion_hidden_dim: int = 256,
        feature_fusion_output_dim: int = 256,
        feature_fusion_activation: str = "relu",
        feature_fusion_add_input_activation: bool = False,
        feature_fusion_add_output_activation: bool = False,
        backbone: DictConfig,
        action_dim: int,
        action_prediction_horizon: int,
        base_action_horizon: Optional[int] = None,
        deployed_action_steps: Optional[int] = None,
        noise_scheduler: DictConfig = None,
        noise_scheduler_step_kwargs: Optional[dict] = None,
        num_denoise_steps_per_inference: int = 16,
        num_latest_obs: int = 1,
        base_action_mask_ratio: float = 0.0,
        zero_mask_base_action_at_inference: bool = False,
        use_intervention_head: bool = False,
        intervention_head_hidden_dim: int = 128,
        intervention_head_hidden_depth: int = 1,
        intervention_head_activation: str = "relu",
        intervention_min_duration_steps: int = 1,
        force_intervention_on_at_inference: bool = False,
        deterministic_inference: bool = True,
        intervention_loss_weight: float = 1.0,
        action_loss_on_intervention_only: bool = True,
        exclude_pre_intervention_from_action_loss: bool = False,
        training_loss_mode: Optional[str] = None,
        freeze_feature_extractor: bool = False,
        freeze_backbone: bool = False,
        lr: float = 1e-4,
        use_cosine_lr: bool = True,
        lr_warmup_steps: Optional[int] = None,
        lr_cosine_steps: Optional[int] = None,
        lr_cosine_min: Optional[float] = None,
        lr_layer_decay: float = 1.0,
        optimizer: str = "adamw",
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if noise_scheduler is None:
            raise ValueError("noise_scheduler must be provided.")
        if not 0.0 <= base_action_mask_ratio <= 1.0:
            raise ValueError("base_action_mask_ratio must be in [0, 1].")
        if intervention_min_duration_steps < 1:
            raise ValueError("intervention_min_duration_steps must be >= 1.")
        assert sum(action_key_dims.values()) == action_dim
        assert set(action_keys) == set(action_key_dims.keys())

        self._prop_keys = prop_keys
        self._prop_dim = prop_dim
        self._action_keys = action_keys
        self._action_key_dims = action_key_dims
        self._features = set(feature_extractors.keys())
        self.action_dim = action_dim
        self.action_prediction_horizon = action_prediction_horizon
        self.base_action_horizon = (
            action_prediction_horizon
            if base_action_horizon is None
            else base_action_horizon
        )
        self.deployed_action_steps = (
            action_prediction_horizon
            if deployed_action_steps is None
            else deployed_action_steps
        )
        if self.base_action_horizon < 1:
            raise ValueError("base_action_horizon must be >= 1.")
        self.num_latest_obs = num_latest_obs
        self._base_action_mask_ratio = base_action_mask_ratio
        self._zero_mask_base_action_at_inference = zero_mask_base_action_at_inference

        self.feature_extractor = SimpleFeatureFusion(
            extractors={k: instantiate(v) for k, v in feature_extractors.items()},
            hidden_depth=feature_fusion_hidden_depth,
            hidden_dim=feature_fusion_hidden_dim,
            output_dim=feature_fusion_output_dim,
            activation=feature_fusion_activation,
            add_input_activation=feature_fusion_add_input_activation,
            add_output_activation=feature_fusion_add_output_activation,
        )
        self.backbone = instantiate(backbone)
        self.noise_scheduler = instantiate(noise_scheduler)
        self.noise_scheduler_step_kwargs = noise_scheduler_step_kwargs or {}
        self.num_denoise_steps_per_inference = num_denoise_steps_per_inference

        self._use_intervention_head = use_intervention_head
        self.intervention_min_duration_steps = intervention_min_duration_steps
        self.intervention_head = None
        if self._use_intervention_head:
            self.intervention_head = CategoricalNet(
                feature_fusion_output_dim,
                action_dim=2,
                hidden_dim=intervention_head_hidden_dim,
                hidden_depth=intervention_head_hidden_depth,
                activation=intervention_head_activation,
            )
        self._force_intervention_on_at_inference = force_intervention_on_at_inference
        self._deterministic_inference = deterministic_inference
        self._intervention_loss_weight = intervention_loss_weight
        self._action_loss_on_intervention_only = action_loss_on_intervention_only
        self._exclude_pre_intervention_from_action_loss = exclude_pre_intervention_from_action_loss
        self._training_loss_mode = training_loss_mode or ("joint" if use_intervention_head else "action")
        self._freeze_feature_extractor = freeze_feature_extractor
        self._freeze_backbone = freeze_backbone
        self._validate_training_loss_mode()
        self._apply_freeze_config()

        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = lr_layer_decay
        self.optimizer = optimizer
        self.weight_decay = weight_decay

        self.save_hyperparameters()

    def _validate_training_loss_mode(self) -> None:
        valid_modes = {"action", "intervention", "joint"}
        if self._training_loss_mode not in valid_modes:
            raise ValueError(
                f"training_loss_mode must be one of {sorted(valid_modes)}, "
                f"got {self._training_loss_mode!r}."
            )
        if self._training_loss_mode in {"intervention", "joint"} and not self._use_intervention_head:
            raise ValueError(
                f"training_loss_mode={self._training_loss_mode!r} requires "
                "use_intervention_head=True."
            )

    def _apply_freeze_config(self) -> None:
        (freeze_params if self._freeze_feature_extractor else unfreeze_params)(self.feature_extractor)
        (freeze_params if self._freeze_backbone else unfreeze_params)(self.backbone)
        if self.intervention_head is not None:
            unfreeze_params(self.intervention_head)

    def set_training_stage(
        self,
        *,
        loss_mode: Optional[str] = None,
        freeze_feature_extractor: Optional[bool] = None,
        freeze_backbone: Optional[bool] = None,
        intervention_loss_weight: Optional[float] = None,
        action_loss_on_intervention_only: Optional[bool] = None,
        exclude_pre_intervention_from_action_loss: Optional[bool] = None,
        **kwargs,
    ) -> None:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unknown BaseChunkDiffusionPolicy stage option(s): {unknown}")
        if loss_mode is not None:
            self._training_loss_mode = loss_mode
            self._validate_training_loss_mode()
        if freeze_feature_extractor is not None:
            self._freeze_feature_extractor = bool(freeze_feature_extractor)
        if freeze_backbone is not None:
            self._freeze_backbone = bool(freeze_backbone)
        if intervention_loss_weight is not None:
            self._intervention_loss_weight = float(intervention_loss_weight)
        if action_loss_on_intervention_only is not None:
            self._action_loss_on_intervention_only = bool(action_loss_on_intervention_only)
        if exclude_pre_intervention_from_action_loss is not None:
            self._exclude_pre_intervention_from_action_loss = bool(exclude_pre_intervention_from_action_loss)
        self._apply_freeze_config()

    def enforce_training_stage(self) -> None:
        self._apply_freeze_config()

    def forward(self, obs, noisy_traj, diffusion_timesteps, compute_intervention: bool = True):
        obs_feature = self._encode_obs(obs)
        if self.training and self._freeze_backbone:
            with torch.no_grad():
                pred = self.backbone(
                    sample=noisy_traj,
                    timestep=diffusion_timesteps,
                    cond=obs_feature,
                )
        else:
            pred = self.backbone(
                sample=noisy_traj,
                timestep=diffusion_timesteps,
                cond=obs_feature,
            )
        intervention_feature = obs_feature[:, -1:]
        if self._use_intervention_head and compute_intervention:
            intervention_dist = self._intervention_dist_from_feature(intervention_feature)
        else:
            intervention_dist = _AlwaysInterveneDistribution(intervention_feature)
        return pred, intervention_dist

    @torch.no_grad()
    def act(self, obs, deterministic=None):
        del deterministic
        obs = self.process_data(obs, extract_action=False)
        B = get_batch_size(obs, strict=True)
        noisy_traj = torch.randn(
            size=(B, self.action_prediction_horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_denoise_steps_per_inference)

        intervention_dist = None
        for t in scheduler.timesteps:
            pred, intervention_dist = self.forward(obs, noisy_traj, t)
            noisy_traj = scheduler.step(
                pred, t, noisy_traj, **self.noise_scheduler_step_kwargs
            ).prev_sample

        action = noisy_traj.clone()
        intervention = intervention_dist.mode()
        if self._force_intervention_on_at_inference:
            intervention = torch.ones_like(intervention)
        return action, intervention

    def reset(self) -> None:
        pass

    def policy_training_step(self, batch, batch_idx):
        return self._forward_step(batch, batch_idx, is_train=True)

    def policy_evaluation_step(self, batch, batch_idx):
        return self._forward_step(batch, batch_idx, is_train=False)

    def _forward_step(self, batch, batch_idx, is_train: bool):
        batch["actions"] = any_concat(
            [batch["actions"][k] for k in self._action_keys], dim=-1
        )
        B = batch["actions"].shape[0]
        batch = self.process_data(batch, extract_action=True)

        pad_mask = batch.pop("masks")
        int_state = batch.pop("int_state")
        intervention_mask = int_state == 2
        batch.pop("oracle_action")
        target_action = batch.pop("actions")

        if target_action.dim() != 4:
            raise ValueError(
                "BaseChunkDiffusionPolicy expected chunked targets with shape "
                f"(B, T, {self.action_prediction_horizon}, A), but got {target_action.shape}."
            )
        if target_action.shape[-2:] != (self.action_prediction_horizon, self.action_dim):
            raise ValueError(
                "Target action chunk shape mismatch: expected trailing shape "
                f"({self.action_prediction_horizon}, {self.action_dim}), got {target_action.shape[-2:]}."
            )

        target_action = target_action[:, -1]
        chunk_mask = pad_mask[:, -1]
        current_intervention = self._current_intervention(intervention_mask)
        target_int_state = int_state[:, -1]
        if self._action_loss_on_intervention_only:
            chunk_mask = chunk_mask & current_intervention[:, -1:].expand_as(chunk_mask)
        elif self._exclude_pre_intervention_from_action_loss:
            chunk_mask = chunk_mask & (target_int_state != 1)

        real_batch_size = chunk_mask.sum().clamp_min(1)
        action_loss = target_action.new_zeros(())
        intervention_dist = None
        if self._training_loss_mode in {"action", "joint"}:
            noise = torch.randn(target_action.shape, device=target_action.device)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B,),
                device=target_action.device,
            ).long()
            noisy_trajs = self.noise_scheduler.add_noise(target_action, noise, timesteps)
            pred, intervention_dist = self.forward(
                obs=batch,
                noisy_traj=noisy_trajs,
                diffusion_timesteps=timesteps,
                compute_intervention=self._training_loss_mode == "joint",
            )
            raw_action_loss = F.mse_loss(pred, noise, reduction="none").mean(dim=-1)
            action_loss = raw_action_loss * chunk_mask
            action_loss = action_loss.sum() / real_batch_size

        if self._use_intervention_head and self._training_loss_mode in {"intervention", "joint"}:
            if intervention_dist is None:
                intervention_dist = self._intervention_dist_from_obs(batch)
            intervention_loss_mask = self._current_mask(pad_mask)
            intervention_target = self._current_intervention(intervention_mask)
            raw_intervention_loss = intervention_dist.imitation_loss(
                intervention_target.long(),
                reduction="none",
            ).reshape(intervention_loss_mask.shape)
            intervention_loss = raw_intervention_loss * intervention_loss_mask
            intervention_acc = intervention_dist.imitation_accuracy(
                intervention_target.long(),
                mask=intervention_loss_mask,
            )
            intervention_loss = (
                intervention_loss.sum() / intervention_loss_mask.sum().clamp_min(1)
            )
            if self._training_loss_mode == "intervention":
                loss = self._intervention_loss_weight * intervention_loss
                real_batch_size = intervention_loss_mask.sum().clamp_min(1)
            else:
                loss = action_loss + self._intervention_loss_weight * intervention_loss
        else:
            intervention_loss = action_loss.new_zeros(())
            intervention_acc = action_loss.new_ones(())
            loss = action_loss
        loss = loss + self._unused_parameter_anchor(loss)

        log_dict = {
            "diffusion_loss": action_loss,
            "action_loss": action_loss,
            "intervention_loss": intervention_loss,
            "intervention_acc": intervention_acc,
        }
        if not is_train:
            pred_action = self._sample_action(batch)
            target_action_for_metrics = target_action
            full_future_mask = pad_mask[:, -1]
            if self._action_loss_on_intervention_only:
                full_future_mask = full_future_mask & current_intervention[:, -1:].expand_as(full_future_mask)
            elif self._exclude_pre_intervention_from_action_loss:
                full_future_mask = full_future_mask & (target_int_state != 1)

            l1_full_future_horizon = torch.abs(
                pred_action - target_action_for_metrics
            ).mean(dim=-1)
            l1_full_future_horizon = l1_full_future_horizon * full_future_mask
            l1_full_future_horizon = (
                l1_full_future_horizon.sum() / full_future_mask.sum().clamp_min(1)
            )

            deployed_steps = min(self.deployed_action_steps, target_action_for_metrics.shape[-2])
            pred_action_to_deploy = pred_action[:, :deployed_steps]
            target_action_to_deploy = target_action_for_metrics[:, :deployed_steps]
            deployed_mask = full_future_mask[:, :deployed_steps]
            l1_deployed_steps_only = torch.abs(
                pred_action_to_deploy - target_action_to_deploy
            ).mean(dim=-1)
            l1_deployed_steps_only = l1_deployed_steps_only * deployed_mask
            l1_deployed_steps_only = (
                l1_deployed_steps_only.sum() / deployed_mask.sum().clamp_min(1)
            )

            log_dict["l1"] = l1_deployed_steps_only
            log_dict["l1_full_future_horizon"] = l1_full_future_horizon
            log_dict["l1_deployed_steps_only"] = l1_deployed_steps_only
        return loss, log_dict, real_batch_size

    def _intervention_dist_from_feature(self, feature: torch.Tensor):
        return self.intervention_head(feature)

    def _intervention_dist_from_obs(self, obs: dict):
        obs_feature = self._encode_obs(obs)
        return self._intervention_dist_from_feature(obs_feature[:, -1:])

    def _unused_parameter_anchor(self, loss: torch.Tensor) -> torch.Tensor:
        modules = []
        if self._training_loss_mode == "action" and self.intervention_head is not None:
            modules.append(self.intervention_head)
        elif self._training_loss_mode == "intervention":
            modules.append(self.backbone)
        anchor = None
        for module in modules:
            for param in module.parameters():
                if not param.requires_grad:
                    continue
                term = param.sum() * 0.0
                anchor = term if anchor is None else anchor + term
        if anchor is None:
            return loss.new_zeros(())
        return anchor

    def _encode_obs(self, obs: dict) -> torch.Tensor:
        prop_obs = []
        for prop_key in self._prop_keys:
            if "/" in prop_key:
                group, key = prop_key.split("/")
                prop_obs.append(obs[group][key])
            else:
                prop_obs.append(obs[prop_key])
        obs = dict(obs)
        obs["proprioception"] = torch.cat(prop_obs, dim=-1)
        obs_time = obs["proprioception"].shape[1]

        if "base_action" in obs:
            obs["base_action"] = self._format_base_action_chunk(obs["base_action"])
            obs["base_action"] = self._maybe_mask_base_action(obs["base_action"])
            if obs["base_action"].shape[1] == 1 and obs_time > 1:
                obs["base_action"] = obs["base_action"].expand(-1, obs_time, -1)

        obs = {k: obs[k] for k in self._features}
        if self.training and self._freeze_feature_extractor:
            with torch.no_grad():
                return self.feature_extractor(obs)
        return self.feature_extractor(obs)

    @torch.no_grad()
    def _sample_action(self, obs: dict) -> torch.Tensor:
        B = get_batch_size(obs, strict=True)
        noisy_traj = torch.randn(
            size=(B, self.action_prediction_horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_denoise_steps_per_inference)
        for t in scheduler.timesteps:
            pred, _ = self.forward(obs, noisy_traj, t, compute_intervention=False)
            noisy_traj = scheduler.step(
                pred, t, noisy_traj, **self.noise_scheduler_step_kwargs
            ).prev_sample
        return noisy_traj

    def _maybe_mask_base_action(self, base_action: torch.Tensor) -> torch.Tensor:
        if not self.training:
            if self._zero_mask_base_action_at_inference:
                return torch.zeros_like(base_action)
            return base_action
        if self._base_action_mask_ratio <= 0.0:
            return base_action
        mask_shape = base_action.shape[:-1]
        keep_mask = torch.rand(mask_shape, device=base_action.device) >= self._base_action_mask_ratio
        return base_action * keep_mask.unsqueeze(-1).to(base_action.dtype)

    def _format_action_chunk(self, action: torch.Tensor) -> torch.Tensor:
        if action.dim() >= 4:
            return action.reshape(*action.shape[:-2], action.shape[-2] * action.shape[-1])
        return action

    def _format_base_action_chunk(self, action: torch.Tensor) -> torch.Tensor:
        if action.dim() >= 4:
            action = action[..., : self.base_action_horizon, :]
            return action.reshape(*action.shape[:-2], action.shape[-2] * action.shape[-1])
        full_chunk_dim = self.action_prediction_horizon * self.action_dim
        if action.shape[-1] == full_chunk_dim:
            action = action.reshape(
                *action.shape[:-1],
                self.action_prediction_horizon,
                self.action_dim,
            )
            action = action[..., : self.base_action_horizon, :]
            return action.reshape(*action.shape[:-2], action.shape[-2] * action.shape[-1])
        return action

    def _current_mask(self, pad_mask: torch.Tensor) -> torch.Tensor:
        if pad_mask.dim() >= 3:
            return pad_mask[..., 0]
        return pad_mask

    def _current_intervention(self, intervention_mask: torch.Tensor) -> torch.Tensor:
        if intervention_mask.dim() >= 3:
            return intervention_mask[..., 0]
        return intervention_mask

    def process_data(self, data_batch: dict, extract_action: bool = False) -> Any:
        obs_batch = data_batch["obs"] if "obs" in data_batch else data_batch
        data = {"qpos": obs_batch["qpos"], "eef": obs_batch["eef"]}
        if "odom" in obs_batch:
            data["odom"] = obs_batch["odom"]
        if "rgb" in self._features:
            data["rgb"] = {
                k.rsplit("::", 1)[0]: obs_batch[k].float() / 255.0
                for k in obs_batch
                if "rgb" in k
            }
        if "rgbd" in self._features:
            rgb = {
                k.rsplit("::", 1)[0]: obs_batch[k].float() / 255.0
                for k in obs_batch
                if "rgb" in k
            }
            depth = {
                k.rsplit("::", 1)[0]: (obs_batch[k].float() - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
                for k in obs_batch
                if "depth" in k
            }
            data["rgbd"] = {k: {"rgb": rgb[k], "depth": depth[k].unsqueeze(-3)} for k in rgb}
        if "pcd" in self._features:
            data["pcd"] = {
                "rgb": obs_batch["pcd"][..., :3],
                "xyz": obs_batch["pcd"][..., 3:],
            }
        if "task" in self._features:
            data["task"] = obs_batch["task"]
        if "base_action" in self._features and "base_action" in obs_batch:
            data["base_action"] = obs_batch["base_action"]
        if extract_action:
            data.update(
                {
                    "actions": data_batch["actions"],
                    "int_state": data_batch["policy"]["int_state"],
                    "base_action": data_batch["policy"]["base_action"],
                    "oracle_action": data_batch["policy"]["oracle_action"],
                    "masks": data_batch["masks"],
                }
            )
        return data

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise NotImplementedError

        if self.use_cosine_lr:
            scheduler_kwargs = dict(
                base_value=1.0,
                final_value=self.lr_cosine_min / self.lr,
                epochs=self.lr_cosine_steps,
                warmup_start_value=self.lr_cosine_min / self.lr,
                warmup_epochs=self.lr_warmup_steps,
                steps_per_epoch=1,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=CosineScheduleFunction(**scheduler_kwargs),
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        return optimizer
