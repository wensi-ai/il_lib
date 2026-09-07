import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig

from il_lib.nn.distributions import CategoricalNet
from il_lib.nn.features import SimpleFeatureFusion
from il_lib.optim import CosineScheduleFunction
from il_lib.policies.policy_base import BasePolicy
from il_lib.utils.training_utils import freeze_params, load_state_dict, unfreeze_params
from omnigibson.learning.utils.eval_utils import ACTION_QPOS_INDICES
from omnigibson.learning.utils.obs_utils import MAX_DEPTH, MIN_DEPTH


class _AlwaysInterveneDistribution:
    """Minimal distribution interface for the no-intervention head residual mode."""

    def __init__(self, reference: torch.Tensor):
        self._reference = reference

    def mode(self) -> torch.Tensor:
        return torch.ones(
            self._reference.shape[:-1],
            device=self._reference.device,
            dtype=torch.long,
        )

    def sample(self) -> torch.Tensor:
        return self.mode()


class _SimpleResidualMLP(nn.Module):
    """Deterministic MLP residual head modeled after the cr-dagger residual MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_tanh_output: bool = False,
        output_scale: Any = 1.0,
        last_layer_init_scale: Optional[float] = None,
    ):
        super().__init__()
        if hidden_depth < 1:
            raise ValueError("action_net_hidden_depth must be >= 1 for SimpleResidualPolicy.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        activation = activation.lower()
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("SimpleResidualPolicy supports action_net_activation in {relu, gelu}.")

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_depth - 1)
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.use_tanh_output = use_tanh_output
        if isinstance(output_scale, (list, tuple)):
            if len(output_scale) != output_dim:
                raise ValueError(
                    f"Per-dimension output_scale has length {len(output_scale)}, "
                    f"expected {output_dim}."
                )
            self.output_scale = tuple(float(scale) for scale in output_scale)
        else:
            self.output_scale = float(output_scale)

        if last_layer_init_scale is not None:
            nn.init.normal_(self.output_layer.weight, mean=0.0, std=float(last_layer_init_scale))
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.activation(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        for layer in self.hidden_layers:
            h = layer(h)
            h = self.activation(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.output_layer(h)
        if self.use_tanh_output:
            output_scale = torch.as_tensor(
                self.output_scale,
                device=out.device,
                dtype=out.dtype,
            )
            out = torch.tanh(out) * output_scale
        return out


class SimpleResidualPolicy(BasePolicy):
    def __init__(
        self,
        *args,
        prop_dim: int,
        prop_keys: List[str],
        feature_extractors: Dict[str, DictConfig],
        feature_fusion_hidden_depth: int = 1,
        feature_fusion_hidden_dim: int = 256,
        feature_fusion_output_dim: int = 256,
        feature_fusion_activation: str = "relu",
        feature_fusion_add_input_activation: bool = False,
        feature_fusion_add_output_activation: bool = False,
        action_dim: int = 7,
        action_prediction_horizon: int = 1,
        action_net_hidden_dim: int = 128,
        action_net_hidden_depth: int = 3,
        action_net_activation: str = "gelu",
        action_net_use_tanh_output: bool = False,
        action_scale: float = 1.0,
        actor_last_layer_init_scale: Optional[float] = None,
        dropout: float = 0.0,
        learn_gripper_action: bool = True,
        include_robot_gripper_action_input: bool = True,
        gripper_action_mode: str = "absolute",
        gripper_action_scale: float = 1.0,
        use_intervention_head: bool = True,
        intervention_head_hidden_dim: int = 128,
        intervention_head_hidden_depth: int = 1,
        intervention_head_activation: str = "relu",
        intervention_min_duration_steps: int = 1,
        deterministic_inference: bool = True,
        intervention_loss_weight: float = 1.0,
        supervise_zero_residual_off_intervention: bool = False,
        zero_residual_loss_weight: float = 0.0,
        off_intervention_residual_l1_weight: float = 0.0,
        predict_direct_action: bool = False,
        direct_action_loss_on_all_valid: bool = True,
        stage2_ckpt_path: Optional[str] = None,
        pretrained_vision_encoder_ckpt_path: Optional[str] = None,
        freeze_pretrained_vision_encoder: bool = False,
        residual_target_normalization: str = "none",
        residual_target_normalization_eps: float = 1e-6,
        lr: float = 1e-4,
        use_cosine_lr: bool = True,
        lr_warmup_steps: Optional[int] = None,
        lr_cosine_steps: Optional[int] = None,
        lr_cosine_min: Optional[float] = None,
        lr_layer_decay: float = 1.0,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._prop_dim = prop_dim
        self._prop_keys = prop_keys
        self._features = set(feature_extractors.keys())

        self.feature_extractor = SimpleFeatureFusion(
            extractors={k: instantiate(v) for k, v in feature_extractors.items()},
            hidden_depth=feature_fusion_hidden_depth,
            hidden_dim=feature_fusion_hidden_dim,
            output_dim=feature_fusion_output_dim,
            activation=feature_fusion_activation,
            add_input_activation=feature_fusion_add_input_activation,
            add_output_activation=feature_fusion_add_output_activation,
        )

        self.action_dim = action_dim
        self.action_prediction_horizon = action_prediction_horizon
        self._learn_gripper_action = learn_gripper_action
        self._include_robot_gripper_action_input = include_robot_gripper_action_input
        self._gripper_action_mode = gripper_action_mode.lower()
        if self._gripper_action_mode not in {"absolute", "delta"}:
            raise ValueError("gripper_action_mode must be one of {absolute, delta}.")

        gripper_action_mask = torch.zeros(action_dim, dtype=torch.bool)
        for joint_key, indices in ACTION_QPOS_INDICES[self.robot_type].items():
            if "gripper" in joint_key:
                gripper_action_mask[indices] = True
        self._gripper_action_indices = torch.where(gripper_action_mask)[0].tolist()

        output_scale = torch.full(
            (action_prediction_horizon, action_dim),
            float(action_scale),
        )
        if self._learn_gripper_action and self._gripper_action_indices:
            output_scale[:, self._gripper_action_indices] = float(gripper_action_scale)
        self.action_net = _SimpleResidualMLP(
            input_dim=feature_fusion_output_dim,
            output_dim=action_dim * action_prediction_horizon,
            hidden_dim=action_net_hidden_dim,
            hidden_depth=action_net_hidden_depth,
            activation=action_net_activation,
            dropout=dropout,
            use_tanh_output=action_net_use_tanh_output,
            output_scale=output_scale.reshape(-1).tolist(),
            last_layer_init_scale=actor_last_layer_init_scale,
        )

        self._use_intervention_head = use_intervention_head
        if intervention_min_duration_steps < 1:
            raise ValueError("intervention_min_duration_steps must be >= 1.")
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

        self._deterministic_inference = deterministic_inference
        self._intervention_loss_weight = intervention_loss_weight
        self._supervise_zero_residual_off_intervention = (
            supervise_zero_residual_off_intervention
        )
        self._zero_residual_loss_weight = zero_residual_loss_weight
        self._off_intervention_residual_l1_weight = off_intervention_residual_l1_weight
        self._predict_direct_action = bool(predict_direct_action)
        self._direct_action_loss_on_all_valid = bool(direct_action_loss_on_all_valid)
        self._residual_target_normalization = residual_target_normalization.lower()
        if self._predict_direct_action and self._uses_residual_target_normalization():
            raise ValueError(
                "residual_target_normalization is only supported for residual targets, "
                "not predict_direct_action=True."
            )
        self._residual_target_normalization_eps = residual_target_normalization_eps
        if self._residual_target_normalization not in {"none", "min_max", "transic"}:
            raise ValueError(
                "SimpleResidualPolicy supports residual_target_normalization in "
                "{none, min_max, transic}."
            )
        residual_target_mask = ~gripper_action_mask
        self.register_buffer("_residual_target_normalization_min", torch.zeros(action_dim))
        self.register_buffer("_residual_target_normalization_max", torch.ones(action_dim))
        self.register_buffer("_residual_target_normalization_mask", residual_target_mask)
        self.register_buffer("_residual_target_normalization_ready", torch.tensor(False, dtype=torch.bool))

        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = lr_layer_decay
        self.optimizer = optimizer
        self.weight_decay = weight_decay

        if stage2_ckpt_path is not None:
            self._load_stage2_checkpoint(stage2_ckpt_path)
        if pretrained_vision_encoder_ckpt_path is not None:
            self._load_pretrained_vision_encoder(
                pretrained_vision_encoder_ckpt_path,
                freeze=freeze_pretrained_vision_encoder,
            )

    def _load_stage2_checkpoint(self, ckpt_path: str) -> None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"stage2_ckpt_path does not exist: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]

        def _extract_weights(prefixes: List[str]) -> tuple[Dict[str, Any], str]:
            for prefix in prefixes:
                weights = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
                if weights:
                    return weights, prefix
            available_keys = list(state_dict.keys())[:10]
            raise KeyError(
                f"Could not find checkpoint weights for any of prefixes {prefixes}. "
                f"Sample checkpoint keys: {available_keys}"
            )

        feature_weights, feature_prefix = _extract_weights(
            [
                "simple_residual_policy.feature_extractor.",
                "residual_policy.feature_extractor.",
                "feature_extractor.",
            ]
        )
        load_state_dict(
            self.feature_extractor,
            feature_weights,
            strip_prefix=feature_prefix,
            strict=True,
        )
        freeze_params(self.feature_extractor)

        action_weights, action_prefix = _extract_weights(
            [
                "simple_residual_policy.action_net.",
                "residual_policy.action_net.",
                "action_net.",
            ]
        )
        load_state_dict(
            self.action_net,
            action_weights,
            strip_prefix=action_prefix,
            strict=True,
        )
        freeze_params(self.action_net)

    def _load_pretrained_vision_encoder(self, ckpt_path: str, *, freeze: bool) -> None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"pretrained_vision_encoder_ckpt_path does not exist: {ckpt_path}"
            )
        extractors = getattr(self.feature_extractor, "_extractors", {})
        if "rgb" not in extractors:
            raise KeyError("pretrained_vision_encoder_ckpt_path requires an rgb feature extractor.")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        prefixes = [
            "base_chunk_diffusion_policy.feature_extractor._extractors.rgb.",
            "residual_diffusion_policy.feature_extractor._extractors.rgb.",
            "diffusion_policy.feature_extractor._extractors.rgb.",
            "feature_extractor._extractors.rgb.",
        ]
        for prefix in prefixes:
            weights = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
            if weights:
                load_state_dict(extractors["rgb"], weights, strip_prefix=prefix, strict=True)
                (freeze_params if freeze else unfreeze_params)(extractors["rgb"])
                return
        available_keys = list(state_dict.keys())[:10]
        raise KeyError(
            "Could not find rgb vision encoder weights in checkpoint. "
            f"Tried prefixes {prefixes}. Sample checkpoint keys: {available_keys}"
        )

    def forward(self, obs):
        prop_obs = []
        for prop_key in self._prop_keys:
            if "/" in prop_key:
                group, key = prop_key.split("/")
                prop_obs.append(obs[group][key])
            else:
                prop_obs.append(obs[prop_key])
        prop_obs = torch.cat(prop_obs, dim=-1)
        obs_time = prop_obs.shape[1]

        obs = dict(obs)
        obs["proprioception"] = prop_obs
        if "rgb" in self._features and "rgb" not in obs:
            rgb_obs = {
                k.rsplit("::", 1)[0]: v
                for k, v in obs.items()
                if isinstance(k, str) and k.endswith("::rgb")
            }
            if rgb_obs:
                obs["rgb"] = rgb_obs
        obs = {k: obs[k] for k in self._features}
        if "base_action" in obs:
            obs["base_action"] = self._format_action_chunk(obs["base_action"])
            if obs["base_action"].shape[1] == 1 and obs_time > 1:
                obs["base_action"] = obs["base_action"].expand(-1, obs_time, -1)

        obs_feature = self.feature_extractor(obs)
        if obs_feature.dim() >= 3:
            obs_feature = obs_feature[:, -1:]

        residual_action = self.action_net(obs_feature)
        if self._use_intervention_head:
            intervention_dist = self.intervention_head(obs_feature)
        else:
            intervention_dist = _AlwaysInterveneDistribution(obs_feature)
        return residual_action, intervention_dist

    @torch.no_grad()
    def act(self, obs, deterministic=None):
        if deterministic is None:
            deterministic = self._deterministic_inference

        residual_action, intervention_dist = self.forward(obs)
        intervention = intervention_dist.mode() if deterministic else intervention_dist.sample()
        action = self._unflatten_action_chunk(residual_action)
        if not self._predict_direct_action:
            action = self._denormalize_residual_target(action)
        return action, intervention

    def reset(self) -> None:
        pass

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

    def policy_training_step(self, batch, batch_idx):
        return self._residual_forward_step(batch, batch_idx, is_train=True)

    def policy_evaluation_step(self, batch, batch_idx):
        return self._residual_forward_step(batch, batch_idx, is_train=False)

    def _build_action_target(
        self,
        robot_policy_action: torch.Tensor,
        oracle_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build mixed continuous-residual / binary-gripper action targets."""
        gripper_indices = self._gripper_action_indices
        robot_policy_gripper_action = torch.where(
            robot_policy_action[..., gripper_indices] >= 0,
            1.0,
            -1.0,
        )
        oracle_gripper_action = torch.where(
            oracle_action[..., gripper_indices] >= 0,
            1.0,
            -1.0,
        )

        if self._predict_direct_action:
            target_action = oracle_action.clone()
            if self._learn_gripper_action:
                target_action[..., gripper_indices] = oracle_gripper_action
        else:
            target_action = oracle_action - robot_policy_action
            if self._learn_gripper_action:
                if self._gripper_action_mode == "absolute":
                    target_action[..., gripper_indices] = oracle_gripper_action
                else:
                    target_action[..., gripper_indices] = (
                        oracle_gripper_action - robot_policy_gripper_action
                    )
        return target_action, robot_policy_gripper_action, oracle_gripper_action

    def _residual_forward_step(self, batch, batch_idx, is_train: bool):
        batch = self.process_data(batch, extract_action=True)

        pad_mask = batch.pop("masks")
        intervention_mask = batch.pop("int_state") == 2

        robot_policy_action = batch["base_action"]
        oracle_action = batch["oracle_action"]
        gripper_indices = self._gripper_action_indices
        target_action, robot_policy_gripper_action, oracle_gripper_action = (
            self._build_action_target(robot_policy_action, oracle_action)
        )

        if self._include_robot_gripper_action_input:
            batch["robot_policy_gripper_action"] = robot_policy_gripper_action

        action_label_mask = (
            torch.ones_like(intervention_mask, dtype=torch.bool)
            if self._predict_direct_action and self._direct_action_loss_on_all_valid
            else intervention_mask
        )
        action_valid_mask = self._action_valid_mask(
            pad_mask,
            action_label_mask if not self._supervise_zero_residual_off_intervention else torch.ones_like(intervention_mask, dtype=torch.bool),
        )
        off_intervention_mask = self._action_valid_mask(pad_mask, ~intervention_mask)

        if self._supervise_zero_residual_off_intervention and not self._predict_direct_action:
            target_action = target_action * intervention_mask.unsqueeze(-1).to(target_action.dtype)
            if self._learn_gripper_action and self._gripper_action_mode == "absolute":
                target_action[..., gripper_indices] = torch.where(
                    intervention_mask.unsqueeze(-1),
                    oracle_gripper_action,
                    robot_policy_gripper_action,
                )

        zero_target_action = torch.zeros_like(target_action)
        if target_action.dim() == 4:
            if not self._predict_direct_action:
                target_action = self._normalize_residual_target(target_action)
                zero_target_action = self._normalize_residual_target(zero_target_action)
            target_action = self._format_action_chunk(target_action)
            zero_target_action = self._format_action_chunk(zero_target_action)
            action_valid_mask = self._current_mask(action_valid_mask) & pad_mask.all(dim=-1)
            off_intervention_mask = self._current_mask(off_intervention_mask) & pad_mask.all(dim=-1)
        elif self.action_prediction_horizon != 1:
            raise ValueError(
                "SimpleResidualPolicy expected chunked correction targets with shape "
                f"(B, T, {self.action_prediction_horizon}, A), but got {target_action.shape}."
            )
        else:
            if not self._predict_direct_action:
                target_action = self._normalize_residual_target(target_action)
                zero_target_action = self._normalize_residual_target(zero_target_action)

        pred_action, intervention_dist = self.forward(batch)
        pred_action = pred_action.reshape_as(target_action)
        pred_action_denormalized = (
            pred_action
            if self._predict_direct_action
            else self._denormalize_residual_target(pred_action)
        )

        action_loss_mask = action_valid_mask.unsqueeze(-1).expand_as(pred_action).to(
            pred_action.dtype
        )
        action_sq_error = F.mse_loss(pred_action, target_action, reduction="none")
        action_loss_denom = action_loss_mask.sum().clamp_min(1.0)
        action_loss = (action_sq_error * action_loss_mask).sum() / action_loss_denom
        real_batch_size = action_valid_mask.sum().clamp_min(1)

        zero_residual_loss = pred_action.new_zeros(())
        if self._zero_residual_loss_weight > 0 and not self._predict_direct_action:
            zero_mask = off_intervention_mask.unsqueeze(-1).expand_as(pred_action).to(
                pred_action.dtype
            )
            if self._learn_gripper_action and self._gripper_action_mode == "absolute":
                zero_mask = zero_mask.clone()
                if self.action_prediction_horizon > 1:
                    gripper_flat_indices = [
                        step * self.action_dim + index
                        for step in range(self.action_prediction_horizon)
                        for index in gripper_indices
                    ]
                    zero_mask[..., gripper_flat_indices] = 0
                else:
                    zero_mask[..., gripper_indices] = 0
            zero_loss_denom = zero_mask.sum().clamp_min(1.0)
            zero_residual_loss = (
                F.mse_loss(pred_action, zero_target_action, reduction="none") * zero_mask
            ).sum() / zero_loss_denom

        off_intervention_l1 = pred_action.new_zeros(())
        if self._off_intervention_residual_l1_weight > 0 and not self._predict_direct_action:
            off_mask = off_intervention_mask.unsqueeze(-1).expand_as(pred_action).to(
                pred_action.dtype
            )
            off_l1_denom = off_mask.sum().clamp_min(1.0)
            off_intervention_l1 = (pred_action_denormalized.abs() * off_mask).sum() / off_l1_denom

        if self._use_intervention_head:
            intervention_target = self._current_intervention(intervention_mask)
            intervention_loss_mask = self._current_mask(pad_mask)
            raw_intervention_loss = intervention_dist.imitation_loss(
                intervention_target.long(), reduction="none"
            ).reshape(intervention_loss_mask.shape)
            intervention_loss = raw_intervention_loss * intervention_loss_mask
            intervention_acc = intervention_dist.imitation_accuracy(
                intervention_target.long(),
                mask=intervention_loss_mask,
            )
            intervention_loss = torch.sum(intervention_loss) / intervention_loss_mask.sum().clamp_min(1)
        else:
            intervention_loss = pred_action.new_zeros(())
            intervention_acc = pred_action.new_ones(())

        loss = action_loss
        loss = loss + self._zero_residual_loss_weight * zero_residual_loss
        loss = loss + self._off_intervention_residual_l1_weight * off_intervention_l1
        if self._use_intervention_head:
            loss = loss + self._intervention_loss_weight * intervention_loss

        log_dict = {
            "action_loss": action_loss,
            "zero_residual_loss": zero_residual_loss,
            "off_intervention_residual_l1": off_intervention_l1,
            "intervention_loss": intervention_loss,
            "intervention_acc": intervention_acc,
        }
        if not is_train:
            log_dict["l1"] = loss
        return loss, log_dict, real_batch_size

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self._maybe_initialize_residual_target_normalization()

    def load_state_dict(self, state_dict, strict: bool = True):
        if strict and self._uses_residual_target_normalization():
            result = super().load_state_dict(state_dict, strict=False)
            allowed_missing = {
                "_residual_target_normalization_min",
                "_residual_target_normalization_max",
                "_residual_target_normalization_mask",
                "_residual_target_normalization_ready",
            }
            missing = [key for key in result.missing_keys if key not in allowed_missing]
            unexpected = list(result.unexpected_keys)
            if missing or unexpected:
                raise RuntimeError(
                    "Error(s) in loading state_dict for SimpleResidualPolicy: "
                    f"missing_keys={missing}, unexpected_keys={unexpected}"
                )
            return result
        return super().load_state_dict(state_dict, strict=strict)

    def _uses_residual_target_normalization(self) -> bool:
        return self._residual_target_normalization != "none"

    def _reshape_action_for_residual_target_normalization(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
        original_shape = action.shape
        if action.shape[-1] == self.action_dim:
            return action, original_shape
        if (
            self.action_prediction_horizon > 1
            and action.shape[-1] == self.action_dim * self.action_prediction_horizon
        ):
            reshaped = action.reshape(
                *action.shape[:-1],
                self.action_prediction_horizon,
                self.action_dim,
            )
            return reshaped, original_shape
        raise ValueError(
            "Residual target normalization expected the last dimension to equal "
            f"{self.action_dim} or {self.action_dim * self.action_prediction_horizon}, got {action.shape}."
        )

    def _normalize_residual_target(self, action: torch.Tensor) -> torch.Tensor:
        return self._apply_residual_target_normalization(action, inverse=False)

    def _denormalize_residual_target(self, action: torch.Tensor) -> torch.Tensor:
        return self._apply_residual_target_normalization(action, inverse=True)

    def _apply_residual_target_normalization(
        self,
        action: torch.Tensor,
        *,
        inverse: bool,
    ) -> torch.Tensor:
        if not self._uses_residual_target_normalization():
            return action
        reshaped_action, original_shape = self._reshape_action_for_residual_target_normalization(action)
        normalized_action = reshaped_action.clone()
        mask = self._residual_target_normalization_mask.to(device=action.device)
        if not bool(mask.any().item()):
            return normalized_action.reshape(original_shape)

        if self._residual_target_normalization == "transic":
            # Each normalized joint command is in [-1, 1], so the largest
            # physically meaningful oracle-minus-base delta is in [-2, 2].
            # TRANSIC trains on this theoretical residual range, then clamps
            # the composed command to physical joint limits at deployment.
            if inverse:
                normalized_action[..., mask] = normalized_action[..., mask] * 2.0
            else:
                normalized_action[..., mask] = (
                    normalized_action[..., mask].clamp(-2.0, 2.0) / 2.0
                )
            return normalized_action.reshape(original_shape)

        if not bool(self._residual_target_normalization_ready.item()):
            raise RuntimeError(
                "Residual target normalization is enabled but stats are not initialized."
            )

        min_vals = self._residual_target_normalization_min.to(
            device=action.device,
            dtype=action.dtype,
        )
        max_vals = self._residual_target_normalization_max.to(
            device=action.device,
            dtype=action.dtype,
        )
        range_vals = max_vals - min_vals
        safe_range_vals = torch.where(
            range_vals.abs() < self._residual_target_normalization_eps,
            torch.ones_like(range_vals),
            range_vals,
        )

        if inverse:
            normalized_action[..., mask] = (
                (normalized_action[..., mask] + 1.0) / 2.0 * safe_range_vals[mask]
                + min_vals[mask]
            )
            constant_mask = mask & (range_vals.abs() < self._residual_target_normalization_eps)
            if bool(constant_mask.any().item()):
                normalized_action[..., constant_mask] = min_vals[constant_mask]
        else:
            normalized_action[..., mask] = (
                2.0 * (normalized_action[..., mask] - min_vals[mask]) / safe_range_vals[mask]
                - 1.0
            )
            constant_mask = mask & (range_vals.abs() < self._residual_target_normalization_eps)
            if bool(constant_mask.any().item()):
                normalized_action[..., constant_mask] = 0.0
        return normalized_action.reshape(original_shape)

    def _maybe_initialize_residual_target_normalization(self) -> None:
        if self._residual_target_normalization != "min_max":
            return
        if bool(self._residual_target_normalization_ready.item()):
            return
        train_dataset = getattr(getattr(self.trainer, "datamodule", None), "_train_dataset", None)
        if train_dataset is None:
            raise RuntimeError(
                "Could not access the training dataset to initialize residual target normalization."
            )

        action_mask = self._residual_target_normalization_mask.cpu()
        residual_min = torch.full((self.action_dim,), float("inf"), dtype=torch.float32)
        residual_max = torch.full((self.action_dim,), float("-inf"), dtype=torch.float32)
        found_valid_residual = False

        for demo in getattr(train_dataset, "_all_demos", []):
            policy_demo = demo.get("policy", {})
            if not {"base_action", "oracle_action", "int_state"}.issubset(policy_demo.keys()):
                continue

            residual = (policy_demo["oracle_action"] - policy_demo["base_action"]).to(torch.float32)
            valid_mask = (policy_demo["int_state"] == 2)
            if residual.dim() >= 3 and "action_masks" in demo:
                valid_mask = valid_mask & demo["action_masks"]

            residual = residual.reshape(-1, self.action_dim)
            valid_mask = valid_mask.reshape(-1)
            if not bool(valid_mask.any().item()):
                continue

            valid_residual = residual[valid_mask]
            found_valid_residual = True
            residual_min[action_mask] = torch.minimum(
                residual_min[action_mask],
                valid_residual[:, action_mask].amin(dim=0),
            )
            residual_max[action_mask] = torch.maximum(
                residual_max[action_mask],
                valid_residual[:, action_mask].amax(dim=0),
            )

        if not found_valid_residual:
            raise RuntimeError(
                "Residual target normalization is enabled, but no intervention residual targets were found."
            )

        residual_min[~action_mask] = 0.0
        residual_max[~action_mask] = 1.0
        self._residual_target_normalization_min.copy_(
            residual_min.to(device=self._residual_target_normalization_min.device)
        )
        self._residual_target_normalization_max.copy_(
            residual_max.to(device=self._residual_target_normalization_max.device)
        )
        self._residual_target_normalization_ready.fill_(True)
        self.print(
            "Initialized residual target normalization stats: "
            f"min={self._residual_target_normalization_min.tolist()}, "
            f"max={self._residual_target_normalization_max.tolist()}"
        )

    def _format_action_chunk(self, action: torch.Tensor) -> torch.Tensor:
        if action.dim() >= 4:
            return action.reshape(*action.shape[:-2], action.shape[-2] * action.shape[-1])
        return action

    def _unflatten_action_chunk(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_prediction_horizon == 1:
            return action
        return action.reshape(
            *action.shape[:-1],
            self.action_prediction_horizon,
            self.action_dim,
        )

    def _current_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() >= 3:
            return mask[..., 0]
        return mask

    def _current_intervention(self, intervention_mask: torch.Tensor) -> torch.Tensor:
        if intervention_mask.dim() >= 3:
            return intervention_mask[..., 0]
        return intervention_mask

    def _action_valid_mask(
        self,
        pad_mask: torch.Tensor,
        intervention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if pad_mask.dim() >= 3 and intervention_mask.dim() == pad_mask.dim() - 1:
            intervention_mask = intervention_mask.unsqueeze(-1).expand_as(pad_mask)
        elif intervention_mask.dim() >= 3 and pad_mask.dim() == intervention_mask.dim() - 1:
            pad_mask = pad_mask.unsqueeze(-1).expand_as(intervention_mask)
        return intervention_mask & pad_mask

    def process_data(self, data_batch: dict, extract_action: bool = False) -> Any:
        data = {"qpos": data_batch["obs"]["qpos"], "eef": data_batch["obs"]["eef"]}
        if "odom" in data_batch["obs"]:
            data["odom"] = data_batch["obs"]["odom"]
        if "rgb" in self._features:
            data["rgb"] = {
                k.rsplit("::", 1)[0]: data_batch["obs"][k].float() / 255.0
                for k in data_batch["obs"]
                if "rgb" in k
            }
        if "rgbd" in self._features:
            rgb = {
                k.rsplit("::", 1)[0]: data_batch["obs"][k].float() / 255.0
                for k in data_batch["obs"]
                if "rgb" in k
            }
            depth = {
                k.rsplit("::", 1)[0]: (data_batch["obs"][k].float() - MIN_DEPTH)
                / (MAX_DEPTH - MIN_DEPTH)
                for k in data_batch["obs"]
                if "depth" in k
            }
            data["rgbd"] = {
                k: {"rgb": rgb[k], "depth": depth[k].unsqueeze(-3)} for k in rgb
            }
        if "pcd" in self._features:
            data["pcd"] = {
                "rgb": data_batch["obs"]["pcd"][..., :3],
                "xyz": data_batch["obs"]["pcd"][..., 3:],
            }
        if "task" in self._features:
            data["task"] = data_batch["obs"]["task"]
        if extract_action:
            data.update(
                {
                    "int_state": data_batch["policy"]["int_state"],
                    "base_action": data_batch["policy"]["base_action"],
                    "oracle_action": data_batch["policy"]["oracle_action"],
                    "masks": data_batch["masks"],
                }
            )
        return data
