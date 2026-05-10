import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from il_lib.nn.common.mlp import build_mlp
from il_lib.policies.base_chunk_diffusion_policy import BaseChunkDiffusionPolicy
from il_lib.utils.array_tensor_utils import any_concat, get_batch_size
from il_lib.utils.training_utils import freeze_params, load_state_dict, unfreeze_params


class GatedDiffusionPolicy(BaseChunkDiffusionPolicy):
    """
    Base-action-conditioned diffusion policy with a learned action gate.

    The diffusion backbone predicts a candidate action chunk. A separate gate
    blends that candidate with the base policy action chunk in normalized action
    space:

        action = gate * diffusion_action + (1 - gate) * base_action

    This policy intentionally disables the base chunk intervention head.
    """

    is_sequence_policy = True

    def __init__(
        self,
        *args,
        gate_type: str = "per_step_action",
        gate_hidden_dim: int = 128,
        gate_hidden_depth: int = 1,
        gate_activation: str = "relu",
        gate_bias_init: float = 0.0,
        gate_temperature: float = 1.0,
        diffusion_loss_weight: float = 1.0,
        blend_loss_weight: float = 1.0,
        training_stage: str = "single_stage",
        stage1_steps: int = 0,
        stage1_gate_value: float = 1.0,
        stage1_base_action_mask_ratio: Optional[float] = None,
        stage2_base_action_mask_ratio: Optional[float] = 0.0,
        stage2_ckpt_path: Optional[str] = None,
        stage2_diffusion_loss_weight: float = 0.0,
        stage2_blend_loss_weight: Optional[float] = None,
        stage2_gate_bce_loss_weight: float = 1.0,
        **kwargs,
    ):
        kwargs["use_intervention_head"] = False
        super().__init__(*args, **kwargs)

        if gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be > 0.")
        if diffusion_loss_weight < 0.0 or blend_loss_weight < 0.0:
            raise ValueError("Gate policy loss weights must be non-negative.")

        self.gate_type = gate_type
        self.gate_temperature = gate_temperature
        self.diffusion_loss_weight = diffusion_loss_weight
        self.blend_loss_weight = blend_loss_weight
        self.training_stage = training_stage
        self.stage1_steps = stage1_steps
        self.stage1_gate_value = stage1_gate_value
        self.stage1_base_action_mask_ratio = stage1_base_action_mask_ratio
        self.stage2_base_action_mask_ratio = stage2_base_action_mask_ratio
        self.stage2_diffusion_loss_weight = stage2_diffusion_loss_weight
        self.stage2_blend_loss_weight = (
            blend_loss_weight if stage2_blend_loss_weight is None else stage2_blend_loss_weight
        )
        self.stage2_gate_bce_loss_weight = stage2_gate_bce_loss_weight

        valid_training_stages = {"single_stage", "stage1", "stage2", "two_stage"}
        if training_stage not in valid_training_stages:
            raise ValueError(f"training_stage must be one of {valid_training_stages}.")
        if stage1_steps < 0:
            raise ValueError("stage1_steps must be non-negative.")
        if not 0.0 <= stage1_gate_value <= 1.0:
            raise ValueError("stage1_gate_value must be in [0, 1].")
        for name, ratio in {
            "stage1_base_action_mask_ratio": stage1_base_action_mask_ratio,
            "stage2_base_action_mask_ratio": stage2_base_action_mask_ratio,
        }.items():
            if ratio is not None and not 0.0 <= ratio <= 1.0:
                raise ValueError(f"{name} must be in [0, 1] when provided.")
        if (
            stage2_diffusion_loss_weight < 0.0
            or self.stage2_blend_loss_weight < 0.0
            or stage2_gate_bce_loss_weight < 0.0
        ):
            raise ValueError("Stage 2 loss weights must be non-negative.")

        self.gate_net = build_mlp(
            input_dim=self.feature_extractor.output_dim,
            hidden_dim=gate_hidden_dim,
            output_dim=self._gate_output_dim(gate_type),
            hidden_depth=gate_hidden_depth,
            activation=gate_activation,
        )
        last_layer = self._last_gate_layer()
        if last_layer is not None:
            nn.init.constant_(last_layer.bias, gate_bias_init)

        if stage2_ckpt_path is not None:
            self._load_stage2_checkpoint(stage2_ckpt_path)
        self._apply_trainable_stage(self._current_stage())

        self.save_hyperparameters()

    def forward(self, obs, noisy_traj, diffusion_timesteps):
        obs_feature = self._encode_obs(obs)
        pred_noise = self.backbone(
            sample=noisy_traj,
            timestep=diffusion_timesteps,
            cond=obs_feature,
        )
        gate = self._predict_gate(obs_feature)
        return pred_noise, gate

    @torch.no_grad()
    def act(self, obs, deterministic=None):
        del deterministic
        diffusion_action = self._sample_diffusion_action(obs)
        base_action = self._base_action_from_obs(obs)
        gate = self._predict_gate(self._encode_obs(obs))
        return self._blend_actions(diffusion_action, base_action, gate)

    def _forward_step(self, batch, batch_idx, is_train: bool):
        del batch_idx
        batch["actions"] = any_concat(
            [batch["actions"][k] for k in self._action_keys], dim=-1
        )
        batch_size = batch["actions"].shape[0]
        batch = self.process_data(batch, extract_action=True)

        pad_mask = batch.pop("masks")
        int_state = batch.pop("int_state")
        batch.pop("oracle_action")
        target_action = batch.pop("actions")

        if target_action.dim() != 4:
            raise ValueError(
                "GatedDiffusionPolicy expected chunked targets with shape "
                f"(B, T, {self.action_prediction_horizon}, A), got {target_action.shape}."
            )
        if target_action.shape[-2:] != (self.action_prediction_horizon, self.action_dim):
            raise ValueError(
                "Target action chunk shape mismatch: expected trailing shape "
                f"({self.action_prediction_horizon}, {self.action_dim}), got "
                f"{target_action.shape[-2:]}."
            )

        target_action = target_action[:, -1]
        base_action = self._base_action_from_obs(batch)
        chunk_mask = pad_mask[:, -1]
        target_int_state = int_state[:, -1]
        if self._action_loss_on_intervention_only:
            intervention_mask = self._current_intervention(int_state == 2)
            chunk_mask = chunk_mask & intervention_mask[:, -1:].expand_as(chunk_mask)
        elif self._exclude_pre_intervention_from_action_loss:
            chunk_mask = chunk_mask & (target_int_state != 1)

        noise = torch.randn(target_action.shape, device=target_action.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=target_action.device,
        ).long()
        noisy_trajs = self.noise_scheduler.add_noise(target_action, noise, timesteps)
        pred_noise, gate = self.forward(
            obs=batch,
            noisy_traj=noisy_trajs,
            diffusion_timesteps=timesteps,
        )
        effective_stage = self._current_stage()
        if effective_stage == "stage1":
            gate = self._constant_gate_like(target_action, self.stage1_gate_value)

        raw_diffusion_loss = F.mse_loss(pred_noise, noise, reduction="none").mean(dim=-1)
        diffusion_loss = self._masked_mean(raw_diffusion_loss, chunk_mask)

        pred_action = self._predict_original_sample(noisy_trajs, pred_noise, timesteps)
        if effective_stage == "stage2":
            pred_action = pred_action.detach()
        blended_action = self._blend_actions(pred_action, base_action, gate)
        raw_blend_loss = F.l1_loss(blended_action, target_action, reduction="none").mean(dim=-1)
        blend_loss = self._masked_mean(raw_blend_loss, chunk_mask)

        gate_bce_loss = target_action.new_zeros(())
        if effective_stage == "stage2" and self.stage2_gate_bce_loss_weight > 0.0:
            gate_target = (target_int_state == 2).to(dtype=target_action.dtype)
            raw_gate_bce_loss = F.binary_cross_entropy(
                gate.expand_as(target_action),
                gate_target.unsqueeze(-1).expand_as(target_action),
                reduction="none",
            ).mean(dim=-1)
            gate_bce_loss = self._masked_mean(raw_gate_bce_loss, chunk_mask)

        diffusion_loss_weight, blend_loss_weight = self._stage_loss_weights(effective_stage)
        loss = (
            diffusion_loss_weight * diffusion_loss
            + blend_loss_weight * blend_loss
            + self.stage2_gate_bce_loss_weight * gate_bce_loss
        )
        real_batch_size = chunk_mask.sum().clamp_min(1)

        gate_for_log = gate.expand_as(target_action).mean(dim=-1)
        log_dict = {
            "diffusion_loss": diffusion_loss,
            "action_loss": blend_loss,
            "blend_loss": blend_loss,
            "gate_mean": self._masked_mean(gate_for_log, chunk_mask),
            "stage": target_action.new_tensor(self._stage_id(effective_stage)),
            "diffusion_loss_weight": target_action.new_tensor(diffusion_loss_weight),
            "blend_loss_weight": target_action.new_tensor(blend_loss_weight),
            "gate_bce_loss": gate_bce_loss,
            "gate_bce_loss_weight": target_action.new_tensor(
                self.stage2_gate_bce_loss_weight if effective_stage == "stage2" else 0.0
            ),
        }
        if not is_train:
            sampled_diffusion_action = self._sample_diffusion_action(batch)
            sampled_gate = self._predict_gate(self._encode_obs(batch))
            pred_action_eval = self._blend_actions(sampled_diffusion_action, base_action, sampled_gate)
            l1_full_future_horizon = torch.abs(pred_action_eval - target_action).mean(dim=-1)
            l1_full_future_horizon = self._masked_mean(l1_full_future_horizon, chunk_mask)

            deployed_steps = min(self.deployed_action_steps, target_action.shape[-2])
            deployed_mask = chunk_mask[:, :deployed_steps]
            l1_deployed_steps_only = torch.abs(
                pred_action_eval[:, :deployed_steps] - target_action[:, :deployed_steps]
            ).mean(dim=-1)
            l1_deployed_steps_only = self._masked_mean(l1_deployed_steps_only, deployed_mask)

            log_dict["l1"] = l1_deployed_steps_only
            log_dict["l1_full_future_horizon"] = l1_full_future_horizon
            log_dict["l1_deployed_steps_only"] = l1_deployed_steps_only
        return loss, log_dict, real_batch_size

    def on_train_batch_start(self, batch, batch_idx, *args, **kwargs) -> None:
        del batch, batch_idx, args, kwargs
        self._apply_trainable_stage(self._current_stage())

    def _current_stage(self) -> str:
        if self.training_stage == "two_stage":
            if self.stage1_steps <= 0:
                return "stage2"
            return "stage1" if int(self.global_step) < self.stage1_steps else "stage2"
        return self.training_stage

    def _stage_id(self, stage: str) -> int:
        return {"single_stage": 0, "stage1": 1, "stage2": 2}[stage]

    def _stage_loss_weights(self, stage: str) -> tuple[float, float]:
        if stage == "stage2":
            return self.stage2_diffusion_loss_weight, self.stage2_blend_loss_weight
        return self.diffusion_loss_weight, self.blend_loss_weight

    def _apply_trainable_stage(self, stage: str) -> None:
        if stage == getattr(self, "_active_trainable_stage", None):
            return
        if stage == "stage1":
            unfreeze_params(self.feature_extractor)
            unfreeze_params(self.backbone)
            freeze_params(self.gate_net)
        elif stage == "stage2":
            freeze_params(self.feature_extractor)
            freeze_params(self.backbone)
            unfreeze_params(self.gate_net)
        else:
            unfreeze_params(self.feature_extractor)
            unfreeze_params(self.backbone)
            unfreeze_params(self.gate_net)
        self._active_trainable_stage = stage

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
            ["gated_diffusion_policy.feature_extractor.", "feature_extractor."]
        )
        load_state_dict(
            self.feature_extractor,
            feature_weights,
            strip_prefix=feature_prefix,
            strict=True,
        )

        backbone_weights, backbone_prefix = _extract_weights(
            ["gated_diffusion_policy.backbone.", "backbone."]
        )
        load_state_dict(
            self.backbone,
            backbone_weights,
            strip_prefix=backbone_prefix,
            strict=True,
        )

    def _constant_gate_like(self, action: torch.Tensor, value: float) -> torch.Tensor:
        if self.gate_type == "scalar":
            shape = (action.shape[0], 1, 1)
        elif self.gate_type == "per_step":
            shape = (action.shape[0], self.action_prediction_horizon, 1)
        elif self.gate_type == "per_action":
            shape = (action.shape[0], 1, self.action_dim)
        else:
            shape = (action.shape[0], self.action_prediction_horizon, self.action_dim)
        return action.new_full(shape, value)

    def _gate_output_dim(self, gate_type: str) -> int:
        if gate_type == "scalar":
            return 1
        if gate_type == "per_step":
            return self.action_prediction_horizon
        if gate_type == "per_action":
            return self.action_dim
        if gate_type == "per_step_action":
            return self.action_prediction_horizon * self.action_dim
        raise ValueError(
            "gate_type must be one of {'scalar', 'per_step', 'per_action', 'per_step_action'}."
        )

    def _last_gate_layer(self) -> Optional[nn.Linear]:
        for module in reversed(self.gate_net):
            if isinstance(module, nn.Linear):
                return module
        return None

    def _predict_gate(self, obs_feature: torch.Tensor) -> torch.Tensor:
        if obs_feature.dim() >= 3:
            obs_feature = obs_feature[:, -1]
        logits = self.gate_net(obs_feature) / self.gate_temperature
        gate = torch.sigmoid(logits)
        if self.gate_type == "scalar":
            return gate.view(-1, 1, 1)
        if self.gate_type == "per_step":
            return gate.view(-1, self.action_prediction_horizon, 1)
        if self.gate_type == "per_action":
            return gate.view(-1, 1, self.action_dim)
        return gate.view(-1, self.action_prediction_horizon, self.action_dim)

    def _base_action_from_obs(self, obs: dict) -> torch.Tensor:
        if "base_action" not in obs:
            raise KeyError("GatedDiffusionPolicy requires obs['base_action'].")
        base_action = obs["base_action"]
        if base_action.dim() >= 4:
            base_action = base_action[:, -1, : self.action_prediction_horizon]
        elif base_action.dim() == 3:
            base_action = base_action[:, : self.action_prediction_horizon]
        else:
            raise ValueError(
                "base_action must have shape (B, T, H, A) or (B, H, A), "
                f"got {base_action.shape}."
            )
        if base_action.shape[1] < self.action_prediction_horizon:
            pad = base_action[:, -1:].repeat(
                1,
                self.action_prediction_horizon - base_action.shape[1],
                1,
            )
            base_action = torch.cat([base_action, pad], dim=1)
        return base_action

    def _maybe_mask_base_action(self, base_action: torch.Tensor) -> torch.Tensor:
        if not self.training:
            if self._zero_mask_base_action_at_inference:
                return torch.zeros_like(base_action)
            return base_action

        ratio = self._base_action_mask_ratio
        stage = self._current_stage()
        if stage == "stage1" and self.stage1_base_action_mask_ratio is not None:
            ratio = self.stage1_base_action_mask_ratio
        elif stage == "stage2" and self.stage2_base_action_mask_ratio is not None:
            ratio = self.stage2_base_action_mask_ratio

        if ratio <= 0.0:
            return base_action
        mask_shape = base_action.shape[:-1]
        keep_mask = torch.rand(mask_shape, device=base_action.device) >= ratio
        return base_action * keep_mask.unsqueeze(-1).to(base_action.dtype)

    def _blend_actions(
        self,
        diffusion_action: torch.Tensor,
        base_action: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        gate = gate.to(device=diffusion_action.device, dtype=diffusion_action.dtype)
        base_action = base_action.to(device=diffusion_action.device, dtype=diffusion_action.dtype)
        return gate * diffusion_action + (1.0 - gate) * base_action

    def _predict_original_sample(
        self,
        noisy_trajs: torch.Tensor,
        pred_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=noisy_trajs.device,
            dtype=noisy_trajs.dtype,
        )
        alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        prediction_type = getattr(self.noise_scheduler.config, "prediction_type", "epsilon")
        if prediction_type == "epsilon":
            pred_original = (noisy_trajs - beta_prod_t.sqrt() * pred_noise) / alpha_prod_t.sqrt()
        elif prediction_type == "sample":
            pred_original = pred_noise
        elif prediction_type == "v_prediction":
            pred_original = alpha_prod_t.sqrt() * noisy_trajs - beta_prod_t.sqrt() * pred_noise
        else:
            raise ValueError(f"Unsupported diffusion prediction_type: {prediction_type}")
        return pred_original.clamp(-1.0, 1.0)

    @torch.no_grad()
    def _sample_diffusion_action(self, obs: dict) -> torch.Tensor:
        batch_size = get_batch_size(obs, strict=True)
        noisy_traj = torch.randn(
            size=(batch_size, self.action_prediction_horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_denoise_steps_per_inference)
        for t in scheduler.timesteps:
            pred_noise, _ = self.forward(obs, noisy_traj, t)
            noisy_traj = scheduler.step(
                pred_noise, t, noisy_traj, **self.noise_scheduler_step_kwargs
            ).prev_sample
        return noisy_traj

    def _masked_mean(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=value.device, dtype=value.dtype)
        return (value * mask).sum() / mask.sum().clamp_min(1)
