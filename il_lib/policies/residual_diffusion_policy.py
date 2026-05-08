from typing import Any, List, Optional

import torch
import torch.nn.functional as F

from ..utils.array_tensor_utils import any_concat
from .base_chunk_diffusion_policy import BaseChunkDiffusionPolicy


class ResidualDiffusionPolicy(BaseChunkDiffusionPolicy):
    """
    Diffusion residual policy that predicts correction chunks in the normalized
    action space while reusing the base-chunk diffusion backbone.
    """

    def __init__(
        self,
        *args,
        learn_gripper_action: bool = True,
        include_robot_gripper_action_input: bool = True,
        **kwargs,
    ):
        base_action_mask_ratio = kwargs.get("base_action_mask_ratio", 0.0)
        zero_mask_base_action_at_inference = kwargs.get(
            "zero_mask_base_action_at_inference", False
        )
        if base_action_mask_ratio != 0.0:
            raise ValueError(
                "ResidualDiffusionPolicy does not support base_action_mask_ratio. "
                "Residual policies must always condition on the base action."
            )
        if zero_mask_base_action_at_inference:
            raise ValueError(
                "ResidualDiffusionPolicy does not support zero_mask_base_action_at_inference. "
                "Residual policies must always condition on the base action."
            )
        super().__init__(*args, **kwargs)
        self._learn_gripper_action = learn_gripper_action
        self._include_robot_gripper_action_input = include_robot_gripper_action_input

    def _build_residual_target(
        self,
        *,
        robot_policy_action: torch.Tensor,
        oracle_action: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        robot_policy_action, robot_policy_gripper_action = (
            robot_policy_action[..., :-1],
            robot_policy_action[..., -1:],
        )
        oracle_action, oracle_gripper_action = (
            oracle_action[..., :-1],
            oracle_action[..., -1:],
        )

        robot_policy_gripper_action = torch.where(
            robot_policy_gripper_action >= 0, 1, 0
        )
        oracle_gripper_action = torch.where(oracle_gripper_action >= 0, 1, 0)

        residual_q = oracle_action - robot_policy_action
        residual_gripper = oracle_gripper_action - robot_policy_gripper_action

        if self._learn_gripper_action:
            target_action = torch.cat([residual_q, residual_gripper], dim=-1)
        else:
            target_action = residual_q

        return target_action, robot_policy_gripper_action

    def _forward_step(self, batch, batch_idx, is_train: bool):
        batch["actions"] = any_concat(
            [batch["actions"][k] for k in self._action_keys], dim=-1
        )
        batch = self.process_data(batch, extract_action=True)

        pad_mask = batch.pop("masks")
        int_state = batch.pop("int_state")
        intervention_mask = int_state == 2
        batch.pop("actions")
        robot_policy_action = batch["base_action"]
        oracle_action = batch.pop("oracle_action")

        target_action, robot_policy_gripper_action = self._build_residual_target(
            robot_policy_action=robot_policy_action,
            oracle_action=oracle_action,
        )
        if not self._use_intervention_head:
            target_action = target_action * intervention_mask.unsqueeze(-1).to(
                target_action.dtype
            )
        if self._include_robot_gripper_action_input:
            batch["robot_policy_gripper_action"] = robot_policy_gripper_action

        if target_action.dim() != 4:
            raise ValueError(
                "ResidualDiffusionPolicy expected chunked correction targets with shape "
                f"(B, T, {self.action_prediction_horizon}, A), but got {target_action.shape}."
            )
        if target_action.shape[-2:] != (self.action_prediction_horizon, self.action_dim):
            raise ValueError(
                "Target residual chunk shape mismatch: expected trailing shape "
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

        B = target_action.shape[0]
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
        )
        raw_action_loss = F.mse_loss(pred, noise, reduction="none").mean(dim=-1)
        action_loss = raw_action_loss * chunk_mask
        real_batch_size = chunk_mask.sum().clamp_min(1)
        action_loss = action_loss.sum() / real_batch_size

        if self._use_intervention_head:
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
            loss = action_loss + self._intervention_loss_weight * intervention_loss
        else:
            intervention_loss = action_loss.new_zeros(())
            intervention_acc = action_loss.new_ones(())
            loss = action_loss

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
                full_future_mask = full_future_mask & current_intervention[:, -1:].expand_as(
                    full_future_mask
                )
            elif self._exclude_pre_intervention_from_action_loss:
                full_future_mask = full_future_mask & (target_int_state != 1)

            l1_full_future_horizon = torch.abs(
                pred_action - target_action_for_metrics
            ).mean(dim=-1)
            l1_full_future_horizon = l1_full_future_horizon * full_future_mask
            l1_full_future_horizon = (
                l1_full_future_horizon.sum() / full_future_mask.sum().clamp_min(1)
            )

            deployed_steps = min(
                self.deployed_action_steps, target_action_for_metrics.shape[-2]
            )
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

    def process_data(self, data_batch: dict, extract_action: bool = False) -> Any:
        data = super().process_data(data_batch, extract_action=extract_action)
        if extract_action and "base_action" not in data:
            raise KeyError("ResidualDiffusionPolicy requires policy/base_action in the batch.")
        if extract_action and "oracle_action" not in data:
            raise KeyError("ResidualDiffusionPolicy requires policy/oracle_action in the batch.")
        return data
