from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F

from il_lib.utils.array_tensor_utils import any_concat, get_batch_size

from .base_chunk_diffusion_policy import BaseChunkDiffusionPolicy
from .base_chunk_diffusion_policy import MAX_DEPTH, MIN_DEPTH


class BaseChunkFlowMatchingPolicy(BaseChunkDiffusionPolicy):
    """
    Chunked action policy trained with the pi0-style flow matching objective.

    The model predicts the velocity field from a noisy action chunk to the data
    chunk: x_t = t * noise + (1 - t) * action, target velocity = noise - action.
    Inference starts from Gaussian noise and integrates backward from t=1 to t=0.
    Unlike BaseChunkDiffusionPolicy, this policy does not condition on predicted
    base action chunks.
    """

    def __init__(
        self,
        *args,
        num_flow_steps_per_inference: int = 10,
        time_beta_alpha: float = 1.5,
        time_beta_beta: float = 1.0,
        time_min: float = 0.001,
        time_max: float = 1.0,
        base_action_mask_ratio: float = 0.0,
        zero_mask_base_action_at_inference: bool = False,
        noise_scheduler: Optional[dict] = None,
        **kwargs,
    ):
        del base_action_mask_ratio, zero_mask_base_action_at_inference
        if num_flow_steps_per_inference < 1:
            raise ValueError("num_flow_steps_per_inference must be >= 1.")
        if not 0.0 <= time_min < time_max <= 1.0:
            raise ValueError("Expected 0 <= time_min < time_max <= 1.")

        # The parent class owns the common feature extractors, backbone, optimizer,
        # and intervention head. Its DDIM scheduler is unused here, but providing
        # one keeps initialization backwards-compatible.
        if noise_scheduler is None:
            noise_scheduler = {
                "_target_": "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
                "num_train_timesteps": 100,
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "beta_schedule": "squaredcos_cap_v2",
                "clip_sample": True,
                "set_alpha_to_one": True,
                "steps_offset": 0,
                "prediction_type": "epsilon",
            }
        feature_extractors = kwargs.get("feature_extractors")
        if feature_extractors is not None and "base_action" in feature_extractors:
            feature_extractors = deepcopy(feature_extractors)
            feature_extractors.pop("base_action")
            kwargs["feature_extractors"] = feature_extractors
        super().__init__(
            *args,
            noise_scheduler=noise_scheduler,
            base_action_mask_ratio=0.0,
            zero_mask_base_action_at_inference=False,
            **kwargs,
        )
        self.num_flow_steps_per_inference = num_flow_steps_per_inference
        self.time_beta_alpha = time_beta_alpha
        self.time_beta_beta = time_beta_beta
        self.time_min = time_min
        self.time_max = time_max

    @torch.no_grad()
    def act(self, obs, deterministic=None):
        del deterministic
        obs = self.process_data(obs, extract_action=False)
        action = self._sample_action(obs)
        intervention_dist = self._intervention_dist_from_obs(obs)
        intervention = intervention_dist.mode()
        if self._force_intervention_on_at_inference:
            intervention = torch.ones_like(intervention)
        return action, intervention

    def _forward_step(self, batch, batch_idx, is_train: bool):
        del batch_idx
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
                "BaseChunkFlowMatchingPolicy expected chunked targets with shape "
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
            noise = torch.randn_like(target_action)
            time = self._sample_time(B, target_action.device, target_action.dtype)
            time_expanded = time[:, None, None]
            noisy_trajs = time_expanded * noise + (1.0 - time_expanded) * target_action
            target_velocity = noise - target_action
            pred, intervention_dist = self.forward(
                obs=batch,
                noisy_traj=noisy_trajs,
                diffusion_timesteps=time,
                compute_intervention=self._training_loss_mode == "joint",
            )
            raw_action_loss = F.mse_loss(pred, target_velocity, reduction="none").mean(dim=-1)
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
            "flow_matching_loss": action_loss,
            "action_loss": action_loss,
            "intervention_loss": intervention_loss,
            "intervention_acc": intervention_acc,
        }
        if not is_train:
            pred_action = self._sample_action(batch)
            full_future_mask = pad_mask[:, -1]
            if self._action_loss_on_intervention_only:
                full_future_mask = full_future_mask & current_intervention[:, -1:].expand_as(full_future_mask)
            elif self._exclude_pre_intervention_from_action_loss:
                full_future_mask = full_future_mask & (target_int_state != 1)

            l1_full_future_horizon = torch.abs(pred_action - target_action).mean(dim=-1)
            l1_full_future_horizon = l1_full_future_horizon * full_future_mask
            l1_full_future_horizon = (
                l1_full_future_horizon.sum() / full_future_mask.sum().clamp_min(1)
            )

            deployed_steps = min(self.deployed_action_steps, target_action.shape[-2])
            pred_action_to_deploy = pred_action[:, :deployed_steps]
            target_action_to_deploy = target_action[:, :deployed_steps]
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

    @torch.no_grad()
    def _sample_action(self, obs: dict) -> torch.Tensor:
        B = get_batch_size(obs, strict=True)
        x_t = torch.randn(
            size=(B, self.action_prediction_horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        dt = -1.0 / float(self.num_flow_steps_per_inference)
        time = 1.0
        while time >= -dt / 2.0:
            timesteps = torch.full((B,), time, device=self.device, dtype=self.dtype)
            velocity, _ = self.forward(
                obs,
                x_t,
                timesteps,
                compute_intervention=False,
            )
            x_t = x_t + dt * velocity
            time += dt
        return x_t

    def _sample_time(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        alpha = torch.as_tensor(self.time_beta_alpha, device=device, dtype=torch.float32)
        beta = torch.as_tensor(self.time_beta_beta, device=device, dtype=torch.float32)
        time = torch.distributions.Beta(alpha, beta).sample((batch_size,))
        time = time * (self.time_max - self.time_min) + self.time_min
        return time.to(device=device, dtype=dtype)

    def process_data(self, data_batch: dict, extract_action: bool = False):
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
        if extract_action:
            data.update(
                {
                    "actions": data_batch["actions"],
                    "int_state": data_batch["policy"]["int_state"],
                    "oracle_action": data_batch["policy"]["oracle_action"],
                    "masks": data_batch["masks"],
                }
            )
        return data
