import torch
from hydra.utils import instantiate
from il_lib.nn.distributions import CategoricalNet, GMMHead
from il_lib.nn.features import SimpleFeatureFusion
from il_lib.optim import CosineScheduleFunction
from il_lib.policies.policy_base import BasePolicy
from il_lib.utils.array_tensor_utils import any_concat
from omnigibson.learning.utils.obs_utils import MAX_DEPTH, MIN_DEPTH
from typing import Any, Dict, List


class _AlwaysInterveneDistribution:
    """Fallback distribution used when the intervention head is disabled."""

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


class BaseChunkPolicy(BasePolicy):
    def __init__(
        self,
        *args,
        prop_dim: int,
        prop_keys: List[str],
        action_keys: List[str],
        feature_extractors: Dict[str, Dict],
        feature_fusion_hidden_depth: int = 1,
        feature_fusion_hidden_dim: int = 256,
        feature_fusion_output_dim: int = 256,
        feature_fusion_activation: str = "relu",
        feature_fusion_add_input_activation: bool = False,
        feature_fusion_add_output_activation: bool = False,
        action_dim: int,
        action_net_gmm_n_modes: int = 5,
        action_net_hidden_dim: int = 128,
        action_net_hidden_depth: int = 1,
        action_net_activation: str = "relu",
        action_prediction_horizon: int = 1,
        deployed_action_steps: int | None = None,
        gmm_low_noise_eval: bool = True,
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
        lr: float = 1e-4,
        use_cosine_lr: bool = True,
        lr_warmup_steps: int | None = None,
        lr_cosine_steps: int | None = None,
        lr_cosine_min: float | None = None,
        lr_layer_decay: float = 1.0,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not 0.0 <= base_action_mask_ratio <= 1.0:
            raise ValueError("base_action_mask_ratio must be in [0, 1].")

        self._prop_dim = prop_dim
        self._prop_keys = prop_keys
        self._action_keys = action_keys
        self._features = set(feature_extractors.keys())
        self.action_dim = action_dim
        self.action_prediction_horizon = action_prediction_horizon
        self.deployed_action_steps = (
            action_prediction_horizon
            if deployed_action_steps is None
            else deployed_action_steps
        )
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
        self.action_net = GMMHead(
            input_dim=feature_fusion_output_dim,
            n_modes=action_net_gmm_n_modes,
            action_dim=action_dim * action_prediction_horizon,
            hidden_dim=action_net_hidden_dim,
            hidden_depth=action_net_hidden_depth,
            activation=action_net_activation,
            low_noise_eval=gmm_low_noise_eval,
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
        self._force_intervention_on_at_inference = force_intervention_on_at_inference
        self._deterministic_inference = deterministic_inference
        self._intervention_loss_weight = intervention_loss_weight

        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = lr_layer_decay
        self.optimizer = optimizer
        self.weight_decay = weight_decay

    def forward(self, obs):
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
        obs = {k: obs[k] for k in self._features}

        if "base_action" in obs:
            obs["base_action"] = self._format_action_chunk(obs["base_action"])
            obs["base_action"] = self._maybe_mask_base_action(obs["base_action"])
            if obs["base_action"].shape[1] == 1 and obs_time > 1:
                obs["base_action"] = obs["base_action"].expand(-1, obs_time, -1)

        obs_feature = self.feature_extractor(obs)
        if obs_feature.dim() >= 3:
            obs_feature = obs_feature[:, -1:]
        action_dist = self.action_net(obs_feature)
        if self._use_intervention_head:
            intervention_dist = self.intervention_head(obs_feature)
        else:
            intervention_dist = _AlwaysInterveneDistribution(obs_feature)
        return action_dist, intervention_dist

    @torch.no_grad()
    def act(self, obs, deterministic=None):
        if deterministic is None:
            deterministic = self._deterministic_inference
        action_dist, intervention_dist = self.forward(obs)
        action = action_dist.mode() if deterministic else action_dist.sample()
        intervention = intervention_dist.mode() if deterministic else intervention_dist.sample()
        if self._force_intervention_on_at_inference:
            intervention = torch.ones_like(intervention)
        return self._unflatten_action_chunk(action), intervention

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
        return self._forward_step(batch, batch_idx, is_train=True)

    def policy_evaluation_step(self, batch, batch_idx):
        return self._forward_step(batch, batch_idx, is_train=False)

    def _forward_step(self, batch, batch_idx, is_train: bool):
        batch["actions"] = any_concat(
            [batch["actions"][k] for k in self._action_keys], dim=-1
        )
        batch = self.process_data(batch, extract_action=True)

        pad_mask = batch.pop("masks")
        intervention_mask = batch.pop("int_state") == 2
        batch.pop("oracle_action")
        target_action = batch.pop("actions")

        if target_action.dim() == 4:
            target_action = self._format_action_chunk(target_action)
            if self._use_intervention_head:
                action_valid_mask = self._current_mask(intervention_mask) & pad_mask.all(dim=-1)
            else:
                action_valid_mask = pad_mask.all(dim=-1)
        elif self.action_prediction_horizon != 1:
            raise ValueError(
                "BaseChunkPolicy expected chunked targets with shape "
                f"(B, T, {self.action_prediction_horizon}, A), but got {target_action.shape}."
            )
        else:
            action_valid_mask = pad_mask & intervention_mask if self._use_intervention_head else pad_mask

        pi, intervention_dist = self.forward(batch)
        raw_action_loss = pi.imitation_loss(
            target_action,
            reduction="none",
        ).reshape(action_valid_mask.shape)
        action_loss = raw_action_loss * action_valid_mask
        real_batch_size = action_valid_mask.sum().clamp_min(1)
        action_loss = torch.sum(action_loss) / real_batch_size

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
            intervention_loss = torch.sum(intervention_loss) / intervention_loss_mask.sum().clamp_min(1)
            loss = action_loss + self._intervention_loss_weight * intervention_loss
        else:
            intervention_loss = action_loss.new_zeros(())
            intervention_acc = action_loss.new_ones(())
            loss = action_loss

        log_dict = {
            "action_loss": action_loss,
            "intervention_loss": intervention_loss,
            "intervention_acc": intervention_acc,
        }
        if not is_train:
            pred_action = self._unflatten_action_chunk(pi.mode())
            target_action_for_metrics = self._unflatten_action_chunk(target_action)

            if target_action_for_metrics.dim() >= 4:
                full_future_mask = pad_mask
                if self._use_intervention_head:
                    full_future_mask = full_future_mask & intervention_mask
                l1_full_future_horizon = torch.abs(
                    pred_action - target_action_for_metrics
                ).mean(dim=-1)
                l1_full_future_horizon = l1_full_future_horizon * full_future_mask
                l1_full_future_horizon = (
                    l1_full_future_horizon.sum() / full_future_mask.sum().clamp_min(1)
                )

                deployed_steps = min(self.deployed_action_steps, target_action_for_metrics.shape[-2])
                pred_action_to_deploy = pred_action[..., :deployed_steps, :]
                target_action_to_deploy = target_action_for_metrics[..., :deployed_steps, :]
                deployed_mask = full_future_mask[..., :deployed_steps]
                l1_deployed_steps_only = torch.abs(
                    pred_action_to_deploy - target_action_to_deploy
                ).mean(dim=-1)
                l1_deployed_steps_only = l1_deployed_steps_only * deployed_mask
                l1_deployed_steps_only = (
                    l1_deployed_steps_only.sum() / deployed_mask.sum().clamp_min(1)
                )
            else:
                l1_deployed_steps_only = torch.abs(
                    pred_action - target_action_for_metrics
                ).mean(dim=-1)
                l1_deployed_steps_only = l1_deployed_steps_only * action_valid_mask
                l1_deployed_steps_only = (
                    l1_deployed_steps_only.sum() / action_valid_mask.sum().clamp_min(1)
                )
                l1_full_future_horizon = l1_deployed_steps_only

            log_dict["l1"] = l1_deployed_steps_only
            log_dict["l1_full_future_horizon"] = l1_full_future_horizon
            log_dict["l1_deployed_steps_only"] = l1_deployed_steps_only
        return loss, log_dict, real_batch_size

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

    def _unflatten_action_chunk(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_prediction_horizon == 1:
            return action
        return action.reshape(
            *action.shape[:-1],
            self.action_prediction_horizon,
            self.action_dim,
        )

    def _current_mask(self, pad_mask: torch.Tensor) -> torch.Tensor:
        if pad_mask.dim() >= 3:
            return pad_mask[..., 0]
        return pad_mask

    def _current_intervention(self, intervention_mask: torch.Tensor) -> torch.Tensor:
        if intervention_mask.dim() >= 3:
            return intervention_mask[..., 0]
        return intervention_mask

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
                k.rsplit("::", 1)[0]: (data_batch["obs"][k].float() - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
                for k in data_batch["obs"]
                if "depth" in k
            }
            data["rgbd"] = {k: {"rgb": rgb[k], "depth": depth[k].unsqueeze(-3)} for k in rgb}
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
                    "actions": data_batch["actions"],
                    "int_state": data_batch["policy"]["int_state"],
                    "base_action": data_batch["policy"]["base_action"],
                    "oracle_action": data_batch["policy"]["oracle_action"],
                    "masks": data_batch["masks"],
                }
            )
        return data
