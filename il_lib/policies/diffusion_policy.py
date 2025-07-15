
import torch
from hydra.utils import instantiate
from il_lib.nn.features import SimpleFeatureFusion
from il_lib.optim import check_optimizer_groups
from il_lib.policies.policy_base import BasePolicy
from il_lib.training.trainer import rank_zero_info
from il_lib.utils.array_tensor_utils import any_slice, get_batch_size
from il_lib.utils.functional_utils import call_once
from omegaconf import DictConfig
from typing import Optional, List


class DiffusionPolicy(BasePolicy):
    """
    Class for:
        - Diffusion Policy from Chi et. al. https://arxiv.org/abs/2303.04137v5
        - 3D Diffusion Policy from Ze et. al. https://arxiv.org/abs/2403.03954
    """
    is_sequence_policy = True

    def __init__(
        self,
        *,
        prop_dim: int,
        prop_keys: List[str],
        # ====== Feature Extractors ======
        feature_extractors: DictConfig,
        feature_fusion_hidden_depth: int = 1,
        feature_fusion_hidden_dim: int = 256,
        feature_fusion_output_dim: int = 256,
        feature_fusion_activation: str = "relu",
        feature_fusion_add_input_activation: bool = False,
        feature_fusion_add_output_activation: bool = False,
        # ====== Backbone ======
        backbone: DictConfig,
        action_dim: int,
        action_keys: List[str],
        action_key_dims: dict[str, int],
        num_latest_obs: int,
        # ====== Diffusion ======
        noise_scheduler: DictConfig,
        noise_scheduler_step_kwargs: Optional[dict] = None,
        num_denoise_steps_per_inference: int,
        horizon: int,
    ):
        super().__init__()

        self._prop_keys = prop_keys
        self.feature_extractor = SimpleFeatureFusion(
            extractors={
                k: instantiate(v) for k, v in feature_extractors.items()
            },
            hidden_depth=feature_fusion_hidden_depth,
            hidden_dim=feature_fusion_hidden_dim,
            output_dim=feature_fusion_output_dim,
            activation=feature_fusion_activation,
            add_input_activation=feature_fusion_add_input_activation,
            add_output_activation=feature_fusion_add_output_activation,
        )

        self.backbone = instantiate(backbone)

        self.action_dim = action_dim
        assert sum(action_key_dims.values()) == action_dim
        assert set(action_keys) == set(action_key_dims.keys())
        self._action_keys = action_keys
        self._action_key_dims = action_key_dims

        self.noise_scheduler = instantiate(noise_scheduler)
        self.noise_scheduler_step_kwargs = noise_scheduler_step_kwargs or {}
        self.num_denoise_steps_per_inference = num_denoise_steps_per_inference

        self.horizon = horizon
        self.num_latest_obs = num_latest_obs

    def forward(self, obs, noisy_traj, diffusion_timesteps):
        """
        obs: dict of (B, L, ...), where L = num_latest_obs
        noisy_traj: (B, L, ...), where L = horizon
        diffusion_timesteps: (B,)
        """
        # construct prop obs
        prop_obs = []
        for prop_key in self._prop_keys:
            if "/" in prop_key:
                group, key = prop_key.split("/")
                prop_obs.append(obs[group][key])
            else:
                prop_obs.append(obs[prop_key])
        prop_obs = torch.cat(prop_obs, dim=-1)  # (B, L, Prop_dim)
        obs = {
            "proprioception": prop_obs,
            "multi_view_cameras": obs["multi_view_cameras"],
        }

        self._check_forward_input_shape(obs, noisy_traj, diffusion_timesteps)
        obs_feature = self.feature_extractor(obs)  # (B, T_O, D)

        pred = self.backbone(
            sample=noisy_traj,
            timestep=diffusion_timesteps,
            cond=obs_feature,
        )

        return pred

    @torch.no_grad()
    def act(self, obs):
        B = get_batch_size(obs, strict=True)
        noisy_traj = torch.randn(
            size=(B, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_denoise_steps_per_inference)

        for t in scheduler.timesteps:
            pred = self.forward(obs, noisy_traj, t)
            # denosing
            noisy_traj = scheduler.step(
                pred, t, noisy_traj, **self.noise_scheduler_step_kwargs
            ).prev_sample  # (B, L, action_dim)
        action_prediction = noisy_traj[
            :, self.num_latest_obs - 1 :
        ].clone()  # (B, L, action_dim)
        split_sections = [self._action_key_dims[k] for k in self._action_keys]
        pred = torch.split(action_prediction, split_sections, dim=-1)
        pred = {k: v for k, v in zip(self._action_keys, pred)}
        return pred

    def reset(self) -> None:
        pass
    
    @call_once
    def _check_forward_input_shape(self, obs, noisy_traj, diffusion_timesteps):
        L_obs = get_batch_size(any_slice(obs, 0), strict=True)
        assert (
            L_obs == self.num_latest_obs
        ), f"obs must have length {self.num_latest_obs}"
        L_traj = get_batch_size(any_slice(noisy_traj, 0), strict=True)
        assert L_traj == self.horizon, f"noisy_traj must have length {self.horizon}"

        B_obs = get_batch_size(obs, strict=True)
        B_traj = get_batch_size(noisy_traj, strict=True)
        if diffusion_timesteps.ndim == 0:
            # for inference
            assert B_obs == B_traj, "Batch size must match"
        else:
            B_t = get_batch_size(diffusion_timesteps, strict=True)
            assert B_obs == B_traj == B_t, "Batch size must match"

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        (
            feature_encoder_pg,
            feature_encoder_pid,
        ) = self.feature_extractor.get_optimizer_groups(
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )
        backbone_pg, backbone_pid = self.backbone.get_optimizer_groups(
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )
        all_groups = feature_encoder_pg + backbone_pg
        _, table_str = check_optimizer_groups(self, all_groups, verbose=True)
        rank_zero_info(table_str)
        return all_groups
