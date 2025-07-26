import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from il_lib.nn.features import SimpleFeatureFusion
from il_lib.optim import CosineScheduleFunction
from il_lib.policies.policy_base import BasePolicy
from il_lib.utils.array_tensor_utils import any_slice, get_batch_size, any_concat
from il_lib.utils.functional_utils import call_once
from omegaconf import DictConfig
from typing import Any, Dict, Optional, List

from omnigibson.learning.utils.eval_utils import JOINT_RANGE_ARRAY


class DiffusionPolicy(BasePolicy):
    """
    Class for:
        - Diffusion Policy from Chi et. al. https://arxiv.org/abs/2303.04137v5
        - 3D Diffusion Policy from Ze et. al. https://arxiv.org/abs/2403.03954
    """
    is_sequence_policy = True

    def __init__(
        self,
        *args,
        prop_dim: int,
        prop_keys: List[str],
        # ====== Feature Extractors ======
        feature_extractors: Dict[str, DictConfig],
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
        deployed_action_steps: int,
        # ====== Diffusion ======
        noise_scheduler: DictConfig,
        noise_scheduler_step_kwargs: Optional[dict] = None,
        num_denoise_steps_per_inference: int,
        horizon: int,
        # ====== Learning ======
        lr: float,
        use_cosine_lr: bool = False,
        lr_warmup_steps: Optional[int] = None,
        lr_cosine_steps: Optional[int] = None,
        lr_cosine_min: Optional[float] = None,
        lr_layer_decay: float = 1.0,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._prop_keys = prop_keys
        self._features = set(feature_extractors.keys())
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
        
        assert sum(action_key_dims.values()) == action_dim
        assert set(action_keys) == set(action_key_dims.keys())
        self.action_dim = action_dim
        self._action_keys = action_keys
        self._action_key_dims = action_key_dims

        self.noise_scheduler = instantiate(noise_scheduler)
        self.noise_scheduler_step_kwargs = noise_scheduler_step_kwargs or {}
        self.num_denoise_steps_per_inference = num_denoise_steps_per_inference

        self.horizon = horizon
        self.num_latest_obs = num_latest_obs
        self.deployed_action_steps = deployed_action_steps
        # ====== Learning ======
        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = lr_layer_decay
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        # Save hyperparameters
        self.save_hyperparameters()

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
        obs["proprioception"] = prop_obs
        obs = {k: obs[k] for k in self._features}  # filter obs to only include features we have
        self._check_forward_input_shape(obs, noisy_traj, diffusion_timesteps)
        obs_feature = self.feature_extractor(obs)  # (B, T_O, D)

        pred = self.backbone(
            sample=noisy_traj,
            timestep=diffusion_timesteps,
            cond=obs_feature,
        )

        return pred

    @torch.no_grad()
    def act(self, obs: dict) -> torch.Tensor:
        obs = self.process_data(obs, extract_action=False)
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
        action = noisy_traj[:, self.num_latest_obs - 1:].clone().cpu()  # (B, L, action_dim)
        # denormalize action
        return (action + 1) / 2 * (
            JOINT_RANGE_ARRAY[self.robot_type][1] - JOINT_RANGE_ARRAY[self.robot_type][0]
        ) + JOINT_RANGE_ARRAY[self.robot_type][0]

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

    def policy_training_step(self, batch, batch_idx):
        batch["actions"] = any_concat(
            [batch["actions"][k] for k in self._action_keys], dim=-1
        )  # (B, ctx_len, A)
        B = batch["actions"].shape[0]
        batch = self.process_data(batch, extract_action=True)

        # get padding mask
        pad_mask = batch.pop("masks")
        trajectories = batch.pop("actions")  # already normalized in [-1, 1], (B, T, A)
        # sample noise
        noise = torch.randn(trajectories.shape, device=trajectories.device)
        # sample diffusion timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=trajectories.device,
        ).long()
        noisy_trajs = self.noise_scheduler.add_noise(
            trajectories, noise, timesteps
        )
        pred = self.forward(
            obs=batch, noisy_traj=noisy_trajs, diffusion_timesteps=timesteps
        )
        action_loss = F.mse_loss(pred, noise, reduction="none")  # (B, L, A)
        action_loss = action_loss.mean(dim=-1).reshape(pad_mask.shape)  # (B, L)
        # reduce the loss according to the action mask
        # "True" indicates should calculate the loss
        action_loss = action_loss * pad_mask
        real_batch_size = pad_mask.sum()
        action_loss = action_loss.sum() / real_batch_size
        log_dict = {"diffusion_loss": action_loss}
        loss = action_loss
        return loss, log_dict, real_batch_size

    def policy_evaluation_step(self, batch, batch_idx):
        """
        Will denoise as if it is in rollout
        but no env interaction
        """
        batch["actions"] = any_concat(
            [batch["actions"][k] for k in self._action_keys], dim=-1
        )  # (B, ctx_len, A)
        batch = self.process_data(batch, extract_action=True)
        # get padding mask
        pad_mask = batch.pop("masks")
        gt_actions = batch.pop("actions")  # already normalized in [-1, 1]

        noisy_traj = torch.randn(
            size=gt_actions.shape,
            device=self.device,
            dtype=self.dtype,
        )
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_denoise_steps_per_inference)
        for t in scheduler.timesteps:
            pred = self.forward(batch, noisy_traj, t)
            # denosing
            noisy_traj = scheduler.step(
                pred, t, noisy_traj, **self.noise_scheduler_step_kwargs
            ).prev_sample  # (B, T, action_dim)
        pred_actions = noisy_traj[:, self.num_latest_obs - 1:]
        gt_actions = gt_actions[:, self.num_latest_obs - 1:]
        pad_mask = pad_mask[:, self.num_latest_obs - 1:]
        l1_full_future_horizon = F.l1_loss(
            pred_actions, gt_actions, reduction="none"
        )  # (B, L, A)
        l1_full_future_horizon = l1_full_future_horizon.mean(dim=-1).reshape(
            pad_mask.shape
        )  # (B, L)
        l1_full_future_horizon = l1_full_future_horizon * pad_mask
        real_batch_size_full_future_horizon = pad_mask.sum()

        deployed_start_t = self.num_latest_obs - 1
        deployed_end_t = deployed_start_t + self.deployed_action_steps
        pred_actions_to_deploy = pred_actions[:, deployed_start_t:deployed_end_t]
        gt_actions = gt_actions[:, deployed_start_t:deployed_end_t]
        pad_mask = pad_mask[:, deployed_start_t:deployed_end_t]
        l1_deployed_steps_only = F.l1_loss(
            pred_actions_to_deploy, gt_actions, reduction="none"
        )  # (B, L, A)
        l1_deployed_steps_only = l1_deployed_steps_only.mean(dim=-1).reshape(
            pad_mask.shape
        )  # (B, L)
        l1_deployed_steps_only = l1_deployed_steps_only * pad_mask
        real_batch_size_deployed_steps_only = pad_mask.sum()

        l1_full_future_horizon = l1_full_future_horizon.sum() / real_batch_size_full_future_horizon
        l1_deployed_steps_only = l1_deployed_steps_only.sum() / real_batch_size_deployed_steps_only
        return (
            l1_full_future_horizon,
            {
                "l1": l1_full_future_horizon,
                "l1_full_future_horizon": l1_full_future_horizon,
                "l1_deployed_steps_only": l1_deployed_steps_only,
            },
            1,
        )

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
                base_value=1.0,  # anneal from the original LR value
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
            return (
                [optimizer],
                [{"scheduler": scheduler, "interval": "step"}],
            )

        return optimizer

    def process_data(self, data_batch: dict, extract_action: bool = False) -> Any:
        # process observation data
        data = {
            "qpos": data_batch["obs"]["qpos"],
            "odom": data_batch["obs"]["odom"],
        }
        if "rgb" in self._features:
            data["rgb"] = {k: data_batch["obs"][k].float() / 255.0 for k in data_batch["obs"] if "rgb" in k}
        if "pcd" in self._features:
            data["pcd"] = {
                "rgb": data_batch["obs"]["pcd"][..., :3],
                "xyz": data_batch["obs"]["pcd"][..., 3:],
            }
        if extract_action:
            # extract action from data_batch
            data.update({
                "actions": data_batch["actions"],
                "masks": data_batch["masks"],
            })
        return data