import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from hydra.utils import instantiate
from il_lib.nn.features import ObsTokenizer
from il_lib.nn.gpt import GPT
from il_lib.nn.diffusion import WholeBodyUNetDiffusionHead
from il_lib.optim import CosineScheduleFunction, default_optimizer_groups, check_optimizer_groups
from il_lib.policies.policy_base import BasePolicy
from il_lib.training.trainer import rank_zero_info
from il_lib.utils.array_tensor_utils import any_slice, get_batch_size
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from typing import Any, Dict, List, Optional

from omnigibson.learning.utils.eval_utils import CAMERA_INTRINSICS, JOINT_RANGE_ARRAY, ROBOT_CAMERA_NAMES
from omnigibson.learning.utils.obs_utils import process_fused_point_cloud, color_pcd_vis


class WBVIMA(BasePolicy):
    """
    WB-VIMA policy from Jiang et al. https://arxiv.org/abs/2503.05652
    """

    def __init__(
        self,
        *args,
        prop_dim: int,
        prop_keys: List[str],
        num_latest_obs: int,
        # ====== Obs Tokenizer ======
        feature_extractors: Dict[str, DictConfig],
        use_modality_type_tokens: bool,
        # ====== Transformer ======
        xf_n_embd: int,
        xf_n_layer: int,
        xf_n_head: int,
        xf_dropout_rate: float,
        xf_use_geglu: bool,
        # ====== Action Decoding ======
        learnable_action_readout_token: bool,
        action_dim: int,
        action_prediction_horizon: int,
        diffusion_step_embed_dim: int,
        unet_down_dims: List[int],
        unet_kernel_size: int,
        unet_n_groups: int,
        unet_cond_predict_scale: bool,
        action_keys: List[str],
        action_key_dims: dict[str, int],
        # ====== Diffusion ======
        noise_scheduler: DictConfig,
        noise_scheduler_step_kwargs: Optional[dict] = None,
        num_denoise_steps_per_inference: int,
        # ====== learning ======
        lr: float = 1e-5,
        use_cosine_lr: bool = False,
        lr_warmup_steps: Optional[int] = None,
        lr_cosine_steps: Optional[int] = None,
        lr_cosine_min: float = 5e-6,
        lr_layer_decay: float = 1.0,
        weight_decay: float = 0.0,
        loss_on_latest_obs_only: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._prop_keys = prop_keys
        self.obs_tokenizer = ObsTokenizer(
            extractors={
                k: instantiate(v) for k, v in feature_extractors.items()
            },
            use_modality_type_tokens=use_modality_type_tokens,
            token_dim=xf_n_embd,
            token_concat_order=list(feature_extractors.keys()),
            strict=True,
        )
        self.num_latest_obs = num_latest_obs
        if learnable_action_readout_token:
            self.action_readout_token = nn.Parameter(torch.zeros(xf_n_embd))
        else:
            self.action_readout_token = torch.zeros(xf_n_embd)
        self.transformer = GPT(
            n_embd=xf_n_embd,
            n_layer=xf_n_layer,
            n_head=xf_n_head,
            dropout=xf_dropout_rate,
            use_geglu=xf_use_geglu,
        )
        self.action_decoder = WholeBodyUNetDiffusionHead(
            whole_body_decoding_order=["base", "torso", "arms"],
            action_dim_per_part={"base": 3, "torso": 4, "arms": 16},
            obs_dim=xf_n_embd,
            action_horizon=action_prediction_horizon,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            noise_scheduler=instantiate(noise_scheduler),
            noise_scheduler_step_kwargs=noise_scheduler_step_kwargs,
            inference_denoise_steps=num_denoise_steps_per_inference,
            unet_down_dims=unet_down_dims,
            unet_kernel_size=unet_kernel_size,
            unet_n_groups=unet_n_groups,
            unet_cond_predict_scale=unet_cond_predict_scale,
        )
        self.action_dim = action_dim
        self.action_prediction_horizon = action_prediction_horizon
        assert sum(action_key_dims.values()) == action_dim
        assert set(action_keys) == set(action_key_dims.keys())
        self._action_keys = action_keys
        self._action_key_dims = action_key_dims

        # Learning specific parameters
        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = lr_layer_decay
        self.weight_decay = weight_decay
        self.loss_on_latest_obs_only = loss_on_latest_obs_only
        # Save hyperparameters
        self.save_hyperparameters()
        # camera intrinsics
        self.camera_intrinsics = dict()
        # TODO: PROPER PROCESSING 
        for camera_id, camera_name in ROBOT_CAMERA_NAMES.items():
            camera_intrinsics = torch.from_numpy(CAMERA_INTRINSICS[camera_id]) / 4.0
            camera_intrinsics[-1, -1] = 1.0  # make it homogeneous
            self.camera_intrinsics[camera_name] = camera_intrinsics

    def forward(self, obs: dict) -> torch.Tensor:
        # construct prop obs
        prop_obs = []
        for prop_key in self._prop_keys:
            if "/" in prop_key:
                group, key = prop_key.split("/")
                prop_obs.append(obs[group][key])
            else:
                prop_obs.append(obs[prop_key])
        prop_obs = torch.cat(prop_obs, dim=-1)  # (B, L, Prop_dim)

        obs_tokens = self.obs_tokenizer(
            {
                "proprioception": prop_obs,
                "pointcloud": obs["pointcloud"],
            }
        )  # (B, L, E), where L is interleaved modalities tokens
        B, _, E = obs_tokens.shape
        action_readout_tokens = self.action_readout_token.view(1, 1, -1).expand(
            B, self.num_latest_obs, -1
        )

        n_tokens_per_step = self.obs_tokenizer.num_tokens_per_step + 1
        n_total_tokens = self.num_latest_obs * n_tokens_per_step
        tokens_in = torch.zeros(
            (B, n_total_tokens, E),
            device=obs_tokens.device,
            dtype=obs_tokens.dtype,
        )
        # insert obs tokens
        for j in range(self.obs_tokenizer.num_tokens_per_step):
            tokens_in[:, j::n_tokens_per_step] = obs_tokens[
                :, j :: self.obs_tokenizer.num_tokens_per_step
            ]
        # insert action readout tokens
        tokens_in[:, self.obs_tokenizer.num_tokens_per_step :: n_tokens_per_step] = (
            action_readout_tokens
        )

        # construct attention mask
        mask = torch.ones(B, n_total_tokens, dtype=torch.bool, device=self.device)
        # we mask action readout tokens
        mask[:, self.obs_tokenizer.num_tokens_per_step :: n_tokens_per_step] = False

        # construct position ids, which starts from 0
        # for all obs tokens in the same step, they share the same position id
        position_ids = torch.zeros(
            (B, n_total_tokens), device=self.device, dtype=torch.long
        )
        p_id = 0
        for t in range(self.num_latest_obs):
            obs_st = t * n_tokens_per_step
            obs_end = obs_st + self.obs_tokenizer.num_tokens_per_step
            action_readout_p = obs_st + self.obs_tokenizer.num_tokens_per_step
            position_ids[:, obs_st:obs_end] = p_id
            p_id += 1
            position_ids[:, action_readout_p] = p_id
            p_id += 1

        # run transformer forward
        tokens_in = rearrange(tokens_in, "B T E -> T B E")
        mask = mask.unsqueeze(1)  # (B, 1, T)
        tokens_out = self.transformer(
            tokens_in, custom_mask=mask, batch_first=False, position_ids=position_ids
        )
        assert tokens_out.shape == (n_total_tokens, B, E)
        tokens_out = rearrange(tokens_out, "T B E -> B T E")
        return tokens_out

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        obs = self.process_data(data_batch=obs, extract_action=False)
        self._action_traj_pred = self._inference(obs=obs, return_last_timestep_only=True)  # dict of (B = 1, T_A, ...)
        self._action_traj_pred = {
            k: v[0].detach().cpu() for k, v in self._action_traj_pred.items()
        }  # dict of (T_A, ...)
        action = torch.cat(list(self._action_traj_pred.values()), dim=1)  # (T_A, A)
        # denormalize action
        return action * (JOINT_RANGE_ARRAY[self.robot_type][1] - JOINT_RANGE_ARRAY[self.robot_type][0]) + JOINT_RANGE_ARRAY[self.robot_type][0]

    def reset(self) -> None:
        pass

    def policy_training_step(self, batch, batch_idx) -> Any:
        batch = self.process_data(data_batch=batch, extract_action=True)
        B = get_batch_size(
            any_slice(batch["actions"], np.s_[0]),
            strict=True,
        )
        # get padding mask
        pad_mask = batch.pop("masks")  # (B, window_size, L_pred_horizon)
        target_action = batch.pop("actions")  # (B, window_size, L_pred_horizon, A)
        gt_action = torch.cat([target_action[k] for k in self._action_keys], dim=-1)
        transformer_output = self.forward(batch)  # (B, L, E), where L is interleaved time and modality tokens
        loss = self._compute_loss(
            transformer_output=transformer_output,
            gt_action=gt_action,
        )  # (B, T_obs, T_act)
        if self.loss_on_latest_obs_only:
            mask = torch.zeros_like(pad_mask)
            mask[:, -1] = 1
            pad_mask = pad_mask * mask
        loss = loss * pad_mask
        action_loss = torch.sum(loss) / pad_mask.sum()
        # sum over action_prediction_horizon dim instead of avg
        action_loss = action_loss * self.action_prediction_horizon
        log_dict = {"diffusion_loss": action_loss}
        loss = action_loss
        return loss, log_dict, B

    def policy_evaluation_step(self, batch, batch_idx) -> Any:
        """
        Will denoise as if it is in rollout
        but no env interaction
        """
        batch = self.process_data(data_batch=batch, extract_action=True)
        B = get_batch_size(
            any_slice(batch["actions"], np.s_[0]),
            strict=True,
        )
        # get padding mask
        pad_mask = batch.pop("masks")  # (B, window_size, L_pred_horizon)
        target_action = batch.pop("actions")  # (B, window_size, L_pred_horizon, A)
        transformer_output = self.forward(batch)  # (B, L, E), where L is interleaved time and modality tokens
        pred_actions = self._inference(
            transformer_output=transformer_output,
            return_last_timestep_only=False,
        )  # dict of (B, window_size, L_pred_horizon, A)
        all_l1 = dict()
        for action_k in pred_actions:
            pred = pred_actions[action_k]
            gt = target_action[action_k]
            l1 = F.l1_loss(pred, gt, reduction="none")  # (B, window_size, L_pred_horizon, A)
            # sum over action dim
            l1 = l1.sum(dim=-1).reshape(pad_mask.shape)  # (B, window_size, L_pred_horizon)
            if self.loss_on_latest_obs_only:
                mask = torch.zeros_like(pad_mask)
                mask[:, -1] = 1
                pad_mask = pad_mask * mask
            all_l1[action_k] = l1 * pad_mask
        # avg on chunks dim, batch dim, and obs window dim so we can compare under different training settings
        all_loss = {
            f"l1_{k}": torch.sum(v) / pad_mask.sum() * self.action_prediction_horizon
            for k, v in all_l1.items()
        }
        summed_l1 = sum(all_loss.values())
        all_loss["l1"] = summed_l1
        return summed_l1, all_loss, B

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer_groups = self._get_optimizer_groups(
            weight_decay=self.weight_decay,
            lr_layer_decay=self.lr_layer_decay,
            lr_scale=1.0,
        )

        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

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
            return ([optimizer], [{"scheduler": scheduler, "interval": "step"}])

        return optimizer

    def _get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        (
            feature_encoder_pg,
            feature_encoder_pid,
        ) = self.obs_tokenizer.get_optimizer_groups(
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )
        transformer_pg, transformer_pid = self.transformer.get_optimizer_groups(
            weight_decay=weight_decay,
            lr_layer_decay=lr_layer_decay,
            lr_scale=lr_scale,
        )
        action_decoder_pg, action_decoder_pid = (
            self.action_decoder.get_optimizer_groups(
                weight_decay=weight_decay,
                lr_layer_decay=lr_layer_decay,
                lr_scale=lr_scale,
            )
        )
        other_pg, _ = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=[
                "action_readout_token",
            ],
            exclude_filter=lambda name, p: id(p)
            in feature_encoder_pid + transformer_pid + action_decoder_pid,
        )
        all_groups = feature_encoder_pg + transformer_pg + action_decoder_pg + other_pg
        _, table_str = check_optimizer_groups(self, all_groups, verbose=True)
        rank_zero_info(table_str)
        return all_groups
    
    @torch.no_grad()
    def _inference(
        self,
        *,
        obs: dict[str, torch.Tensor] | None = None,
        transformer_output: torch.Tensor | None = None,
        return_last_timestep_only: bool,
    ) -> dict[str, torch.Tensor]:
        """
        Compute prediction, should either provide obs or transformer_output
        Args:
            obs: dict of (B, T, ...), where T = num_latest_obs
            transformer_output: (B, L, E), where L = num_latest_obs * n_tokens_per_step
            return_last_timestep_only: Whether to return only the action chunks corresponding to the last timestep.
        Returns:
            dict of (B, T_obs, T_act, A) or (B, T_act, A), where T_act = action prediction horizon.
        """
        assert not (
            obs is None and transformer_output is None
        ), "Provide either obs or transformer_output"
        if transformer_output is None:
            transformer_output = self.forward(obs)
        action_readout_tokens = self._get_action_readout_tokens(transformer_output)
        pred = self.action_decoder.inference(
            obs=action_readout_tokens,
            return_last_timestep_only=return_last_timestep_only,
        )  # (B, T_obs, T_act, A) or (B, T_act, A)
        return {
            "base": pred["base"],
            "torso": pred["torso"],
            "left_arm": pred["arms"][..., :7],
            "left_gripper": pred["arms"][..., 7:8],
            "right_arm": pred["arms"][..., 8:15],
            "right_gripper": pred["arms"][..., 15:16],
        }
    
    def _compute_loss(
        self,
        *,
        obs: dict[str, torch.Tensor] | None = None,
        transformer_output: torch.Tensor | None = None,
        gt_action: torch.Tensor,
    ):
        """
        Compute loss, should either provide obs or transformer_output

        Args:
            obs: dict of (B, T, ...), where T = num_latest_obs
            transformer_output: (B, L, E), where L = num_latest_obs * n_tokens_per_step
            gt_action: Ground truth action of size (B, T_obs, T_act, A), where T_act = action prediction horizon.
                i.e., for each obs, the model predicts T_act future actions.
            mask: Mask of size (B, T_obs, T_act), indicating whether the action is valid.
        """
        assert not (
            obs is None and transformer_output is None
        ), "Provide either obs or transformer_output"
        if transformer_output is None:
            transformer_output = self.forward(obs)
        action_readout_tokens = self._get_action_readout_tokens(transformer_output)
        mobile_base_action = gt_action[..., :3]
        torso_action = gt_action[..., 3:7]
        arms_action = gt_action[..., 7:]
        loss = self.action_decoder.compute_loss(
            obs=action_readout_tokens,
            gt_action={
                "base": mobile_base_action,
                "torso": torso_action,
                "arms": arms_action,
            },
        )
        return loss
    
    def _get_action_readout_tokens(self, transformer_output: torch.Tensor):
        B, _, E = transformer_output.shape
        n_tokens_per_step = self.obs_tokenizer.num_tokens_per_step + 1
        action_readout_tokens = transformer_output[
            :, self.obs_tokenizer.num_tokens_per_step :: n_tokens_per_step
        ]  # (B, T_obs, E)
        assert action_readout_tokens.shape == (B, self.num_latest_obs, E)
        return action_readout_tokens
    
    def process_data(self, data_batch: dict, extract_action: bool = False) -> Any:
        # first, move rgb dim back to the last dim
        for key in data_batch["obs"]:
            if "rgb" in key:
                data_batch["obs"][key] = data_batch["obs"][key].movedim(-3, -1)
        # process observation data
        fused_pcd = process_fused_point_cloud(
            obs=data_batch["obs"],
            camera_intrinsics=self.camera_intrinsics,
            pcd_num_points=4096,
        )[0]
        # if fused_pcd is 2D, we need to expand it to 4D
        if fused_pcd.ndim == 2:
            color_pcd_vis(fused_pcd)
            fused_pcd = fused_pcd.unsqueeze(0).unsqueeze(0).to(self.device)
        data = {
            "pointcloud": {
                "rgb": fused_pcd[..., :3],
                "xyz": fused_pcd[..., 3:],
            },
            "qpos": data_batch["obs"]["qpos"],
            "odom": data_batch["obs"]["odom"],
        }
        if extract_action:
            # extract action from data_batch
            data.update({
                "actions": data_batch["actions"],
                "masks": data_batch["masks"],
            })
        return data

