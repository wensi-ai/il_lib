import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from il_lib.optim import CosineScheduleFunction, default_optimizer_groups
from il_lib.nn.transformers import (
    build_position_encoding,
    Transformer, TransformerEncoderLayer, TransformerEncoder
)
from il_lib.nn.features import MultiviewResNet18
from il_lib.policies.policy_base import BasePolicy
from il_lib.utils.array_tensor_utils import any_concat
from omegaconf import DictConfig, OmegaConf
from omnigibson.learning.utils.obs_utils import MAX_DEPTH, MIN_DEPTH
from torch.autograd import Variable
from typing import Any, List, Optional

__all__ = ["ACT"]


class ACT(BasePolicy):
    """
    Action Chunking with Transformers (ACT) policy from Zhao et. al. https://arxiv.org/abs/2304.13705 
    """
    def __init__(
        self,
        *args,
        prop_dim: int,
        prop_keys: List[str],
        action_dim: int,
        action_keys: List[str],
        obs_backbone: DictConfig,
        pos_encoding: DictConfig,
        # ====== policy ======
        num_queries: int,
        hidden_dim: int,
        dropout: float,
        n_heads: int,
        dim_feedforward: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        pre_norm: bool,
        kl_weight: float,
        # ====== learning ======
        lr: float,
        use_cosine_lr: bool = False,
        lr_warmup_steps: Optional[int] = None,
        lr_cosine_steps: Optional[int] = None,
        lr_cosine_min: Optional[float] = None,
        lr_layer_decay: float = 1.0,
        weight_decay: float = 0.0,
        # ====== eval ======
        temporal_aggregate: bool,
        temporal_aggregation_factor: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._prop_keys = prop_keys
        self._action_keys = action_keys 
        self.action_dim = action_dim
        self.obs_backbone = instantiate(obs_backbone)

        self.transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=True,
        )
        self.encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu",
                normalize_before=pre_norm,
            ),
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(hidden_dim) if pre_norm else None,
        )
        self.position_embedding = build_position_encoding(OmegaConf.to_container(pos_encoding))
        self.num_queries = num_queries
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
        self.input_proj_robot_state = nn.Linear(prop_dim, hidden_dim)
        self.pos = torch.nn.Embedding(2, hidden_dim)
        # encoder extra parameters
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_prop_proj = nn.Linear(prop_dim, hidden_dim) # project prop to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)  # project hidden state to latent std, var
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

        # ====== learning ======
        self.kl_weight = kl_weight

        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = lr_layer_decay
        self.weight_decay = weight_decay

        self._temporal_aggregate = temporal_aggregate
        if temporal_aggregate:
            assert temporal_aggregation_factor > 0
        self._temporal_aggregation_factor = temporal_aggregation_factor

        self._obs_statistics = None
        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, obs: dict, actions: Optional[torch.Tensor]=None) -> torch.Tensor:
        is_training = actions is not None
        bs = obs["prop"].shape[0]
        # construct prop obs
        prop_obs = []
        for prop_key in self._prop_keys:
            if "/" in prop_key:
                group, key = prop_key.split("/")
                prop_obs.append(obs[group][key])
            else:
                prop_obs.append(obs[prop_key])
        prop_obs = torch.cat(prop_obs, dim=-1)  # (B, L, Prop_dim)
        # flatten first two dims
        prop_obs = prop_obs.reshape(-1, prop_obs.shape[-1])  # (B * L, Prop_dim)

        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (B, seq, hidden_dim)
            prop_embed = self.encoder_prop_proj(prop_obs)  # (B, hidden_dim)
            prop_embed = torch.unsqueeze(prop_embed, dim=1)  # (B, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, dim=0).repeat(
                bs, 1, 1
            )  # (B, 1, hidden_dim)
            encoder_input = torch.cat(
                [cls_embed, prop_embed, action_embed], dim=1
            )  # (B, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, B, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(
                prop_obs.device
            )  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], dim=1)  # (B, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = self._reparametrize(mu, logvar)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(self.device)

        latent_input = self.latent_out_proj(latent_sample)
        all_cam_features = []
        all_cam_pos = []
        resnet_output = self.obs_backbone(obs["rgb"])  # dict of (B, C, H, W)
        for features in resnet_output.values():
            pos = self.position_embedding(features)
            all_cam_features.append(self.input_proj(features))
            all_cam_pos.append(pos)
        # proprioception features
        proprio_input = self.input_proj_robot_state(prop_obs)
        # fold camera dimension into width dimension
        src = torch.cat(all_cam_features, axis=3)
        pos = torch.cat(all_cam_pos, axis=3)
        hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        a_hat = self.action_head(hs)
        return a_hat, [mu, logvar]

    @torch.no_grad()
    def act(self, obs: dict) -> torch.Tensor:
        a_hat = self.forward(obs=obs)[0]
        return a_hat
    
    def reset(self) -> None:
        pass
    
    def policy_training_step(self, batch, batch_idx) -> Any:
        batch["actions"] = any_concat(
            [batch["actions"][k] for k in self._action_keys], dim=-1
        )  # (B, ctx_len, A)
        B = batch["actions"].shape[0]
        batch = self.process_data(batch, extract_action=True)

        # get padding mask
        pad_mask = batch.pop("masks")  # (B, obs_window_size, L_pred_horizon)
        pad_mask = pad_mask.reshape(-1, pad_mask.shape[-1])  # (B * obs_window_size, L_pred_horizon)
        # ACT assumes true for padding, false for not padding
        pad_mask = ~pad_mask

        gt_actions = batch.pop("actions")  # already normalized in [-1, 1], (B, T, L_pred_horizon, A)
        # flatten first two dims
        gt_actions = gt_actions.reshape(
            -1, gt_actions.shape[-2], gt_actions.shape[-1]
        )  # (B * obs_window_size, L_pred_horizon, A)

        loss_dict = self.policy._compute_loss(
            obs=obs,
            actions=gt_actions,
            is_pad=pad_mask,
        )

        loss = loss_dict["loss"]
        log_dict = {
            "l1": loss_dict["l1"],
            "kl": loss_dict["kl"],
        }
        return loss, log_dict, B

    def policy_evaluation_step(self, batch, batch_idx) -> Any:
        with torch.no_grad():
            return self.policy_training_step(batch, batch_idx)

    def configure_optimizers(self):
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
            return (
                [optimizer],
                [{"scheduler": scheduler, "interval": "step"}],
            )

        return optimizer
    
    def process_data(self, data_batch: dict, extract_action: bool = False) -> Any:
        # process observation data
        data = {"qpos": data_batch["obs"]["qpos"], "eef": data_batch["obs"]["eef"]}
        if "odom" in data_batch["obs"]:
            data["odom"] = data_batch["obs"]["odom"]
        if "rgb" in self._features:
            data["rgb"] = {k.rsplit("::", 1)[0]: data_batch["obs"][k].float() / 255.0 for k in data_batch["obs"] if "rgb" in k}
        if "rgbd" in self._features:
            rgb = {k.rsplit("::", 1)[0]: data_batch["obs"][k].float() / 255.0 for k in data_batch["obs"] if "rgb" in k}
            depth = {k.rsplit("::", 1)[0]: (data_batch["obs"][k].float() - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH) for k in data_batch["obs"] if "depth" in k}
            data["rgbd"] = {k: torch.cat([rgb[k], depth[k].unsqueeze(-3)], dim=-3) for k in rgb}
        if "task" in self._features:
            data["task"] = data_batch["obs"]["task"]
        if extract_action:
            # extract action from data_batch
            data.update({
                "actions": data_batch["actions"],
                "masks": data_batch["masks"],
            })
        return data
    
    def _get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        head_pg, _ = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
        )
        return head_pg

    def _compute_loss(self, obs, actions, is_pad):
        """
        Forward pass for computing the loss.
        """
        actions = actions[:, : self.num_queries]
        is_pad = is_pad[:, : self.num_queries]

        a_hat, (mu, logvar) = self.forward(
            obs=obs,
            actions=actions,
            is_pad=is_pad,
        )
        total_kld = self._kl_divergence(mu, logvar)[0]
        loss_dict = dict()
        all_l1 = F.l1_loss(actions, a_hat, reduction="none")
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        loss_dict["l1"] = l1
        loss_dict["kl"] = total_kld[0]
        loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
        return loss_dict

    def _reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparametrization trick to sample from a Gaussian distribution.
        """
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps
    
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def _kl_divergence(self, mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld, dimension_wise_kld, mean_kld
