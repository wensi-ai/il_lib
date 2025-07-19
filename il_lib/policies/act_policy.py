import copy

import numpy as np
import torch
import torch.nn.functional as F
from il_lib.optim import CosineScheduleFunction, default_optimizer_groups
from torch import nn, Tensor
from tqdm import tqdm
from typing import Any, List, Literal, Optional
from il_lib.policies.policy_base import BasePolicy
from il_lib.utils.array_tensor_utils import any_concat, any_stack, get_batch_size
from il_lib.utils.functional_utils import unstack_sequence_fields
from il_lib.utils.convert_utils import any_to_torch_tensor
from il_lib.utils.print_utils import color_text


__all__ = ["ACT"]


class ACT(BasePolicy):
    """
    Action Chunking with Transformers (ACT) policy from Zhao et. al. https://arxiv.org/abs/2304.13705 
    """
    def __init__(
        self,
        *args,
        # ====== policy ======
        hidden_dim: int,
        dropout: float,
        n_heads,
        num_encoder_layers: int,
        num_decoder_layers: int,
        pre_norm: bool,
        dim_feedforward: int,
        prop_dim: int,
        goal_dim: int,
        action_dim: int,
        num_queries: int,
        kl_weight: float,
        prop_obs_keys: List[str],
        goal_obs_keys: List[str],
        action_prediction_horizon: int,
        # ====== learning ======
        lr: float,
        use_cosine_lr: bool = False,
        lr_warmup_steps: Optional[int] = None,
        lr_cosine_steps: Optional[int] = None,
        lr_cosine_min: Optional[float] = None,
        lr_layer_decay: float = 1.0,
        weight_decay: float = 0.0,
        action_part_order: List[Literal["torso", "left_arm", "right_arm"]],
        # ====== eval ======
        action_steps_to_deploy: int,
        temporal_aggregate: bool,
        temporal_aggregation_factor: float,
        **kwargs,
    ) -> None:
        """
        Initializes the ACT policy.
        Args:
            hidden_dim (int): Dimension of the hidden layers.
            dropout (float): Dropout rate.
            n_heads (int): Number of attention heads.
            num_encoder_layers (int): Number of encoder layers.
            num_decoder_layers (int): Number of decoder layers.
            pre_norm (bool): Whether to use pre-norm in Transformer layers.
            dim_feedforward (int): Dimension of the feedforward network.
            prop_dim (int): Dimension of the robot state input.
            goal_dim (int): Dimension of the goal state input.
            action_dim (int): Dimension of the action output.
            num_queries (int): Number of queries for the Transformer decoder.
            kl_weight (float): Weight for KL divergence loss.
        """
        super().__init__(*args, **kwargs)

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
            TransformerEncoderLayer(
                hidden_dim,
                n_heads,
                dim_feedforward,
                dropout,
                "relu",
                pre_norm,
            ),
            num_encoder_layers,
            nn.LayerNorm(hidden_dim) if pre_norm else None,
        )
        self.num_queries = num_queries
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj_robot_state = nn.Linear(prop_dim, hidden_dim)
        self.input_proj_goal_state = nn.Linear(goal_dim, hidden_dim)
        self.pos = torch.nn.Embedding(2, hidden_dim)

        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_prop_proj = nn.Linear(
            prop_dim, hidden_dim
        )  # project prop to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table", _get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)
        )  # [CLS], qpos, a_seq

        self.kl_weight = kl_weight
        self.action_dim = action_dim

        # ====== learning ======
        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = lr_layer_decay
        self.weight_decay = weight_decay

        self._action_steps_to_deploy = action_steps_to_deploy
        self._temporal_aggregate = temporal_aggregate
        if temporal_aggregate:
            assert temporal_aggregation_factor > 0
        self._temporal_aggregation_factor = temporal_aggregation_factor

        self._obs_statistics = None

    def forward(self, prop: torch.Tensor, goal: torch.Tensor, actions=None, is_pad=None):
        """
        Forward pass of the ACT policy.
        Args:
            prop (torch.Tensor): Robot state input of shape (B, prop_dim).
            goal (torch.Tensor): Goal state input of shape (B, goal_dim).
            actions (torch.Tensor, optional): Action sequence of shape (B, seq_len, action_dim). If None, the model is in evaluation mode.
            is_pad (torch.Tensor, optional): Padding mask of shape (B, seq_len). If None, the model assumes no padding.
        """
        is_training = actions is not None  # train or val
        bs = prop.shape[0]

        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            prop_embed = self.encoder_prop_proj(prop)  # (bs, hidden_dim)
            prop_embed = torch.unsqueeze(prop_embed, dim=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, dim=0).repeat(
                bs, 1, 1
            )  # (bs, 1, hidden_dim)
            encoder_input = torch.cat(
                [cls_embed, prop_embed, action_embed], dim=1
            )  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(
                prop.device
            )  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], dim=1)  # (bs, seq+1)
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
        else:
            mu = logvar = None

        prop = self.input_proj_robot_state(prop)  # (bs, hidden_dim)
        goal = self.input_proj_goal_state(goal)
        transformer_input = torch.stack([prop, goal], dim=1)  # seq length = 2
        hs = self.transformer(
            transformer_input, None, self.query_embed.weight, self.pos.weight
        )[0]
        a_hat = self.action_head(hs)
        return a_hat, [mu, logvar]

    @torch.no_grad()
    def act(self, prop, goal, *args, **kwargs) -> torch.Tensor:
        a_hat = self.forward(prop=prop, goal=goal)[0]
        return a_hat
    
    def reset(self) -> None:
        pass
    
    def policy_training_step(self, batch, batch_idx) -> Any:
        B = batch["actions"].shape[0]
        # obs data is dict of (B, window_size, ...)
        # action chunks is (B, window_size, action_prediction_horizon, A)
        batch = self.process_data(batch, extract_action=True)

        # get padding mask
        pad_mask = batch.pop("masks")  # (B, window_size, L_pred_horizon)
        pad_mask = pad_mask.reshape(-1, pad_mask.shape[-1])  # (B * window_size, L_pred_horizon)
        # ACT assumes true for padding, false for not padding
        pad_mask = ~pad_mask

        prop_obs = any_concat(
            [
                any_to_torch_tensor(
                    main_data_chunk[k], device=self.device, dtype=self.dtype
                )
                for k in self.prop_obs_keys
            ],
            dim=-1,
        )  # (B, window_size, D)
        # flatten first two dims
        prop_obs = prop_obs.reshape(-1, prop_obs.shape[-1])  # (B * window_size, D)

        goal_obs = any_concat(
            [
                any_to_torch_tensor(
                    main_data_chunk[k], device=self.device, dtype=self.dtype
                )
                for k in self.goal_obs_keys
            ],
            dim=-1,
        )  # (B, window_size, D)
        # flatten first two dims
        goal_obs = goal_obs.reshape(-1, goal_obs.shape[-1])  # (B * window_size, D)

        gt_actions = main_data_chunk[
            "action_chunks"
        ]  # already normalized in [-1, 1], shape (B, window_size, L_pred_horizon, A)
        # reindex to get policy actions
        gt_actions = gt_actions[
            ..., self._policy_train_gt_action_reindex
        ]  # (B, T, L_pred_horizon, A)
        # flatten first two dims
        gt_actions = gt_actions.reshape(
            -1, gt_actions.shape[-2], gt_actions.shape[-1]
        )

        loss_dict = self.policy._compute_loss(
            prop=prop_obs,
            goal=goal_obs,
            actions=gt_actions,
            is_pad=pad_mask,
        )
        for k, v in loss_dict.items():
            if k not in all_loss_dict:
                all_loss_dict[k] = []
            all_loss_dict[k].append(v)


        avg_all_loss_dict = {}
        for k, v in all_loss_dict.items():
            avg_all_loss_dict[k] = torch.mean(torch.stack(v), dim=0)
        loss = avg_all_loss_dict["loss"]
        log_dict = {
            "l1": avg_all_loss_dict["l1"],
            "kl": avg_all_loss_dict["kl"],
        }
        return loss, log_dict, B

    def policy_evaluation_step(self, batch, batch_idx) -> Any:
        # handle the case when rollout eval during training, where obs_statistics is still none
        if self._obs_statistics is None:
            print(
                color_text(
                    "\n[INFO] module.obs_statistics is None, using data_module.obs_statistics",
                    color="green",
                )
            )
            self._obs_statistics = self.data_module.obs_statistics

        # pre-compute for standardization purposes
        right_palm_position_mean = any_to_torch_tensor(
            self._obs_statistics["right_palm_position_mean"],
            device=self.device,
        )
        right_palm_position_std = any_to_torch_tensor(
            self._obs_statistics["right_palm_position_std"],
            device=self.device,
        )
        right_palm_linear_vel_mean = any_to_torch_tensor(
            self._obs_statistics["right_palm_linear_vel_mean"],
            device=self.device,
        )
        right_palm_linear_vel_std = any_to_torch_tensor(
            self._obs_statistics["right_palm_linear_vel_std"],
            device=self.device,
        )
        right_palm_angular_vel_mean = any_to_torch_tensor(
            self._obs_statistics["right_palm_angular_vel_mean"],
            device=self.device,
        )
        right_palm_angular_vel_std = any_to_torch_tensor(
            self._obs_statistics["right_palm_angular_vel_std"],
            device=self.device,
        )

        episode_count = 0
        eval_metrics = None
        pbar = tqdm(total=self._n_eval_episodes, desc="Evaluating")

        action_reindex = None

        obs, _ = self.env.reset()
        all_pred_actions_trajs = None
        all_pred_actions_trajs_mask = None
        curr_t = 0
        if self._temporal_aggregate:
            all_pred_actions_trajs = torch.zeros(
                (
                    self.env_cfg.horizon,
                    self.env_cfg.horizon + self.action_prediction_horizon - 1,
                    self.policy.action_dim,
                ),
                dtype=self.dtype,
                device=self.device,
            )
            all_pred_actions_trajs_mask = torch.zeros(
                (
                    self.env_cfg.horizon,
                    self.env_cfg.horizon + self.action_prediction_horizon - 1,
                ),
                dtype=bool,
                device=self.device,
            )

        pred_actions_trajs = torch.zeros(
            (
                self.action_prediction_horizon,
                self.policy.action_dim,
            ),
            dtype=self.dtype,
            device=self.device,
        )
        deployed_action_pointer = None
        obs = obs["custom"]
        goal_position = self.env.goal_position.clone()
        goal_orientation = self.env.goal_orientation.clone()
        # ====== preprocess obs ======
        # create cos q, sin q
        obs["cos_q"] = torch.cos(obs["q"])
        obs["sin_q"] = torch.sin(obs["q"])
        # fill in goal eef pose
        obs["goal_eef_position"] = goal_position
        obs["goal_eef_orientation"] = goal_orientation
        # compute the difference between eef pose and curr pose
        obs["goal_eef_position_delta"] = goal_position.to(device=self.device) - obs[
            "right_palm_position"
        ].to(device=self.device)
        obs["goal_eef_orientation_delta"] = T.quat_distance(
            goal_orientation.to(device=self.device),
            obs["right_palm_orientation"].to(device=self.device),
        )

        # ====== normalization and standardization ======
        # normalize q to [-1, 1] according to joint limits
        obs["q"] = (obs["q"].to(device=self.device) - self.joint_lower_limits) / (
            self.joint_upper_limits - self.joint_lower_limits
        ) * 2 - 1
        # standardize `right_palm_position`
        obs["right_palm_position"] = obs["right_palm_position"].to(device=self.device)
        obs["right_palm_position"] = (
            obs["right_palm_position"] - right_palm_position_mean
        ) / right_palm_position_std
        # standardize `right_palm_linear_vel`
        obs["right_palm_linear_vel"] = obs["right_palm_linear_vel"].to(
            device=self.device
        )
        obs["right_palm_linear_vel"] = (
            obs["right_palm_linear_vel"] - right_palm_linear_vel_mean
        ) / right_palm_linear_vel_std
        # standardize `right_palm_angular_vel`
        obs["right_palm_angular_vel"] = obs["right_palm_angular_vel"].to(
            device=self.device
        )
        obs["right_palm_angular_vel"] = (
            obs["right_palm_angular_vel"] - right_palm_angular_vel_mean
        ) / right_palm_angular_vel_std
        # standardize `goal_eef_position`, using statistics for `right_palm_position`
        obs["goal_eef_position"] = obs["goal_eef_position"].to(device=self.device)
        obs["goal_eef_position"] = (
            obs["goal_eef_position"] - right_palm_position_mean
        ) / right_palm_position_std

        while True:
            prop_obs = any_concat(
                [
                    any_to_torch_tensor(obs[k], device=self.device)
                    for k in self.prop_obs_keys
                ],
                dim=-1,
            )
            goal_obs = any_concat(
                [
                    any_to_torch_tensor(obs[k], device=self.device)
                    for k in self.goal_obs_keys
                ],
                dim=-1,
            )
            prop_obs = prop_obs.unsqueeze(0)
            goal_obs = goal_obs.unsqueeze(0)
            policy_obs_dict = {
                "prop": prop_obs,
                "goal": goal_obs,
            }
            with torch.no_grad():
                new_pred_actions_trajs = self.policy.act(
                    **policy_obs_dict
                )  # (B = 1, action_prediction_horizon, A)
                new_pred_actions_trajs = new_pred_actions_trajs[
                    0
                ]  # (action_prediction_horizon, A)
                pred_actions_trajs[:] = new_pred_actions_trajs[:]
                deployed_action_pointer = 0
            if self._temporal_aggregate:
                all_pred_actions_trajs[
                    curr_t, curr_t : curr_t + self.action_prediction_horizon
                ] = new_pred_actions_trajs[:]
                all_pred_actions_trajs_mask[
                    curr_t, curr_t : curr_t + self.action_prediction_horizon
                ] = True

                all_pred_actions_trajs_curr_t_and_future = all_pred_actions_trajs[
                    :, curr_t : curr_t + self.action_prediction_horizon
                ]  # (T_max, action_prediction_horizon, A)
                all_pred_actions_trajs_mask_curr_t_and_future = (
                    all_pred_actions_trajs_mask[
                        :, curr_t : curr_t + self.action_prediction_horizon
                    ]
                )  # (T_max, action_prediction_horizon)
                aggregated_actions_trajs = []
                for t in range(self.action_prediction_horizon):
                    mask = all_pred_actions_trajs_mask_curr_t_and_future[
                        :, t
                    ]  # (T_max,)
                    valid_pred_actions = all_pred_actions_trajs_curr_t_and_future[
                        mask, t
                    ]  # (T_valid, A)
                    weights = torch.exp(
                        self._temporal_aggregation_factor
                        * torch.arange(
                            len(valid_pred_actions),
                            device=valid_pred_actions.device,
                            dtype=valid_pred_actions.dtype,
                        )
                    )
                    weights = weights / weights.sum()
                    aggregated_action = torch.sum(
                        valid_pred_actions * weights[:, None], dim=0
                    )  # (A,)
                    aggregated_actions_trajs.append(aggregated_action)
                aggregated_actions_trajs = torch.stack(
                    aggregated_actions_trajs
                )  # (action_prediction_horizon, A)
                pred_actions_trajs[:] = aggregated_actions_trajs[:]
            action = pred_actions_trajs[deployed_action_pointer]
            deployed_action_pointer += 1
            action = action.cpu()
            # clip to +- 1
            action = torch.clamp(action, -1, 1)
            action = any_concat(
                [torso_neutral_pos[:2], action, left_arm_neutral_pos], dim=-1
            )  # (16,)
            if action_reindex is None:
                action_reindex = self.env.robot.get_action_reindex(
                    [
                        "torso_joint1",
                        "torso_joint2",
                        "torso_joint3",
                        "torso_joint4",
                        "right_arm_joint1",
                        "right_arm_joint2",
                        "right_arm_joint3",
                        "right_arm_joint4",
                        "right_arm_joint5",
                        "right_arm_joint6",
                        "left_arm_joint1",
                        "left_arm_joint2",
                        "left_arm_joint3",
                        "left_arm_joint4",
                        "left_arm_joint5",
                        "left_arm_joint6",
                    ]
                )
            obs, _, terminated, truncated, _ = self.env.step(action[action_reindex])
            curr_t += 1
            obs = obs["custom"]
            # ====== preprocess obs ======
            # create cos q, sin q
            obs["cos_q"] = torch.cos(obs["q"])
            obs["sin_q"] = torch.sin(obs["q"])
            # fill in goal eef pose
            obs["goal_eef_position"] = goal_position
            obs["goal_eef_orientation"] = goal_orientation
            # compute the difference between eef pose and curr pose
            obs["goal_eef_position_delta"] = goal_position.to(device=self.device) - obs[
                "right_palm_position"
            ].to(device=self.device)
            obs["goal_eef_orientation_delta"] = T.quat_distance(
                goal_orientation.to(device=self.device),
                obs["right_palm_orientation"].to(device=self.device),
            )

            # ====== normalization and standardization ======
            # normalize q to [-1, 1] according to joint limits
            obs["q"] = (obs["q"].to(device=self.device) - self.joint_lower_limits) / (
                self.joint_upper_limits - self.joint_lower_limits
            ) * 2 - 1
            # standardize `right_palm_position`
            obs["right_palm_position"] = obs["right_palm_position"].to(
                device=self.device
            )
            obs["right_palm_position"] = (
                obs["right_palm_position"] - right_palm_position_mean
            ) / right_palm_position_std
            # standardize `right_palm_linear_vel`
            obs["right_palm_linear_vel"] = obs["right_palm_linear_vel"].to(
                device=self.device
            )
            obs["right_palm_linear_vel"] = (
                obs["right_palm_linear_vel"] - right_palm_linear_vel_mean
            ) / right_palm_linear_vel_std
            # standardize `right_palm_angular_vel`
            obs["right_palm_angular_vel"] = obs["right_palm_angular_vel"].to(
                device=self.device
            )
            obs["right_palm_angular_vel"] = (
                obs["right_palm_angular_vel"] - right_palm_angular_vel_mean
            ) / right_palm_angular_vel_std
            # standardize `goal_eef_position`, using statistics for `right_palm_position`
            obs["goal_eef_position"] = obs["goal_eef_position"].to(device=self.device)
            obs["goal_eef_position"] = (
                obs["goal_eef_position"] - right_palm_position_mean
            ) / right_palm_position_std

            if terminated or truncated:
                episode_count += 1
                pbar.update(1)
                metrics = self.env.metrics
                if eval_metrics is None:
                    eval_metrics = {k: [] for k in metrics.keys()}
                for k, v in metrics.items():
                    eval_metrics[k].append(v)

                if episode_count >= self._n_eval_episodes:
                    break

                obs, _ = self.env.reset()
                all_pred_actions_trajs = None
                all_pred_actions_trajs_mask = None
                curr_t = 0
                if self._temporal_aggregate:
                    all_pred_actions_trajs = torch.zeros(
                        (
                            self.env_cfg.horizon,
                            self.env_cfg.horizon + self.action_prediction_horizon - 1,
                            self.policy.action_dim,
                        ),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    all_pred_actions_trajs_mask = torch.zeros(
                        (
                            self.env_cfg.horizon,
                            self.env_cfg.horizon + self.action_prediction_horizon - 1,
                        ),
                        dtype=bool,
                        device=self.device,
                    )

                pred_actions_trajs = torch.zeros(
                    (
                        self.action_prediction_horizon,
                        self.policy.action_dim,
                    ),
                    dtype=self.dtype,
                    device=self.device,
                )
                deployed_action_pointer = None
                obs = obs["custom"]
                goal_position = self.env.goal_position.clone()
                goal_orientation = self.env.goal_orientation.clone()
                # ====== preprocess obs ======
                # create cos q, sin q
                obs["cos_q"] = torch.cos(obs["q"])
                obs["sin_q"] = torch.sin(obs["q"])
                # fill in goal eef pose
                obs["goal_eef_position"] = goal_position
                obs["goal_eef_orientation"] = goal_orientation
                # compute the difference between eef pose and curr pose
                obs["goal_eef_position_delta"] = goal_position.to(
                    device=self.device
                ) - obs["right_palm_position"].to(device=self.device)
                obs["goal_eef_orientation_delta"] = T.quat_distance(
                    goal_orientation.to(device=self.device),
                    obs["right_palm_orientation"].to(device=self.device),
                )

                # ====== normalization and standardization ======
                # normalize q to [-1, 1] according to joint limits
                obs["q"] = (
                    obs["q"].to(device=self.device) - self.joint_lower_limits
                ) / (self.joint_upper_limits - self.joint_lower_limits) * 2 - 1
                # standardize `right_palm_position`
                obs["right_palm_position"] = obs["right_palm_position"].to(
                    device=self.device
                )
                obs["right_palm_position"] = (
                    obs["right_palm_position"] - right_palm_position_mean
                ) / right_palm_position_std
                # standardize `right_palm_linear_vel`
                obs["right_palm_linear_vel"] = obs["right_palm_linear_vel"].to(
                    device=self.device
                )
                obs["right_palm_linear_vel"] = (
                    obs["right_palm_linear_vel"] - right_palm_linear_vel_mean
                ) / right_palm_linear_vel_std
                # standardize `right_palm_angular_vel`
                obs["right_palm_angular_vel"] = obs["right_palm_angular_vel"].to(
                    device=self.device
                )
                obs["right_palm_angular_vel"] = (
                    obs["right_palm_angular_vel"] - right_palm_angular_vel_mean
                ) / right_palm_angular_vel_std
                # standardize `goal_eef_position`, using statistics for `right_palm_position`
                obs["goal_eef_position"] = obs["goal_eef_position"].to(
                    device=self.device
                )
                obs["goal_eef_position"] = (
                    obs["goal_eef_position"] - right_palm_position_mean
                ) / right_palm_position_std

        # aggregate the metrics
        aggregated_metrics = {
            k: torch.mean(any_stack(v).float()) for k, v in eval_metrics.items()
        }
        pbar.close()
        return 0, aggregated_metrics, 1

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

    @property
    def joint_lower_limits(self):
        return torch.tensor(self._q_lo, device=self.device, dtype=torch.float32)

    @property
    def joint_upper_limits(self):
        return torch.tensor(self._q_hi, device=self.device, dtype=torch.float32)
    
    def _get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        head_pg, _ = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
        )
        return head_pg

    def _compute_loss(
        self,
        *,
        prop,
        goal,
        actions,
        is_pad,
    ):
        actions = actions[:, : self.num_queries]
        is_pad = is_pad[:, : self.num_queries]

        a_hat, (mu, logvar) = self.forward(
            prop=prop,
            goal=goal,
            actions=actions,
            is_pad=is_pad,
        )
        total_kld, dim_wise_kld, mean_kld = _kl_divergence(mu, logvar)
        loss_dict = dict()
        all_l1 = F.l1_loss(actions, a_hat, reduction="none")
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        loss_dict["l1"] = l1
        loss_dict["kl"] = total_kld[0]
        loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
        return loss_dict
    

    def process_data(self, data_batch: dict, extract_action: bool = False) -> Any:
        # process observation data
        data = {}
        for k in self.prop_obs_keys:
            data[k] = data_batch["obs"][k]
        return data


class Transformer(nn.Module):

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src,
        mask,
        query_embed,
        pos_embed,
        latent_input=None,
        proprio_input=None,
        additional_pos_embed=None,
    ):
        # TODO flatten only when input has H and W
        if len(src.shape) == 4:  # has H and W
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            # mask = mask.flatten(1)

            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(
                1, bs, 1
            )  # seq, bs, dim
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

            addition_input = torch.stack([latent_input, proprio_input], axis=0)
            src = torch.cat([addition_input, src], axis=0)
        else:
            assert len(src.shape) == 3
            # flatten NxHWxC to HWxNxC
            bs, hw, c = src.shape
            src = src.permute(1, 0, 2)
            pos_embed = pos_embed.unsqueeze(1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        hs = hs.transpose(1, 2)
        return hs


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _get_sinusoid_encoding_table(n_position, d_hid):
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


def _kl_divergence(mu, logvar):
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
