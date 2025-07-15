import torch
import torch.nn as nn
from il_lib.optim.lr_schedule import CosineScheduleFunction
from il_lib.policies.policy_base import BasePolicy
from typing import Any, List, Union, Optional, Tuple
from il_lib.nn.common import MLP
from il_lib.nn.distributions import GMMHead
from il_lib.nn.features import SimpleFeatureFusion, MultiviewResNet18
from il_lib.utils.array_tensor_utils import any_slice, get_batch_size, any_concat
from il_lib.utils.functional_utils import unstack_sequence_fields


class BC_RNN(BasePolicy):
    """
    BC-RNN policy from Mandlekar et. al. https://arxiv.org/abs/2108.03298
    """
    def __init__(
        self,
        *args,
        # ====== policy ======
        prop_dim: int,
        prop_keys: List[str],
        action_keys: List[str],
        obs_mlp_hidden_depth: int,
        obs_mlp_hidden_dim: int,
        resnet_output_dim: int,
        resnet_token_dim: int,
        resnet_enable_random_crop: bool,
        resnet_random_crop_size: Optional[Union[int, List[int]]],
        feature_fusion_hidden_depth: int = 1,
        feature_fusion_hidden_dim: int = 256,
        feature_fusion_output_dim: int = 256,
        feature_fusion_activation: str = "relu",
        feature_fusion_add_input_activation: bool = False,
        feature_fusion_add_output_activation: bool = False,
        # ====== RNN ======
        rnn_n_layers: int = 2,
        rnn_hidden_dim: int = 256,
        # ====== GMM Head ======
        action_dim: int,
        action_net_gmm_n_modes: int = 5,
        action_net_hidden_dim: int,
        action_net_hidden_depth: int,
        action_net_activation: str = "relu",
        deterministic_inference: bool = True,
        gmm_low_noise_eval: bool = True,
        # ====== learning ======
        lr: float,
        use_cosine_lr: bool = False,
        lr_warmup_steps: Optional[int] = None,
        lr_cosine_steps: Optional[int] = None,
        lr_cosine_min: Optional[float] = None,
        lr_layer_decay: float = 1.0,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._prop_keys = prop_keys
        self.feature_extractor = SimpleFeatureFusion(
            extractors={
                "proprioception": MLP(
                    prop_dim,
                    hidden_dim=obs_mlp_hidden_dim,
                    output_dim=obs_mlp_hidden_dim,
                    hidden_depth=obs_mlp_hidden_depth,
                    add_output_activation=True,
                ),
                "multi_view_cameras": MultiviewResNet18(
                    ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"],
                    resnet_output_dim=resnet_output_dim,
                    token_dim=resnet_token_dim,
                    load_pretrained=False,
                    pretrained_ckpt_path=None,
                    enable_random_crop=resnet_enable_random_crop,
                    random_crop_size=resnet_random_crop_size,
                ),
            },
            hidden_depth=feature_fusion_hidden_depth,
            hidden_dim=feature_fusion_hidden_dim,
            output_dim=feature_fusion_output_dim,
            activation=feature_fusion_activation,
            add_input_activation=feature_fusion_add_input_activation,
            add_output_activation=feature_fusion_add_output_activation,
        )

        self.rnn = nn.LSTM(
            input_size=feature_fusion_output_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_n_layers,
            batch_first=True,
        )
        self.action_net = GMMHead(
            rnn_hidden_dim,
            n_modes=action_net_gmm_n_modes,
            action_dim=action_dim,
            hidden_dim=action_net_hidden_dim,
            hidden_depth=action_net_hidden_depth,
            activation=action_net_activation,
            low_noise_eval=gmm_low_noise_eval,
        )
        self._deterministic_inference = deterministic_inference
    
        self.action_keys = action_keys

        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = lr_layer_decay
        self.optimizer = optimizer
        self.weight_decay = weight_decay

    def forward(self, obs: dict, policy_state: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Forward pass of the ACT policy.
        Args:
            obs: dict of (B, L=1, ...) observations
            policy_state: rnn_state of shape (h_0, c_0) or h_0
        Returns:
            action distribution, policy_state
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

        x = self.feature_extractor(obs)
        x, policy_state = self.rnn(x, policy_state)
        return self.action_net(x), policy_state

    @torch.no_grad()
    def act(self, obs: dict, policy_state: torch.Tensor, deterministic: bool=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: dict of (B, L=1, ...) observations
            policy_state: rnn_state of shape (h_0, c_0) or h_0
            deterministic: if True, use mode of the distribution, otherwise sample
        Returns:
            action: (B, A) tensor of actions
            policy_state: updated rnn_state
        """
        assert (
            get_batch_size(any_slice(obs, 0), strict=True) == 1
        ), "Use L=1 for act"
        dist, policy_state = self.forward(obs, policy_state)
        if deterministic is None:
            deterministic = self._deterministic_inference
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        # action is (B, L=1, A), reduce to (B, A)
        action = action[:, 0]
        return action, policy_state

    def reset(self) -> None:
        pass

    def policy_training_step(self, batch, batch_idx) -> Any:
        batch["actions"] = any_concat(
            [batch["actions"][k] for k in self.action_keys], dim=-1
        )  # (N_chunks, B, ctx_len, A)
        B, T = batch["actions"].shape[1:3]
        # main data is dict of (N_chunks, B, ctx_len, ...)
        # we loop over chunk dim
        main_data = unstack_sequence_fields(
            batch, batch_size=get_batch_size(batch, strict=True)
        )
        all_loss = []
        all_l1 = []
        real_batch_size = []
        for i, main_data_chunk in enumerate(main_data):
            policy_state = self._get_initial_state(B)
            # get padding mask
            pad_mask = main_data_chunk.pop("pad_mask")

            obs = {
                "pointcloud": main_data_chunk["pointcloud"],
            }
            if "multi_view_cameras" in main_data_chunk:
                obs["multi_view_cameras"] = main_data_chunk["multi_view_cameras"]
            for k in self._prop_keys:
                if "/" in k:
                    group, key = k.split("/")
                    if group not in obs:
                        obs[group] = {}
                    obs[group][key] = main_data_chunk[group][key]
                else:
                    obs[k] = main_data_chunk[k]

            trajectories = main_data_chunk[
                "actions"
            ]  # already normalized in [-1, 1], (B, T, A)
            pi = self.forward(obs, policy_state)[0]
            action_loss = pi.imitation_loss(trajectories, reduction="none").reshape(
                pad_mask.shape
            )
            # reduce the loss according to the action mask
            # "True" indicates should calculate the loss
            action_loss = action_loss * pad_mask
            all_loss.append(action_loss)
            # minus because imitation_accuracy returns negative l1 distance
            l1 = -pi.imitation_accuracy(trajectories, pad_mask)
            all_l1.append(l1)
            real_batch_size.append(pad_mask.sum())
        real_batch_size = torch.sum(torch.stack(real_batch_size)).item()
        action_loss = torch.sum(torch.stack(all_loss)) / real_batch_size
        l1 = torch.mean(torch.stack(all_l1))
        log_dict = {"gmm_loss": action_loss, "l1": l1}
        loss = action_loss
        return loss, log_dict, real_batch_size

    def policy_evaluation_step(self, *args, **kwargs) -> Any:
        with torch.no_grad():
            return self.policy_training_step(*args, **kwargs)

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
    
    def _get_initial_state(self, batch_size: int):
        h_0 = torch.zeros(
            self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=self.device
        )
        c_0 = torch.zeros_like(h_0)
        return h_0, c_0