import torch
import torch.nn as nn
from hydra.utils import instantiate
from il_lib.optim.lr_schedule import CosineScheduleFunction
from il_lib.policies.policy_base import BasePolicy
from typing import Any, Dict, List, Optional, Tuple
from omegaconf import DictConfig
from omnigibson.learning.utils.obs_utils import MAX_DEPTH, MIN_DEPTH
from il_lib.nn.distributions import GMMHead, MixtureOfGaussian
from il_lib.nn.features import SimpleFeatureFusion
from il_lib.utils.array_tensor_utils import any_slice, get_batch_size, any_concat


class BC_RNN(BasePolicy):
    """
    BC-RNN policy from Mandlekar et. al. https://arxiv.org/abs/2108.03298
    """
    def __init__(
        self,
        *args,
        prop_dim: int,
        prop_keys: List[str],
        action_keys: List[str],
        # ====== Feature Extractors ======
        feature_extractors: Dict[str, DictConfig],
        feature_fusion_hidden_depth: int = 1,
        feature_fusion_hidden_dim: int = 256,
        feature_fusion_output_dim: int = 256,
        feature_fusion_activation: str = "relu",
        feature_fusion_add_input_activation: bool = False,
        feature_fusion_add_output_activation: bool = False,
        # ====== RNN ======
        rnn_n_layers: int = 2,
        rnn_hidden_dim: int = 256,
        rnn_horizon: int = 10,
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
        self._features = set(feature_extractors.keys())
        self.feature_extractor = SimpleFeatureFusion(
            extractors={k: instantiate(v) for k, v in feature_extractors.items()},
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
        self._action_keys = action_keys
        self._rnn_counter = 0
        self._rnn_horizon = rnn_horizon
        self._policy_state = None # (h_0, c_0) for the rnn

        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = lr_layer_decay
        self.optimizer = optimizer
        self.weight_decay = weight_decay

    def forward(self, 
        obs: dict,
        policy_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[MixtureOfGaussian, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the BC-RNN policy.
        Args:
            obs: dict of (B, L=1, ...) observations
            policy_state: rnn_state of shape (h_0, c_0)
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
        obs["proprioception"] = prop_obs
        obs = {k: obs[k] for k in self._features}  # filter obs to only include features we have
        x = self.feature_extractor(obs)
        x, policy_state = self.rnn(x, policy_state)
        return self.action_net(x), policy_state

    @torch.no_grad()
    def act(self, 
        obs: dict,
        deterministic: Optional[bool]=None
    ) -> torch.Tensor:
        """
        Args:
            obs: dict of (B, L=1, ...) observations
            deterministic: if True, use mode of the distribution, otherwise sample
        Returns:
            action: (B, A) tensor of actions
        """
        # process obs
        obs = self.process_data(obs, extract_action=False)
        assert (
            get_batch_size(any_slice(obs, 0), strict=True) == 1
        ), "Use L=1 for act"
        if self._rnn_counter % self._rnn_horizon == 0:
            # reset the rnn state
            self._policy_state = self._get_initial_state(batch_size=1)
        self._rnn_counter += 1
        dist, self._policy_state = self.forward(obs, self._policy_state)
        if deterministic is None:
            deterministic = self._deterministic_inference
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        # denormalize action
        action = action.cpu()   # (B, T, A)
        return self._denormalize_action(action)

    def reset(self) -> None:
        self._rnn_counter = 0

    def policy_training_step(self, batch, batch_idx) -> Any:
        batch["actions"] = any_concat(
            [batch["actions"][k] for k in self._action_keys], dim=-1
        )  # (B, ctx_len, A)
        B = batch["actions"].shape[0]
        batch = self.process_data(batch, extract_action=True)

        policy_state = self._get_initial_state(B)
        # get padding mask
        pad_mask = batch.pop("masks")
        trajectories = batch.pop("actions")  # already normalized in [-1, 1], (B, T, A)
        pi = self.forward(batch, policy_state)[0]
        action_loss = pi.imitation_loss(trajectories, reduction="none").reshape(
            pad_mask.shape
        )
        # reduce the loss according to the action mask
        # "True" indicates should calculate the loss
        action_loss = action_loss * pad_mask
        # minus because imitation_accuracy returns negative l1 distance
        l1 = -pi.imitation_accuracy(trajectories, pad_mask)
        real_batch_size = pad_mask.sum()
        action_loss = action_loss.sum() / real_batch_size
        l1 = torch.mean(l1)
        log_dict = {"gmm_loss": action_loss, "l1": l1}
        loss = action_loss
        return loss, log_dict, real_batch_size

    def policy_evaluation_step(self, batch, batch_idx) -> Any:
        with torch.no_grad():
            return self.policy_training_step(batch, batch_idx)

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
    
    def _get_initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h_0 = torch.zeros(
            self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=self.device
        )
        c_0 = torch.zeros_like(h_0)
        return h_0, c_0

    def process_data(self, data_batch: dict, extract_action: bool = False) -> Any:
        # process observation data
        data = {
            "qpos": data_batch["obs"]["qpos"],
            "odom": data_batch["obs"]["odom"],
        }
        if "rgb" in self._features:
            data["rgb"] = {k.rsplit("::", 1)[0]: data_batch["obs"][k].float() / 255.0 for k in data_batch["obs"] if "rgb" in k}
        if "rgbd" in self._features:
            rgb = {k.rsplit("::", 1)[0]: data_batch["obs"][k].float() / 255.0 for k in data_batch["obs"] if "rgb" in k}
            depth = {k.rsplit("::", 1)[0]: (data_batch["obs"][k].float() - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH) for k in data_batch["obs"] if "depth" in k}
            data["rgbd"] = {k: torch.cat([rgb[k], depth[k].unsqueeze(-3)], dim=-3) for k in rgb}
        if extract_action:
            # extract action from data_batch
            data.update({
                "actions": data_batch["actions"],
                "masks": data_batch["masks"],
            })
        return data