import os
import torch
from hydra.utils import instantiate
from il_lib.nn.distributions import GMMHead, CategoricalNet
from il_lib.policies.policy_base import BasePolicy
from il_lib.nn.features import SimpleFeatureFusion
from il_lib.optim import CosineScheduleFunction
from il_lib.utils.training_utils import freeze_params, load_state_dict
from omnigibson.learning.utils.obs_utils import MAX_DEPTH, MIN_DEPTH
from omegaconf import DictConfig
from typing import Any, Dict, List, Optional


class ResidualPolicy(BasePolicy):
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
        # ====== Action ======
        action_dim: int,
        action_net_gmm_n_modes: int = 5,
        action_net_hidden_dim: int,
        action_net_hidden_depth: int,
        action_net_activation: str = "relu",
        gmm_low_noise_eval: bool = True,
        # ====== Intervention ======
        learn_gripper_action: bool = True,
        include_robot_gripper_action_input: bool = True,
        intervention_head_hidden_dim: int,
        intervention_head_hidden_depth: int,
        intervention_head_activation: str = "relu",
        deterministic_inference: bool = True,
        update_intervention_head_only: bool = False,
        ckpt_path_if_update_intervention_head_only: Optional[str] = None,
        intervention_loss_weight: float = 1.0,
        # ====== Learning ======
        lr: float,
        use_cosine_lr: bool = True,
        lr_warmup_steps: Optional[int] = None,
        lr_cosine_steps: Optional[int] = None,
        lr_cosine_min: Optional[float] = None,
        lr_layer_decay: float = 1.0,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._prop_dim = prop_dim
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

        self.action_net = GMMHead(
            input_dim=feature_fusion_output_dim,
            n_modes=action_net_gmm_n_modes,
            action_dim=action_dim,
            hidden_dim=action_net_hidden_dim,
            hidden_depth=action_net_hidden_depth,
            activation=action_net_activation,
            low_noise_eval=gmm_low_noise_eval,
        )
        self.intervention_head = CategoricalNet(
            feature_fusion_output_dim,
            action_dim=2,  # intervention or not
            hidden_dim=intervention_head_hidden_dim,
            hidden_depth=intervention_head_hidden_depth,
            activation=intervention_head_activation,
        )
        if update_intervention_head_only:
            assert os.path.exists(ckpt_path_if_update_intervention_head_only)
            ckpt = torch.load(
                ckpt_path_if_update_intervention_head_only, map_location="cpu"
            )

            feature_extractor_weighs = {
                k: v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("residual_policy.feature_extractor")
            }
            load_state_dict(
                self.feature_extractor,
                feature_extractor_weighs,
                strip_prefix="residual_policy.feature_extractor.",
                strict=True,
            )
            freeze_params(self.feature_extractor)

            action_net_weights = {
                k: v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("residual_policy.action_net")
            }
            load_state_dict(
                self.action_net,
                action_net_weights,
                strip_prefix="residual_policy.action_net.",
                strict=True,
            )
            freeze_params(self.action_net)

        self._deterministic_inference = deterministic_inference
        self._intervention_loss_weight = intervention_loss_weight
        self._learn_gripper_action = learn_gripper_action
        self._include_robot_gripper_action_input = include_robot_gripper_action_input

        # ====== Learning ======
        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.lr_layer_decay = lr_layer_decay
        self.optimizer = optimizer
        self.weight_decay = weight_decay

    def forward(self, obs):
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
        obs_feature = self.feature_extractor(obs)  # (B, T_O, D)
        action_dist = self.action_net(obs_feature)
        intervention_dist = self.intervention_head(obs_feature)
        return action_dist, intervention_dist

    @torch.no_grad()
    def act(self, obs, deterministic=None):
        if deterministic is None:
            deterministic = self._deterministic_inference
        
        residual_action_dist, intervention_dist = self.forward(obs)
        if deterministic:
            residual_action = residual_action_dist.mode()
            intervention = intervention_dist.mode()
        else:
            residual_action = residual_action_dist.sample()
            intervention = intervention_dist.sample()

        return residual_action, intervention

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

    def policy_training_step(self, batch, batch_idx):
       return self._residual_forward_step(batch, batch_idx, is_train=True)

    def policy_evaluation_step(self, batch, batch_idx):
       return self._residual_forward_step(batch, batch_idx, is_train=False)

    def _residual_forward_step(self, batch, batch_idx, is_train: bool):
        """
        batch data are (B, ctx_len, ...)
        """
        batch = self.process_data(batch, extract_action=True)

        pad_mask = batch.pop("masks")
        intervention_mask = batch.pop("int_state") == 2  # intervention happened
        # only valid when both intervention and pad mask are True
        action_valid_mask = intervention_mask & pad_mask
        # get residual action target
        robot_policy_action = batch["base_action"]
        oracle_action = batch["oracle_action"]
        # separate q action and gripper action
        robot_policy_action, robot_policy_gripper_action = (
            robot_policy_action[..., :-1],
            robot_policy_action[..., -1:],
        )
        oracle_action, oracle_gripper_action = (
            oracle_action[..., :-1],
            oracle_action[..., -1:],
        )
        # rectify gripper action from [-1, 1] to {0, 1}
        robot_policy_gripper_action = torch.where(
            robot_policy_gripper_action >= 0, 1, 0
        )
        oracle_gripper_action = torch.where(oracle_gripper_action >= 0, 1, 0)
        residual_q = oracle_action - robot_policy_action
        residual_gripper = oracle_gripper_action - robot_policy_gripper_action

        # TODO: normalize residual_q to [-1, 1]
        # residual_q = (residual_q - delta_q_lower_limits) / (
        #     delta_q_upper_limits - delta_q_lower_limits
        # ) * 2 - 1
        if self._learn_gripper_action:
            # gripper change action is already in {-1, 1} so we can use GMM to optimize both q and gripper action
            target_action = torch.cat([residual_q, residual_gripper], dim=-1)
        else:
            target_action = residual_q
        if self._include_robot_gripper_action_input:
            batch["robot_policy_gripper_action"] = robot_policy_gripper_action
        # forward pass
        pi, intervention_dist = self.forward(batch)
        raw_action_loss = pi.imitation_loss(
            target_action, reduction="none"
        ).reshape(action_valid_mask.shape)
        action_loss = raw_action_loss * action_valid_mask
        raw_intervention_loss = intervention_dist.imitation_loss(
            intervention_mask.long(), reduction="none"
        ).reshape(pad_mask.shape)
        intervention_loss = raw_intervention_loss * pad_mask
        intervention_acc = intervention_dist.imitation_accuracy(
            intervention_mask.long(),
            mask=pad_mask,
        )
        real_batch_size = action_valid_mask.sum()
        action_loss = torch.sum(action_loss) / real_batch_size
        intervention_loss = (torch.sum(intervention_loss) / pad_mask.sum())
        loss = action_loss + self._intervention_loss_weight * intervention_loss
        log_dict = {
            "action_loss": action_loss,
            "intervention_loss": intervention_loss,
            "intervention_acc": intervention_acc,
        }
        if not is_train:
            # use the combined loss as a proxy for evaluation
            log_dict.update({
                "l1": action_loss + intervention_loss,
            })
        return loss, log_dict, real_batch_size
    
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
            data["rgbd"] = {k: {"rgb": rgb[k], "depth": depth[k].unsqueeze(-3)} for k in rgb}
        if "pcd" in self._features:
            data["pcd"] = {
                "rgb": data_batch["obs"]["pcd"][..., :3],
                "xyz": data_batch["obs"]["pcd"][..., 3:],
            }
        if "task" in self._features:
            data["task"] = data_batch["obs"]["task"]
        if extract_action:
            # extract (correction-related) action from data_batch
            data.update({
                "int_state": data_batch["policy"]["int_state"],
                "base_action": data_batch["policy"]["base_action"],
                "oracle_action": data_batch["policy"]["oracle_action"],
                "masks": data_batch["masks"],
            })
        return data
