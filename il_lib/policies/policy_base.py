import logging
import os
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from typing import Any, Optional
from omegaconf import DictConfig, OmegaConf
from omnigibson.learning.utils.obs_utils import create_video_writer
from omnigibson.macros import gm
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler


logger = logging.getLogger("BasePolicy")


class BasePolicy(LightningModule, ABC):
    """
    Base class for policies that is used for training and rollout
    """

    def __init__(self, eval: Optional[DictConfig] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # require evaluator for online testing
        self.eval_config = eval
        if self.eval_config is not None:
            OmegaConf.resolve(self.eval_config)
        else:
            logger.warning("No evaluation config provided, online evaluation will not be performed during testing.")
        self.evaluator = None
        self.test_id = 0
        self.robot_type = "R1Pro"

    @abstractmethod
    def forward(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of the policy.
        This is used for inference and should return the action.
        """
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def act(self, obs, policy_state, deterministic=None) -> torch.Tensor:
        """
        Args:
            obs: dict of (B, L=1, ...)
            policy_state: (h_0, c_0) or h_0
            deterministic: whether to use deterministic action or not
        Returns:
            action: (B, L=1, A) where A is the action dimension
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the policy
        """
        raise NotImplementedError

    @abstractmethod
    def policy_training_step(self, batch, batch_idx) -> Any:
        raise NotImplementedError

    @abstractmethod
    def policy_evaluation_step(self, batch, batch_idx) -> Any:
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Get optimizers, which are subsequently used to train.
        """
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        loss, log_dict, batch_size = self.policy_training_step(*args, **kwargs)
        log_dict = {f"train/{k}": v for k, v in log_dict.items()}
        log_dict["train/loss"] = loss
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def validation_step(self, *args, **kwargs):
        loss, log_dict, real_batch_size = self.policy_evaluation_step(*args, **kwargs)
        log_dict = {f"val/{k}": v for k, v in log_dict.items()}
        log_dict["val/loss"] = loss
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=real_batch_size,
            sync_dist=True,
        )
        return log_dict

    def on_validation_end(self):
        # only run test for global zero rank
        if self.trainer.is_global_zero:
            if self.eval_config is not None and self.eval_config.online_eval:
                # evaluator for online evaluation should only be created once
                if self.evaluator is None:
                    self.evaluator = self.create_evaluator()
                if not self.trainer.sanity_checking:
                    self.log_dict(self.run_online_evaluation(), sync_dist=True)

    def create_evaluator(self):
        """
        Create a evaluator parameter config containing vectorized distributed envs.
        This will be used to spawn the OmniGibson environments for online evaluation
        """
        # For performance optimization
        gm.DEFAULT_VIEWER_WIDTH = 128
        gm.DEFAULT_VIEWER_HEIGHT = 128
        gm.HEADLESS = self.eval_config.headless

        # update parameters with policy cfg file
        assert self.eval_config is not None, "eval_config must be provided to create evaluator!"
        from omnigibson.learning.eval import Evaluator
        evaluator = Evaluator(self.eval_config)
        # set the policy for the evaluator
        evaluator.policy.policy = self
        evaluator.policy.device = self.device
        return evaluator

    def run_online_evaluation(self):
        """
        Run online evaluation using the evaluator.
        """
        assert self.evaluator is not None, "evaluator is not created!"
        self.evaluator.reset()
        self.evaluator.env._current_episode = 0
        if self.eval_config.write_video:
            video_name = f"videos/test_{self.test_id}.mp4"
            os.makedirs("videos", exist_ok=True)
            self.evaluator.video_writer = create_video_writer(
                fpath=video_name,
                resolution=(720, 1080),
            )
        done = False
        while not done:
            terminated, truncated = self.evaluator.step()
            if self.eval_config.write_video:
                self.evaluator._write_video()
            if terminated or truncated:
                done = True
                self.evaluator.env.reset()
        if self.eval_config.write_video:
            self.evaluator.video_writer = None
        self.test_id += 1
        results = {"eval/success_rate": self.evaluator.n_success_trials / self.evaluator.n_trials}
        return results