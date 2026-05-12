from typing import List
import logging
import os
import socket
import time
from copy import deepcopy
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, ListConfig
import torch
import il_lib.utils.file_utils as FU
import il_lib.utils.config_utils as CU
import il_lib.utils.print_utils as PU
from il_lib.utils.training_utils import load_torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import Callback, TQDMProgressBar
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.rank_zero import rank_zero_debug as rank_zero_debug_pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info as rank_zero_info_pl
from pytorch_lightning.callbacks import ModelCheckpoint


__all__ = [
    "Trainer",
    "CustomProgressBar",
    "StagedTrainingCallback",
    "rank_zero_info",
    "rank_zero_debug",
    "rank_zero_warn",
    "rank_zero_info_pl",
    "rank_zero_debug_pl",
]

logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)
logging.getLogger("torch.distributed.nn.jit.instantiator").setLevel(logging.WARNING)
PU.logging_exclude_pattern("root", patterns="*Reducer buckets have been rebuilt in this iteration*")


class Trainer:
    def __init__(self, cfg: DictConfig, eval_only=False):
        """
        Args:
            eval_only: if True, will not save any model dir
        """
        cfg = deepcopy(cfg)
        OmegaConf.set_struct(cfg, False)
        CU.register_omegaconf_resolvers()
        run_name = self.generate_run_name(cfg)
        self.run_name = run_name
        self.run_dir = FU.f_join(cfg.exp_root_dir, run_name)
        self._eval_only = eval_only
        self._resume_mode = None  # 'full state' or 'model only'
        self._maybe_apply_staged_training_max_steps(cfg)
        if eval_only:
            rank_zero_info("Eval only, will not save any model dir")
        else:
            if "resume" in cfg and "ckpt_path" in cfg.resume and cfg.resume.ckpt_path:
                cfg.resume.ckpt_path = FU.f_expand(
                    cfg.resume.ckpt_path.replace("_RUN_DIR_", self.run_dir).replace("_RUN_NAME_", run_name)
                )
                self._resume_mode = "full state" if cfg.resume.get("full_state", False) else "model only"
                rank_zero_info(
                    "=" * 80,
                    "=" * 80 + "\n",
                    f"Resume training from {cfg.resume.ckpt_path}",
                    f"\t({self._resume_mode})\n",
                    "=" * 80,
                    "=" * 80,
                    sep="\n",
                    end="\n\n",
                )
                time.sleep(3)
                assert FU.f_exists(cfg.resume.ckpt_path), "resume ckpt_path does not exist"

            rank_zero_print("Run name:", run_name, "\nExp dir:", self.run_dir)
            FU.f_mkdir(self.run_dir)
            FU.f_mkdir(FU.f_join(self.run_dir, "tb"))
            FU.f_mkdir(FU.f_join(self.run_dir, "logs"))
            FU.f_mkdir(FU.f_join(self.run_dir, "ckpt"))
            CU.omegaconf_save(cfg, self.run_dir, "conf.yaml")
            rank_zero_print("Checkpoint cfg:", CU.omegaconf_to_dict(cfg.trainer.checkpoint))
        self.cfg = cfg
        self.ckpt_cfg = cfg.trainer.pop("checkpoint")
        self.data_module = self.create_data_module(cfg)
        self._monkey_patch_add_info(self.data_module)
        self.trainer = self.create_trainer(cfg)
        self.module = self.create_module(cfg)
        self.module.data_module = self.data_module
        self._monkey_patch_add_info(self.module)

        if not eval_only and self._resume_mode == "model only":
            ret = self.module.load_state_dict(
                load_torch(cfg.resume.ckpt_path)["state_dict"],
                strict=cfg.resume.strict,
            )
            rank_zero_warn("state_dict load status:", ret)

    def create_module(self, cfg):
        return instantiate(cfg.module, _recursive_=False)

    def create_data_module(self, cfg):
        return instantiate(cfg.data)

    def generate_run_name(self, cfg):
        return cfg.run_name + "_" + time.strftime("%Y%m%d-%H%M%S")

    def _maybe_apply_staged_training_max_steps(self, cfg):
        if not cfg.get("training_stages", {}).get("enabled", False):
            return
        stages = cfg.training_stages.get("stages", [])
        start_stage_idx = 0
        stage_ckpt_path = None
        for stage in stages:
            ckpt_path = stage.get("ckpt_path", None)
            if not ckpt_path:
                break
            stage_ckpt_path = ckpt_path
            start_stage_idx += 1
        if start_stage_idx >= len(stages):
            raise ValueError("All configured training stages have ckpt_path set; nothing remains to train.")

        if stage_ckpt_path:
            stage_ckpt_path = FU.f_expand(
                str(stage_ckpt_path)
                .replace("_RUN_DIR_", self.run_dir)
                .replace("_RUN_NAME_", self.run_name)
            )
            cfg.resume.ckpt_path = stage_ckpt_path
            cfg.resume.full_state = False
            rank_zero_info(
                f"Skipping {start_stage_idx} completed training stage(s); "
                f"loading model weights from {stage_ckpt_path}"
            )

        cfg.training_stages.start_stage_idx = start_stage_idx
        total_steps = sum(int(stage.get("num_steps", 0)) for stage in stages[start_stage_idx:])
        if total_steps <= 0:
            raise ValueError("training_stages.enabled=true requires stages with positive num_steps.")
        cfg.max_steps = total_steps
        cfg.trainer.max_steps = total_steps

    def _monkey_patch_add_info(self, obj):
        """
        Add useful info to module and data_module so they can access directly
        """
        # our own info
        obj.run_config = self.cfg
        obj.run_name = self.run_name
        # add properties from trainer
        for attr in [
            "global_rank",
            "local_rank",
            "world_size",
            "num_nodes",
            "num_processes",
            "node_rank",
            "num_gpus",
            "data_parallel_device_ids",
        ]:
            if hasattr(obj, attr):
                continue
            setattr(
                obj.__class__,
                attr,
                # force capture 'attr'
                property(lambda self, attr=attr: getattr(self.trainer, attr)),
            )

    def create_loggers(self, cfg) -> List[pl_loggers.Logger]:
        if self._eval_only:
            loggers = []
        else:
            loggers = [
                pl_loggers.CSVLogger(self.run_dir, name="logs", version=""),
            ]
        if cfg.use_wandb and self._wandb_supported_in_runtime():
            wandb_kwargs = {
                "name": cfg.wandb_run_name,
                "project": cfg.wandb_project,
                "group": cfg.wandb_group,
                "id": self.run_name,
                "save_dir": self.run_dir,
            }
            if "wandb_entity" in cfg:
                wandb_kwargs["entity"] = cfg.wandb_entity
            loggers.append(pl_loggers.WandbLogger(**wandb_kwargs))
        return loggers

    def _wandb_supported_in_runtime(self) -> bool:
        cache_dir = os.path.expanduser("~/.cache")
        if not os.access(cache_dir, os.W_OK):
            rank_zero_warn(f"W&B disabled because cache directory is not writable: {cache_dir}")
            return False
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
        except OSError as exc:
            rank_zero_warn(f"W&B disabled because local sockets are unavailable: {exc}")
            return False
        return True

    def create_callbacks(self, cfg) -> List[Callback]:
        ModelCheckpoint.FILE_EXTENSION = ".pth"
        callbacks = []
        # Construct ModelCheckpoint callback
        if isinstance(self.ckpt_cfg, DictConfig):
            ckpt = ModelCheckpoint(dirpath=FU.f_join(self.run_dir, "ckpt"), **self.ckpt_cfg)
            callbacks.append(ckpt)
        else:
            assert isinstance(self.ckpt_cfg, ListConfig)
            for _cfg in self.ckpt_cfg:
                ckpt = ModelCheckpoint(dirpath=FU.f_join(self.run_dir, "ckpt"), **_cfg)
                callbacks.append(ckpt)
        if "callbacks" in cfg.trainer:
            extra_callbacks = [instantiate(callback) for callback in cfg.trainer.pop("callbacks")]
            callbacks.extend(extra_callbacks)
        rank_zero_print("Lightning callbacks:", [c.__class__.__name__ for c in callbacks])
        return callbacks
    
    def create_strategy(self, cfg):
        # Instantiate strategy if it's a DictConfig
        if "strategy" in cfg.trainer:
            if isinstance(cfg.trainer["strategy"], DictConfig):
                return instantiate(cfg.trainer.pop("strategy"))
            else:
                return cfg.trainer.pop("strategy")
        return None

    def _normalize_trainer_cfg_for_runtime(self, cfg):
        trainer_cfg = cfg.trainer
        accelerator = trainer_cfg.get("accelerator", "auto")
        if accelerator == "gpu" and not torch.cuda.is_available():
            rank_zero_warn("GPU requested but CUDA is unavailable. Falling back to CPU training.")
            trainer_cfg.accelerator = "cpu"
            trainer_cfg.devices = 1
            trainer_cfg.benchmark = False
            trainer_cfg.strategy = "auto"
        return cfg

    def create_trainer(self, cfg) -> pl.Trainer:
        cfg = self._normalize_trainer_cfg_for_runtime(cfg)
        return pl.Trainer(
            logger=self.create_loggers(cfg),
            callbacks=self.create_callbacks(cfg),
            strategy=self.create_strategy(cfg),
            **cfg.trainer,
        )

    def fit(self):
        return self.trainer.fit(
            self.module,
            datamodule=self.data_module,
            ckpt_path=(self.cfg.resume.ckpt_path if self._resume_mode == "full state" else None),
        )

    def validate(self):
        return self.trainer.validate(self.module, datamodule=self.data_module, ckpt_path=None)

    def test(self):
        return self.trainer.test(self.module, datamodule=self.data_module, ckpt_path=None)


@rank_zero_only
def rank_zero_print(*msg, **kwargs):
    PU.pprint_(*msg, **kwargs)


@rank_zero_only
def rank_zero_info(*msg, **kwargs):
    PU.pprint_(
        PU.color_text("[INFO]", color="green", styles=["reverse", "bold"]),
        *msg,
        **kwargs,
    )


@rank_zero_only
def rank_zero_warn(*msg, **kwargs):
    PU.pprint_(
        PU.color_text("[WARN]", color="yellow", styles=["reverse", "bold"]),
        *msg,
        **kwargs,
    )


@rank_zero_only
def rank_zero_debug(*msg, **kwargs):
    if rank_zero_debug.enabled:
        PU.pprint_(PU.color_text("[DEBUG]", color="blue", bg_color="on_grey"), *msg, **kwargs)


rank_zero_debug.enabled = True


class StagedTrainingCallback(Callback):
    def __init__(self, stage_cfg=None):
        super().__init__()
        cfg = OmegaConf.to_container(stage_cfg, resolve=True) if isinstance(stage_cfg, DictConfig) else stage_cfg
        self.enabled = bool((cfg or {}).get("enabled", False))
        self.start_stage_idx = int((cfg or {}).get("start_stage_idx", 0))
        self.stages = list((cfg or {}).get("stages", []))[self.start_stage_idx :]
        self._active_stage_idx = None
        self._boundaries = []
        total = 0
        for stage in self.stages:
            total += int(stage.get("num_steps", 0))
            self._boundaries.append(total)

    def _stage_index_for_step(self, global_step: int) -> int:
        for idx, boundary in enumerate(self._boundaries):
            if global_step < boundary:
                return idx
        return max(len(self.stages) - 1, 0)

    def _normalize_module_options(self, options: dict) -> dict:
        options = dict(options or {})
        if "freeze_vision_encoders" in options and "freeze_feature_extractor" not in options:
            options["freeze_feature_extractor"] = options.pop("freeze_vision_encoders")
        if "freeze_diffusion" in options and "freeze_backbone" not in options:
            options["freeze_backbone"] = options.pop("freeze_diffusion")
        return options

    def _apply_stage(self, trainer, pl_module, stage_idx: int) -> None:
        if self._active_stage_idx == stage_idx:
            if hasattr(pl_module, "enforce_training_stage"):
                pl_module.enforce_training_stage()
            return
        stage = self.stages[stage_idx]
        if "module" in stage:
            if not hasattr(pl_module, "set_training_stage"):
                raise AttributeError(
                    f"{pl_module.__class__.__name__} does not support staged training."
                )
            pl_module.set_training_stage(**self._normalize_module_options(stage["module"]))
        datamodule = trainer.datamodule
        if datamodule is not None and hasattr(datamodule, "apply_stage_config"):
            datamodule.apply_stage_config(stage.get("data", {}))
        self._active_stage_idx = stage_idx
        rank_zero_info(
            f"Applied training stage {self.start_stage_idx + stage_idx + 1}:",
            stage.get("name", f"stage_{stage_idx + 1}"),
        )

    def on_train_start(self, trainer, pl_module) -> None:
        if not self.enabled:
            return
        if not self.stages:
            raise ValueError("StagedTrainingCallback requires at least one stage.")
        self._apply_stage(trainer, pl_module, self._stage_index_for_step(trainer.global_step))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if not self.enabled:
            return
        self._apply_stage(trainer, pl_module, self._stage_index_for_step(trainer.global_step))


class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items
