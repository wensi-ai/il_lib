from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule

from il_lib.nn.common import build_mlp
from il_lib.nn.features import SimpleFeatureFusion
from il_lib.optim import CosineScheduleFunction


class _SequenceEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        steps: int,
        output_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        pooling: str,
        activation: str,
        dropout: float,
    ):
        super().__init__()
        self.pooling = pooling
        self.steps = steps
        self.dropout = dropout

        if pooling == "flatten":
            self.encoder = build_mlp(
                input_dim=input_dim * steps,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                hidden_depth=hidden_depth,
                activation=activation,
                add_output_activation=True,
            )
            self.step_encoder = None
        elif pooling in {"last", "mean", "max"}:
            self.step_encoder = build_mlp(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                hidden_depth=hidden_depth,
                activation=activation,
                add_output_activation=True,
            )
            self.encoder = None
        else:
            raise ValueError("pooling must be one of {'flatten', 'last', 'mean', 'max'}.")

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.dim() != 3:
            raise ValueError(f"Expected sequence tensor with shape [B, T, D], got {tuple(x.shape)}")
        if x.shape[1] != self.steps:
            raise ValueError(f"Expected {self.steps} steps, got {x.shape[1]}")

        if self.pooling == "flatten":
            x = x.reshape(x.shape[0], -1)
            x = self.encoder(x)
            return F.dropout(x, p=self.dropout, training=self.training)

        x = self.step_encoder(x.reshape(-1, x.shape[-1]))
        x = x.reshape(-1, self.steps, self.output_dim)
        if self.pooling == "last":
            x = x[:, -1]
        elif self.pooling == "mean":
            x = x.mean(dim=1)
        else:
            x = x.max(dim=1).values
        return F.dropout(x, p=self.dropout, training=self.training)


class InterventionClassifier(LightningModule):
    """
    Lightweight offline classifier for intervention prediction research.

    The goal is to make input combinations and small architectural choices easy
    to sweep from Hydra config, not to lock in one final modeling decision yet.
    """

    def __init__(
        self,
        *,
        input_keys: Optional[List[str]] = None,
        input_configs: Optional[DictConfig | Dict[str, Dict]] = None,
        feature_extractors: Optional[DictConfig | Dict[str, Dict]] = None,
        feature_fusion_hidden_depth: int = 1,
        feature_fusion_hidden_dim: int = 256,
        feature_fusion_output_dim: int = 256,
        feature_fusion_activation: str = "relu",
        feature_fusion_add_input_activation: bool = False,
        feature_fusion_add_output_activation: bool = False,
        encoder_hidden_dim: int = 128,
        encoder_output_dim: int = 64,
        encoder_hidden_depth: int = 1,
        encoder_pooling: str = "flatten",
        classifier_hidden_dim: int = 128,
        classifier_hidden_depth: int = 1,
        classifier_activation: str = "relu",
        dropout: float = 0.0,
        positive_weight: float = 1.0,
        decision_threshold: float = 0.5,
        lr: float = 1e-4,
        use_cosine_lr: bool = False,
        lr_warmup_steps: Optional[int] = None,
        lr_cosine_steps: Optional[int] = None,
        lr_cosine_min: Optional[float] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        raw_input_configs = (
            OmegaConf.to_container(input_configs, resolve=True)
            if isinstance(input_configs, DictConfig)
            else input_configs
        )
        raw_feature_extractors = (
            OmegaConf.to_container(feature_extractors, resolve=False)
            if isinstance(feature_extractors, DictConfig)
            else feature_extractors
        )
        self.input_keys = list(input_keys or [])
        self.input_configs = raw_input_configs or {}
        self.lr = lr
        self.use_cosine_lr = use_cosine_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_cosine_steps = lr_cosine_steps
        self.lr_cosine_min = lr_cosine_min
        self.weight_decay = weight_decay
        self.decision_threshold = decision_threshold
        self._features = set()
        self._use_feature_extractors = raw_feature_extractors is not None
        self.save_hyperparameters(ignore=["input_configs", "feature_extractors"])

        if self._use_feature_extractors:
            self._features = set(raw_feature_extractors.keys())
            if not self.input_keys:
                self.input_keys = list(raw_feature_extractors.keys())
            self.feature_extractor = SimpleFeatureFusion(
                extractors={
                    key: instantiate(cfg)
                    for key, cfg in raw_feature_extractors.items()
                },
                hidden_depth=feature_fusion_hidden_depth,
                hidden_dim=feature_fusion_hidden_dim,
                output_dim=feature_fusion_output_dim,
                activation=feature_fusion_activation,
                add_input_activation=feature_fusion_add_input_activation,
                add_output_activation=feature_fusion_add_output_activation,
            )
            self.encoders = None
            fused_dim = feature_fusion_output_dim
        else:
            if not self.input_keys:
                raise ValueError("input_keys must be provided when feature_extractors is not set.")
            if raw_input_configs is None:
                raise ValueError("input_configs must be provided when feature_extractors is not set.")
            encoders = {}
            for key in self.input_keys:
                if key not in raw_input_configs:
                    raise KeyError(f"Missing input_configs entry for '{key}'")
                cfg = raw_input_configs[key]
                encoders[key] = _SequenceEncoder(
                    input_dim=int(cfg["input_dim"]),
                    steps=int(cfg["steps"]),
                    output_dim=encoder_output_dim,
                    hidden_dim=encoder_hidden_dim,
                    hidden_depth=encoder_hidden_depth,
                    pooling=encoder_pooling,
                    activation=classifier_activation,
                    dropout=dropout,
                )
            self.encoders = nn.ModuleDict(encoders)
            self.feature_extractor = None
            fused_dim = encoder_output_dim * len(self.input_keys)
        self.classifier = build_mlp(
            input_dim=fused_dim,
            hidden_dim=classifier_hidden_dim,
            output_dim=1,
            hidden_depth=classifier_hidden_depth,
            activation=classifier_activation,
        )
        self.register_buffer("positive_weight", torch.tensor(float(positive_weight), dtype=torch.float32))

    def _prepare_feature_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prepared = {}
        obs_time = 1

        for key in self._features:
            if key not in inputs:
                raise KeyError(f"Batch is missing requested input '{key}'")
            value = inputs[key]
            if key == "rgb":
                if not isinstance(value, dict) or len(value) == 0:
                    raise ValueError("RGB inputs must be a non-empty dict of multiview tensors.")
                rgb_inputs = {
                    view: tensor.to(torch.float32)
                    for view, tensor in value.items()
                }
                prepared[key] = rgb_inputs
                obs_time = max(obs_time, next(iter(rgb_inputs.values())).shape[1])
                continue

            value = value.to(torch.float32)
            if value.dim() == 2:
                value = value.unsqueeze(1)
            if key in {"base_action_chunk", "action_history"}:
                seq_len = 1 if value.dim() == 3 else value.shape[1]
                value = value.reshape(value.shape[0], seq_len, -1)
            prepared[key] = value
            obs_time = max(obs_time, value.shape[1])
        for key, value in prepared.items():
            if key == "rgb":
                for view, tensor in value.items():
                    if tensor.shape[1] == 1 and obs_time > 1:
                        value[view] = tensor.expand(-1, obs_time, -1, -1, -1)
                    elif tensor.shape[1] != obs_time:
                        raise ValueError(
                            f"RGB input '{view}' has {tensor.shape[1]} steps, expected {obs_time}."
                        )
                continue

            if value.shape[1] == 1 and obs_time > 1:
                prepared[key] = value.expand(-1, obs_time, *value.shape[2:])
            elif value.shape[1] != obs_time:
                raise ValueError(
                    f"Input '{key}' has {value.shape[1]} steps, expected {obs_time}."
                )
        return prepared

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self._use_feature_extractors:
            fused = self.feature_extractor(self._prepare_feature_inputs(inputs))
            if fused.dim() >= 3:
                fused = fused[:, -1]
        else:
            features = []
            for key in self.input_keys:
                if key not in inputs:
                    raise KeyError(f"Batch is missing requested input '{key}'")
                features.append(self.encoders[key](inputs[key]))
            fused = torch.cat(features, dim=-1)
        return self.classifier(fused).squeeze(-1)

    def _shared_step(self, batch: Dict[str, Dict[str, torch.Tensor]], stage: str):
        if not isinstance(batch, dict) or "inputs" not in batch or "target" not in batch:
            return None
        logits = self.forward(batch["inputs"])
        target = batch["target"].to(torch.float32).reshape_as(logits)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self.positive_weight.to(logits.device),
        )

        probs = torch.sigmoid(logits)
        preds = (probs >= self.decision_threshold).to(torch.float32)
        accuracy = (preds == target).to(torch.float32).mean()
        true_positive = ((preds == 1) & (target == 1)).to(torch.float32).sum()
        pred_positive = (preds == 1).to(torch.float32).sum().clamp_min(1.0)
        actual_positive = (target == 1).to(torch.float32).sum().clamp_min(1.0)
        precision = true_positive / pred_positive
        recall = true_positive / actual_positive

        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/acc": accuracy,
                f"{stage}/precision": precision,
                f"{stage}/recall": recall,
                f"{stage}/pred_positive_rate": preds.mean(),
                f"{stage}/target_positive_rate": target.mean(),
            },
            prog_bar=stage != "train",
            on_step=False,
            on_epoch=True,
            batch_size=target.shape[0],
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if not self.use_cosine_lr:
            return optimizer

        scheduler_kwargs = dict(
            base_value=1.0,
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
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
