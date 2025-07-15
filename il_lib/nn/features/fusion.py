import torch
import torch.nn as nn
import numpy as np
from il_lib.nn.common import build_mlp
from il_lib.utils.functional_utils import call_once
from il_lib.utils.print_utils import color_text
from il_lib.utils.shape_utils import check_shape
from il_lib.utils.array_tensor_utils import get_batch_size, any_slice
from il_lib.optim import default_optimizer_groups
from einops import rearrange


class SimpleFeatureFusion(nn.Module):
    def __init__(
        self,
        extractors: dict[str, nn.Module],
        hidden_depth: int,
        hidden_dim: int,
        output_dim: int,
        activation,
        add_input_activation: bool,
        add_output_activation: bool,
    ):
        super().__init__()
        self._extractors = nn.ModuleDict(extractors)
        extractors_output_dim = sum(e.output_dim for e in extractors.values())
        self.output_dim = output_dim
        self._head = build_mlp(
            input_dim=extractors_output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            weight_init="orthogonal",
            bias_init="zeros",
            norm_type=None,
            # add input activation because we assume upstream extractors do not have activation at the end
            add_input_activation=add_input_activation,
            add_input_norm=False,
            add_output_activation=add_output_activation,
            add_output_norm=False,
        )

        self._obs_groups = None

    @call_once
    def _check_obs_key_match(self, obs: dict, strict: bool = False):
        if strict:
            assert set(self._extractors.keys()) == set(obs.keys())
        elif set(self._extractors.keys()) != set(obs.keys()):
            print(
                color_text(
                    f"[warning] obs key mismatch: {set(self._extractors.keys())} != {set(obs.keys())}",
                    "yellow",
                )
            )

    def forward(self, x):
        x = self._group_obs(x)
        self._check_obs_key_match(x, strict=False)
        x = {k: v.forward(x[k]) for k, v in self._extractors.items()}
        x = torch.cat([x[k] for k in sorted(x.keys())], dim=-1)
        x = self._head(x)
        return x

    def _group_obs(self, obs):
        obs_keys = obs.keys()
        if self._obs_groups is None:
            # group by /
            obs_groups = {k.split("/")[0] for k in obs_keys}
            self._obs_groups = sorted(list(obs_groups))
        obs_rtn = {}
        for g in self._obs_groups:
            is_subgroup = any(k.startswith(f"{g}/") for k in obs_keys)
            if is_subgroup:
                obs_rtn[g] = {
                    k.split("/", 1)[1]: v
                    for k, v in obs.items()
                    if k.startswith(f"{g}/")
                }
            else:
                obs_rtn[g] = obs[g]
        return obs_rtn

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        extractors_pgs, extractor_pids = [], []
        for extractor in self._extractors.values():
            pg, pid = extractor.get_optimizer_groups(
                weight_decay=weight_decay,
                lr_layer_decay=lr_layer_decay,
                lr_scale=lr_scale,
            )
            extractors_pgs.extend(pg)
            extractor_pids.extend(pid)
        head_pg, head_pid = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            exclude_filter=lambda name, p: id(p) in extractor_pids,
        )
        return extractors_pgs + head_pg, extractor_pids + head_pid


class ObsTokenizer(nn.Module):
    def __init__(
        self,
        extractors: dict[str, nn.Module],
        *,
        use_modality_type_tokens: bool,
        token_dim: int,
        token_concat_order: list[str],
        strict: bool = True,
    ):
        assert set(extractors.keys()) == set(token_concat_order)
        super().__init__()
        self._extractors = nn.ModuleDict(extractors)
        self.output_dim = token_dim
        self._token_concat_order = token_concat_order
        self._strict = strict
        self._obs_groups = None
        self._use_modality_type_tokens = use_modality_type_tokens
        self._modality_type_tokens = None
        if use_modality_type_tokens:
            modality_type_tokens = {}
            for k in extractors:
                modality_type_tokens[k] = nn.Parameter(torch.zeros(token_dim))
            self._modality_type_tokens = nn.ParameterDict(modality_type_tokens)

    def forward(self, obs: dict[str, torch.Tensor]):
        """
        x: Dict of (B, T, ...)

        Each encoder should encode corresponding obs field to (B, T, E), where E = token_dim

        The final output interleaves encoded tokens along the time dimension
        """
        obs = self._group_obs(obs)
        self._check_obs_key_match(obs)
        x = {
            k: v.forward(obs[k]) for k, v in self._extractors.items()
        }  # dict of (B, T, E)
        if self._use_modality_type_tokens:
            for k in x:
                x[k] = x[k] + self._modality_type_tokens[k]
        x = rearrange(
            [x[k] for k in self._token_concat_order],
            "F B T E -> B (T F) E",
        )
        self._check_output_shape(obs, x)
        return x

    def _group_obs(self, obs):
        obs_keys = obs.keys()
        if self._obs_groups is None:
            # group by /
            obs_groups = {k.split("/")[0] for k in obs_keys}
            self._obs_groups = sorted(list(obs_groups))
        obs_rtn = {}
        for g in self._obs_groups:
            is_subgroup = any(k.startswith(f"{g}/") for k in obs_keys)
            if is_subgroup:
                obs_rtn[g] = {
                    k.split("/", 1)[1]: v
                    for k, v in obs.items()
                    if k.startswith(f"{g}/")
                }
            else:
                obs_rtn[g] = obs[g]
        return obs_rtn

    @call_once
    def _check_obs_key_match(self, obs: dict):
        if self._strict:
            assert set(self._extractors.keys()) == set(obs.keys())
        elif set(self._extractors.keys()) != set(obs.keys()):
            print(
                color_text(
                    f"[warning] obs key mismatch: {set(self._extractors.keys())} != {set(obs.keys())}",
                    "yellow",
                )
            )

    @call_once
    def _check_output_shape(self, obs, output):
        T = get_batch_size(any_slice(obs, np.s_[0]), strict=True)
        check_shape(
            output, (None, T * len(self._token_concat_order), self.output_dim)
        )

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        pg, pid = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=[
                "_extractors.*",
                "_modality_type_tokens.*",
            ],
        )
        return pg, pid

    @property
    def num_tokens_per_step(self):
        return len(self._token_concat_order)


class ObsTokenizerSingleToken(nn.Module):
    def __init__(
        self,
        extractors: dict[str, nn.Module],
        *,
        use_modality_type_tokens: bool,
        token_dim: int,
        token_concat_order: list[str],
        strict: bool = True,
    ):
        assert set(extractors.keys()) == set(token_concat_order)
        super().__init__()
        self._extractors = nn.ModuleDict(extractors)
        self.output_dim = token_dim
        self._token_concat_order = token_concat_order
        self._strict = strict
        self._obs_groups = None
        self._use_modality_type_tokens = use_modality_type_tokens
        self._modality_type_tokens = None
        if use_modality_type_tokens:
            modality_type_tokens = {}
            for k in extractors:
                modality_type_tokens[k] = nn.Parameter(torch.zeros(token_dim))
            self._modality_type_tokens = nn.ParameterDict(modality_type_tokens)
        self._output_fc = nn.Linear(token_dim * len(token_concat_order), token_dim)

    def forward(self, obs: dict[str, torch.Tensor]):
        """
        x: Dict of (B, T, ...)

        Each encoder should encode corresponding obs field to (B, T, E), where E = token_dim

        The final output interleaves encoded tokens along the time dimension
        """
        obs = self._group_obs(obs)
        self._check_obs_key_match(obs)
        x = {
            k: v.forward(obs[k]) for k, v in self._extractors.items()
        }  # dict of (B, T, E)
        if self._use_modality_type_tokens:
            for k in x:
                x[k] = x[k] + self._modality_type_tokens[k]
        x = torch.concat([x[k] for k in self._token_concat_order], dim=-1)
        x = self._output_fc(x)
        return x

    def _group_obs(self, obs):
        obs_keys = obs.keys()
        if self._obs_groups is None:
            # group by /
            obs_groups = {k.split("/")[0] for k in obs_keys}
            self._obs_groups = sorted(list(obs_groups))
        obs_rtn = {}
        for g in self._obs_groups:
            is_subgroup = any(k.startswith(f"{g}/") for k in obs_keys)
            if is_subgroup:
                obs_rtn[g] = {
                    k.split("/", 1)[1]: v
                    for k, v in obs.items()
                    if k.startswith(f"{g}/")
                }
            else:
                obs_rtn[g] = obs[g]
        return obs_rtn

    @call_once
    def _check_obs_key_match(self, obs: dict):
        if self._strict:
            assert set(self._extractors.keys()) == set(obs.keys())
        elif set(self._extractors.keys()) != set(obs.keys()):
            print(
                    color_text(
                    f"[warning] obs key mismatch: {set(self._extractors.keys())} != {set(obs.keys())}",
                    "yellow",
                )
            )

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        pg, pid = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=[
                "_extractors.*",
                "_modality_type_tokens.*",
            ],
        )
        return pg, pid

    @property
    def num_tokens_per_step(self):
        return 1