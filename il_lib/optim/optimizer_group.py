"""
Generate Optimizer groups
"""

from typing import Callable, Union, List, Tuple
import torch
import torch.nn as nn

from il_lib.utils.misc_utils import match_patterns, getattr_nested
from il_lib.utils.training_utils import freeze_params


__all__ = [
    "default_optimizer_groups",
    "transformer_lr_decay_optimizer_groups",
    "transformer_freeze_layers",
    "transformer_freeze_except_last_layers",
    "check_optimizer_groups",
]

FilterType = Union[
    Callable[[str, torch.Tensor], bool], List[str], Tuple[str], str, None
]

HAS_REMOVEPREFIX = hasattr(str, "removeprefix")


def default_optimizer_groups(
    model: nn.Module,
    weight_decay: float,
    lr_scale: float = 1.0,
    no_decay_filter: FilterType = None,
    exclude_filter: FilterType = None,
):
    """
    lr_scale is only effective when using with enlight.learn.lr_schedule.LambdaLRWithScale

    Returns:
        [{'lr_scale': 1.0, 'weight_decay': weight_decay, 'params': decay_group},
         {'lr_scale': 1.0, 'weight_decay': 0.0, 'params': no_decay_group}],
        list of all param_ids processed
    """
    no_decay_filter = _transform_filter(no_decay_filter)
    exclude_filter = _transform_filter(exclude_filter)
    decay_group = []
    no_decay_group = []
    all_params_id = []
    for n, p in model.named_parameters():
        all_params_id.append(id(p))
        if not p.requires_grad or exclude_filter(n, p):
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or no_decay_filter(n, p):
            no_decay_group.append(p)
        else:
            decay_group.append(p)
    return [
        {"weight_decay": weight_decay, "params": decay_group, "lr_scale": lr_scale},
        {"weight_decay": 0.0, "params": no_decay_group, "lr_scale": lr_scale},
    ], all_params_id


def _get_transformer_blocks(model, block_sequence_name):
    block_sequence_name = block_sequence_name.rstrip(".")
    return getattr_nested(model, block_sequence_name)


def transformer_freeze_layers(
    model,
    layer_0_params: List[str],
    block_sequence_name,
    freeze_layers: List[int],
    extra_freeze_filter: FilterType = None,
):
    """
    Args:
        model: transformer model with pos embed and other preprocessing parts as layer 0
            and `block_sequence_name` that is a sequence of transformer blocks
        layer_0_params: list of parameter names before the first transformer
          block, which will be assigned layer 0
        block_sequence_name: name of the sequence module that contains transformer layers
          such that "block.0", "block.1", ... will share one LR within each block
        freeze_layers: list of layer indices to freeze. Include 0 to freeze
          preprocessing layers. Use negative indices to freeze from the last layers,
          e.g. -1 to freeze the last layer. Note that any nn.Module after the transformer
          block will NOT be frozen.
        extra_freeze_filter: filter to apply to the rest of the parameters
    """
    extra_freeze_filter = _transform_filter(extra_freeze_filter)
    freeze_layers = list(freeze_layers)
    layer_0_params = _transform_filter(layer_0_params)
    blocks = _get_transformer_blocks(model, block_sequence_name)
    assert max(freeze_layers) <= len(blocks), f"max({freeze_layers}) > {len(blocks)}"
    # convert all negative indices to last <N>
    freeze_layers = [(L if L >= 0 else len(blocks) + L + 1) for L in freeze_layers]
    for i, block in enumerate(blocks):
        if i + 1 in freeze_layers:
            freeze_params(block)

    for n, p in model.named_parameters():
        if layer_0_params(n, p) and 0 in freeze_layers:
            freeze_params(p)
        if extra_freeze_filter(n, p):
            freeze_params(p)


def transformer_freeze_except_last_layers(
    model,
    layer_0_params: List[str],
    block_sequence_name,
    num_last_layers: int,
    extra_freeze_filter: FilterType = None,
):
    """
    According to Kaiming's MAE paper, finetune ONLY the last <N> layers is typically
    as good as finetuning all layers, while being much more compute and memory efficient.

    Args:
        model: transformer model with pos embed and other preprocessing parts as layer 0
            and `block_sequence_name` that is a sequence of transformer blocks
        layer_0_params: list of parameter names before the first transformer
          block, which will be assigned layer 0
        block_sequence_name: name of the sequence module that contains transformer layers
          such that "block.0", "block.1", ... will share one LR within each block
        num_last_layers: number of last N layers to unfreeze (finetune)
        extra_freeze_filter: filter to apply to the rest of the parameters
    """
    num_blocks = len(_get_transformer_blocks(model, block_sequence_name))
    # blocks start from 1, because 0th block is preprocessing
    # get a range of the first num_blocks - num_last_layers blocks
    return transformer_freeze_layers(
        model,
        layer_0_params=layer_0_params,
        block_sequence_name=block_sequence_name,
        freeze_layers=range(num_blocks - num_last_layers + 1),
        extra_freeze_filter=extra_freeze_filter,
    )


def transformer_lr_decay_optimizer_groups(
    model,
    layer_0_params: List[str],
    block_sequence_name,
    *,
    weight_decay,
    lr_scale=1.0,
    lr_layer_decay,
    no_decay_filter: FilterType = None,
    exclude_filter: FilterType = None,
):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    lr_scale is only effective when using with enlight.learn.lr_schedule.LambdaLRWithScale

    Args:
        model: transformer model with pos embed and other preprocessing parts as layer 0,
          then the blocks start from layer 1 to layer N. LR will be progressively smaller
          from the last layer to the first layer. Layer N will be lr*lr_scale,
          N-1 will be lr*lr_scale*lr_layer_decay,
          N-2 will be lr*lr_scale*lr_layer_decay^2, ...
        layer_0_params: list of parameter names before the first transformer
          block, which will be assigned layer 0
        block_sequence_name: name of the sequence module that contains transformer layers
          such that "block.0", "block.1", ... will share one LR within each block
    """
    no_decay_filter = _transform_filter(no_decay_filter)
    exclude_filter = _transform_filter(exclude_filter)
    layer_0_params = _transform_filter(layer_0_params)
    block_sequence_name = block_sequence_name.rstrip(".")

    param_group_names = {}
    param_groups = {}

    num_layers = len(getattr_nested(model, block_sequence_name)) + 1

    layer_scales = [
        lr_scale * lr_layer_decay ** (num_layers - i) for i in range(num_layers + 1)
    ]
    all_params_id = []

    for n, p in model.named_parameters():
        all_params_id.append(id(p))
        if not p.requires_grad or exclude_filter(n, p):
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or no_decay_filter(n, p):
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        # get the layer index of the param
        if layer_0_params(n, p):
            layer_id = 0
        elif n.startswith(block_sequence_name + "."):
            # blocks.0 -> layer 1; blocks.1 -> layer 2; ...
            try:
                if HAS_REMOVEPREFIX:
                    layer_id = (
                        int(n.removeprefix(block_sequence_name + ".").split(".")[0]) + 1
                    )
                else:
                    layer_id = int(n[len(block_sequence_name) + 1 :].split(".")[0]) + 1

            except ValueError:
                raise ValueError(
                    f"{n} must have the format {block_sequence_name}.<layer_id>... "
                    f"where <layer_id> is an integer"
                )
        else:
            layer_id = num_layers
        group_name = f"layer_{layer_id}_{g_decay}"

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values()), all_params_id


def check_optimizer_groups(
    model, param_groups: List[dict], verbose=True, order_by="param"
):
    """
    For debugging purpose, check which param belongs to which param group
    This groups by param groups first

    Args:
        order_by: 'params' or 'groups', either by optimization group order or
            by nn.Module parameter list order.

    Returns:
        {param_name (str): group_idx (int)}, table (str, for ASCII print)
    """
    from tabulate import tabulate

    group_configs = [
        ", ".join(f"{k}={v:.6g}" for k, v in sorted(group.items()) if k != "params")
        for group in param_groups
    ]
    display_table = []
    name_to_group_idx = {}
    pid_to_group_id = {
        id(p): i for i, group in enumerate(param_groups) for p in group["params"]
    }
    for n, p in model.named_parameters(recurse=True):
        if id(p) in pid_to_group_id:
            gid = pid_to_group_id[id(p)]
            display_table.append((n, gid, group_configs[gid]))
            if n in name_to_group_idx:
                if verbose:
                    print(
                        f"WARNING: {n} is in both group "
                        f"{name_to_group_idx[n]} and {gid}"
                    )
            name_to_group_idx[n] = gid
        else:
            display_table.append((n, "_", "excluded"))
            name_to_group_idx[n] = None
    if order_by == "group":
        display_table.sort(key=lambda x: 1e10 if x[1] == "_" else x[1])
    table_str = tabulate(
        display_table, headers=["param", "i", "config"], tablefmt="presto"
    )
    return name_to_group_idx, table_str


def _transform_filter(filter: FilterType):
    """
    Filter can be:
        - None: always returns False
        - function(name, p) -> True to activate, False to deactivate
        - list of strings to match, can have wildcard
    """
    if filter is None:
        return lambda name, p: False
    elif callable(filter):
        return filter
    elif isinstance(filter, (str, list, tuple)):
        if isinstance(filter, str):
            filter = [filter]
        return lambda name, p: match_patterns(name, include=filter)
    else:
        raise ValueError(f"Invalid filter: {filter}")