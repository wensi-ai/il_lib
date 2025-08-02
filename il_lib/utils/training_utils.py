import os
import random
import numpy as np
import torch
import torch.nn as nn
import tree
from il_lib.utils.functional_utils import implements_method
from il_lib.utils.tree_utils import tree_value_at_path
from il_lib.utils.file_utils import f_join
from typing import Union, List, Tuple


def seed_everywhere(seed, torch_deterministic=False, rank=0):
    """set seed across modules"""
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def sequential_split_dataset(dataset: torch.utils.data.Dataset, split_portions: list[float]):
    """
    Split a dataset into multiple datasets, each with a different portion of the
    original dataset. Uses torch.utils.data.Subset.
    """
    from il_lib.utils.functional_utils import accumulate

    assert len(split_portions) > 0, "split_portions must be a non-empty list"
    assert all(0.0 <= p <= 1.0 for p in split_portions), f"{split_portions=}"
    assert abs(sum(split_portions) - 1.0) < 1e-6, f"{sum(split_portions)=} != 1.0"
    L = len(dataset)
    assert L > 0, "dataset must be non-empty"
    # split the list with proportions
    lengths = [int(p * L) for p in split_portions]
    # make sure the last split fills the full dataset
    lengths[-1] += L - sum(lengths)
    indices = list(range(L))

    return [
        torch.utils.data.Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(accumulate(lengths), lengths)
    ]


def load_torch(*fpath: str, map_location="cpu") -> dict:
    """
    Default maps to "cpu"
    """
    fpath = str(f_join(fpath))
    try:
        return torch.load(fpath, map_location=map_location, weights_only=False)
    except RuntimeError as e:
        raise RuntimeError(f"{e}\n\n --- Error loading {fpath}")



def set_requires_grad(model, requires_grad):
    if torch.is_tensor(model):
        model.requires_grad = requires_grad
    else:
        for param in model.parameters():
            param.requires_grad = requires_grad



def freeze_params(model):
    set_requires_grad(model, False)
    if not torch.is_tensor(model):
        model.eval()


def unfreeze_params(model):
    set_requires_grad(model, True)
    if not torch.is_tensor(model):
        model.train()


def classify_accuracy(
    output,
    target,
    topk: Union[int, List[int], Tuple[int]] = 1,
    mask=None,
    reduction="mean",
    scale_100=False,
):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    Accuracy is a float between 0.0 and 1.0

    Args:
        topk: if int, return a single acc. If tuple, return a tuple of accs
        mask: shape [batch_size,], binary mask of whether to include this sample or not
    """
    if isinstance(topk, int):
        topk = [topk]
        is_int = True
    else:
        is_int = False

    batch_size = target.size(0)
    assert output.size(0) == batch_size
    if mask is not None:
        assert mask.dim() == 1
        assert mask.size(0) == batch_size

    assert reduction in ["sum", "mean", "none"]
    if reduction != "mean":
        assert not scale_100, f"reduce={reduction} does not support scale_100=True"

    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        if mask is not None:
            correct = mask * correct

        mult = 100.0 if scale_100 else 1.0
        res = []
        for k in topk:
            correct_k = correct[:k].int().sum(dim=0)
            if reduction == "mean":
                if mask is not None:
                    # fmt: off
                    res.append(
                        float(correct_k.float().sum().mul_(mult / (mask.sum().item() + 1e-6)).item())
                    )
                    # fmt: on
                else:
                    res.append(
                        float(correct_k.float().sum().mul_(mult / batch_size).item())
                    )
            elif reduction == "sum":
                res.append(int(correct_k.sum().item()))
            elif reduction == "none":
                res.append(correct_k)
            else:
                raise NotImplementedError(f"Unknown reduce={reduction}")

    if is_int:
        assert len(res) == 1, "INTERNAL"
        return res[0]
    else:
        return res


def load_state_dict(objects, states, strip_prefix=None, strict=False):
    """
    Args:
        strict: objects and states must match exactly
        strip_prefix: only match the keys that have the prefix, and strip it
    """

    def _load(paths, obj):
        if not implements_method(obj, "load_state_dict"):
            raise ValueError(
                f"Object {type(obj)} does not support load_state_dict() method"
            )
        try:
            state = tree_value_at_path(states, paths)
        except ValueError:  # paths do not exist in `states` structure
            if strict:
                raise
            else:
                return
        if strip_prefix:
            assert isinstance(strip_prefix, str)
            state = {
                k[len(strip_prefix) :]: v
                for k, v in state.items()
                if k.startswith(strip_prefix)
            }
        if isinstance(obj, nn.Module):
            return obj.load_state_dict(state, strict=strict)
        else:
            return obj.load_state_dict(state)

    return tree.map_structure_with_path(_load, objects)
