from typing import Union, List, Tuple
import torch
import numpy as np
import warnings


def check_shape(
    value: Union[Tuple, List, torch.Tensor, np.ndarray],
    expected: Union[Tuple, List, torch.Tensor, np.ndarray],
    err_msg="",
    mode="raise",
):
    """
    Args:
        value: np array or torch Tensor
        expected:
          - list[int], tuple[int]: if any value is None, will match any dim
          - np array or torch Tensor: must have the same dimensions
        mode:
          - "raise": raise ValueError, shape mismatch
          - "return": returns True if shape matches, otherwise False
          - "warning": warnings.warn
    """
    assert mode in ["raise", "return", "warning"]
    if torch.is_tensor(value):
        actual_shape = value.size()
    elif hasattr(value, "shape"):
        actual_shape = value.shape
    else:
        assert isinstance(value, (list, tuple))
        actual_shape = value
        assert all(
            isinstance(s, int) for s in actual_shape
        ), f"actual shape: {actual_shape} is not a list of ints"

    if torch.is_tensor(expected):
        expected_shape = expected.size()
    elif hasattr(expected, "shape"):
        expected_shape = expected.shape
    else:
        assert isinstance(expected, (list, tuple))
        expected_shape = expected

    err_msg = f" for {err_msg}" if err_msg else ""

    if len(actual_shape) != len(expected_shape):
        err_msg = (
            f"Dimension mismatch{err_msg}: actual shape {actual_shape} "
            f"!= expected shape {expected_shape}."
        )
        if mode == "raise":
            raise ValueError(err_msg)
        elif mode == "warning":
            warnings.warn(err_msg)
        return False

    for s_a, s_e in zip(actual_shape, expected_shape):
        if s_e is not None and s_a != s_e:
            err_msg = (
                f"Shape mismatch{err_msg}: actual shape {actual_shape} "
                f"!= expected shape {expected_shape}."
            )
            if mode == "raise":
                raise ValueError(err_msg)
            elif mode == "warning":
                warnings.warn(err_msg)
            return False
    return True