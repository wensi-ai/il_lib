from typing import Tuple


def tree_value_at_path(obj, paths: Tuple):
    try:
        for p in paths:
            obj = obj[p]
        return obj
    except Exception as e:
        raise ValueError(f"{e}\n\n-- Incorrect nested path {paths} for object: {obj}.")


def tree_assign_at_path(obj, paths: Tuple, value):
    try:
        for p in paths[:-1]:
            obj = obj[p]
        if len(paths) > 0:
            obj[paths[-1]] = value
    except Exception as e:
        raise ValueError(f"{e}\n\n-- Incorrect nested path {paths} for object: {obj}.")
