import importlib
import math
import os
from il_lib.datas.dataset import DummyDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from pathlib import Path
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any, Optional


def _expand_path_value(value: Any):
    if isinstance(value, (ListConfig, DictConfig)):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, (list, tuple)):
        return [os.path.expanduser(str(item)) for item in value]
    return os.path.expanduser(str(value))


def _parse_per_file_limits(value) -> dict[str, int]:
    if value in (None, "", {}):
        return {}
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, ListConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        limits = {}
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(
                    "per_file_train_demo_limits entries must look like filename.hdf5:count"
                )
            key, raw_count = item.rsplit(":", 1)
            limit = int(raw_count)
            if limit < 0:
                raise ValueError("per_file_train_demo_limits counts must be non-negative.")
            limits[key.strip()] = limit
        return limits
    if isinstance(value, dict):
        limits = {str(key): int(limit) for key, limit in value.items()}
        if any(limit < 0 for limit in limits.values()):
            raise ValueError("per_file_train_demo_limits counts must be non-negative.")
        return limits
    limits = {}
    for item in value:
        if isinstance(item, str):
            key, raw_count = item.rsplit(":", 1)
            limits[key.strip()] = int(raw_count)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            limits[str(item[0])] = int(item[1])
        else:
            raise ValueError(f"Unsupported per-file limit entry: {item!r}")
    if any(limit < 0 for limit in limits.values()):
        raise ValueError("per_file_train_demo_limits counts must be non-negative.")
    return limits


def _parse_file_names(value) -> set[str]:
    if value in (None, "", [], {}):
        return set()
    if isinstance(value, (ListConfig, DictConfig)):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        names = [item.strip() for item in value.split(",")]
    else:
        names = [str(item).strip() for item in value]
    return {name for name in names if name}


def _demo_source_name(demo_key) -> str:
    if isinstance(demo_key, dict) and "source" in demo_key:
        return str(demo_key["source"])
    filename = getattr(getattr(demo_key, "file", None), "filename", "")
    return str(filename)


def _demo_label(demo_key) -> str:
    if isinstance(demo_key, dict):
        source = str(demo_key.get("source", ""))
        episode = demo_key.get("episode_index")
        return f"{source}:episode_{episode}" if episode is not None else source
    source = _demo_source_name(demo_key)
    name = getattr(demo_key, "name", "")
    source_label = Path(source).name if source else ""
    return f"{source_label}:{Path(name).name}" if source_label and name else source_label or str(demo_key)


def _limit_for_source(source_name: str, limits: dict[str, int]) -> Optional[int]:
    if source_name in limits:
        return limits[source_name]
    source_path = Path(source_name)
    if source_path.name in limits:
        return limits[source_path.name]
    source_stem = source_path.stem
    if source_stem in limits:
        return limits[source_stem]
    return None


def _source_is_selected(source_name: str, selected_names: set[str]) -> bool:
    if not selected_names:
        return True
    source_path = Path(source_name)
    return (
        source_name in selected_names
        or source_path.name in selected_names
        or source_path.stem in selected_names
    )


def _ordered_train_val_split(items, test_size: float):
    if not 0 < test_size < 1:
        raise ValueError("val_split_ratio must be between 0 and 1 when validation splitting is enabled.")
    val_count = math.ceil(len(items) * test_size)
    train_count = len(items) - val_count
    if train_count <= 0:
        raise ValueError(
            f"val_split_ratio={test_size} leaves no training demos from {len(items)} total demos."
        )
    return items[:train_count], items[train_count:]


class BehaviorDataModule(LightningDataModule):
    def __init__(
        self,
        *args,
        data_path: str,
        task_name: str,
        batch_size: int,
        val_batch_size: Optional[int],
        val_split_ratio: float,
        dataloader_num_workers: int,
        seed: int,
        shuffle: bool,
        max_num_demos: Optional[int] = None,
        included_data_files=None,
        per_file_train_demo_limits=None,
        use_limit_overflow_as_val: bool = False,
        dataset_class: str,
        val_data_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self._data_path = _expand_path_value(data_path)
        self._val_data_path = _expand_path_value(val_data_path) if val_data_path else None
        self._task_name = task_name
        self._batch_size = batch_size
        self._val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self._dataloader_num_workers = dataloader_num_workers
        self._val_split_ratio = val_split_ratio
        self._max_num_demos = max_num_demos
        self._seed = seed
        self._shuffle = shuffle
        self._dataset_class = dataset_class
        self._included_data_files = _parse_file_names(included_data_files)
        self._per_file_train_demo_limits = _parse_per_file_limits(per_file_train_demo_limits)
        self._use_limit_overflow_as_val = use_limit_overflow_as_val
        # store args and kwargs for dataset initialization
        self._args = args
        self._kwargs = kwargs

        self._train_dataset, self._val_dataset = None, None

    @property
    def _uses_separate_val_data_path(self) -> bool:
        return self._val_data_path is not None

    @property
    def _supports_separate_val_data_path(self) -> bool:
        return self._dataset_class in {
            "iiil.datas.IIILInterventionDataset",
            "iiil.datas.IIILLeRobotInterventionDataset",
        }

    def _get_dataset_class(self):
        module_path, class_name = self._dataset_class.rsplit(".", 1)
        return getattr(importlib.import_module(module_path), class_name)

    def _filter_demo_keys_by_source(self, demo_keys):
        if not self._included_data_files:
            return demo_keys
        filtered = [
            demo_key
            for demo_key in demo_keys
            if _source_is_selected(_demo_source_name(demo_key), self._included_data_files)
        ]
        if not filtered:
            selected = ", ".join(sorted(self._included_data_files))
            available = sorted({_demo_source_name(demo_key) for demo_key in demo_keys})
            raise ValueError(
                f"data.included_data_files selected no demos. Requested: {selected}. "
                f"Available files: {available}"
            )
        return filtered

    def _log_selected_demo_keys(self) -> None:
        train_by_source = {}
        val_by_source = {}
        for demo_key in getattr(self, "_train_demo_keys", []):
            source = _demo_source_name(demo_key)
            train_by_source[source] = train_by_source.get(source, 0) + 1
        for demo_key in getattr(self, "_val_demo_keys", []):
            source = _demo_source_name(demo_key)
            val_by_source[source] = val_by_source.get(source, 0) + 1
        print(
            "BehaviorDataModule demo split: "
            f"train={len(getattr(self, '_train_demo_keys', []))} {train_by_source}, "
            f"val={len(getattr(self, '_val_demo_keys', []))} {val_by_source}"
        )

    def _select_train_demo_keys(self, all_demo_keys):
        all_demo_keys = self._filter_demo_keys_by_source(all_demo_keys)
        if not self._per_file_train_demo_limits:
            if self._max_num_demos is not None:
                return all_demo_keys[: self._max_num_demos]
            return all_demo_keys

        source_counts = {}
        train_demo_keys = []
        for demo_key in all_demo_keys:
            source_name = _demo_source_name(demo_key)
            limit = _limit_for_source(source_name, self._per_file_train_demo_limits)
            if limit is None:
                train_demo_keys.append(demo_key)
                continue
            count = source_counts.get(source_name, 0)
            source_counts[source_name] = count + 1
            if count < limit:
                train_demo_keys.append(demo_key)

        if self._max_num_demos is not None:
            train_demo_keys = train_demo_keys[: self._max_num_demos]
        return train_demo_keys

    def _split_demo_keys(self, all_demo_keys):
        all_demo_keys = self._filter_demo_keys_by_source(all_demo_keys)
        if not self._per_file_train_demo_limits:
            if self._max_num_demos is not None:
                all_demo_keys = all_demo_keys[: self._max_num_demos]
            if self._val_split_ratio <= 0:
                return all_demo_keys, []
            return _ordered_train_val_split(all_demo_keys, self._val_split_ratio)

        source_counts = {}
        train_demo_keys, val_demo_keys = [], []
        for demo_key in all_demo_keys:
            source_name = _demo_source_name(demo_key)
            limit = _limit_for_source(source_name, self._per_file_train_demo_limits)
            if limit is None:
                train_demo_keys.append(demo_key)
                continue
            count = source_counts.get(source_name, 0)
            source_counts[source_name] = count + 1
            if count < limit:
                train_demo_keys.append(demo_key)
            elif self._use_limit_overflow_as_val:
                val_demo_keys.append(demo_key)

        if self._max_num_demos is not None:
            train_demo_keys = train_demo_keys[: self._max_num_demos]
        if val_demo_keys:
            return train_demo_keys, val_demo_keys
        if self._val_split_ratio <= 0:
            return train_demo_keys, []
        return _ordered_train_val_split(train_demo_keys, self._val_split_ratio)

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            if self._uses_separate_val_data_path and not self._supports_separate_val_data_path:
                raise ValueError(
                    "data.val_data_path is only supported for "
                    "iiil.datas.IIILInterventionDataset and "
                    "iiil.datas.IIILLeRobotInterventionDataset."
                )
            DatasetClassModule = self._get_dataset_class()
            all_demo_keys = DatasetClassModule.get_all_demo_keys(self._data_path, self._task_name)
            if self._uses_separate_val_data_path:
                self._train_demo_keys = self._select_train_demo_keys(all_demo_keys)
                self._val_demo_keys = DatasetClassModule.get_all_demo_keys(
                    self._val_data_path,
                    self._task_name,
                )
                val_data_path = self._val_data_path
            else:
                self._train_demo_keys, self._val_demo_keys = self._split_demo_keys(all_demo_keys)
                val_data_path = self._data_path
            if not self._train_demo_keys:
                selected = sorted(self._included_data_files) if self._included_data_files else "all files"
                examples = [_demo_label(demo_key) for demo_key in all_demo_keys[:10]]
                raise ValueError(
                    "No training demos were selected. Check data.included_data_files, "
                    f"data.per_file_train_demo_limits, and max_num_demos. Selection: {selected}. "
                    f"Example available demos: {examples}"
                )
            self._log_selected_demo_keys()
            # initialize datasets
            self._train_dataset = DatasetClassModule(
                *self._args,
                **self._kwargs,
                data_path=self._data_path,
                demo_keys=self._train_demo_keys,
                seed=self._seed,
            )
            if self._val_demo_keys:
                self._val_dataset = DatasetClassModule(
                    *self._args,
                    **self._kwargs,
                    data_path=val_data_path,
                    demo_keys=self._val_demo_keys,
                    seed=self._seed,
                )

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=min(self._batch_size, self._dataloader_num_workers),
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            shuffle=self._shuffle,
        )

    def _rebuild_validation_dataset(self) -> None:
        if self._uses_separate_val_data_path and not self._supports_separate_val_data_path:
            raise ValueError(
                "data.val_data_path is only supported for "
                "iiil.datas.IIILInterventionDataset and "
                "iiil.datas.IIILLeRobotInterventionDataset."
            )

        DatasetClassModule = self._get_dataset_class()
        all_demo_keys = DatasetClassModule.get_all_demo_keys(self._data_path, self._task_name)
        if self._uses_separate_val_data_path:
            self._val_demo_keys = DatasetClassModule.get_all_demo_keys(
                self._val_data_path,
                self._task_name,
            )
            val_data_path = self._val_data_path
        else:
            _, self._val_demo_keys = self._split_demo_keys(all_demo_keys)
            val_data_path = self._data_path

        self._val_dataset = None
        if self._val_demo_keys:
            self._val_dataset = DatasetClassModule(
                *self._args,
                **self._kwargs,
                data_path=val_data_path,
                demo_keys=self._val_demo_keys,
                seed=self._seed,
            )

    def _apply_validation_data_config(self, validation_data_config: dict) -> None:
        if not validation_data_config:
            return

        rebuild_validation_dataset = False
        if "val_data_path" in validation_data_config:
            val_data_path = validation_data_config["val_data_path"]
            self._val_data_path = _expand_path_value(val_data_path) if val_data_path else None
            rebuild_validation_dataset = True
        if "val_split_ratio" in validation_data_config:
            self._val_split_ratio = float(validation_data_config["val_split_ratio"])
            rebuild_validation_dataset = True
        if "val_batch_size" in validation_data_config:
            self._val_batch_size = int(validation_data_config["val_batch_size"])

        if rebuild_validation_dataset and self._train_dataset is not None:
            self._rebuild_validation_dataset()

    def apply_stage_config(self, stage_config: dict) -> None:
        data_config = stage_config.get("data", stage_config) if stage_config else {}
        validation_config = stage_config.get("validation", {}) if stage_config else {}
        self._apply_validation_data_config(validation_config.get("data", {}))
        if "reweight_strategy" in data_config:
            reweight_strategy = data_config["reweight_strategy"]
            self._kwargs["reweight_strategy"] = reweight_strategy
            for dataset in (self._train_dataset, self._val_dataset):
                if dataset is not None and hasattr(dataset, "set_reweight_strategy"):
                    dataset.set_reweight_strategy(reweight_strategy)

    def val_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            return None
        return DataLoader(
            self._val_dataset,
            batch_size=self._val_batch_size,
            num_workers=min(self._val_batch_size, self._dataloader_num_workers),
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        For test_step(), simply returns a dummy dataset.
        """
        return DataLoader(DummyDataset())

    def on_train_epoch_start(self) -> None:
        # set epoch for train dataset, which will trigger shuffling
        assert self._train_dataset is not None and self.trainer is not None
        self._train_dataset.epoch = self.trainer.current_epoch
