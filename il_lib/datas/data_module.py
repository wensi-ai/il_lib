import importlib
import os
from il_lib.datas.dataset import DummyDataset
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional


def _parse_per_file_limits(value) -> dict[str, int]:
    if value in (None, "", {}):
        return {}
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
            limits[key.strip()] = int(raw_count)
        return limits
    if isinstance(value, dict):
        return {str(key): int(limit) for key, limit in value.items()}
    limits = {}
    for item in value:
        if isinstance(item, str):
            key, raw_count = item.rsplit(":", 1)
            limits[key.strip()] = int(raw_count)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            limits[str(item[0])] = int(item[1])
        else:
            raise ValueError(f"Unsupported per-file limit entry: {item!r}")
    return limits


def _demo_source_name(demo_key) -> str:
    filename = getattr(getattr(demo_key, "file", None), "filename", "")
    return Path(filename).name


def _limit_for_source(source_name: str, limits: dict[str, int]) -> Optional[int]:
    if source_name in limits:
        return limits[source_name]
    source_stem = Path(source_name).stem
    if source_stem in limits:
        return limits[source_stem]
    return None


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
        per_file_train_demo_limits=None,
        use_limit_overflow_as_val: bool = False,
        dataset_class: str,
        **kwargs,
    ):
        super().__init__()
        self._data_path = os.path.expanduser(data_path)
        self._task_name = task_name
        self._batch_size = batch_size
        self._val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self._dataloader_num_workers = dataloader_num_workers
        self._val_split_ratio = val_split_ratio
        self._max_num_demos = max_num_demos
        self._seed = seed
        self._shuffle = shuffle
        self._dataset_class = dataset_class
        self._per_file_train_demo_limits = _parse_per_file_limits(per_file_train_demo_limits)
        self._use_limit_overflow_as_val = use_limit_overflow_as_val
        # store args and kwargs for dataset initialization
        self._args = args
        self._kwargs = kwargs

        self._train_dataset, self._val_dataset = None, None

    def _split_demo_keys(self, all_demo_keys):
        if not self._per_file_train_demo_limits:
            if self._max_num_demos is not None:
                all_demo_keys = all_demo_keys[: self._max_num_demos]
            return train_test_split(
                all_demo_keys,
                test_size=self._val_split_ratio,
                shuffle=False,
            )

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
        return train_test_split(
            train_demo_keys,
            test_size=self._val_split_ratio,
            shuffle=False,
        )

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            # get dataset class module
            module_path, class_name = self._dataset_class.rsplit(".", 1)
            DatasetClassModule = getattr(importlib.import_module(module_path), class_name)
            all_demo_keys = DatasetClassModule.get_all_demo_keys(self._data_path, self._task_name)
            self._train_demo_keys, self._val_demo_keys = self._split_demo_keys(all_demo_keys)
            # initialize datasets
            self._train_dataset = DatasetClassModule(
                *self._args,
                **self._kwargs,
                data_path=self._data_path,
                demo_keys=self._train_demo_keys,
                seed=self._seed,
            )
            self._val_dataset = DatasetClassModule(
                *self._args,
                **self._kwargs,
                data_path=self._data_path,
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

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None
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
