import importlib
from il_lib.datas.dataset import DummyDataset
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Optional


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
        max_num_demos: Optional[int] = None,
        dataset_class: str,
        **kwargs,
    ):
        super().__init__()
        self._data_path = data_path
        self._task_name = task_name
        self._batch_size = batch_size
        self._val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self._dataloader_num_workers = dataloader_num_workers
        self._val_split_ratio = val_split_ratio
        self._max_num_demos = max_num_demos
        self._seed = seed
        self._dataset_class = dataset_class
        # store args and kwargs for dataset initialization
        self._args = args
        self._kwargs = kwargs

        self._train_dataset, self._val_dataset = None, None

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            # get dataset class module
            module_path, class_name = self._dataset_class.rsplit(".", 1)
            DatasetClassModule = getattr(importlib.import_module(module_path), class_name)
            all_demo_keys = DatasetClassModule.get_all_demo_keys(self._data_path, self._task_name)
            # limit number of demos
            if self._max_num_demos is not None:
                all_demo_keys = all_demo_keys[: self._max_num_demos]
            self._train_demo_keys, self._val_demo_keys = train_test_split(
                all_demo_keys,
                test_size=self._val_split_ratio,
                random_state=self._seed,
            )
            # initialize datasets
            self._train_dataset = DatasetClassModule(
                *self._args,
                **self._kwargs,
                data_path=self._data_path,
                task_name=self._task_name,
                demo_keys=self._train_demo_keys,
                seed=self._seed,
            )
            self._val_dataset = DatasetClassModule(
                *self._args,
                **self._kwargs,
                data_path=self._data_path,
                task_name=self._task_name,
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
