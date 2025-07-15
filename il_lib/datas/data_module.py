import torch
from il_lib.datas.dataset import BehaviorDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple
from il_lib.utils.array_tensor_utils import any_stack, make_recursive_func
from il_lib.utils.convert_utils import any_to_torch_tensor
from sklearn.model_selection import train_test_split


class BehaviorDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        val_batch_size: Optional[int],
        val_split_ratio: float,
        dataloader_num_workers: int,
        seed: int,
        max_num_demos: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self._data_path = data_path
        self._batch_size = batch_size
        self._val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self._dataloader_num_workers = dataloader_num_workers
        self._val_split_ratio = val_split_ratio
        self._max_num_demos = max_num_demos
        self._seed = seed
        # store args and kwargs for dataset initialization
        self._args = args
        self._kwargs = kwargs

        self._train_dataset, self._val_dataset = None, None

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            all_demo_keys = BehaviorDataset.get_all_demo_keys(self._data_path)
            # limit number of demos
            if self._max_num_demos is not None:
                all_demo_keys = all_demo_keys[: self._max_num_demos]
            self._train_demo_keys, self._val_demo_keys = train_test_split(
                all_demo_keys,
                test_size=self._val_split_ratio,
                random_state=self._seed,
            )
            
            self._train_dataset = BehaviorDataset(*self._args, **self._kwargs, seed=self._seed, demo_keys=self._train_demo_keys)
            self._val_dataset = BehaviorDataset(*self._args, **self._kwargs, seed=self._seed, demo_keys=self._val_demo_keys)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            num_workers=min(self._batch_size, self._dataloader_num_workers),
            pin_memory=True,
            persistent_workers=True,
            collate_fn=_seq_chunk_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self._val_batch_size,
            num_workers=min(self._val_batch_size, self._dataloader_num_workers),
            pin_memory=True,
            persistent_workers=True,
            collate_fn=_seq_chunk_collate_fn,
        )

    def on_train_epoch_start(self):
        # set epoch for train dataset, which will trigger shuffling
        self._train_dataset.epoch = self.trainer.current_epoch


def _seq_chunk_collate_fn(sample_list: List[Tuple]) -> dict:
    """
    sample_list: list of (T, ...). PyTorch's native collate_fn can stack all data.
    But here we also add a leading singleton dimension, so it won't break the compatibility with episode data format.
    """
    stacked_list = any_stack(sample_list, dim=0)  # (B, T, ...)
    return _nested_th_expand_dims(stacked_list, dim=0)  # (1, B, T, ...)


@make_recursive_func
def _nested_th_expand_dims(x, dim):
    return torch.unsqueeze(x, dim=dim)
