import logging
import numpy as np
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    """
    Dummy dataset for test_step().
    Does absolutely nothing since we will do online evaluation.
    """

    def __init__(self, batch_size: int=1, epoch_len: int=1):
        """
        Still set batch_size because pytorch_lightning tracks it
        """
        self.n = epoch_len
        self._batch_size = batch_size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return np.zeros((self._batch_size,), dtype=bool)
