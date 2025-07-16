import h5py
import numpy as np
import os
import torch
from copy import deepcopy
from torch.utils.data import IterableDataset
from il_lib.utils.array_tensor_utils import any_concat, any_ones_like, any_slice, any_stack, get_batch_size
from il_lib.utils.eval_utils import ACTION_QPOS_INDICES, JOINT_RANGE
from typing import Optional, List, Tuple

from omnigibson.learning.utils.obs_utils import OBS_LOADER_MAP


class BehaviorDataset(IterableDataset):

    @classmethod
    def get_all_demo_keys(cls, data_path: str, task_name: str):
        assert os.path.exists(f"{data_path}/data/{task_name}"), f"Data path does not exist!"
        demo_keys = []
        for file_name in sorted(os.listdir(f"{data_path}/data/{task_name}")):
            if file_name.endswith(".hdf5"):
                base_name = file_name.split(".")[0]
                with h5py.File(os.path.join(data_path, "data", task_name, file_name), "r", swmr=True, libver="latest") as f:
                    for demo_id in f["data"]:
                        demo_keys.append(f"{base_name}::{demo_id}")
        return demo_keys

    def __init__(
        self,
        *args,
        data_path: str,
        task_name: str,
        demo_keys: List[str],
        obs_window_size: int,
        ctx_len: int,
        use_action_chunks: bool = False,
        action_prediction_horizon: Optional[int] = None,
        visual_obs_types: List[str],
        multi_view_cameras: Optional[List[str]] = None,
        load_task_info: bool = False,
        seed: int = 42,
        shuffle: bool = True,
        # dataset parameters
        pcd_downsample_points: Optional[int] = None,
    ):
        """
        Args:
            data_path (str): Path to the data directory.
            obs_window_size (int): Size of the observation window.
            ctx_len (int): Context length.
            use_action_chunks (bool): Whether to use action chunks.
                Action will be from (T, A) to (T, L_pred_horizon, A)
            action_prediction_horizon (Optional[int]): Horizon of the action prediction.
                Must not be None if use_action_chunks is True.
            visual_obs_types (List[str]): List of visual observation types to load.
                Valid options are: "rgb", "depth", "pcd", "seg".
            multi_view_cameras (Optional[List[str]]): List of multi-view camera names to load obs from.
            load_task_info (bool): Whether to load privileged task information.
            seed (int): Random seed.
            shuffle (bool): Whether to shuffle the dataset.
        """
        super().__init__()
        self._data_path = data_path
        self._task_name = task_name
        self._demo_keys = demo_keys
        self._obs_window_size = obs_window_size
        self._ctx_len = ctx_len
        self._use_action_chunks = use_action_chunks
        self._action_prediction_horizon = action_prediction_horizon
        assert self._action_prediction_horizon is not None if self._use_action_chunks else True, \
            "action_prediction_horizon must be provided if use_action_chunks is True!"
        self._load_task_info = load_task_info
        self._seed = seed
        self._shuffle = shuffle
        self._epoch = 0

        assert set(visual_obs_types).issubset({"rgb", "depth", "pcd", "seg"}), \
            "visual_obs_types must be a subset of {'rgb', 'depth', 'pcd', 'seg'}!"
        self._visual_obs_types = visual_obs_types
        self._pcd_downsample_points = pcd_downsample_points
        self._multi_view_cameras = multi_view_cameras
        self.robot_type = None

        # get all demo keys
        self._demo_keys = []
        for file_name in sorted(os.listdir(f"{self._data_path}/data/{self._task_name}")):
            if file_name.endswith(".hdf5"):
                base_name = file_name.split(".")[0]
                with h5py.File(os.path.join(data_path, "data", task_name, file_name), "r", swmr=True, libver="latest") as f:
                    for demo_id in f["data"]:
                        if self.robot_type is None:
                            self.robot_type = f["data"][demo_id].attrs["robot_type"]
                            self._joint_range = JOINT_RANGE[self.robot_type]
                        else:
                            assert self.robot_type == f["data"][demo_id].attrs["robot_type"]
                        self._demo_keys.append(f"{base_name}::{demo_id}")
        self._demo_indices = list(range(len(self._demo_keys)))
        # Preload demos into memory 
        self._all_demos = [self._preload_demo(demo_key) for demo_key in self._demo_keys]

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, epoch: int):
        self._epoch = epoch
        if self._shuffle:
            # deterministically shuffle the demos
            g = torch.Generator()
            g.manual_seed(epoch + self._seed)
            self._demo_indices = torch.randperm(len(self._demo_keys), generator=g).tolist()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        for demo_ptr in self._demo_indices[worker_id::num_workers]:
            yield from self.get_streamed_data(demo_ptr)

    def get_streamed_data(self, demo_ptr: int):
        data_chunks, mask_chunks = self._chunk_demo(self._all_demos[demo_ptr])
        # Initialize obs loaders
        obs_loaders = dict()
        for obs_type in self._visual_obs_types:
            if obs_type == "pcd":
                continue # TODO: add pcd loader
            else:
                base_name, demo_id = self._demo_keys[demo_ptr].split("::")
                for camera_name in self._multi_view_cameras:
                    kwargs = dict()
                    if obs_type == "seg":
                        kwargs["num_classes"] = 100
                    obs_loaders[f"{camera_name}::{obs_type}"] = iter(OBS_LOADER_MAP[obs_type](
                        data_path=f"{self._data_path}/videos/{self._task_name}",
                        base_name=base_name,
                        demo_id=demo_id, 
                        camera_name=camera_name,
                        batch_size=self._obs_window_size,
                        stride=1,
                        **kwargs,
                    ))
        for i in range(len(data_chunks)):
            data, mask = data_chunks[i], mask_chunks[i]
            # load visual obs
            for obs_type in self._visual_obs_types:
                if obs_type == "pcd":
                    continue # TODO: add pcd loader
                else:
                    for camera_name in self._multi_view_cameras:
                        data["obs"][f"{camera_name}::{obs_type}"] = next(obs_loaders[f"{camera_name}::{obs_type}"])
            data["masks"] = data["action_chunk_masks"] & mask[:, None] if self._use_action_chunks else mask
            yield data
        for obs_type in self._visual_obs_types:
            for camera_name in self._multi_view_cameras:
                obs_loaders[f"{camera_name}::{obs_type}"].close()

    def _preload_demo(self, demo_key: str) -> dict:
        """
        Preload a single demo into memory. Currently it loads action, proprio, and task info.
        Args:
            demo_key (str): Key of the demo to preload.
        Returns:
            demo (dict): Preloaded demo.
        """
        demo = dict()
        demo["obs"] = dict()
        base_name, demo_id = demo_key.split("::")
        
        # load low_dim data
        action_dict = dict()
        with h5py.File(os.path.join(self._data_path, "data", self._task_name, f"{base_name}.hdf5"), "r", swmr=True, libver="latest") as f:
            # TODO: Fix this (remove -1) with the new data format
            demo["obs"]["robot_r1::proprio"] = f["data"][demo_id]["obs"]["robot_r1::proprio"][:-1].astype(np.float32) # remove last frame

            for key, indices in ACTION_QPOS_INDICES[self.robot_type].items():
                action_dict[key] = f["data"][demo_id]["action"][:, indices]
                # action normalization
                action_dict[key] = (action_dict[key] - self._joint_range[key][0]) / (self._joint_range[key][1] - self._joint_range[key][0])
            if self._load_task_info:
                demo["obs"]["task::low_dim"] = f["data"][demo_id]["obs"]["task::low_dim"][:]
        if self._use_action_chunks:
            # make actions from (T, A) to (T, L_pred_horizon, A)
            # need to construct a mask
            action_chunks = []
            action_chunk_masks = []
            action_structure = deepcopy(any_slice(action_dict, np.s_[0:1]))  # (1, A)
            for t in range(get_batch_size(action_dict, strict=True)):
                action_chunk = any_slice(action_dict, np.s_[t : t + self._action_prediction_horizon])
                action_chunk_size = get_batch_size(action_chunk, strict=True)
                pad_size = self._action_prediction_horizon - action_chunk_size
                mask = any_concat(
                    [
                        np.ones((action_chunk_size,), dtype=bool),
                        np.zeros((pad_size,), dtype=bool),
                    ],
                    dim=0,
                )  # (L_pred_horizon,)
                action_chunk = any_concat(
                    [
                        action_chunk,
                    ]
                    + [any_ones_like(action_structure)] * pad_size,
                    dim=0,
                )  # (L_pred_horizon, A)
                action_chunks.append(action_chunk)
                action_chunk_masks.append(mask)
            action_chunks = any_stack(action_chunks, dim=0)  # (T, L_pred_horizon, A)
            action_chunk_masks = np.stack(action_chunk_masks, axis=0)  # (T, L_pred_horizon)
            demo["actions"] = action_chunks
            demo["action_masks"] = action_chunk_masks
        else:
            demo["actions"] = action_dict

        return demo

    def _chunk_demo(self, demo: dict) -> Tuple[List[dict], List[torch.Tensor]]:
        data_chunks, mask_chunks = [], []
        L = get_batch_size(demo, strict=True)
        assert L >= self._obs_window_size >= 1
        N_chunks = L - self._obs_window_size + 1
        # split obs into chunks
        for chunk_idx in range(N_chunks):
            s = np.s_[chunk_idx : chunk_idx + self._obs_window_size]
            data = dict()
            for k in demo:
                if k == "actions":
                    data[k] = any_slice(demo[k], np.s_[chunk_idx: chunk_idx + self._ctx_len])
                    action_chunk_size = get_batch_size(data[k], strict=True)
                    pad_size = self._ctx_len - action_chunk_size
                    # pad action chunks to equal length of ctx_len
                    data[k] = any_concat(
                        [
                            data[k],
                        ]
                        + [any_ones_like(any_slice(data[k], np.s_[0:1]))] * pad_size,
                        dim=0,
                    )
                    mask_chunks.append(torch.cat([
                        torch.ones((action_chunk_size,), dtype=torch.bool),
                        torch.zeros((pad_size,), dtype=torch.bool),
                    ], dim=0))
                else:
                    data[k] = any_slice(demo[k], s)
            data_chunks.append(data)
        return data_chunks, mask_chunks
