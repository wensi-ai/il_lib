import h5py
import logging
import numpy as np
import os
import pandas as pd
import torch
import torch.distributed as dist
from copy import deepcopy
from il_lib.utils.array_tensor_utils import any_concat, any_ones_like, any_slice, any_stack, get_batch_size
from il_lib.utils.training_utils import sequential_sum_balanced_partitioning
from torch.utils.data import IterableDataset, Dataset, get_worker_info
from typing import Any, Optional, List, Tuple, Dict, Generator

from omnigibson.learning.utils.eval_utils import ACTION_QPOS_INDICES, JOINT_RANGE, PROPRIO_QPOS_INDICES, PROPRIOCEPTION_INDICES
from omnigibson.learning.utils.obs_utils import OBS_LOADER_MAP


logger = logging.getLogger("BehaviorDataset")


class BehaviorDataset(IterableDataset):

    @classmethod
    def get_all_demo_keys(cls, data_path: str, task_id: int):
        task_dir_name = f"task-{task_id:04d}"
        assert os.path.exists(f"{data_path}/data/{task_dir_name}"), f"Data path does not exist!"
        demo_keys = sorted([
            file_name.split(".")[0].split("_")[-1] for file_name in sorted(os.listdir(f"{data_path}/data/{task_dir_name}"))
            if file_name.endswith(".parquet")
        ])
        return demo_keys

    def __init__(
        self,
        *args,
        data_path: str,
        task_id: int,
        demo_keys: List[str],
        obs_window_size: int,
        ctx_len: int,
        use_action_chunks: bool = False,
        action_prediction_horizon: Optional[int] = None,
        downsample_factor: int = 1,
        visual_obs_types: List[str],
        multi_view_cameras: Optional[Dict[str, Any]] = None,
        load_task_info: bool = False,
        seed: int = 42,
        shuffle: bool = True,
        # dataset parameters
        online_pcd_generation: bool = False,
        **kwargs,
    ):
        """
        Args:
            data_path (str): Path to the data directory.
            task_id (int): Task ID.
            demo_keys (List[str]): List of demo keys.
            obs_window_size (int): Size of the observation window.
            ctx_len (int): Context length.
            use_action_chunks (bool): Whether to use action chunks.
                Action will be from (T, A) to (T, L_pred_horizon, A)
            action_prediction_horizon (Optional[int]): Horizon of the action prediction.
                Must not be None if use_action_chunks is True.
            downsample_factor (int): Downsample factor for the data (with uniform temporal subsampling). 
                Note that the original data is at 30Hz, so if factor=3 then data will be at 10Hz.
                Default is 1 (no downsampling), must be >= 1.
            visual_obs_types (List[str]): List of visual observation types to load.
                Valid options are: "rgb", "depth", "seg".
            multi_view_cameras (Optional[Dict[str, Any]]): Dict of id-camera pairs to load obs from.
            load_task_info (bool): Whether to load privileged task information.
            seed (int): Random seed.
            shuffle (bool): Whether to shuffle the dataset.
            online_pcd_generation (bool): Whether to generate point clouds online or using pre-generated ones.
        """
        super().__init__(*args, **kwargs)
        self._data_path = data_path
        self._task_id = task_id
        self._demo_keys = demo_keys
        self._obs_window_size = obs_window_size
        self._ctx_len = ctx_len
        self._use_action_chunks = use_action_chunks
        self._action_prediction_horizon = action_prediction_horizon
        assert self._action_prediction_horizon is not None if self._use_action_chunks else True, \
            "action_prediction_horizon must be provided if use_action_chunks is True!"
        self._downsample_factor = downsample_factor
        assert self._downsample_factor >= 1, "downsample_factor must be >= 1!"
        self._load_task_info = load_task_info
        self._seed = seed
        self._shuffle = shuffle
        self._epoch = 0

        assert set(visual_obs_types).issubset({"rgb", "depth_linear", "seg_instance_id", "pcd"}), \
            "visual_obs_types must be a subset of {'rgb', 'depth_linear', 'seg_instance_id', 'pcd'}!"
        self._visual_obs_types = set(visual_obs_types)

        self._multi_view_cameras = multi_view_cameras
        self._online_pcd_generation = online_pcd_generation
        if self._online_pcd_generation and "pcd" in self._visual_obs_types:
            self._visual_obs_types.add("rgb")
            self._visual_obs_types.add("depth_linear")

        self.robot_type = "R1Pro"

        self._demo_indices = list(range(len(self._demo_keys)))
        # Preload demos into memory 
        self._all_demos = [self._preload_demo(demo_key) for demo_key in self._demo_keys]
        # get demo lengths (N_chunks)
        self._demo_lengths = []
        for demo in self._all_demos:
            L = get_batch_size(demo, strict=True)
            assert L >= self._obs_window_size >= 1
            self._demo_lengths.append(L - self._obs_window_size + 1)
        logger.info(f"Dataset chunk length: {sum(self._demo_lengths)}")

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

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        global_worker_id, total_global_workers = self._get_global_worker_id()
        demo_lengths_shuffled = [self._demo_lengths[i] for i in self._demo_indices]
        start_demo_id, start_demo_idx, end_demo_id, end_demo_idx = sequential_sum_balanced_partitioning(
            demo_lengths_shuffled, total_global_workers, global_worker_id
        )
        for demo_idx, demo_ptr in enumerate(self._demo_indices[start_demo_id:end_demo_id+1]):
            start_idx = start_demo_idx if demo_idx == 0 else 0
            end_idx = end_demo_idx if demo_idx == end_demo_id - start_demo_id else self._demo_lengths[demo_ptr]
            yield from self.get_streamed_data(demo_ptr, start_idx, end_idx)

    def get_streamed_data(self, demo_ptr: int, start_idx: int, end_idx: int) -> Generator[Dict[str, Any], None, None]:
        chunk_generator = self._chunk_demo(demo_ptr, start_idx, end_idx)
        # Initialize obs loaders
        obs_loaders = dict()
        for obs_type in self._visual_obs_types:
            if obs_type == "pcd" and not self._online_pcd_generation:
                # pcd_generator
                f_pcd = h5py.File(f"{self._data_path}/pcd/task-{self._task_id:04d}/episode_{self._demo_keys[demo_ptr]}.hdf5", "r", swmr=True, libver='latest')
                # Create a generator that yields sliding windows of point clouds
                pcd_data = f_pcd["data/demo_0/robot_r1::fused_pcd"]
                def pcd_window_generator(start_idx, end_idx):
                    for i in range(start_idx, end_idx):
                        yield torch.from_numpy(pcd_data[i * self._downsample_factor : (i + self._obs_window_size) * self._downsample_factor : self._downsample_factor])
                pcd_generator = pcd_window_generator(start_idx=start_idx, end_idx=end_idx)
            else:
                # calculate the start a
                for camera_id in self._multi_view_cameras.keys():
                    camera_name = self._multi_view_cameras[camera_id]["name"]
                    stride = 1
                    # TODO: ADD KWARGS
                    obs_loaders[f"{camera_name}::{obs_type}"] = iter(OBS_LOADER_MAP[obs_type](
                        data_path=self._data_path,
                        task_id=self._task_id,
                        camera_id=camera_id, 
                        demo_id=self._demo_keys[demo_ptr],
                        batch_size=self._obs_window_size,
                        stride=stride,
                        start_idx=start_idx * stride * self._downsample_factor,
                        end_idx=((end_idx - 1) * stride + self._obs_window_size) * self._downsample_factor,
                        output_size=tuple(self._multi_view_cameras[camera_id]["resolution"]),
                    ))
        for _ in range(start_idx, end_idx):
            data, mask = next(chunk_generator)
            # load visual obs
            for obs_type in self._visual_obs_types:
                if obs_type == "pcd":
                    # get file from 
                    data["obs"]["pcd"] = next(pcd_generator)
                else:
                    for camera in self._multi_view_cameras.values():
                        data["obs"][f"{camera['name']}::{obs_type}"] = next(obs_loaders[f"{camera['name']}::{obs_type}"])
            data["masks"] = mask
            yield data
        for obs_type in self._visual_obs_types:
            if obs_type == "pcd":
                f_pcd.close()
            else:
                for camera in self._multi_view_cameras.values():
                    obs_loaders[f"{camera['name']}::{obs_type}"].close()

    def _preload_demo(self, demo_key: str) -> Dict[str, Any]:
        """
        Preload a single demo into memory. Currently it loads action, proprio, and task info.
        Args:
            demo_key (str): Key of the demo to preload.
        Returns:
            demo (dict): Preloaded demo.
        """
        demo = dict()
        demo["obs"] = dict()
        # load low_dim data
        action_dict = dict()
        df = pd.read_parquet(os.path.join(self._data_path, "data", f"task-{self._task_id:04d}", f"episode_{demo_key}.parquet"))
        proprio = torch.from_numpy(np.array(df["observation.state"][::self._downsample_factor].tolist(), dtype=np.float32))
        demo["obs"] = {
            "qpos": dict(),
            "odom": {
                "base_velocity": 2 * (
                    proprio[..., PROPRIOCEPTION_INDICES[self.robot_type]["base_qvel"]] - JOINT_RANGE[self.robot_type]["base"][0]
                ) / (JOINT_RANGE[self.robot_type]["base"][1] - JOINT_RANGE[self.robot_type]["base"][0]) - 1.0
            },
        }
        for key in PROPRIO_QPOS_INDICES[self.robot_type]:
            if "gripper" in key:
                # rectify gripper actions to {-1, 1}
                demo["obs"]["qpos"][key] = torch.mean(proprio[..., PROPRIO_QPOS_INDICES[self.robot_type][key]], dim=-1, keepdim=True)
                demo["obs"]["qpos"][key] = torch.where(
                    demo["obs"]["qpos"][key] > (JOINT_RANGE[self.robot_type][key][0] + JOINT_RANGE[self.robot_type][key][1]) / 2, 1.0, -1.0
                )
            else:
                # normalize the qpos to [-1, 1]
                demo["obs"]["qpos"][key] = 2 * (
                    proprio[..., PROPRIO_QPOS_INDICES[self.robot_type][key]] - JOINT_RANGE[self.robot_type][key][0]
                ) / (JOINT_RANGE[self.robot_type][key][1] - JOINT_RANGE[self.robot_type][key][0]) - 1.0
        demo["obs"]["cam_rel_poses"] = torch.from_numpy(np.array(df["observation.cam_rel_poses"][::self._downsample_factor].tolist(), dtype=np.float32))
        # Note that we need to take the action at the timestamp before the next observation
        action_arr = torch.from_numpy(np.array(df["action"].tolist(), dtype=np.float32))
        # First pad the action array so that it is divisible by the downsample factor
        if action_arr.shape[0] % self._downsample_factor != 0:
            pad_size = self._downsample_factor - (action_arr.shape[0] % self._downsample_factor)
            # pad with the last action
            action_arr = torch.cat([action_arr, action_arr[-1:].repeat(pad_size, 1)], dim=0)
        # Now downsample the action array
        action_arr = action_arr[self._downsample_factor - 1::self._downsample_factor]
        for key, indices in ACTION_QPOS_INDICES[self.robot_type].items():
            action_dict[key] = action_arr[:, indices]
            # action normalization
            if not "gripper" in key:    # Gripper actions are already normalized to [-1, 1]
                action_dict[key] = 2 * (
                    action_dict[key] - JOINT_RANGE[self.robot_type][key][0]
                ) / (JOINT_RANGE[self.robot_type][key][1] - JOINT_RANGE[self.robot_type][key][0]) - 1.0
        if self._load_task_info:
            demo["obs"]["task::low_dim"] = torch.from_numpy(np.array(df["observation.task_info"][::self._downsample_factor].tolist(), dtype=np.float32))
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
                        torch.ones((action_chunk_size,), dtype=torch.bool),
                        torch.zeros((pad_size,), dtype=torch.bool),
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
            action_chunk_masks = torch.stack(action_chunk_masks, dim=0)  # (T, L_pred_horizon)
            demo["actions"] = action_chunks
            demo["action_masks"] = action_chunk_masks
        else:
            demo["actions"] = action_dict

        return demo

    def _chunk_demo(self, demo_ptr: int, start_idx: int, end_idx: int) -> Generator[Tuple[dict, torch.Tensor], None, None]:
        demo = self._all_demos[demo_ptr]
        # split obs into chunks
        for chunk_idx in range(start_idx, end_idx):
            data, mask = [], []
            s = np.s_[chunk_idx : chunk_idx + self._obs_window_size]
            data = dict()
            for k in demo:
                if k == "actions":
                    data[k] = any_slice(demo[k], np.s_[chunk_idx: chunk_idx + self._ctx_len])
                    action_chunk_size = get_batch_size(data[k], strict=True)
                    pad_size = self._ctx_len - action_chunk_size
                    if self._use_action_chunks:
                        assert pad_size == 0, "pad_size should be 0 if use_action_chunks is True!"
                        mask = demo["action_masks"][chunk_idx: chunk_idx + self._ctx_len]
                    else:
                        # pad action chunks to equal length of ctx_len
                        data[k] = any_concat(
                            [
                                data[k],
                            ]
                            + [any_ones_like(any_slice(data[k], np.s_[0:1]))] * pad_size,
                            dim=0,
                        )
                        mask = torch.cat([
                            torch.ones((action_chunk_size,), dtype=torch.bool),
                            torch.zeros((pad_size,), dtype=torch.bool),
                        ], dim=0)
                elif k != "action_masks":
                    data[k] = any_slice(demo[k], s)
                else:
                    # action_masks has already been processed
                    pass
            yield data, mask

    def _get_global_worker_id(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            num_workers = worker_info.num_workers if worker_info else 1
            global_worker_id = rank * num_workers + worker_id
            total_global_workers = world_size * num_workers
        else:
            global_worker_id = worker_id
            total_global_workers = worker_info.num_workers if worker_info else 1
        return global_worker_id, total_global_workers
    

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
