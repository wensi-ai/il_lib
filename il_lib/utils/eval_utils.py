import numpy as np
import torch as th
from collections import OrderedDict


# Robot parameters
SUPPORTED_ROBOTS = ["R1Pro"]
DEFAULT_ROBOT_TYPE = (
    "R1Pro"  # This should always be our robot generally since GELLO is designed for this specific robot
)
ROBOT_NAME = "robot_r1"
RESOLUTION = [240, 240]  # Resolution for RGB and depth images


# Action indices
ACTION_QPOS_INDICES = {
    "R1Pro": OrderedDict(
        {
            "base": np.s_[0:3],
            "torso": np.s_[3:7],
            "left_arm": np.s_[7:14],
            "left_gripper": np.s_[14:15],
            "right_arm": np.s_[15:22],
            "right_gripper": np.s_[22:23],
        }
    )
}


# Proprioception configuration
PROPRIOCEPTION_INDICES = {
    "R1Pro": OrderedDict(
        {
            "joint_qpos": np.s_[0:28],
            "joint_qpos_sin": np.s_[28:56],
            "joint_qpos_cos": np.s_[56:84],
            "joint_qvel": np.s_[84:112],
            "joint_qeffort": np.s_[112:140],
            "robot_pos": np.s_[140:143],
            "robot_ori_cos": np.s_[143:146],
            "robot_ori_sin": np.s_[146:149],
            "robot_2d_ori": np.s_[149:150],
            "robot_2d_ori_cos": np.s_[150:151],
            "robot_2d_ori_sin": np.s_[151:152],
            "robot_lin_vel": np.s_[152:155],
            "robot_ang_vel": np.s_[155:158],
            "arm_left_qpos": np.s_[158:165],
            "arm_left_qpos_sin": np.s_[165:172],
            "arm_left_qpos_cos": np.s_[172:179],
            "arm_left_qvel": np.s_[179:186],
            "eef_left_pos": np.s_[186:189],
            "eef_left_quat": np.s_[189:193],
            "grasp_left": np.s_[193:194],
            "gripper_left_qpos": np.s_[194:196],
            "gripper_left_qvel": np.s_[196:198],
            "arm_right_qpos": np.s_[198:205],
            "arm_right_qpos_sin": np.s_[205:212],
            "arm_right_qpos_cos": np.s_[212:219],
            "arm_right_qvel": np.s_[219:226],
            "eef_right_pos": np.s_[226:229],
            "eef_right_quat": np.s_[229:233],
            "grasp_right": np.s_[233:234],
            "gripper_right_qpos": np.s_[234:236],
            "gripper_right_qvel": np.s_[236:238],
            "trunk_qpos": np.s_[238:242],
            "trunk_qvel": np.s_[242:246],
            "base_qpos": np.s_[246:249],
            "base_qpos_sin": np.s_[249:252],
            "base_qpos_cos": np.s_[252:255],
            "base_qvel": np.s_[255:258],
        }
    )
}

# Proprioception indices
PROPRIO_QPOS_INDICES = {
    "R1Pro": OrderedDict(
        {
            "torso": np.s_[6:10],
            "left_arm": np.s_[10:24:2],
            "right_arm": np.s_[11:24:2],
            "left_gripper": np.s_[24:26],
            "right_gripper": np.s_[26:28],
        }
    )
}


# Joint limits
JOINT_RANGE = {
    "R1Pro": {
        "base": (
            np.array([-0.75, -0.75, -1.0]),
            np.array([0.75, 0.75, 1.0])
        ),
        "torso": (
            np.array([-1.1345, -2.7925, -1.8326, -3.0543]),
            np.array([1.8326, 2.5307, 1.5708, 3.0543])
        ),
        "left_arm": (
            np.array([-4.4506, -0.1745, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708]),
            np.array([1.3090, 3.1416, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708])
        ),
        "right_arm": (
            np.array([-4.4506, -3.1416, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708]),
            np.array([1.3090, 0.1745, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708])
        ),
        "left_gripper": (
            np.array([-1]),
            np.array([1])
        ),
        "right_gripper": (
            np.array([-1]),
            np.array([1])
        ),
    }
}


# PCD range
PCD_RANGE = (
    np.array([-0.5, -0.5, -0.5]),
    np.array([0.5, 0.5, 0.5])
)


def flatten_obs_dict(obs: dict, parent_key: str = "") -> dict:
    """
    Process the observation dictionary by recursively flattening the keys.
    so obs["robot_r1"]["camera"]["rgb"] will become obs["robot_r1::camera:::rgb"].
    """
    processed_obs = {}
    for key, value in obs.items():
        new_key = f"{parent_key}::{key}" if parent_key else key
        if isinstance(value, dict):
            processed_obs.update(flatten_obs_dict(value, parent_key=new_key))
        else:
            processed_obs[new_key] = value
    return processed_obs


def find_start_point(base_vel):
    """
    Find the first point where the base velocity is non-zero.
    This is used to skip the initial part of the dataset where the robot is not moving.
    """
    start_idx = np.where(np.linalg.norm(base_vel, axis=-1) > 1e-5)[0]
    if len(start_idx) == 0:
        return 0
    return min(start_idx[0], 500)  # Limit to the first 100 points to avoid long initial periods
