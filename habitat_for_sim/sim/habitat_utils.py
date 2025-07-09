import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import magnum as mn
import math
import random
from omegaconf import DictConfig
from habitat.articulated_agents.humanoids.kinematic_humanoid import (
    KinematicHumanoid,
)
from habitat.articulated_agent_controllers import (
    HumanoidRearrangeController,
    HumanoidSeqPoseController,
)

def wrap_pi(angle):
    """把任意角度包到 (-π, π] 区间"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def local2world_position_yaw(local_path: np.ndarray,
                              qpos: np.ndarray,
                              start_pos: np.ndarray,
                              start_yaw: float) -> np.ndarray:
    """
    将 [x_local, y_local, z_local, yaw_local] 变换为世界坐标路径
    输入:
        local_path: (N, 4) ndarray，每行 [x, y, z, yaw]，在起始点局部系下
        start_pos:  (3,) ndarray，世界系起始位置
        start_yaw:  float，世界系起始朝向 (rad)
    返回:
        world_path: (N, 4) ndarray，每行 [x_world, y_world, z_world, yaw_world]
    """
    cos_yaw = math.cos(start_yaw+qpos[2])
    sin_yaw = math.sin(start_yaw+qpos[2])

    rot_mat = np.array([
        [ cos_yaw, 0, -sin_yaw],
        [     0.0, 1,     0.0 ],
        [ sin_yaw, 0,  cos_yaw]
    ])  # 绕Y轴右手旋转

    out = np.zeros_like(local_path)

    for i, (x_l, y_l, z_l, yaw_l) in enumerate(local_path):
        pos_local = np.array([x_l, y_l, z_l])
        pos_world = rot_mat @ pos_local + start_pos
        yaw_world = wrap_pi(yaw_l + start_yaw)

        out[i, :3] = pos_world
        out[i, 3] = yaw_world

    return out.astype(np.float32)


def load_humanoid(sim):
    names = ["female_0", "female_1", "female_2", "female_3", "male_0", "male_1", "male_2", "male_3"]
    humanoid_name =  random.choice(names) 
    data_root = "/home/wangzejin/habitat/ON-MLLM/human_follower/habitat_humanoids" #wzjpath
    urdf_path = f"{data_root}/{humanoid_name}/{humanoid_name}.urdf"
    motion_pkl = f"{data_root}/{humanoid_name}/{humanoid_name}_motion_data_smplx.pkl"

    agent_cfg = DictConfig(
        {
            "articulated_agent_urdf": urdf_path,
            "motion_data_path": motion_pkl,
            "auto_update_sensor_transform": True,
        }
    )
    humanoid = KinematicHumanoid(agent_cfg, sim)
    humanoid.reconfigure()
    humanoid.update()

    controller = HumanoidRearrangeController(walk_pose_path=motion_pkl)
    controller.reset(humanoid.base_transformation)

    return humanoid, controller

def shortest_angle_diff(a, b):
    """
    返回 b−a 的最短角差，范围 (-π, π]
    """
    diff = (b - a + math.pi) % (2 * math.pi) - math.pi
    return diff



def to_vec3(v) -> mn.Vector3:
    """接受 magnum.Vector3 或 list/tuple/np.ndarray"""
    if isinstance(v, mn.Vector3):
        return v
    return mn.Vector3(float(v[0]), float(v[1]), float(v[2]))

# # ---------- 辅助转换 ---------- #
# def to_vec3(arr_like):
#     """
#     任意 Vector3 表示 → np.float32[3]
#     支持：Magnum Vector3 / ndarray / list / tuple
#     """
#     if isinstance(arr_like, mn.Vector3):
#         return np.array([arr_like.x, arr_like.y, arr_like.z], dtype=np.float32)
#     arr = np.asarray(arr_like, dtype=np.float32).reshape(3)
#     return arr

def to_quat(arr_like):
    """
    任意四元数 → np.float32[4]  (w, x, y, z)
    支持：Magnum Quaternion / numpy-quaternion / list / tuple / ndarray
    """
    if isinstance(arr_like, mn.Quaternion):
        return np.array([arr_like.scalar,
                         arr_like.vector.x,
                         arr_like.vector.y,
                         arr_like.vector.z], dtype=np.float32)
    if isinstance(arr_like, qt.quaternion):
        return np.array([arr_like.w, arr_like.x, arr_like.y, arr_like.z],
                        dtype=np.float32)
    arr = np.asarray(arr_like, dtype=np.float32).reshape(4)
    # 若给成 (x,y,z,w) 可自动调整 —— 以最后一个元素绝对值最大视为 w
    if abs(arr[0]) < abs(arr[3]):           # 猜测是 [x,y,z,w]
        arr = arr[[3, 0, 1, 2]]
    return arr.astype(np.float32)