
from habitat.utils.visualizations.utils import (
    append_text_underneath_image,
    images_to_video,
)
from habitat.articulated_agents.humanoids.kinematic_humanoid import (
    KinematicHumanoid,
)
from habitat.articulated_agent_controllers import (
    HumanoidRearrangeController,
    HumanoidSeqPoseController,
)
from habitat_sim.utils.common import quat_from_coeffs, quat_from_two_vectors , quat_from_angle_axis, quat_to_angle_axis # 把 [w,x,y,z] 转 Quaternion

import numpy as np
import habitat_sim
import math
import sys
import magnum as mn
from tqdm import tqdm
from PIL import Image
import imageio
import random
from omegaconf import DictConfig

def get_humanoid_id(id_exception=None):
    while True:
        gender =  random.choice(['female','male'])
        if gender == 'female':
            num = random.randint(0, 34)
        else:
            num = random.randint(0, 64)
        humanoid_name = gender+'_'+f'{num}'
        if id_exception is None or humanoid_name!=id_exception:
            break
    return humanoid_name

class AgentHumanoid:
    def __init__(self, sim, base_pos: mn.Vector3, name ,motion_path=None, is_target=False):
        self.sim = sim
        self.is_target = is_target
        self.humanoid = None
        self.controller = None
        self.remove_from_scene()
        self.humanoid, self.controller = self._load_humanoid(sim,name, motion_path)
        self.humanoid.base_pos = base_pos
        # self.pos = base_pos
        # self.yaw = 0.0
        # self.seg_vec = None
        # self.quat = None
        self.time_step = 0
        self.name = name

    def reset(self, name):
        self.time_step = 0
        self.humanoid, self.controller = self._load_humanoid(self.sim, name, None)
        return
    def remove_from_scene(self):
        if hasattr(self.humanoid, "sim_obj") and self.humanoid.sim_obj is not None:
            self.sim.remove_articulated_object(self.humanoid.sim_obj.object_id)
            self.humanoid.sim_obj = None  # 防止重复删除
    def _load_humanoid(self, sim, humanoid_name,motion_path=None):
        self.remove_from_scene()
        # data_root = "human_follower/habitat_humanoids"
        data_root = "human_follower/humanoid_data"
        urdf_path = f"{data_root}/{humanoid_name}/{humanoid_name}.urdf"
        motion_pkl = motion_path or f"{data_root}/{humanoid_name}/{humanoid_name}_motion_data_smplx.pkl"
        agent_cfg = DictConfig({
            "articulated_agent_urdf": urdf_path,
            "motion_data_path": motion_pkl,
            "auto_update_sensor_transform": True,
        })
        humanoid = KinematicHumanoid(agent_cfg, sim)
        humanoid.reconfigure()
        humanoid.update()
        controller = HumanoidRearrangeController(walk_pose_path=motion_pkl)
        controller.reset(humanoid.base_transformation)
        return humanoid, controller

    def _random_offset_pos(self, goal_pos: mn.Vector3, radius=2.0):
        angle = np.random.uniform(0, 2 * math.pi)
        offset = mn.Vector3(math.cos(angle), 0, math.sin(angle)) * radius
        return goal_pos + offset

    def place_near_goal(self, goal_pos: mn.Vector3, radius=2.0):
        pos = self._random_offset_pos(goal_pos, radius)
        self.yaw = np.random.uniform(-math.pi, math.pi)
        self.humanoid.base_pos = pos
        self.humanoid.base_rot = self.yaw

    def reset_path(self,path):
        self.interfere_path = path
        self.time_step = 0

    def get_pose(self):
        position = self.humanoid.base_pos
        yaw = self.humanoid.base_rot
        quat = quat_from_angle_axis(yaw, np.array([0, 1, 0]))

        return position, yaw, quat

    def step_directly(self, target_pos: mn.Vector3, target_yaw: float):
        self.humanoid.base_pos = target_pos
        self.humanoid.base_rot = target_yaw

    def step_with_controller(self, target_pos: mn.Vector3, target_yaw: float, direction: mn.Vector3):

        direction = direction.normalized()

        self.humanoid.base_pos = target_pos
        # 计算方向向量的 yaw（绕 y 轴的旋转角）
        # target_yaw = math.atan2(-direction.x, -direction.z)  # habitat 中朝 -Z 为前方
        self.humanoid.base_rot = target_yaw

        self.controller.calculate_walk_pose(direction)
        new_pose = self.controller.get_pose()
        new_joints = new_pose[:-16]
        new_pos_transform_base = new_pose[-16:]
        new_pos_transform_offset = new_pose[-32:-16]

        if np.array(new_pos_transform_offset).sum() != 0:
            vecs_base = [mn.Vector4(new_pos_transform_base[i * 4: (i + 1) * 4]) for i in range(4)]
            vecs_offset = [mn.Vector4(new_pos_transform_offset[i * 4: (i + 1) * 4]) for i in range(4)]
            self.humanoid.set_joint_transform(
                new_joints, mn.Matrix4(*vecs_offset), mn.Matrix4(*vecs_base)
            )
            self.humanoid.base_pos = target_pos  # 再次同步位置

        # self.sim.step_physics(1.0 / fps)
