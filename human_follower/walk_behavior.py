import os
import gzip
import json
import yaml
import numpy as np
import habitat_sim
import math
import sys
import magnum as mn
from tqdm import tqdm
from PIL import Image
import imageio
import random
import logging
from human_follower.hybrid_a.planner import HybridAStar

from habitat_sim.utils.common import quat_from_coeffs, quat_from_two_vectors , quat_from_angle_axis, quat_to_angle_axis # 把 [w,x,y,z] 转 Quaternion
from human_follower.save_data import to_quat
from habitat_for_sim.agent.path_generator import generate_path
from habitat_for_sim.utils.frontier_exploration import FrontierExploration
from habitat_sim.utils import viz_utils as vut

def to_vec3(v) -> mn.Vector3:
    """接受 magnum.Vector3 或 list/tuple/np.ndarray"""
    if isinstance(v, mn.Vector3):
        return v
    return mn.Vector3(float(v[0]), float(v[1]), float(v[2]))

def quat_from_two_vectors_mn(v0, v1, eps=1e-8):
    a = to_vec3(v0).normalized()
    b = to_vec3(v1).normalized()
    c = mn.math.clamp(mn.math.dot(a, b), -1.0, 1.0)

    if c < -1.0 + eps:                       # 反向
        orth = mn.Vector3(1,0,0) if abs(a.x) < 0.9 else mn.Vector3(0,1,0)
        axis = mn.math.cross(a, orth).normalized()
        return mn.Quaternion.rotation(mn.Rad(math.pi), axis)

    axis  = mn.math.cross(a, b)
    s     = math.sqrt((1.0 + c) * 2.0)
    inv_s = 1.0 / s
    # 这里用 (vector, scalar) 构造
    q = mn.Quaternion(
            mn.Vector3(axis.x * inv_s,
                       axis.y * inv_s,
                       axis.z * inv_s),
            s * 0.5
        ).normalized()
    return q     


def shortest_angle_diff(a, b):
    """
    返回 b−a 的最短角差，范围 (-π, π]
    """
    diff = (b - a + math.pi) % (2 * math.pi) - math.pi
    return diff


def simulate(sim, dt, get_observations=False):
    r"""Runs physics simulation at 60FPS for a given duration (dt) optionally collecting and returning sensor observations."""
    observations = []
    target_time = sim.get_world_time() + dt
    while sim.get_world_time() < target_time:
        sim.step_physics(0.1 / 60.0)
        if get_observations:
            observations.append(sim.get_sensor_observations())
    return observations

def walk_along_path_multi(
    all_index,
    sim,
    humanoid_agent,  # AgentHumanoid 实例
    human_path,
    fps=10,
    forward_speed=0.7,
    interfering_humanoids=None,
):
    output = {"obs": [], "follow_paths": []}
    height_bias = 0
    keep_distance = 0.7

    # 修正高度偏移
    for i in range(len(human_path)):
        pos, quat, yaw = human_path[i]
        new_pos = mn.Vector3(pos.x, pos.y - height_bias, pos.z)
        human_path[i] = (new_pos, quat, yaw)

    observations = []
    humanoid_agent.humanoid.base_pos = human_path[0][0]
    humanoid_agent.humanoid.base_rot = human_path[0][2]
    humanoid_agent.step_directly(
        target_pos=human_path[0][0],
        target_yaw=human_path[0][2],
    )
    follow_state = sim.agents[0].get_state()
    follow_yaw = human_path[0][2]
    follow_state.position = human_path[0][0]
    follow_state.rotation = to_quat(human_path[0][1])
    sim.agents[0].set_state(follow_state)

    move_dis = 0
    sample_list = [random.uniform(2, 2.5), random.uniform(3, 3.5)]
    record_range = random.uniform(3, 5)
    
    for k in range(1, len(human_path)):
        goal_pos, goal_quat, goal_yaw = human_path[k]

        start_pos, start_yaw, quat = humanoid_agent.get_pose()

        seg_vec = goal_pos - start_pos
        seg_len = seg_vec.length()
        move_dis += seg_len
        if seg_len < 1e-4:
            continue

        yaw_diff = shortest_angle_diff(start_yaw, goal_yaw)
        direction = seg_vec.normalized()
        step_dist = forward_speed / fps
        n_steps = int(np.ceil(seg_len / step_dist))

        for step in range(n_steps):
            start_pos = start_pos + direction * step_dist
            
            frac = (step + 1) / n_steps
            start_yaw = start_yaw + yaw_diff * frac
            # ▶ 调用 step_to 控制主行人移动
            humanoid_agent.step_with_controller(
                target_pos=goal_pos,
                target_yaw=goal_yaw,
                direction=direction,
            )

            sim.step_physics(1.0 / fps)



        # ▶ 插入干扰人形位置
        if interfering_humanoids:
            for interferer in interfering_humanoids:
                interferer.place_near_goal(goal_pos, radius=2.0)

        # ▶ 记录轨迹与观察
        shortest_path = habitat_sim.ShortestPath()
        if move_dis > record_range:
            shortest_path.requested_start = follow_state.position
            shortest_path.requested_end = goal_pos
            if sim.pathfinder.find_path(shortest_path):
                move_dis = 0
                record_range = random.uniform(3, 5)
                sample_list = []
                new_path = generate_path(shortest_path.points, sim.pathfinder, filt_distance=keep_distance, visualize=False)
                for i in range(len(new_path)):
                    follow_state.position = new_path[i][0]
                    follow_state.rotation = to_quat(new_path[i][1])
                    follow_yaw = new_path[i][2]
                    sim.agents[0].set_state(follow_state)
                    obs = sim.get_sensor_observations(0)
                    observations.append(obs.copy())
                    follow_data = {
                        "obs_idx": len(observations) - 1,
                        "follow_state": new_path[i],
                        "human_state": human_path[k],
                        "path": new_path[i:],
                        "type": 0,
                    }
                    output["follow_paths"].append(follow_data)
            else:
                observations.append(sim.get_sensor_observations(0).copy())
        else:
            observations.append(sim.get_sensor_observations(0).copy())
            shortest_path.requested_start = follow_state.position
            shortest_path.requested_end = goal_pos
            if len(sample_list) > 0 and move_dis > sample_list[0]:
                if sim.pathfinder.find_path(shortest_path):
                    del sample_list[0]
                    new_path = generate_path(shortest_path.points, sim.pathfinder, filt_distance=keep_distance, visualize=False)
                    follow_data = {
                        "obs_idx": len(observations) - 1,
                        "follow_state": (follow_state.position, follow_state.rotation, follow_yaw),
                        "human_state": human_path[k],
                        "path": new_path,
                        "type": 1,
                    }
                    output["follow_paths"].append(follow_data)

    output["obs"] = observations
    if all_index < 20:
        os.makedirs("results2", exist_ok=True)
        vut.make_video(
            observations,
            "color_0_0",
            "color",
            f"results2/humanoid_wrapper_{all_index}",
            open_vid=False,
        )
    print("walk done")
    return output

