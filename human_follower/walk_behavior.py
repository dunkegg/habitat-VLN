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
from habitat_for_sim.utils.goat import calculate_euclidean_distance
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

def clip_by_distance2target(path, distance, target_pos=None):
    if target_pos is None:
        target_pos = np.array(path[-1][0]) 
     # 获取目标点的位置
    clipped_path = [
        (pos, quat, yaw) for pos, quat, yaw in path
        if np.linalg.norm(np.array(pos) - target_pos) > distance  # wzj
    ]

    return clipped_path


# def get_path_with_time(raw_path, time_step = 0.1, speed = 0.5):
#     #raw path : postion, quat, yaw
def slerp_quaternion(q1: mn.Quaternion, q2: mn.Quaternion, t: float) -> mn.Quaternion:
    """
    对两个 Magnum 四元数执行球面线性插值（SLERP）
    保证数值稳定性，避免 NaN
    """
    q1 = q1.normalized()
    q2 = q2.normalized()

    dot = mn.math.dot(q1.vector, q2.vector)

    if dot < 0.0:
        q2 = mn.Quaternion(-q2.vector)
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        # 太接近，使用线性插值并归一化
        interp = (q1.vector * (1 - t) + q2.vector * t).normalized()
        return mn.Quaternion(interp)

    theta_0 = math.acos(dot)
    sin_theta = math.sin(theta_0)

    w1 = math.sin((1 - t) * theta_0) / sin_theta
    w2 = math.sin(t * theta_0) / sin_theta

    interp_vector = q1.vector * w1 + q2.vector * w2
    return mn.Quaternion(interp_vector.normalized())

def get_path_with_time(raw_path, time_step=0.1, speed=0.5):
    """
    对原始路径按速度插值，生成细化的 (position, quat, yaw) 路径

    Args:
        raw_path: List of (position: mn.Vector3, quat: mn.Quaternion, yaw: float) tuples
        time_step: float, 插值时间间隔（单位：秒）
        speed: float, 行进速度（单位：米/秒）

    Returns:
        new_path: List of (position: mn.Vector3, quat: mn.Quaternion, yaw: float) 插值后的路径
    """
    new_path = []
    if len(raw_path) < 2:
        return new_path

    step_dist = speed * time_step

    for i in range(len(raw_path) - 1):
        start_pos, start_quat, start_yaw = raw_path[i]
        end_pos, end_quat, end_yaw = raw_path[i + 1]

        seg_vec = end_pos - start_pos
        seg_len = seg_vec.length()
        
        if seg_len < 1e-4:
            continue

        direction = seg_vec.normalized()
        yaw_diff = shortest_angle_diff(start_yaw, end_yaw)

        n_steps = int(np.ceil(seg_len / step_dist))
        for step in range(n_steps):
            frac = (step * step_dist) / seg_len

            interp_pos = start_pos + direction * (frac * seg_len)
            interp_yaw = start_yaw + yaw_diff * frac
            interp_quat = slerp_quaternion(start_quat, end_quat, frac)

            new_path.append((interp_pos, interp_quat, interp_yaw))

    # 确保最后一个点包含在内
    last_pos, last_quat, last_yaw = raw_path[-1]
    new_path.append((last_pos, last_quat, last_yaw))

    return new_path


def generate_interfere_path_from_target_path(follow_path, agent, time_step=0.1, speed=0.5, radius=1.0):
    """
    根据目标人的轨迹生成干扰人的扰动路径
    Args:
        target_path: 原始路径 [(pos, quat, yaw)]
        agent: AgentHumanoid 实例（用于初始位置）
        time_step: 时间分辨率
        speed: 插值速度
        radius: 生成扰动点的偏移半径

    Returns:
        List of (pos, yaw) 干扰人路径
    """
    # dense_path = get_path_with_time(follow_path, time_step, speed)

    interfere_path = []
    distance = 0
    start_pos = follow_path[0][0]
    threshold= 0.5
    for pos, quat, yaw in follow_path:
        distance += calculate_euclidean_distance([start_pos.x, start_pos.y,start_pos.z],[pos.x, pos.y,pos.z])
        start_pos = pos
        if distance > threshold:
            distance = 0
            # 在主路径每个点附近生成一个扰动点
            offset = mn.Vector3(
                random.uniform(-radius, radius),
                0,
                random.uniform(-radius, radius)
            )
            new_pos = pos + offset
            interfere_path.append((new_pos, quat, yaw))

    dense_path = get_path_with_time(interfere_path, time_step, speed)
    return dense_path

def generate_interfere_sample_from_target_path(follow_path, pathfinder, radius=1.0):
    """
    沿主路径采样点生成干扰人轨迹，每个点在正前方向一定角度范围内偏移
    """
    interfere_path = []
    sample_distance = 0
    total_distance = 0
    threshold = 0.5
    start_pos = follow_path[0][0]
    circle_random = random.choice([True, False])
    for i in range(1, len(follow_path)):
        pos, _, yaw = follow_path[i]
        dis_diff = calculate_euclidean_distance(
            [start_pos.x, start_pos.y, start_pos.z],
            [pos.x, pos.y, pos.z]
        )
        sample_distance += dis_diff
        total_distance += dis_diff
        start_pos = pos

        if sample_distance < threshold:
            continue
        sample_distance = 0

        # 增大随机幅度
        weight_radius = math.log(total_distance + 1) * radius

        
        if circle_random:
            for _ in range(10):
                offset = mn.Vector3(
                    random.uniform(-weight_radius, weight_radius),
                    0,
                    random.uniform(-weight_radius, weight_radius)
                )
                new_pos = pos + offset
                real_coords = np.array([new_pos.x, new_pos.y, new_pos.z])
                if pathfinder.is_navigable(real_coords):
                    interfere_path.append(new_pos)
                    break
        else:
            # 方向向量（从 i-1 到 i）
            prev_pos = follow_path[i - 1][0]
            forward_vec = (pos - prev_pos).normalized()

            # 生成前方 ±60° 扇形内随机扰动向量
            max_angle = math.radians(60)
            for _ in range(10):
                angle = random.uniform(-max_angle, max_angle)
                rot_mat = mn.Matrix4.rotation_y(mn.Rad(angle))
                new_dir = rot_mat.transform_vector(forward_vec)
                offset = new_dir * random.uniform(0.1, weight_radius)

                new_pos = pos + offset
                real_coords = np.array([new_pos.x, new_pos.y, new_pos.z])
                if pathfinder.is_navigable(real_coords):
                    interfere_path.append(new_pos)
                    break

    return interfere_path

def generate_interfer_path(interfering_humanoids, human_path, time_step=1/10, speed=0.7, radius=1.5):
    """
    为所有干扰人生成扰动轨迹，并初始化路径状态

    Args:
        interfering_humanoids (list): AgentHumanoid 实例列表
        human_path (list): 目标人的轨迹 [(pos, quat, yaw)]
        time_step (float): 插值时间步长
        speed (float): 干扰人移动速度
        radius (float): 偏移扰动半径
    """
    for interferer in interfering_humanoids:
        path = generate_interfere_path_from_target_path(
            human_path, interferer,
            time_step=time_step, speed=speed, radius=radius
        )
        interferer.reset_path(path)

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
    
    keep_distance = 0.7


    observations = []
    humanoid_agent.step_directly(
        target_pos=human_path[0][0],
        target_yaw=human_path[0][2],
    )
    sim.step_physics(1.0 / fps)

    follow_state = sim.agents[0].get_state()


    # obs_quat = quat_from_angle_axis(human_path[5][2]+math.pi, np.array([0, 1, 0]))
    # follow_state.position = human_path[5][0]
    # follow_state.rotation = obs_quat
    # sim.agents[0].set_state(follow_state)
    # obs = sim.get_sensor_observations(0)
    # observations.append(obs.copy())
    # observations.append(obs.copy())


    follow_yaw = human_path[0][2]
    follow_state.position = human_path[0][0]
    follow_state.rotation = to_quat(human_path[0][1])
    sim.agents[0].set_state(follow_state)


    follow_timestep = 0

    move_dis = 0
    sample_list = [random.uniform(1, 1.5), random.uniform(1.5, 2),random.uniform(2, 2.5),random.uniform(2.5, 3.5), random.uniform(3, 3.5),random.uniform(3.5, 4)]
    long_follow_range = random.uniform(4, 5)
    
    
    for time_step in range(2, len(human_path)):
        goal_pos, goal_quat, goal_yaw = human_path[time_step]

        # 获取 humanoid 当前状态
        start_pos, start_yaw, _ = humanoid_agent.get_pose()

        seg_vec = goal_pos - start_pos
        seg_len = seg_vec.length()
        move_dis += seg_len
        if seg_len < 1e-4:
            continue

        direction = seg_vec.normalized()

        # 调用控制器移动一步
        humanoid_agent.step_with_controller(
            target_pos=goal_pos,
            target_yaw=goal_yaw,
            direction=direction,
        )


        # ▶ 插入干扰人形位置 todo
        if interfering_humanoids:
            # for interferer in interfering_humanoids:
            #     interferer.place_near_goal(goal_pos, radius=2.0)
        
            for interferer in interfering_humanoids:
                if hasattr(interferer, "interfere_path") and time_step < len(interferer.interfere_path):
                    pos, quat,yaw = interferer.interfere_path[time_step]
                    if time_step>0:
                        interferer_direction = (interferer.interfere_path[time_step][0] - interferer.interfere_path[time_step-1][0]).normalized()
                    else:
                        interferer_direction = direction
                    interferer.step_with_controller(pos, yaw, interferer_direction)
                    interferer.time_step += 1

        # 更新物理引擎
        sim.step_physics(1.0 / fps)
        
        # ▶ 记录轨迹与观察
        shortest_path = habitat_sim.ShortestPath()
        if move_dis > long_follow_range:
            shortest_path.requested_start = follow_state.position
            shortest_path.requested_end = goal_pos
            if sim.pathfinder.find_path(shortest_path) and follow_timestep< time_step:
                move_dis = 0
                long_follow_range = random.uniform(3, 5)
                sample_list = []

                # new_path = generate_path(shortest_path.points, sim.pathfinder, filt_distance=keep_distance, visualize=False)
                # for j in range(len(new_path)):
                #     follow_state.position = new_path[j][0]
                #     follow_state.rotation = to_quat(new_path[j][1])
                #     follow_yaw = new_path[j][2]
                #     sim.agents[0].set_state(follow_state)
                #     obs = sim.get_sensor_observations(0)
                #     observations.append(obs.copy())
                #     follow_data = {
                #         "obs_idx": len(observations) - 1,
                #         "follow_state": new_path[j],
                #         "human_state": human_path[time_step],
                #         "path": new_path[j:],
                #         "type": 0,
                #     }
                #     output["follow_paths"].append(follow_data)
                
                cur_follow_timestep = follow_timestep
                for t in range(cur_follow_timestep, time_step):
                    follow_state.position = human_path[t][0]
                    follow_state.rotation = to_quat(human_path[t][1])
                    follow_yaw = human_path[t][2]
                    sim.agents[0].set_state(follow_state)
                    if t%5 ==0:
                        observations.append(sim.get_sensor_observations(0).copy())
                        follow_data = {
                            "obs_idx": len(observations) - 1,
                            "follow_state": human_path[follow_timestep],
                            "human_state": human_path[time_step],
                            "path": clip_by_distance2target(human_path[follow_timestep:time_step], keep_distance),
                            "desc": humanoid_agent.get_desc(),
                            "type": 0,
                        }
                        output["follow_paths"].append(follow_data)
                    follow_timestep+=1

        elif len(sample_list) > 0 and move_dis > sample_list[0]:
            shortest_path.requested_start = follow_state.position
            shortest_path.requested_end = goal_pos
            if sim.pathfinder.find_path(shortest_path):
                del sample_list[0]
                # new_path = generate_path(shortest_path.points, sim.pathfinder, filt_distance=keep_distance, visualize=False)
                # follow_data = {
                #     "obs_idx": len(observations) - 1,
                #     "follow_state": (follow_state.position, follow_state.rotation, follow_yaw),
                #     "human_state": human_path[time_step],
                #     "path": new_path,
                #     "type": 1,
                # }
                # output["follow_paths"].append(follow_data)

                observations.append(sim.get_sensor_observations(0).copy())
                follow_data = {
                    "obs_idx": len(observations) - 1,
                    "follow_state": human_path[follow_timestep],
                    "human_state": human_path[time_step],
                    "path": clip_by_distance2target(human_path[follow_timestep:time_step], keep_distance),
                    "desc": humanoid_agent.get_desc(),
                    "type": 1,
                }
                output["follow_paths"].append(follow_data)
        elif move_dis < 3 and follow_timestep%3==0:
            shortest_path.requested_start = follow_state.position
            shortest_path.requested_end = goal_pos
            if sim.pathfinder.find_path(shortest_path) and follow_timestep< time_step-1:
                follow_timestep+=1
                follow_state.position = human_path[follow_timestep][0]
                follow_state.rotation = to_quat(human_path[follow_timestep][1])
                follow_yaw = human_path[follow_timestep][2]
                sim.agents[0].set_state(follow_state)

                observations.append(sim.get_sensor_observations(0).copy())           
                follow_data = {
                    "obs_idx": len(observations) - 1,
                    "follow_state": human_path[follow_timestep],
                    "human_state": human_path[time_step],
                    "path": clip_by_distance2target(human_path[follow_timestep:time_step], keep_distance),
                    "desc": humanoid_agent.get_desc(),
                    "type": 2,
                }
                
                output["follow_paths"].append(follow_data)

    output["obs"] = observations
    if all_index < 10:
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

