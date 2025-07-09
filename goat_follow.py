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

from omegaconf import DictConfig
# 将上级目录加入 Python 搜索路径
sys.path.append(os.path.abspath(".."))
from utils.explore.explore_habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
# from utils.project_direction import DirectionProjector
from human_follower.save_data import save_output_to_h5, to_quat
from utils.goat import read_yaml, extract_dict_from_folder, get_current_scene, process_episodes_and_goals, convert_to_scene_objects, find_scene_path, calculate_euclidean_distance
from agent.path_generator import generate_path
from utils.frontier_exploration import FrontierExploration
from habitat_sim.utils import viz_utils as vut
def simulate(sim, dt, get_observations=False):
    r"""Runs physics simulation at 60FPS for a given duration (dt) optionally collecting and returning sensor observations."""
    observations = []
    target_time = sim.get_world_time() + dt
    while sim.get_world_time() < target_time:
        sim.step_physics(0.1 / 60.0)
        if get_observations:
            observations.append(sim.get_sensor_observations())
    return observations
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

# def quaternion_to_yaw(q):
#     """
#     q : mn.Quaternion  (w + xyz)
#     返回绕 Y 轴的航向角（弧度），范围 (-π, π]
#     """
#     # y-up, z-forward 右手坐标系
#     siny_cosp = 2.0 * (q.scalar * q.vector.y + q.vector.x * q.vector.z)
#     cosy_cosp = 1.0 - 2.0 * (q.vector.y**2 + q.vector.z**2)
#     return math.atan2(siny_cosp, cosy_cosp)

# def convert_path(raw_path):
#     """
#     raw_path: [(pos_list, quat_wxyz), ...]
#       pos_list -> [x,y,z]  or np.ndarray
#       quat_wxyz -> [w,x,y,z] list / np.ndarray
#     返回: [(mn.Vector3, float_yaw)]
#     """
#     out = []
#     for pos_raw, quat_raw in raw_path:
#         # 1) 位置
#         pos_vec = mn.Vector3(pos_raw)
#         # pos_vec.y += 1
#         # 2) 四元数 → Magnum.Quaternion
#         quat_raw = np.asarray(quat_raw, dtype=float)
#         if quat_raw.shape != (4,):
#             raise ValueError("四元数必须是长度 4 的 [w,x,y,z]")
#         quat = mn.Quaternion(
#             mn.Vector3(quat_raw[1], quat_raw[2], quat_raw[3]),
#             quat_raw[0],
#         )

#         # 3) 取 yaw
#         yaw = quaternion_to_yaw(quat)

#         out.append((pos_vec,quat,yaw))
#     return out
def load_humanoid(sim):
    names = ["female_0", "female_1", "female_2", "female_3", "male_0", "male_1", "male_2", "male_3"]
    humanoid_name =  random.choice(names) 
    data_root = "human_follower/habitat_humanoids"
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


def walk_along_path(all_index, sim, humanoid, controller, human_path, fps=10, forward_speed=0.7):
    
    output = {"obs":[], "follow_paths":[]}
    """
    path: [(mn.Vector3 pos, float yaw_rad), ...]
    """
    height_bias = 0
    keep_distance = 0.7
    for i in range(len(human_path)):
        pos, quat, yaw = human_path[i]            # 解包原 tuple
        new_pos  = mn.Vector3(pos.x, pos.y- height_bias, pos.z)
        human_path[i]  = (new_pos,quat, yaw)   


    observations=[]
    humanoid.base_pos = human_path[0][0]
    humanoid.base_rot = human_path[0][2]
    count = 0
    human_pos = []
    follow_pos = humanoid.base_pos
    follow_state = sim.agents[0].get_state()
    human_state = sim.agents[0].get_state()
    follow_state.position = humanoid.base_pos
    # follow_state.position.y = 0.2
    follow_state.rotation = to_quat(human_path[0][1])
    follow_yaw = human_path[0][2]
    sim.agents[0].set_state(follow_state)
    planner = HybridAStar(
        xy_resolution=0.1,
        yaw_resolution=math.radians(5),
        sim=sim,
        height = follow_state.position.y,
    )

    follow_index = 0
    move_dis = 0
    sample_list = [random.uniform(2, 2.5), random.uniform(3, 3.5)]
    record_range =  random.uniform(3, 5)
    for k in range(1, len(human_path)):
        goal_pos, goal_quat,goal_yaw = human_path[k]

        # 位移向量
        start_pos = humanoid.base_pos
        seg_vec   = goal_pos - start_pos
        seg_len   = seg_vec.length()
        move_dis += seg_len
        if seg_len < 1e-4:
            continue

        # 朝向增量
        start_yaw = humanoid.base_rot            # 当前 yaw (float)
        yaw_diff  = shortest_angle_diff(start_yaw, goal_yaw)

        # 行走分段
        direction = seg_vec.normalized()
        step_dist = forward_speed / fps
        n_steps   = int(np.ceil(seg_len / step_dist))
        
        for step in range(n_steps):
            # --- 1) 平移 ---
            humanoid.base_pos += direction * step_dist

            # --- 2) 线性插值 yaw ---
            frac = (step + 1) / n_steps        # 0→1
            humanoid.base_rot = start_yaw + yaw_diff * frac
            move_pos = humanoid.base_pos
            move_yaw = humanoid.base_rot
            move_quat = quat_from_angle_axis(humanoid.base_rot, np.array([0, 1, 0]))
            human_pos.append([move_pos,move_quat])
            human_state.position = move_pos
            human_state.rotation = move_quat
            # --- 3) 步行动画，一帧 pose ---
            controller.calculate_walk_pose(seg_vec)   # 方向矢量给即可
            new_pose = controller.get_pose()

            new_joints = new_pose[:-16]
            new_pos_transform_base = new_pose[-16:]
            new_pos_transform_offset = new_pose[-32:-16]

            if np.array(new_pos_transform_offset).sum() != 0:
                vecs_base = [
                    mn.Vector4(new_pos_transform_base[i * 4 : (i + 1) * 4])
                    for i in range(4)
                ]
                vecs_offset = [
                    mn.Vector4(new_pos_transform_offset[i * 4 : (i + 1) * 4])
                    for i in range(4)
                ]
                new_transform_offset = mn.Matrix4(*vecs_offset)
                new_transform_base = mn.Matrix4(*vecs_base)
                humanoid.set_joint_transform(
                    new_joints, new_transform_offset, new_transform_base
                )
                humanoid.base_pos = move_pos


            sim.step_physics(1.0 / fps)
        count+=1

        follow_rot, _ = quat_to_angle_axis(follow_state.rotation)
        # path = planner.plan((follow_state.position.x,follow_state.position.y, follow_rot), (humanoid.base_pos.x,humanoid.base_pos.y,humanoid.base_rot))
        if len(sample_list)==0:
            sample_list = [random.uniform(2, 2.5), random.uniform(3, 3.5)]
        shortest_path = habitat_sim.ShortestPath()


        if move_dis > record_range:
            shortest_path.requested_start = follow_state.position
            shortest_path.requested_end = goal_pos
            if sim.pathfinder.find_path(shortest_path):
                move_dis = 0
                record_range = random.uniform(3, 5)
                sample_list=[]

                new_path = generate_path(shortest_path.points,sim.pathfinder, filt_distance = keep_distance, visualize=False)
                for i in range(len(new_path)):
                    follow_state.position = new_path[i][0]
                    follow_state.rotation = to_quat(new_path[i][1])
                    follow_yaw  = new_path[i][2]
                    sim.agents[0].set_state(follow_state)
                    obs = sim.get_sensor_observations(0)          
                    observations.append(obs.copy())
                    follow_data = {"obs_idx": len(observations)-1, "follow_state": new_path[i], "human_state": human_path[k],
                                   "path": new_path[i:], "type": 0
                                    }
                    output["follow_paths"].append(follow_data)  
            else:
                obs = sim.get_sensor_observations(0)          
                observations.append(obs.copy())                  
        else:
            obs = sim.get_sensor_observations(0)         
            observations.append(obs.copy())  
            shortest_path.requested_start = follow_state.position
            shortest_path.requested_end = goal_pos
            if len(sample_list)>0 and move_dis>sample_list[0]:
                if sim.pathfinder.find_path(shortest_path):
                    del sample_list[0]
                    new_path = generate_path(shortest_path.points,sim.pathfinder, filt_distance = keep_distance, visualize=False)
                    follow_data = {"obs_idx": len(observations)-1, "follow_state":(follow_state.position, follow_state.rotation, follow_yaw), "human_state": human_path[k],
                    "path": new_path, "type": 1
                    }
                    output["follow_paths"].append(follow_data)   

    
    output["obs"] = observations
    if all_index <20:
        vut.make_video(
            observations,
            "color_0_0",
            "color",
            f"results/humanoid_wrapper_{all_index}",
            open_vid=False,
        )
    print("walk done")
    return output
# Example usage
if __name__ == "__main__":
    episodes_count = 0
    
    folder = "/home/wangzejin/habitat/goat_bench/data/datasets/goat_bench/hm3d/v1/train/content"
    yaml_file_path = "/home/wangzejin/habitat/ON-MLLM/human_follower/cfg/exp.yaml"
    
    # 初始化目标文件列表
    target_files = []   

    # 遍历文件夹并将相对路径添加到目标文件列表
    for root, dirs, files in os.walk(folder):
        for file in files:
            # 计算相对路径并加入列表
            relative_path = os.path.relpath(os.path.join(root, file), folder)
            target_files.append(relative_path)
    
    cfg = read_yaml(yaml_file_path)
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    cfg.scenes_data_path = "/home/wangzejin/habitat/goat_bench/data/scene_datasets/hm3d/train"
    
    data = extract_dict_from_folder(folder, target_files)
    
    all_index = 0
    for file_name, content in data.items():
        if all_index > 10000:
            break
        
        if episodes_count > 50000:
            break
        print(f"Processing {file_name}:")
        structured_data,  filtered_episodes = process_episodes_and_goals(content)
        episodes = convert_to_scene_objects(structured_data, filtered_episodes)
        
        # unique_episodes = {}
        # for ep in episodes:
        #     if ep["object_environment"] not in unique_episodes:
        #         unique_episodes[ep["object_environment"]] = ep

        # # 随机选择 5 个（如果少于 5 个，取全部）
        # random.seed(42)
        # episodes = random.sample(list(unique_episodes.values()), min(5, len(unique_episodes)))

        
        scene = cfg.current_scene = get_current_scene(structured_data)
        
        
        # Set up scene in Habitat
        try:
            simulator.close()
        except:
            pass
        scene_mesh_dir = find_scene_path(cfg, cfg.current_scene)
        sim_settings = {
            "scene": scene_mesh_dir,
            "default_agent": [0],
            "sensor_height": 1.5,
            "width": 640,
            "height": 480,
            "hfov": 120,
        }
        sim_cfg = make_simple_cfg(sim_settings)
        simulator = habitat_sim.Simulator(sim_cfg)
        
        # 从 sim_cfg 中获取 agent 配置
        agent_cfg = sim_cfg.agents[0]  # 获取默认代理的配置
        # 获取 NavMeshSettings 对象
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.agent_radius = agent_cfg.radius             # 设置 agent 碰撞半径
        navmesh_settings.agent_height = agent_cfg.height             # 设置 agent 碰撞高度
        navmesh_settings.agent_max_climb = 1          # 设置最大爬升高度
        navmesh_settings.agent_max_slope = 45.0         # 设置最大坡度角度（单位：度）

        # 重新生成导航网格
        navmesh_success = simulator.recompute_navmesh(simulator.pathfinder, navmesh_settings)

        # 验证导航网格是否成功生成
        if not navmesh_success or not simulator.pathfinder.is_loaded:
            raise RuntimeError("Navmesh recomputation failed. Cannot proceed with pathfinding.")

        semantic_scene = simulator.semantic_scene
        pathfinder = simulator.pathfinder
        pathfinder.seed(cfg.seed)
        if not simulator.pathfinder.is_loaded:
            print("Failed to load or generate navmesh.")
            continue
            raise RuntimeError("Failed to load or generate navmesh.")   

        # direction_projector = DirectionProjector(cfg.hfov, 'depth_sensor', 7)
        # human_agent = simulator.initialize_agent(sim_settings["default_agent"][0])
        # follow_agent = simulator.initialize_agent(sim_settings["default_agent"][1])
        # human_state = habitat_sim.AgentState()
        # follow_state = habitat_sim.AgentState()
        # obj_mgr = simulator.get_object_template_manager()
        # humanoid_template_ids = obj_mgr.load_configs("human_follower/human_model/as")
        # # handle = obj_mgr.get_template_handles("human_follower/human_model")
        # # humanoid_template_id = obj_mgr.load_template(handle)
        # agent_node = human_agent.scene_node

        # # 2. 添加刚体对象，并设置 parent 为 agent_node
        # rigid_mgr = simulator.get_rigid_object_manager()
        # human = rigid_mgr.add_object_by_template_id(humanoid_template_ids[0], attachment_node=agent_node)
        # human.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        humanoid, controller = load_humanoid(simulator)

        # # 起点对齐
        # humanoid.base_pos = custom_path[0][0]
        # humanoid.base_rot = custom_path[0][1]





        print("begin")
        for episode_id, obj_data in enumerate(tqdm(episodes)):
            #print("obj_data:", obj_data)
            episode_data = obj_data
            
            # 设置起始位置和旋转
            start_position = obj_data.start_position
            start_rotation = obj_data.start_rotation
            distance = obj_data.info['euclidean_distance']
            goal_position = obj_data.goal["position"]

            start_normal = pos_habitat_to_normal(start_position)
            start_floor_height = start_normal[-1]



            if goal_position is None or start_position is None:
                continue
            # 使用 ShortestPath 对象生成避免穿墙的最短路径
            shortest_path = habitat_sim.ShortestPath()
            shortest_path.requested_start = start_position
            shortest_path.requested_end = goal_position

            # 查找最短路径 # 模拟前沿探索过程
            if pathfinder.find_path(shortest_path):
                
                # 检查起始点和目标点是否在同一楼层 (高度差小于1m)
                start_floor_height = start_position[1]  # y 值代表高度
                goal_floor_height = goal_position[1]
                if abs(start_floor_height - goal_floor_height) > 1:
                    print(f"Skipping episode due to height difference: {start_floor_height} vs {goal_floor_height}")
                    continue
                path = shortest_path.points
                all_distance = 0
                for i in range(len(path)-1):
                    distance = calculate_euclidean_distance(path[i], path[i+1])
                    all_distance+=distance
                
                if all_distance < 5:
                    print(f"Skipping episode due to short distance: {all_distance}m")
                    continue
                

                # 检查路径是否跨楼层 (高度差小于1m)
                floor_heights = [point[1] for point in path]  # 获取所有路径点的高度
                if max(floor_heights) - min(floor_heights) > 1:
                    print("Skipping episode due to multi-floor path")
                    continue
                
                # 初始化探索类
                if not cfg.shortest_path:

                    num_frontiers = random.randint(1, 4)
                    #print("original path:",path)
                    explorer = FrontierExploration(simulator)
                    explorer.explore_until_target(
                                start_position = start_position,
                                target_position = goal_position,
                                num_frontiers = num_frontiers)
                    path = explorer.trail
                #print("frontier_exploration path:", path)
                

                #轨迹优化流水线
                new_path = generate_path(path, pathfinder, visualize=False)
                
                #print(path[0])
                #print(path)
            else:
                print("No valid path found:", obj_data["object_category"])
                continue 
            
            # new_path = convert_path(path)

            try:
                output_data = walk_along_path(all_index, simulator, humanoid, controller, new_path,fps=10)
                save_output_to_h5(output_data, f"results/new_hdf5/episode_{all_index}.hdf5")
                episodes_count += len(output_data["follow_paths"])
                all_index+=1
                print(f"Already has {episodes_count} cases")
            except Exception as e:   
                print(f"Error !!!!!!!: {e}")
                continue


            print("done")
