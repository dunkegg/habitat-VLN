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


# 将上级目录加入 Python 搜索路径
sys.path.append(os.path.abspath(".."))
from habitat_for_sim.utils.explore.explore_habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
# from utils.project_direction import DirectionProjector
from human_follower.save_data import save_output_to_h5, to_quat
from habitat_for_sim.utils.goat import read_yaml, extract_dict_from_folder, get_current_scene, process_episodes_and_goals, convert_to_scene_objects, find_scene_path, calculate_euclidean_distance
from habitat_for_sim.agent.path_generator import generate_path
from habitat_for_sim.utils.frontier_exploration import FrontierExploration
from habitat_sim.utils import viz_utils as vut


from human_follower.walk_behavior import walk_along_path_multi, generate_interfere_path_from_target_path, get_path_with_time,generate_interfer_path, generate_interfere_sample_from_target_path
from human_follower.human_agent import AgentHumanoid, get_humanoid_id


# Example usage
if __name__ == "__main__":
    episodes_count = 0
    
    folder = "/home/wangzejin/habitat/goat_bench/data/datasets/goat_bench/hm3d/v1/train/content"
    yaml_file_path = "habitat_for_sim/cfg/exp.yaml"
    
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
        if all_index > 100:
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


        # interfering_humanoids, interfering_controllers = load_interfering_humanoids(simulator, num=2)

        # names = ["female_0", "female_1", "female_2", "female_3", "male_0", "male_1", "male_2", "male_3"]
        # humanoid_name = random.choice(names)

        humanoid_name = get_humanoid_id()
        ###label
        folders = [f"female_{i}" for i in range(35)] + [f"male_{i}" for i in range(65)]

        humanoid_name = folders[all_index]
        # 原主目标人
        target_humanoid = AgentHumanoid(
            simulator,
            base_pos=mn.Vector3(0, 0.083, 0),
            name = humanoid_name,
            # motion_path="human_follower/habitat_humanoids/female_0/female_0_motion_data_smplx.pkl",
            is_target=True
        )

        # # 干扰人可定义多个实例
        # interferer_1 = AgentHumanoid(simulator, base_pos=mn.Vector3(0, 0.083, 0), name = get_humanoid_id(humanoid_name))
        # interferer_2 = AgentHumanoid(simulator, base_pos=mn.Vector3(0, 0.083, 0), name = get_humanoid_id(humanoid_name))
        # interferer_3 = AgentHumanoid(simulator, base_pos=mn.Vector3(0, 0.083, 0), name = get_humanoid_id(humanoid_name))
        # max_humanoids=[interferer_1, interferer_2,interferer_3]


        interfering_humanoids = []
        for idx in range(random.randint(1, 3)):
            break
            # max_humanoids[idx].reset(name = get_humanoid_id(humanoid_name))
            interferer = AgentHumanoid(simulator, base_pos=mn.Vector3(0, 0.083, 0), name = get_humanoid_id(humanoid_name))
            interfering_humanoids.append(interferer)

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
                
                # # 初始化探索类
                # if not cfg.shortest_path:

                #     num_frontiers = random.randint(1, 4)
                #     #print("original path:",path)
                #     explorer = FrontierExploration(simulator)
                #     explorer.explore_until_target(
                #                 start_position = start_position,
                #                 target_position = goal_position,
                #                 num_frontiers = num_frontiers)
                #     path = explorer.trail
                #print("frontier_exploration path:", path)
                

                #轨迹优化流水线
                new_path = generate_path(path, pathfinder, visualize=False)
                
                #print(path[0])
                #print(path)
            else:
                print("No valid path found:", obj_data["object_category"])
                continue 
            

         
            # 目标人移动
            human_fps = 10
            human_speed = 0.7
            dense_path = get_path_with_time(new_path, time_step=1/human_fps, speed=human_speed)
            
            # 修正高度偏移
            height_bias = 0
            for i in range(len(dense_path)):
                pos, quat, yaw = dense_path[i]
                new_pos = mn.Vector3(pos.x, pos.y - height_bias, pos.z)
                dense_path[i] = (new_pos,quat,yaw)

            # 插入高度跳变检测（连续跳变次数超过阈值才判为异常）
            height_jump_threshold = 0.04
            max_allowed_jumps = 5
            jump_count = 0
            normal_height = 0.083759
            for i in range(1, len(dense_path)):
                curr_height = dense_path[i][0].y

                if curr_height - normal_height > height_jump_threshold:
                    jump_count += 1
                    if jump_count >= max_allowed_jumps:
                        break

            if jump_count >= max_allowed_jumps:
                print(f"Skipping episode due to {jump_count} large height jumps.")
                continue

            
            #
            for interfering_humanoid in interfering_humanoids:
                sample_path = generate_interfere_sample_from_target_path(dense_path,pathfinder, 1)
                list_pos = [[point.x,point.y,point.z] for point in sample_path]
                interfering_path = generate_path(list_pos, pathfinder, visualize=False)
                interfering_path = get_path_with_time(interfering_path, time_step=1/human_fps, speed=0.9)
                # todo
                interfering_humanoid.reset_path(interfering_path)
                
            
            # # 干扰人移动
            # generate_interfer_path(
            #     interfering_humanoids=interfering_humanoids,
            #     human_path=dense_path,
            #     # time_step=1/human_fps,
            #     time_step=1/human_fps,
            #     speed=0.9,  # 比主行人快
            #     radius=0.5  # 可调扰动幅度
            # )

            output_data = walk_along_path_multi(
                all_index=all_index,
                sim=simulator,
                humanoid_agent=target_humanoid,
                human_path=dense_path,
                fps=10,
                interfering_humanoids=interfering_humanoids
            )
            all_index+=1
            print(f"Case {all_index}, {humanoid_name} Done")
            
