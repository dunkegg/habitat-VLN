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
from habitat_for_sim.utils.load_scene import load_simulator, generate_path_from_scene
from habitat_sim.utils import viz_utils as vut


from human_follower.walk_behavior import walk_along_path_multi, generate_interfere_path_from_target_path, get_path_with_time,generate_interfer_path, generate_interfere_sample_from_target_path
from human_follower.human_agent import AgentHumanoid, get_humanoid_id


# Example usage
if __name__ == "__main__":

    yaml_file_path = "habitat_for_sim/cfg/exp.yaml"
    cfg = read_yaml(yaml_file_path)
    json_data = cfg.json_file_path
    
    # 初始化目标文件列表
    target_files = []   

    # 遍历文件夹并将相对路径添加到目标文件列表
    for root, dirs, files in os.walk(json_data):
        for file in files:
            # 计算相对路径并加入列表
            relative_path = os.path.relpath(os.path.join(root, file), json_data)
            target_files.append(relative_path)
    
    
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    
    # cfg.scenes_data_path = "/home/wangzejin/habitat/goat_bench/data/scene_datasets/hm3d/train"
    
    data = extract_dict_from_folder(json_data, target_files)
    
    max_episodes = cfg.max_episodes

    all_index = 0
    episodes_count = 0
    for file_name, content in data.items():
        if episodes_count > max_episodes:
            break
        

        print(f"Processing {file_name}:")
        structured_data,  filtered_episodes = process_episodes_and_goals(content)
        episodes = convert_to_scene_objects(structured_data, filtered_episodes)
                
        cfg.current_scene = get_current_scene(structured_data)
        
        
        # Set up scene in Habitat
        try:
            simulator.close()
        except:
            pass

        simulator = load_simulator(cfg)

        semantic_scene = simulator.semantic_scene
        pathfinder = simulator.pathfinder
        pathfinder.seed(cfg.seed)
        if not simulator.pathfinder.is_loaded:
            print("Failed to load or generate navmesh.")
            continue
            raise RuntimeError("Failed to load or generate navmesh.")   


        with open("character_descriptions.json", "r") as f:
            id_dict = json.load(f)

        # ###label
        # folders = [f"female_{i}" for i in range(35)] + [f"male_{i}" for i in range(65)]
        # humanoid_name = folders[all_index]

        humanoid_name = get_humanoid_id(id_dict, name_exception=None) 
        
        # 原主目标人
        description = id_dict[humanoid_name]["description"]
        target_humanoid = AgentHumanoid(simulator,base_pos=mn.Vector3(0, 0.083, 0), base_yaw = 0,name = humanoid_name,description = description, is_target=True)
        
        all_interfering_humanoids = []
        for idx in range(3):
            # break
            # max_humanoids[idx].reset(name = get_humanoid_id(humanoid_name))
            interferer_name = get_humanoid_id(id_dict, name_exception = humanoid_name)
            interferer_description = id_dict[humanoid_name]["description"]
            interferer = AgentHumanoid(simulator, base_pos=mn.Vector3(0, 0.083, 0), base_yaw = 0, name = interferer_name, description = interferer_description, is_target=False)
            all_interfering_humanoids.append(interferer)


        print("begin")
        for episode_id, episode_data in enumerate(tqdm(episodes)):

            if episodes_count > max_episodes:
                break

            human_fps = 5
            human_speed = 0.7
            followed_path = generate_path_from_scene(episode_data, pathfinder, human_fps, human_speed)
            if followed_path is None:
                continue
            
            #
            k = random.randint(1, 3) 
            interfering_humanoids = random.sample(all_interfering_humanoids, k)
            ##
            for interfering_humanoid in interfering_humanoids:
                sample_path = generate_interfere_sample_from_target_path(followed_path,pathfinder, 1)
                list_pos = [[point.x,point.y,point.z] for point in sample_path]
                interfering_path = generate_path(list_pos, pathfinder, visualize=False)
                interfering_path = get_path_with_time(interfering_path, time_step=1/human_fps, speed=0.9)
                interfering_humanoid.reset_path(interfering_path)
                

            output_data = walk_along_path_multi(
                all_index=all_index,
                sim=simulator,
                humanoid_agent=target_humanoid,
                human_path=followed_path,
                fps=10,
                interfering_humanoids=interfering_humanoids
            )
            save_output_to_h5(output_data, f"results/follow_hdf5/episode_{all_index}.hdf5")
            episodes_count += len(output_data["follow_paths"])
            print(f"Case {all_index}, {humanoid_name} Done, Already has {episodes_count} cases")
            all_index+=1
            
            
