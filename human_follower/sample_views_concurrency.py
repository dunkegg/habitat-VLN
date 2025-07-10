"""
Run EQA in Habitat-Sim with VLM exploration.

"""
from baseline.VLMnav.src.vlm import *
import os
import copy
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import numpy as np
import torch
np.set_printoptions(precision=3)
import csv
import ast
import pickle
import logging
import math
import random
import quaternion
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import habitat_sim
# from retrieve.llama3_retrieve import Retriever
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis

import habitat
from habitat.config.default import get_config
from habitat.config import read_write

from utils.explore.explore_habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
from utils.explore.geom import get_cam_intr, get_scene_bnds
from utils.process_json import getScenefromOvon
from utils.project_direction import DirectionProjector
from utils.label import generate_video_from_steps, calculate_action, get_topdown_map, rotation_to_direction
#calculate_angle
from human_follower.utils.path_generator import generate_path,direction_to_quaternion

from scipy.spatial.transform import Rotation as R, Slerp

import json
import warnings
import asyncio



async def main(cfg):
    
    warnings.filterwarnings("ignore", category=UserWarning)

    # vlm_cls = globals()['GeminiVLM']
    # label_VLM: VLM = vlm_cls('gemini-1.5-flash-001', system_instruction='Get information of target object in the image')
    
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)
    direction_projector = DirectionProjector(cfg.hfov, 'depth_sensor', 7)

    # Load VLM
    os.environ["LOCAL_RANK"] = "0"    
    # vlm = VLM(cfg.vlm)

    agent_num = 1
    
    scene = cfg.current_scene
    objects_dataset = getScenefromOvon(scene, cfg)
    episodes = objects_dataset.episodes 

    # Set up scene in Habitat
    try:
        simulator.close()
    except:
        pass
    scene_mesh_dir = os.path.join(
        cfg.scenes_data_path, scene, scene[6:] + ".basis" + ".glb"
    )
    #navmesh_file = os.path.join(
    #    cfg.scenes_data_path, scene, scene[6:] + ".basis" + ".navmesh"
    #)
    sim_settings = {
        "scene": scene_mesh_dir,
        "default_agent": [0],
        "sensor_height": cfg.camera_height,
        "width": img_width,
        "height": img_height,
        "hfov": cfg.hfov,
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
        raise RuntimeError("Failed to load or generate navmesh.")   
    
    # assert len(objects) >=  cfg.scenario_end_index
    print("begin")
    for index, obj_data in enumerate(tqdm(episodes)):

        # 设置起始位置和旋转
        start_position = obj_data.start_position
        start_rotation = obj_data.start_rotation
        distance = obj_data.info['euclidean_distance']
        if distance < 5:
            continue
        #print(f'distance:{distance}')
        # 从 dataset 中选择目标
        scene_id = os.path.basename(obj_data.scene_id)
        object_name = obj_data.object_category
        
        key = f"{scene_id}_{object_name}"
        category_goals = objects_dataset.goals_by_category[key]
        #print("category_goals:", category_goals)
        # 随机选择一个目标
        # selected_goal_data = random.choice(category_goals) #todo 充分利用数据
        selected_goal_data = category_goals[0]
        
        goal_position = selected_goal_data.position
        
        start_normal = pos_habitat_to_normal(start_position)
        start_floor_height = start_normal[-1]
        goal_normal = pos_habitat_to_normal(goal_position)
        goal_floor_height = goal_normal[-1]       
        if abs(goal_floor_height-start_floor_height)>0.5:
            continue
        
        #TODO:未来可利用viewpoint作为终止判断指标而不是goal本身坐标，使得agent停止时目标不会因为过近而无法完整呈现在视场内
        #但是直接使用viewpoint作为终点rotation不正确，故暂不使用
        #viewpoints = selected_goal_data.view_points
        
        # 创建存储文件夹
        scene_dir = os.path.join(cfg.output_dir, scene)
        os.makedirs(scene_dir, exist_ok=True)
        episode_dir = os.path.join(scene_dir, f"episode_{index}")
        os.makedirs(episode_dir, exist_ok=True)

        #print("start_position:", start_position)
        # 存储路径点及图像
        episode_data = {
            #"object_env": obejct_env,
            "start": {"position": start_position.tolist() if isinstance(start_position, np.ndarray) else start_position},
            "goal": {"position": goal_position.tolist() if isinstance(goal_position, np.ndarray) else goal_position},
            "steps": [],
        }
        
        
        """ episode_data_dir = os.path.join(cfg.output_dir, str(index))
        os.makedirs(episode_data_dir, exist_ok=True)        
        # scene = sdata['scene'] """

        sim_agent = simulator.initialize_agent(sim_settings["default_agent"][0])
        agent_state = habitat_sim.AgentState()

        room_names = []
        for region in semantic_scene.regions:
            room_names.append(region.category.name())
        

        # 使用 ShortestPath 对象生成避免穿墙的最短路径
        shortest_path = habitat_sim.ShortestPath()
        shortest_path.requested_start = start_position
        shortest_path.requested_end = goal_position
        
        # 查找最短路径
        if pathfinder.find_path(shortest_path):
            path = shortest_path.points
            #print("path",path)
            path = generate_path(path, pathfinder)
            # print(path[0])
            # print(path)
        else:
            print("No valid path found:", key)
            continue 
        
        #获得每个观察点
        last_point = None
        
        min_move = 0.2
        min_turn = 5
        
        
        for cnt_step, point in enumerate(tqdm(path)):
            #logging.info(f"\n== point: {cnt_step}")

            if cnt_step + 1 >= len(path):
                break
            # agent_state.position = np.array(point[0])
            # agent_state.rotation = np.array(point[1])
            
            agent_state.position = point[0]
            agent_state.rotation = point[1]
            
            sim_agent.set_state(agent_state)
            # agent = self.sim.get_agent(0)
            curr_state = sim_agent.get_state()

            # 获取当前的观测数据
            agent_id = 0
            obs = simulator.get_sensor_observations(agent_id)


        

        # 将生成的数据保存为 JSON 文件
        json_path = os.path.join(episode_dir, "episode_data_for_llamavid_high_level_reasoning.json")
        with open(json_path, "w") as json_file:
            json.dump(episode_data_for_llamavid, json_file, indent=4)
        logging.info(f"{json_path} finishied")
        

    simulator.close()
    logging.info(f"scene: {scene} All episodes processed.")

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    import os
    import logging
    
    # 获取配置路径和参数
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="cfg/exp.yaml", type=str)
    parser.add_argument("-start", "--start_index", default=0, type=int)
    parser.add_argument("-end", "--end_index", default=50, type=int)
    parser.add_argument("-rs", "--random_seed", default=1997, type=int)
    parser.add_argument("-ss", "--step_scale", default=2, type=float)
    parser.add_argument("-sdp", "--scene_list_file_path", required=True, type=str)  # 需要采样场景的列表 txt 文件
    parser.add_argument("-cs", "--current_scene", help="Specify the current scene ID", type=str, required=False)
    args = parser.parse_args()

    # 解析配置文件
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    cfg.scenario_start_index = args.start_index
    cfg.scenario_end_index = args.end_index
    cfg.random_seed = args.random_seed
    cfg.step_scale = args.step_scale

    # 创建输出目录
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # 递归创建
    logging_path = os.path.join(cfg.output_dir, "log.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    cfg.pkl_output_dir = os.path.join(cfg.output_parent_dir, 'pkl')
    if not os.path.exists(cfg.pkl_output_dir):
        os.makedirs(cfg.pkl_output_dir, exist_ok=True)  # 递归创建

    # 从指定的 txt 文件中读取场景列表
    scene_list_file = args.scene_list_file_path
    with open(scene_list_file, "r") as file:
        scene_list = [line.strip() for line in file if line.strip()]  # 读取非空行

    # 逐个处理场景
    import traceback  # 用于获取详细的错误信息

    logging.info(f"***** Running {cfg.exp_name} *****")
    for scene in scene_list:
        logging.info(f"***** Now Loading {scene} for Sampling *****")
        cfg.current_scene = scene
        #try:
        asyncio.run(main(cfg))
        #except Exception as e:  # 捕获所有普通异常
        #    logging.error(f"An error occurred while processing {scene}: {e}")
        #    print(f"Full traceback:\n{traceback.format_exc()}")
        #    logging.debug(f"Full traceback:\n{traceback.format_exc()}")  # 记录详细堆栈信息
        #except BaseException as e:  # 捕获所有其他可能导致程序退出的异常
        #    logging.critical(f"A critical error occurred while processing {scene}: {e}")
        #    logging.debug(f"Full traceback:\n{traceback.format_exc()}")  # 记录详细堆栈信息
        #finally:
        #    if torch.cuda.is_available():
        #        torch.cuda.empty_cache()  # 清理显存缓存
        #    logging.info(f"Finished processing {scene}.")



