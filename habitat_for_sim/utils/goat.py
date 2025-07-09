import gzip
import json
import yaml
import math
import os
import numpy as np
def quaternion_inverse(q):
    """
    计算单位四元数 q 的逆 (w, x, y, z) -> (w, -x, -y, -z).
    假设 q 已归一化。
    """
    w, x, y, z = q
    return (w, -x, -y, -z)

def quaternion_mul(q1, q2):
    """
    计算两个单位四元数 q1 * q2.
    q1, q2 都以 (w, x, y, z) 表示。
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return (w, x, y, z)

def quaternion_rotate_vector(q, v):
    """
    使用单位四元数 q (w, x, y, z) 旋转 3D 向量 v (x, y, z).
    返回一个新的 3D 向量 (x', y', z').
    
    公式: v' = q * v * q^-1
    这里把 v 看成 (0, vx, vy, vz) 的纯四元数。
    """
    w, x, y, z = q
    vx, vy, vz = v
    # 将向量 v 视为四元数 (0, vx, vy, vz)
    v_quat = (0, vx, vy, vz)
    
    # q * v
    tmp = quaternion_mul(q, v_quat)
    # (q * v) * q^-1
    q_inv = quaternion_inverse(q)
    res = quaternion_mul(tmp, q_inv)
    
    # 返回 res 的向量部分
    return (res[1], res[2], res[3])

class Config:
    """
    允许通过属性访问字典内容的配置类。
    """
    def __init__(self, config_dict):
        self._config = config_dict

    def __getattr__(self, name):
        value = self._config.get(name)
        if isinstance(value, dict):
            return Config(value)
        return value

    def __setattr__(self, name, value):
        if name == "_config":
            super().__setattr__(name, value)
        else:
            self._config[name] = value

    def to_dict(self):
        """将 Config 对象转换回字典。"""
        return self._config
    
class DotAccessDict:
    """
    一个允许通过点操作和索引访问的字典类，并支持 JSON 序列化。
    """
    def __init__(self, data):
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary.")
        self._data = {k: self._wrap(v) for k, v in data.items()}

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"Attribute '{name}' not found.")

    def __setattr__(self, name, value):
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = self._wrap(value)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = self._wrap(value)

    def __contains__(self, key):
        return key in self._data

    def to_dict(self):
        return {k: self._unwrap(v) for k, v in self._data.items()}

    def _wrap(self, value):
        if isinstance(value, dict):
            return DotAccessDict(value)
        elif isinstance(value, list):
            return [self._wrap(v) for v in value]
        elif isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _unwrap(self, value):
        if isinstance(value, DotAccessDict):
            return value.to_dict()
        elif isinstance(value, list):
            return [self._unwrap(v) for v in value]
        return value

    def __repr__(self):
        return repr(self._data)

    def __iter__(self):
        return iter(self._data)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def __len__(self):
        return len(self._data)

    def __delitem__(self, key):
        del self._data[key]

    def __eq__(self, other):
        if isinstance(other, DotAccessDict):
            return self._data == other._data
        elif isinstance(other, dict):
            return self.to_dict() == other
        return False

    def __json__(self):
        """使对象支持 JSON 序列化。"""
        return self.to_dict()

    def __reduce__(self):
        """支持对象序列化和反序列化。"""
        return (self.__class__, (self.to_dict(),))

def read_yaml(file_path):
    """
    读取 YAML 文件内容并返回可通过属性访问的配置对象。

    :param file_path: str, YAML 文件路径
    :return: Config, 配置对象
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return Config(data)
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        return None

def load_json_gz(file_path):
    """
    Load a .json.gz file and return its content as a Python dictionary.

    Parameters:
        file_path (str): Path to the .json.gz file.

    Returns:
        dict: Content of the JSON file.
    """
    with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
        return json.load(gz_file)
    
def find_scene_path(cfg, scene_id):
    """
    根据后缀场景 ID 动态查找完整的场景路径。

    :param cfg: Config 对象，包含场景数据路径。
    :param scene_id: str, 后缀场景 ID，例如 "1S7LAXRdDqK"。
    :return: str, 完整的场景路径(.basis.glb)，如果未找到则返回 None。
    """
    scene_dir = cfg.scenes_data_path
    all_scenes = os.listdir(scene_dir)

    # 遍历所有目录名，匹配以场景 ID 结尾的目录
    for full_scene_name in all_scenes:
        if full_scene_name.endswith(f"-{scene_id}"):
            # 构建完整的场景路径
            scene_path = os.path.join(scene_dir, full_scene_name, scene_id + ".basis.glb")
            return scene_path

    print(f"Scene ID '{scene_id}' not found in {scene_dir}")
    return None   

def extract_dict_from_folder(folder_path, target_files):
    """
    Extract and load specified .json.gz files from a folder into a dictionary.

    Parameters:
        folder_path (str): Path to the folder containing .json.gz files.
        target_files (list): List of target .json.gz filenames to extract.

    Returns:
        dict: A dictionary where keys are filenames and values are file contents as dicts.
    """
    extracted_data = {}

    for file_name in target_files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.json.gz'):
            try:
                extracted_data[file_name] = load_json_gz(file_path)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
        else:
            print(f"File not found or invalid format: {file_name}")

    return extracted_data

def print_nested_structure(data, indent=0):
    """
    Recursively print the structure of a nested dictionary.

    Parameters:
        data (dict): The dictionary to print.
        indent (int): Current indentation level.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            print("  " * indent + f"{key}:")
            if isinstance(value, (dict, list)):
                print_nested_structure(value, indent + 1)
            else:
                print("  " * (indent + 1) + f"{type(value).__name__}")
    elif isinstance(data, list):
        print("  " * indent + "[list]")
        if data:
            print_nested_structure(data[0], indent + 1)  # Show structure of the first element
    else:
        print("  " * indent + f"{type(data).__name__}")

def process_episodes_and_goals(data):
    """
    Process the 'goals' and 'episodes' keys in the extracted data to create a structured dictionary and filter episodes.

    Parameters:
        data (dict): The dictionary containing the loaded JSON data.

    Returns:
        tuple: A tuple containing:
            - A structured dictionary with object categories, viewpoints, descriptions, and locations.
            - A filtered list of episodes.
    """
    # Process goals to create structured data
    structured_data = {}

    goals = data.get("goals", {})
    for key, instances in goals.items():
        category = key.split(".")[0] + "_" + key.split("_")[-1]  # Extract the category from the key
        if category not in structured_data:
            structured_data[category] = []

        for instance in instances:
            lang_desc = instance.get("lang_desc")
            if lang_desc is not None:  # Only include instances with a non-None lang_desc
                obj = {
                    "object_category": instance.get("object_category"),
                    "view_points": instance.get("view_points", []),
                    "lang_desc": lang_desc,
                    "goal_location": instance.get("position"),
                    "object_id": instance.get("object_id")  # Add object_id
                }
                structured_data[category].append(obj)

    # Remove categories with no objects
    structured_data = {k: v for k, v in structured_data.items() if v}

    # Process episodes and filter based on the structured data
    filtered_episodes = []
    episodes = data.get("episodes", [])

    for episode in episodes:
        tasks = episode.get("tasks", [])
        valid_tasks = []

        for task in tasks:
            if task[1] == "description":  # Check for description tasks
                object_id = task[2]
                # Verify if the object_id exists in structured_data
                found = any(
                    obj.get("object_id") == object_id
                    for category_objects in structured_data.values()
                    for obj in category_objects
                )
                if found:
                    valid_tasks.append(task)

        # If the episode has valid description tasks, include it in the filtered episodes
        if valid_tasks:
            filtered_episode = {
                "episode_id": episode.get("episode_id"),
                "start_position": episode.get("start_position"),
                "start_rotation": episode.get("start_rotation"),
                "scene_id": episode.get("scene_id"),
                "scene_dataset_config": episode.get("scene_dataset_config"),
                "valid_tasks": valid_tasks  # Include only valid description tasks
            }
            filtered_episodes.append(filtered_episode)

    return structured_data, filtered_episodes

def print_structured_data(structured_data):
    """
    Print the structured data for better readability.

    Parameters:
        structured_data (dict): The structured data to print.
    """
    for category, instances in structured_data.items():
        print(f"Category: {category}")
        for instance in instances:
            print(f"  Object Category: {instance['object_category']}")
            print(f"  Lang Desc: {instance['lang_desc']}")
            print(f"  Goal Location: {instance['goal_location']}")
            print(f"  Number of Viewpoints: {len(instance['view_points'])}")
            print(f"  Example of Viewpoints: {instance['view_points'][0]}")
            print(f"  Object ID: {instance['object_id']}")

def get_current_scene(structured_data):
    """
    Extract the current scene identifier from the structured data.

    Parameters:
        structured_data (dict): The structured data dictionary.

    Returns:
        str: The scene identifier, e.g., "1S7LAXRdDqK".
    """
    for category in structured_data.keys():
        return category.split("_")[0]
    return None


def calculate_euclidean_distance(point1, point2):
    """
    计算两点之间的欧几里得距离。

    参数:
        point1 (list): 第一个点的坐标 [x1, y1, z1]
        point2 (list): 第二个点的坐标 [x2, y2, z2]

    返回:
        float: 欧几里得距离
    """
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))


def convert_to_scene_objects(structured_data, filtered_episodes):
    """
    将 structured_data 和 filtered_episodes 转换为 scene_object 列表。
    每个 episode 对应一个 scene_object，确保一对一的目标和起始点映射。

    参数:
        structured_data (dict): 包含目标信息的数据结构。
        filtered_episodes (list): 包含 episode 的列表，已过滤的有效 episode。

    返回:
        list: 包含所有 scene_object 的列表。
    """
    scene_objects = []

    for episode in filtered_episodes:
        episode_id = episode.get("episode_id")
        start_position = episode.get("start_position")
        start_rotation = episode.get("start_rotation")
        scene_id = episode.get("scene_id")
        valid_tasks = episode.get("valid_tasks", [])

        for task in valid_tasks:
            # Extract task details
            object_id = task[2]
            object_found = None

            # Find the corresponding object in structured_data
            for category, instances in structured_data.items():
                for instance in instances:
                    if instance.get("object_id") == object_id:
                        object_found = instance
                        break
                if object_found:
                    break

            if not object_found:
                continue

            # Extract details from the found object
            object_category = object_found.get("object_category")
            goal_location = object_found.get("goal_location")
            view_points = object_found.get("view_points", [])

            # Assuming each episode has one valid viewpoint
            view_point_count = 0
            for vp in view_points:
                
                view_point_count = view_point_count + 1
                if view_point_count > 1:
                    break
                # 每个episode最多对应两个viewpoint,从而控制（起始，目标）对的数量
                
                vp_agent_state = vp.get("agent_state", {})
                vp_position = vp_agent_state.get("position")
                vp_rotation = vp_agent_state.get("rotation")

                # Create a scene_object
                scene_object = {
                    "episode_id": episode_id,
                    "scene_id": scene_id,
                    "object_environment": object_found.get("lang_desc"),
                    "info": {
                        "geodesic_distance": None,
                        "euclidean_distance": vp.get("Euclidean_distance"),
                        "closest_goal_object_id": object_id
                    },
                    "object_category": object_category,
                    "start_position": start_position,
                    "start_rotation": start_rotation,
                    "start": {
                        "position": start_position,
                        "rotation": start_rotation
                    },
                    "goal": {
                        "position": vp_position,
                        "rotation": vp_rotation
                    },
                    "goals": [],
                    "reference_replay": [],
                    "steps": [],
                    "path": []
                }
                
                scene_objects.append(scene_object)

    return [DotAccessDict(scene_object) for scene_object in scene_objects]
