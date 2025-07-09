from .ovon_dataset import OVONDatasetV1
import gzip
import os

def getScenefromOvon(scene, cfg):
    # s = '00842-hkr2MGpHD6B'
    scene_name = scene.split('-')[1]
    json_file_path = os.path.join(cfg.json_file_path, f"{scene_name}.json.gz")
    scenes_dir = cfg.scenes_dir
    #cfg.json_file_path = f"/data2/zejinw/ON-MLLM/hm3d/ovon_data/hm3d/val_seen/content/"
    # should join : {scene_name}.json.gz
    #cfg.scenes_dir = "/data2/zejinw/data/scene_datasets/hm3d/val"  # 场景文件夹路径

    # 读取 .gz 压缩的 JSON 数据并加载到 OVONDatasetV1 实例中
    with gzip.open(json_file_path, "rt", encoding="utf-8") as f:
        json_data = f.read()

    # 使用 from_json 方法加载数据
    dataset = OVONDatasetV1()
    dataset.from_json(json_data, scenes_dir=scenes_dir)
    # episodes = dataset.episodes  # episodes 已由 from_json 方法创建

    # # 验证 episodes 是否正确加载
    # for episode in episodes:
    #     episode
    
    return dataset


# def getScenefromOvon():
#     # 定义该函数的内容
#     # 定义JSON文件路径
#     json_file_path = "/home/ovon/data/datasets/ovon/hm3d/val_seen/content/4ok3usBNeis.json.gz"
#     scenes_dir = "/root/autodl-tmp/HM3Ddata/scene_datasets"  # 场景文件夹路径

#     # 读取 .gz 压缩的 JSON 数据并加载到 OVONDatasetV1 实例中
#     with gzip.open(json_file_path, "rt", encoding="utf-8") as f:
#         json_data = f.read()

#     # 使用 from_json 方法加载数据
#     dataset = OVONDatasetV1()
#     dataset.from_json(json_data, scenes_dir=scenes_dir)

#     return dataset
                
    