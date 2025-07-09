import time
import logging
import torch
import numpy as np
from prismatic import load
import os

class VLM:
    def __init__(self, cfg):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'{cfg.device}:{local_rank}')
        start_time = time.time()
        self.model = load(cfg.model_id, hf_token=cfg.hf_token)
        self.model.to(device, dtype=torch.bfloat16)
        logging.info(f"Loaded VLM in {time.time() - start_time:.3f}s")

    def generate(self, prompt, image, T=0.4, max_tokens=512):
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()
        generated_text = self.model.generate(
            image,
            prompt_text,
            do_sample=True,
            temperature=T,
            max_new_tokens=max_tokens,
            min_length=1,
        )
        return generated_text

    def get_loss(self, image, prompt, tokens, get_smx=True, T=1):
        "Get unnormalized losses (negative logits) of the tokens"
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()
        losses = self.model.get_loss(
            image,
            prompt_text,
            return_string_probabilities=tokens,
        )[0]
        losses = np.array(losses)
        if get_smx:
            return np.exp(-losses / T) / np.sum(np.exp(-losses / T))
        return losses

    def vlm_generate_objects_from_image(self,image_view, vlm_temperature):

        # user_prompt = """You are in a indoor scenario. Analyze the provided image and list all the objects that could be moved. 
        #                     Provide the name of each object including its color, separated by a comma. 
                            
        #                     For example, if you see a white cup, a red pillow, you should list them as: white cup, red pillow
                            
        #                     Do not output object such as 'floor' since floor cannot be moved!
        #                     """

        user_prompt = """You are in a indoor scenario. List all the movable objects in the image if you are confident what they are. Do not output the same object multiple times.
        
                            Provide the name of each object including its color, separated by a comma. 
                            
                            For example, if you see a white cup, a red pillow, you should list them as: white cup, red pillow
                            
                            Do not output object such as 'floor' since floor cannot be moved!
                            """

        response = self.generate(
                        user_prompt,
                        image_view,
                        T = vlm_temperature
                    )

        items_list = response.split(", ")
        # print(f"{items_list}")
        return items_list

    def vlm_generate_objects_with_location_from_image(self, image_view, vlm_temperature):
        prompt = """List objects that can be removed by hands in this picture one by one. 
                            Also describe their position relationship. 
                            The position relationship should only consider physical contact. """
        
        response = self.generate(prompt, image_view, T=vlm_temperature)
        sentences = response.split('\n')  # 按换行符分割句子
        items_locations = {}  # 创建一个空字典来保存物品和位置信息
        for sentence in sentences:
            sentence = re.sub(r'^\W*', '', sentence)  # 去掉句子开头的任何非字母字符
            match = re.match(r'^(.*?):(.*?)$', sentence)  # 使用正则表达式匹配物品和位置信息
            if match:
                item = match.group(1).strip()  # 提取物品
                locations = match.group(2).strip().split(',')  # 提取位置信息并按逗号分割
                items_locations[item] = locations  # 将物品和位置信息存储到字典中
                print("Item:", item)
                print("Locations:", locations)

        return items_locations

    def prediction_set_from_views(self, retriever, views_images, vlm_temperature):	
        object_pairs = []
        rooms=[]
        for view_image in views_images:
            items_locations = self.vlm_generate_objects_with_location_from_image(view_image, vlm_temperature)
            items = list(items_locations.keys())
            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    item1 = items[i]
                    item2 = items[j]
                    locations1 = items_locations[item1]
                    locations2 = items_locations[item2]
                    object_pairs.append([item1+' '+locations1, item2+' '+locations2])
                    rooms.append('kitchen')

        prediction_set = retriever.relevance_for_prediction(object_pairs,None)
        return prediction_set
					# comparison = {
					# 	"item1": item1,
					# 	"item2": item2,
					# 	"common_locations": list(set(locations1) & set(locations2)),
					# 	"unique_locations_item1": list(set(locations1) - set(locations2)),
					# 	"unique_locations_item2": list(set(locations2) - set(locations1))
					# }
