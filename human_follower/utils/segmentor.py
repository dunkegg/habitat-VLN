import logging
import os
import torch
import numpy as np

import cv2

from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

class Segmentor:
    """
    A class for semantic segmentation using a pre-trained model.
    """

    def __init__(self):
        """
        Initialize the segmentation model and processor.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic").to(self.device)

        # Get class ids for navigable regions
        id2label = self.model.config.id2label
        self.navigability_class_ids = [id for id, label in id2label.items() if 'floor' in label.lower() or 'rug' in label.lower()]

    def get_navigability_mask(self, im: np.array):
        """
        Generate a navigability mask from an input image.

        Parameters
        ----------
        im : np.array
            An RGB image for generating the navigability mask.
        """
        image = Image.fromarray(im[:, :, :3])
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0].cpu().numpy()

        navigability_mask = np.isin(predicted_semantic_map, self.navigability_class_ids)
        return navigability_mask
