import numpy as np
import os

np.set_printoptions(precision=3)
import csv
import ast
import copy
import pickle
import random
import logging
import math
import quaternion
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from ..utils.segmentor import Segmentor

class ON_AGENT:
    """Modified for multi tasks"""
    def __init__(
        self, cfg, hb_agent
    ):
        self.cfg = cfg

        self.output_dir = None

        # scene
        self.question = None
        self.target_object = None
        # pose
        self.hb_agent = hb_agent
        self.agent_state = None



        self.history_path = None
        self.result = None
        self.result_all = []
        self.result_info_all = {"episode": [], "summary": {"weighted success rate": 0.0, "relevance success rate": 0.0}}
        #
        self.reached = None
        #
        self.save_img = True

        # self.logger = logger
        random.seed(1024)
        self.debug = False
        
        
        # models
        self.segmentor = Segmentor()


    def reset(
            self,
            target_object,
            agent_state,
            pathfinder,
            output_dir=None,

            save_img=True,

    ):
        self.target_object = target_object

        self.agent_state = agent_state
        self.hb_agent.set_state(self.agent_state)

        self.history_path = np.empty((0, 2))


        ###
        self.steps = 0
        self.save_img = save_img

    def move_to(self, postion, rotation):
        self.agent_state.position = postion
        self.agent_state.rotation = rotation
        self.hb_agent.set_state(self.agent_state)
