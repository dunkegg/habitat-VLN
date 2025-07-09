import numpy as np
import cv2
from PIL import Image
import copy
from .segmentor import Segmentor
import habitat_sim
from .utils import *
class DirectionProjector:
    """
    A class for semantic segmentation using a pre-trained model.
    """

    def __init__(self, hfov, mode,  num_theta):
        """
        Initialize the segmentation model and processor.
        """
        self.segmentor = Segmentor()
        self.mode = mode
        self.fov = hfov
        self.num_theta = num_theta
        
        res_factor = 1
        self.resolution = (
            480 // res_factor,
            640 // res_factor
        )

        self.focal_length = calculate_focal_length(self.fov, self.resolution[1])
        self.radius = 0.1
        
        # sensor_range =  np.deg2rad(90)
        sensor_range = 90
        all_thetas = np.linspace(-sensor_range, sensor_range, self.num_theta)
        self.theta_list = []
        for theta_i in all_thetas:
            self.theta_list.append((self.radius, theta_i))
        

        
    def _get_navigability_mask(self, rgb_image: np.array, depth_image: np.array, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose):
        """
        Get the navigability mask for the current state, according to the configured navigability mode.
        """
        if self.mode == 'segmentation':
            navigability_mask = self.segmentor.get_navigability_mask(rgb_image)
        else:
            thresh = 0.2
            height_map = depth_to_height(depth_image, self.fov, sensor_state.position, sensor_state.rotation)
            navigability_mask = abs(height_map - (agent_state.position[1] - 0.04)) < thresh

        return navigability_mask        
        
    def _navigability(self, rgb_image: np.array, depth_image: np.array, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose):

        
        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        # sensor_range =  np.deg2rad(self.fov / 2) * 1.5
        # sensor_range =  np.deg2rad(90)
        sensor_range = 90
        
        all_thetas = np.linspace(-sensor_range, sensor_range, self.num_theta)

        theta_list = []
        for theta_i in all_thetas:
            theta_list.append((self.radians, theta_i))
            # r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state, depth_image)
            # if r_i is not None:
            #     self._update_voxel(
            #         r_i, theta_i, agent_state,
            #         clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
            #     )
            #     a_initial.append((r_i, theta_i))

        return theta_list, navigability_mask
    
    def _project(self, obs: dict, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose ,deg_idx: int):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        # agent_state: habitat_sim.AgentState = obs['agent_state']
        # sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs[f'color_sensor_{deg_idx}']
        depth_image = obs[f'depth_sensor_{deg_idx}']
        
        
        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )
      

        projected_image = copy.deepcopy(rgb_image)
        #add navigability_mask onto projected_image
        # 将 navigability_mask 转换为三通道图像
        mask_color = (0, 255, 0)  # 绿色
        # mask_visual = np.zeros_like(projected_image, dtype=np.uint8)
        # mask_visual[navigability_mask > 0, :3] = mask_color

        # 覆盖显示 navigability_mask
        projected_image[navigability_mask > 0, :3] = mask_color
        
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # scale_factor = 1
        # text_size = 0.4 * scale_factor
        # text_thickness = math.ceil(3 * scale_factor)
        
        # start_px = agent_frame_to_image_coords(
        #     [0, 0, 0], agent_state, sensor_state, 
        #     resolution=self.resolution, focal_length=self.focal_length
        # )
        # text_color=GREEN
        # for r_i, theta_i in self.theta_list:

            
        #     agent_point = [r_i * np.sin(np.deg2rad(theta_i)), 0, -r_i * np.cos(np.deg2rad(theta_i))]
        #     end_px = agent_frame_to_image_coords(
        #         agent_point, agent_state, sensor_state, 
        #         resolution=self.resolution, focal_length=self.focal_length
        #     )
            
        #     cv2.arrowedLine(projected_image, tuple(start_px), tuple(end_px), WHITE, math.ceil(1 * scale_factor), tipLength=0.0)
            
        #     text = str(theta_i)
        #     (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
        #     text_position = (end_px[0] - text_width // 2, end_px[1] + text_height // 2)
        #     cv2.putText(projected_image, text, text_position, font, text_size, text_color, text_thickness)
            
        return projected_image
                
        
