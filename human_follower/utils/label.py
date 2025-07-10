
import numpy as np
import cv2
import os
import habitat
from habitat.utils.visualizations.maps import get_topdown_map_from_sim
import copy
from scipy.spatial.transform import Rotation as R

def rotation_to_direction(rotation_quaternion):
    """
    Maps a quaternion rotation to a 2D direction vector on the map.

    Args:
        rotation_quaternion: A list or array of [x, y, z, w] representing the quaternion.

    Returns:
        A tuple (dx, dy) representing the direction vector in 2D.
    """
    # Convert quaternion to [x, y, z, w] format
    if hasattr(rotation_quaternion, 'x'):
        quat_array = [rotation_quaternion.x, rotation_quaternion.y, rotation_quaternion.z, rotation_quaternion.w]
    elif isinstance(rotation_quaternion, list):
        quat_array = [rotation_quaternion[0], rotation_quaternion[1], rotation_quaternion[2], rotation_quaternion[3]]
    else:
        raise ValueError("Unsupported quaternion format", rotation_quaternion)
    # Convert to numpy float array
    quat_array = np.asarray(quat_array, dtype=np.float64)
    
    # Convert quaternion to a 3D rotation matrix
    rot_matrix = R.from_quat(quat_array).as_matrix()

    # Extract forward direction from the rotation matrix
    # the agent's "forward" is along the negative Z-axis in 3D
    forward_3d = -rot_matrix[:, 2]  # Negative Z-axis direction

    # Project forward vector to 2D (X, Z plane)
    dx = forward_3d[0]  # X component
    dy = forward_3d[2]  # Z component
    return dx, dy

def calculate_action(cur_rotation, rele_position, next_rotation, min_dis = 0.2):
    """
    计算从当前朝向到下一个点的旋转角度
    cur_rotation: 当前朝向的二维向量
    rele_position: 相对位置的二维向量
    next_rotation: 下一个点的朝向
    """
    # 标准化向量
    cur_rotation = cur_rotation / np.linalg.norm(cur_rotation)
    next_dis = 0
    # 判断是否需要旋转到目标点
    if np.linalg.norm(rele_position) < min_dis:
        target_direction = next_rotation / np.linalg.norm(next_rotation)
    else:
        target_direction = rele_position / np.linalg.norm(rele_position)
        next_dis = np.linalg.norm(rele_position)

    
    angle_rad = np.arctan2(target_direction[1], target_direction[0]) - np.arctan2(cur_rotation[1], cur_rotation[0])

    angle_deg = np.degrees(angle_rad)

    return next_dis, angle_deg

def generate_video_from_steps(steps_data, video_path, fps=30, prefix=None):
    """
    Generates a video from a sequence of steps containing sensor data.
    Warnning: AutoDL platform only

    Args:
        steps_data (dict): A dictionary containing step data with 'steps' as key.
                           Each step contains position, rotation, and sensor_data.
        video_path (str): Path to save the generated video file.
        fps (int): Frames per second for the video.
    """
    # 图像路径的前缀
    #prefix = "/root/autodl-tmp/results/result_test/view_sample/view_sample"

    # 筛选出每个 step 的 color_sensor_0 图像路径，并转换为绝对路径
    image_paths = []
    for step in steps_data['steps']:
        for sensor in step["sensor_data"]:
            if sensor["uuid"] == "color_sensor_0":
                absolute_path = os.path.join(prefix, sensor["image_path"])
                image_paths.append(absolute_path)
                break

    if not image_paths:
        raise ValueError("No images found for color_sensor_0 in the steps data.")

    # 确定视频帧的宽度和高度
    first_frame_path = image_paths[0]
    if not os.path.exists(first_frame_path):
        raise FileNotFoundError(f"Image file {first_frame_path} does not exist.")
    
    frame = cv2.imread(first_frame_path)
    height, width, layers = frame.shape

    # 定义视频编码器和输出视频文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # 将每张图片写入视频
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} does not exist. Skipping.")
            continue
        frame = cv2.imread(image_path)
        video.write(frame)

    # 释放视频对象
    video.release()
    print(f"Video saved to {video_path}")


def find_tangential_triangle(circle_center, radius,distance, point):
    """
    Find a triangle with a given point as its vertex, and two edges tangent to a circle.

    Args:
        circle_center: tuple (x0, y0), coordinates of the circle's center.
        radius: float, radius of the circle.
        point: tuple (xp, yp), coordinates of the given vertex.

    Returns:
        tuple: Coordinates of the two tangent points and the given vertex.
    """
    x0, y0 = circle_center
    xp, yp = point

    # Vector from point P to the circle center O
    dx, dy = x0 - xp, y0 - yp
    d = distance

    if d < radius:
        raise ValueError("Point is inside the circle. No tangential triangle exists.")

    # Angle between the line OP and the tangent lines
    theta = np.arcsin(radius / d)+0.5

    # Angle of the line OP
    angle_op = np.arctan2(dy, dx)

    # Angles of the tangent lines
    angle_tangent1 = angle_op + theta
    angle_tangent2 = angle_op - theta

    # Tangent points on the circle
    x1 = x0 - radius * np.cos(angle_tangent1)
    y1 = y0 - radius * np.sin(angle_tangent1)

    x2 = x0 - radius * np.cos(angle_tangent2)
    y2 = y0 - radius * np.sin(angle_tangent2)

    tangent_point1 = (x1, y1)
    tangent_point2 = (x2, y2)

    return point, tangent_point1, tangent_point2

from shapely.geometry import Point, Polygon

def is_point_in_triangle_shapely(A, B, C, P):
    """
    Determine if a point P is inside the triangle defined by points A, B, and C using Shapely.

    Args:
        A, B, C: Tuples (x, y) representing the vertices of the triangle.
        P: Tuple (x, y) representing the point to test.

    Returns:
        True if P is inside the triangle, False otherwise.
    """
    triangle = Polygon([A, B, C])
    point = Point(P)
    return triangle.contains(point)

def is_line_circle_intersect(P1, P2, O, R):
    """
    Determine if a line segment intersects with a circle.

    Args:
        P1, P2: Tuples (x, y) representing the endpoints of the line segment.
        O: Tuple (x, y) representing the center of the circle.
        R: Float, radius of the circle.

    Returns:
        True if the line segment intersects the circle, False otherwise.
    """
    # Vector form of the line segment
    x1, y1 = P1
    x2, y2 = P2
    x0, y0 = O

    # Compute the quadratic equation coefficients
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - x0, y1 - y0

    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = fx**2 + fy**2 - R**2

    # Solve the discriminant
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        # No intersection
        return False

    # Calculate the two possible solutions for t
    discriminant = np.sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)

    # Check if either t1 or t2 is within [0, 1]
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)

def is_line_triangle_intersect(P1, P2, A, B, C):
    """
    Determine if a line segment intersects with a triangle.

    Args:
        P1, P2: Tuples (x, y) representing the endpoints of the line segment.
        A, B, C: Tuples (x, y) representing the vertices of the triangle.

    Returns:
        True if the line segment intersects the triangle, False otherwise.
    """
    def cross_product(P, Q, R):
        """Calculate the cross product of vectors PQ and PR."""
        return (Q[0] - P[0]) * (R[1] - P[1]) - (Q[1] - P[1]) * (R[0] - P[0])

    def is_line_segment_intersect(P1, P2, Q1, Q2):
        """Check if two line segments intersect."""
        d1 = cross_product(P1, P2, Q1)
        d2 = cross_product(P1, P2, Q2)
        d3 = cross_product(Q1, Q2, P1)
        d4 = cross_product(Q1, Q2, P2)

        return (d1 * d2 < 0) and (d3 * d4 < 0)
    
    def is_point_in_triangle_via_cross(A, B, C, P):
        """
        Determine if a point P is inside the triangle defined by points A, B, and C using cross product.

        Args:
            A, B, C: Tuples (x, y) representing the vertices of the triangle.
            P: Tuple (x, y) representing the point to test.

        Returns:
            True if P is inside the triangle, False otherwise.
        """
        def cross_product(P, Q, R):
            """Calculate the cross product of vectors PQ and PR."""
            return (Q[0] - P[0]) * (R[1] - P[1]) - (Q[1] - P[1]) * (R[0] - P[0])

        # Calculate cross products
        cross1 = cross_product(A, B, P)
        cross2 = cross_product(B, C, P)
        cross3 = cross_product(C, A, P)

        # Check if all cross products have the same sign
        return (cross1 >= 0 and cross2 >= 0 and cross3 >= 0) or (cross1 <= 0 and cross2 <= 0 and cross3 <= 0)
    # Check if either endpoint of the line segment is inside the triangle
    if is_point_in_triangle_via_cross(A, B, C, P1) or is_point_in_triangle_via_cross(A, B, C, P2):
        return True

    # Check if the line segment intersects any of the triangle's edges
    return (
        is_line_segment_intersect(P1, P2, A, B) or
        is_line_segment_intersect(P1, P2, B, C) or
        is_line_segment_intersect(P1, P2, C, A)
    )

def crop_and_pad_image(rgb_map, cur_position, target_size=(480, 480)):
    """
    Crop a region around cur_position and pad to ensure the final size is target_size.
    
    Args:
        rgb_map (numpy array): The input image (H, W, C).
        cur_position (tuple): The center position (x, y) for cropping.
        radius_in_pixels (int): The radius to crop around the center.
        target_size (tuple): The desired size after cropping and padding (default: 512x512).
    
    Returns:
        numpy array: The cropped and padded image of size target_size.
    """
    h, w, _ = rgb_map.shape
    target_h, target_w = target_size
    
    # # 计算裁剪区域
    # diameter = int(1.5 * radius_in_pixels)
    diameter = int(target_h/2)
    x1 = max(0, cur_position[0] - diameter)
    y1 = max(0, cur_position[1] - diameter)
    x2 = min(w, cur_position[0] + diameter)
    y2 = min(h, cur_position[1] + diameter)

    # 裁剪图像
    cropped = rgb_map[y1:y2, x1:x2]
    
    # 初始化一个空白图像，大小为目标尺寸
    # padded = np.full((target_h, target_w, rgb_map.shape[2]), dtype=rgb_map.dtype)
    padded = np.full((target_h, target_w, rgb_map.shape[2]), fill_value=255, dtype=rgb_map.dtype)  # 白色背景
    # 计算裁剪图像放置在目标图像中的位置
    crop_h, crop_w, _ = cropped.shape

    start_y = (target_h - crop_h) // 2
    start_x = (target_w - crop_w) // 2

    # 将裁剪区域粘贴到空白图像的中心
    padded[start_y:start_y+crop_h, start_x:start_x+crop_w] = cropped

    return padded
def new_apply_visibility_overlay(topdownmap, explored_pixel_path, cur_position, obs_in_pixels, rotation_vec, direction):
    # 计算方向向量

    front_vec = np.array(rotation_vec)
    right_vec = np.array([-front_vec[1], front_vec[0]])
    back_vec = -front_vec
    left_vec = -right_vec

    if direction == 0:
        cur_vec = front_vec
    elif direction == 1:
        cur_vec = left_vec
    elif direction == 2:
        cur_vec = back_vec
    elif direction == 3:
        cur_vec = right_vec
    else:
        raise ValueError("Invalid direction value. Must be 0, 1, 2, or 3.")


    # 创建一个与原图相同大小的透明层

    marker_size = obs_in_pixels
    obs_x = cur_position[0] + int(cur_vec[0] * marker_size)
    obs_y = cur_position[1] + int(cur_vec[1] * marker_size)
    
    
    radius_in_pixels = 100
    p1,p2,p3 = find_tangential_triangle((obs_x, obs_y),radius_in_pixels,obs_in_pixels, (cur_position[0],cur_position[1]))


    # 在透明层上绘制较亮的蓝色圆 (带 Alpha 通道)，仅绘制非白色区域
    circle_color = (173, 216, 230, 200)  # 较亮的蓝色 + 实一点
    tria_color = (173, 216, 230, 32) 
    h, w, _ = topdownmap.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    
    
    # for i in range(h):
    #     for j in range(w):
    #         if np.linalg.norm([i - obs_y, j - obs_x]) <= radius_in_pixels:
    #             # if not np.array_equal(topdownmap[i, j], [255, 255, 255]):  # 检查是否是白色区域
    #             overlay[i, j] = circle_color
    #         elif is_point_in_triangle_shapely(p1,p2,p3,(j,i)):
    #             # if not np.array_equal(topdownmap[i, j], [255, 255, 255]):  # 检查是否是白色区域
    #             overlay[i, j] = circle_color

    # 创建网格
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # 计算圆的掩码
    circle_mask = (np.sqrt((y - obs_y)**2 + (x - obs_x)**2) <= radius_in_pixels)

    # 计算三角形的掩码
    triangle_mask = np.zeros((h, w), dtype=bool)
    triangle_min_x = max(0, min(p1[0], p2[0], p3[0]))
    triangle_max_x = min(w, max(p1[0], p2[0], p3[0]))
    triangle_min_y = max(0, min(p1[1], p2[1], p3[1]))
    triangle_max_y = min(h, max(p1[1], p2[1], p3[1]))

    for i in range(int(triangle_min_y), int(triangle_max_y)):
        for j in range(int(triangle_min_x), int(triangle_max_x)):
            if is_point_in_triangle_shapely(p1, p2, p3, (j, i)):
                triangle_mask[i, j] = True

    # # 合并掩码（并集）
    # combined_mask = circle_mask | triangle_mask

    # 应用掩码到 overlay    
    overlay[triangle_mask] = tria_color
    overlay[circle_mask] = circle_color

   

    # 初始化 RGBA 图像
    rgba_map = np.dstack([topdownmap, np.full((h, w), 255, dtype=np.uint8)])

    # 将透明层叠加到 RGBA 图像上
    alpha_overlay = overlay[:, :, 3] / 255.0  # 归一化透明度
    for c in range(3):  # 对 RGB 通道进行叠加
        rgba_map[:, :, c] = (
            rgba_map[:, :, c] * (1 - alpha_overlay) + overlay[:, :, c] * alpha_overlay
        ).astype(np.uint8)

    # 确保 Alpha 通道不透明
    rgba_map[:, :, 3] = 255

    # 绘制当前位置 (红色圆点，透明度 255)
    cv2.circle(rgba_map, cur_position, 10, (255, 0, 0, 255), thickness=-1)
    
    # cv2.circle(rgba_map, (int(p2[0]),int(p2[1])), 10, (255, 0, 0, 255), thickness=-1)
    # cv2.circle(rgba_map, (int(p3[0]),int(p3[1])), 10, (255, 0, 0, 255), thickness=-1)
    intersect = False
    # # 在图像上绘制探索路径 (橙色，透明度 255)
    # for i in range(len(explored_pixel_path) - 1):
    #     start = explored_pixel_path[i]
    #     end = explored_pixel_path[i + 1]
    #     cv2.line(rgba_map, start, end, (255, 165, 0, 255), thickness=5)
    #     if not intersect:
    #         intersect = is_line_circle_intersect((start[0],start[1]),(end[0],end[1]),(obs_x, obs_y),radius_in_pixels) or is_line_triangle_intersect((start[0],start[1]),(end[0],end[1]),p1,p2,p3)

    def is_bbox_intersect(line_bbox, circle_bbox, triangle_bbox):
        """
        Check if a line's bounding box intersects with the bounding boxes of the circle or triangle.
        """
        def bbox_overlap(bbox1, bbox2):
            return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                        bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

        return bbox_overlap(line_bbox, circle_bbox) or bbox_overlap(line_bbox, triangle_bbox)

    # 计算圆的包围盒
    circle_bbox = (
        obs_x - radius_in_pixels,
        obs_y - radius_in_pixels,
        obs_x + radius_in_pixels,
        obs_y + radius_in_pixels
    )

    # 计算三角形的包围盒
    triangle_bbox = (
        min(p1[0], p2[0], p3[0]),
        min(p1[1], p2[1], p3[1]),
        max(p1[0], p2[0], p3[0]),
        max(p1[1], p2[1], p3[1])
    )

    # 遍历路径的线段
    for i in range(len(explored_pixel_path) - 1):
        start = explored_pixel_path[i]
        end = explored_pixel_path[i + 1]

        # 绘制路径线
        cv2.line(rgba_map, start, end, (255, 165, 0, 255), thickness=5)

        # 跳过明显不相交的线段
        line_bbox = (
            min(start[0], end[0]),
            min(start[1], end[1]),
            max(start[0], end[0]),
            max(start[1], end[1])
        )
        if not is_bbox_intersect(line_bbox, circle_bbox, triangle_bbox):
            continue

        # 检测相交
        if not intersect:
            # intersect = (
            #     is_line_circle_intersect((start[0], start[1]), (end[0], end[1]), (obs_x, obs_y), radius_in_pixels) or
            #     is_line_triangle_intersect((start[0], start[1]), (end[0], end[1]), p1, p2, p3)
            # )
            intersect = is_line_circle_intersect((start[0], start[1]), (end[0], end[1]), (obs_x, obs_y), radius_in_pixels)

    result = crop_and_pad_image(rgba_map, cur_position, target_size=(640,640))
    
    return result, intersect

from .utils import generate_sector_lines, check_lines, generate_mask_and_update_map,count_path_mask_intersections,generate_observed_map, extract_color_mask,calculate_overlap, generate_circle_lines,draw_directional_line,is_pixel_in_mask

def apply_visibility_overlay_2(topdownmap, explored_pixel_path, cur_position, obs_in_pixels, rotation_vec, direction,target_position):
    # 计算方向向量

    front_vec = np.array(rotation_vec)
    right_vec = np.array([-front_vec[1], front_vec[0]])
    back_vec = -front_vec
    left_vec = -right_vec

    if direction == 0:
        cur_vec = front_vec
    elif direction == 1:
        cur_vec = left_vec
    elif direction == 2:
        cur_vec = back_vec
    elif direction == 3:
        cur_vec = right_vec
    else:
        raise ValueError("Invalid direction value. Must be 0, 1, 2, or 3.")


    # 创建一个与原图相同大小的透明层

    marker_size = obs_in_pixels
    obs_x = cur_position[0] + int(cur_vec[0] * marker_size)
    obs_y = cur_position[1] + int(cur_vec[1] * marker_size)
    
    
    radius_in_pixels = 100
    n = 48
    obs_lines = generate_sector_lines((cur_position[0],cur_position[1]),radius_in_pixels+obs_in_pixels,(cur_vec[0],cur_vec[1]), np.pi/3*2, n)
    
    obs_lines = check_lines(topdownmap,obs_lines,5)
    
    updated_map, cur_mask = generate_mask_and_update_map(topdownmap, obs_lines, np.pi/3*2, n)


    # 绘制当前位置 (红色圆点，透明度 255)
    cv2.circle(updated_map, cur_position, 10, (255, 0, 0, 255), thickness=-1)

    target_color = (255, 165, 0)
    observed_mask = extract_color_mask(updated_map, target_color, tolerance=5)

    # intersect = count_path_mask_intersections(mask,explored_pixel_path)

    intersect, cur_union = calculate_overlap(cur_mask, observed_mask)
    intersect = int(intersect)*100
    observed_union = (observed_mask > 0).sum()

    viewed = is_pixel_in_mask(cur_mask, (target_position[0],target_position[1]))
    located = is_pixel_in_mask(observed_mask, (target_position[0],target_position[1]))

    result = crop_and_pad_image(updated_map, cur_position, target_size=(640,640))
    # result = cv2.resize(result, (480, 480), interpolation=cv2.INTER_AREA)
    return result, intersect, viewed, located, cur_union, observed_union

def apply_visibility_raw(topdownmap, explored_pixel_path, cur_position, obs_in_pixels, cur_vec, target_position):


    radius_in_pixels = 100
    # 遍历路径的线段
    new_size = 640
    min_x = cur_position[0] - new_size/2
    max_x = cur_position[0] + new_size/2
    min_y = cur_position[1] - new_size/2
    max_y = cur_position[1] + new_size/2
    gray_color=(128, 128, 128)
    orange_color=(255, 165, 0)
    h, w, _ = topdownmap.shape
    radius = int((radius_in_pixels+obs_in_pixels)/2)
    
    n = 36
    for i in range(len(explored_pixel_path)-1):
        point = explored_pixel_path[i]
        cx, cy = point
        point_in = min_x <= cx <= max_x and min_y <= cy <= max_y

        if not point_in:
            continue
        

        obs_lines = generate_sector_lines(point,radius,(cur_vec[0],cur_vec[1]), np.pi*2, n*4)
        
        obs_lines = check_lines(topdownmap,obs_lines,5)
        
        topdownmap = generate_observed_map(topdownmap, obs_lines, np.pi*2, n*4)

        # # Define the circular area
        # for y in range(max(0, cy - radius), min(h, cy + radius + 1)):
        #     for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
        #         # Check if the point is within the circle
        #         if (x - cx)**2 + (y - cy)**2 <= radius**2:
        #             # Check if the pixel is gray
        #             if np.array_equal(topdownmap[y, x], gray_color):
        #                 # Update to orange
        #                 topdownmap[y, x] = orange_color

        # 绘制路径线
        # cv2.line(topdownmap, start, end, (255, 165, 0, 255), thickness=5)
        
        
    # 绘制当前位置 (红色圆点，透明度 255)
    cv2.circle(topdownmap, cur_position, 10, (255, 0, 0, 255), thickness=-1)
    # cv2.circle(topdownmap, target_position, 10, (255, 0, 255, 255), thickness=-1)
    # topdownmap = draw_directional_line(topdownmap, (cur_position[0], cur_position[1]) ,  (cur_vec[0], cur_vec[1]), 50)
    # result = crop_and_pad_image(topdownmap, cur_position, target_size=(new_size,new_size))
    # result = cv2.resize(result, (480, 480), interpolation=cv2.INTER_AREA)
    return topdownmap

def get_topdown_map(agent_path_info, rotation_vec , sim, target,deg_idx, map_resolution=1024, draw_border=True, meters_per_pixel=None, agent_id=0, showimage=True):
    """
    Generates a top-down map and provides utilities to map:
    1. Real-world coordinates to pixel indices.
    2. Quaternion rotations to 2D directional vectors for visualizing agent orientation.
    
    Returns:
        NParray of final iamge

    """
    agent_state = agent_path_info[0]
    position = agent_state.position
    rotation = agent_state.rotation
    #print("position:",position)
    #print("rotation:", rotation)
    path=[]
    for point_act in agent_path_info[1]:
        path.append(point_act[0])
    
    
    
    # Get the bounds of the navigable area
    bounds_min, bounds_max = sim.pathfinder.get_bounds()

    # Calculate map size in meters and determine meters_per_pixel
    map_size_meters = [
        bounds_max[0] - bounds_min[0],  # Width (X-axis)
        bounds_max[2] - bounds_min[2],  # Length (Z-axis, ignoring Y-axis)
    ]
    
    if meters_per_pixel is None:
        meters_per_pixel = max(map_size_meters) / map_resolution

    # Generate the top-down map
    top_down_map = get_topdown_map_from_sim(sim, map_resolution, draw_border, meters_per_pixel, agent_id)

    # Recolor the map
    recolor_map = np.array(
        [[255, 255, 255],  # Occupied (white)
         [128, 128, 128],  # Unoccupied (gray)
         [0, 0, 0]],       # Border (black)
        dtype=np.uint8
    )
    recolored_map = recolor_map[top_down_map]

    # Function to map world coordinates to pixel indices
    def world_to_pixel(world_coordinates):
        """
        Maps real-world coordinates to pixel indices on the top-down map.

        Args:
            world_coordinates: A list or array of [x, y, z] world coordinates.

        Returns:
            A tuple (px, py) representing pixel indices.
        """
        x, _, z = world_coordinates  # Ignore Y-axis (height)
        px = int((x - bounds_min[0]) / meters_per_pixel)
        py = int((z - bounds_min[2]) / meters_per_pixel)
        return px, py

    

    #映射real loc 到 当前image
    # Map the path to pixel coordinates
    pixel_path = [world_to_pixel(step_point[0]) for step_point in path] 
    cur_position = world_to_pixel(position)
    target_position = world_to_pixel(target)
    
    # Determine the explored path (from start to current position)
    explored_pixel_path = []
    for step_point in path:
        explored_pixel_path.append(world_to_pixel(step_point[0]))
        # Stop adding points once reaching the current position
        if np.allclose(step_point[0], position, atol=1e-2):  # Use a tolerance for floating-point comparison
            break
        
    visibility_radius = 2.5
    # obs_in_pixels = int(visibility_radius / meters_per_pixel)
    obs_in_pixels = 220
    # recolored_map = apply_visibility_overlay(recolored_map,explored_pixel_path ,cur_position, radius_in_pixels, rotation_vec, shift_direction)
    maps = []
    intersects = []
    cur_union_list = []
    observed_union_list = []
    recolored_map = apply_visibility_raw(recolored_map,explored_pixel_path ,cur_position, obs_in_pixels,rotation_vec, target_position)

    viewed = False
    located = False
    for i in range (0,4):
        td_map = copy.deepcopy(recolored_map)
        if i == deg_idx:
            new_map, intersect, viewed, located, cur_union, observed_union = apply_visibility_overlay_2(td_map,explored_pixel_path ,cur_position, obs_in_pixels, rotation_vec, i, target_position)
        else:
            new_map, intersect, _, _ ,cur_union, observed_union= apply_visibility_overlay_2(td_map,explored_pixel_path ,cur_position, obs_in_pixels, rotation_vec, i, target_position)
        maps.append(new_map)
        intersects.append(intersect)
        cur_union_list.append(cur_union*meters_per_pixel)
        observed_union_list.append(observed_union*meters_per_pixel)
        # if intersect:
        #     intersects.append(1)
        # else:
        #     intersects.append(0)

    cv2.circle(recolored_map, target_position, 10, (255, 0, 255, 255), thickness=-1)
    maps.append(recolored_map)
    
    # from concurrent.futures import ThreadPoolExecutor

    # # 定义并行任务函数
    # def process_visibility_overlay(index):
    #     return new_apply_visibility_overlay(
    #         recolored_map, explored_pixel_path, cur_position, obs_in_pixels, rotation_vec, index
    #     )

    # # 使用多线程并行处理
    # maps = []
    # intersects = []
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     results = list(executor.map(process_visibility_overlay, range(4)))

    # # 拆分结果
    # for new_map, intersect in results:
    #     maps.append(new_map)
    #     intersects.append(intersect)
    return maps, intersects, viewed, located , cur_union_list, observed_union_list



    