import logging
import os
import traceback
import cv2
import numpy as np
import math
import quaternion
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (200, 200, 200)

def generate_circle_lines(center, r, n):
    """
    Generate n evenly distributed lines radiating from the center within a circle.

    Args:
        center: Tuple (cx, cy) for the center of the circle.
        r: Radius of the circle.
        n: Number of lines to generate.

    Returns:
        List of tuples [(start_x, start_y, end_x, end_y), ...] for each line.
    """
    cx, cy = center

    # Compute angle increment for evenly distributed lines
    delta_angle = 2 * math.pi / n

    # Generate lines
    lines = []
    for i in range(n):
        angle = i * delta_angle
        end_x = cx + r * math.cos(angle)
        end_y = cy + r * math.sin(angle)
        lines.append((cx, cy, end_x, end_y))

    return lines
def generate_sector_lines(center, r, direction, theta, n):
    
    cx, cy = center
    vx, vy = direction
    """
    Generate n equal segments of a sector and compute the center line for each segment.

    Args:
        cx, cy: Circle center coordinates.
        r: Radius of the circle.
        vx, vy: Direction vector for the symmetry axis of the sector.
        theta: Total angle of the sector (in radians).
        n: Number of equal divisions.

    Returns:
        List of tuples [(start_x, start_y, end_x, end_y), ...] for each segment center line.
    """
    # Normalize the direction vector
    magnitude = math.sqrt(vx**2 + vy**2)
    vx, vy = vx / magnitude, vy / magnitude

    # Compute center angle of the sector
    angle_center = math.atan2(vy, vx)

    # Compute start and end angles of the sector
    angle_start = angle_center - theta / 2
    angle_end = angle_center + theta / 2

    # Compute the angle increment for each segment
    delta_angle = theta / n

    # Generate lines for each segment
    lines = []
    for i in range(n):
        angle_i = angle_start + i * delta_angle + delta_angle / 2  # Center of the i-th segment
        end_x = cx + r * math.cos(angle_i)
        end_y = cy + r * math.sin(angle_i)
        lines.append((cx, cy, end_x, end_y))

    return lines




def check_lines(topdownmap, lines, step=30):
    """
    Check each line segment to see if the extended points are in white areas.

    Args:
        topdownmap: 2D or 3D map where pixel values indicate regions (e.g., RGB).
        lines: List of tuples [(start_x, start_y, end_x, end_y), ...].
        step: Length of each extension step.

    Returns:
        List of results for each line:
        [
            [(x1, y1, is_white), (x2, y2, is_white), ...],
            ...
        ]
    """
    h, w, _ = topdownmap.shape
    results = []

    for start_x, start_y, end_x, end_y in lines:

        # Calculate the direction vector of the line
        dx = end_x - start_x
        dy = end_y - start_y
        line_length = np.sqrt(dx**2 + dy**2)
        if line_length == 0:
            continue  # Skip degenerate lines

        # Normalize direction vector and scale by step
        dx = dx / line_length * step
        dy = dy / line_length * step

        # Extend the line from the start point
        current_x, current_y = start_x, start_y
        while True:
            # Round to nearest integer for indexing
            i, j = int(round(current_y)), int(round(current_x))

            # Check bounds
            if not (0 <= i < h and 0 <= j < w):
                break

            # Check if the point is in a white region
            is_white = np.array_equal(topdownmap[i, j], [255, 255, 255])

            if is_white:
                break  # Stop if not in white area


            # Stop if we reach the end of the line
            if np.sqrt((current_x - start_x)**2 + (current_y - start_y)**2) >= line_length:
                break
            
            # Move to the next point
            current_x += dx
            current_y += dy


        results.append((start_x, start_y, current_x, current_y))

    return results

def generate_mask_and_update_map(topdownmap, lines, theta, n):
    """
    Generate a mask based on circular arcs defined by line segments and update the map.

    Args:
        topdownmap: The input map (H, W, 3).
        lines: List of tuples [(start_x, start_y, end_x, end_y), ...].
        theta: Total angle of the arcs (in radians).
        n: Number of divisions for the arcs.

    Returns:
        Updated map with areas outside the mask turned white.
    """
    h, w, _ = topdownmap.shape

    # Create a blank mask
    mask = np.zeros((h, w), dtype=np.uint8)

    for start_x, start_y, current_x, current_y in lines:
        # Calculate the radius (length of the line segment)
        radius = np.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)

        # Calculate the angle of the line segment
        direction_angle = np.arctan2(current_y - start_y, current_x - start_x)

        # Arc range
        angle_start = np.degrees(direction_angle - theta / (2 * n))
        angle_end = np.degrees(direction_angle + theta / (2 * n))

        # Draw the arc on the mask
        center = (int(round(start_x)), int(round(start_y)))
        axes = (int(round(radius)), int(round(radius)))
        cv2.ellipse(mask, center, axes, 0, angle_start, angle_end, 255, -1)

    # Update the map based on the mask
    updated_map = topdownmap.copy()
    updated_map[mask == 0] = [255, 255, 255]  # Set non-mask areas to white

    return updated_map, mask

def generate_observed_map(topdownmap, lines, theta, n):
    """
    Generate a mask based on circular arcs defined by line segments and update the map.

    Args:
        topdownmap: The input map (H, W, 3).
        lines: List of tuples [(start_x, start_y, end_x, end_y), ...].
        theta: Total angle of the arcs (in radians).
        n: Number of divisions for the arcs.

    Returns:
        Updated map with areas outside the mask turned white.
    """
    h, w, _ = topdownmap.shape

    # Create a blank mask
    mask = np.zeros((h, w), dtype=np.uint8)

    for start_x, start_y, current_x, current_y in lines:
        # Calculate the radius (length of the line segment)
        radius = np.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)

        # Calculate the angle of the line segment
        direction_angle = np.arctan2(current_y - start_y, current_x - start_x)

        # Arc range
        angle_start = np.degrees(direction_angle - theta / (2 * n))
        angle_end = np.degrees(direction_angle + theta / (2 * n))

        # Draw the arc on the mask
        center = (int(round(start_x)), int(round(start_y)))
        axes = (int(round(radius)), int(round(radius)))
        cv2.ellipse(mask, center, axes, 0, angle_start, angle_end, 255, -1)

    # Update the map based on the mask

    topdownmap[mask == 255] = [255, 165, 0]  # Set non-mask areas to white

    return topdownmap
def calculate_overlap(mask1, mask2):
    """
    Calculate the overlap rate (IoU) between two binary masks.

    Args:
        mask1: First binary mask (numpy array, 0 or 255).
        mask2: Second binary mask (numpy array, 0 or 255).

    Returns:
        IoU: Intersection over Union as a float value.
    """
    # Ensure masks are binary
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)

    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Avoid division by zero
    if union == 0:
        return 0.0

    iou = intersection / union
    return iou, union

def extract_color_mask(image, target_color=[255, 165, 0], tolerance=5):
    """
    Extract a mask for a specific color from the image.

    Args:
        image: Input image (H, W, C).
        target_color: Target color as a tuple (B, G, R).
        tolerance: Allowed tolerance for color matching.

    Returns:
        mask: Binary mask where 255 indicates the target color.
    """

    mask = (
        (image[:, :, 0] == target_color[0]) &
        (image[:, :, 1] == target_color[1]) &
        (image[:, :, 2] == target_color[2])
    )

    # Convert boolean mask to binary mask (0 or 255)
    mask = (mask * 255).astype(np.uint8)

    return mask

def is_bbox_intersect(line_bbox, mask_bbox):
    """
    Check if two bounding boxes intersect.

    Args:
        line_bbox: Bounding box of the line segment [x_min, y_min, x_max, y_max].
        mask_bbox: Bounding box of the mask [x_min, y_min, x_max, y_max].

    Returns:
        True if the bounding boxes intersect, False otherwise.
    """
    return not (
        line_bbox[2] < mask_bbox[0] or  # Line bbox is completely to the left of mask bbox
        line_bbox[0] > mask_bbox[2] or  # Line bbox is completely to the right of mask bbox
        line_bbox[3] < mask_bbox[1] or  # Line bbox is completely above mask bbox
        line_bbox[1] > mask_bbox[3]     # Line bbox is completely below mask bbox
    )

def count_path_mask_intersections(mask, explored_pixel_path):
    """
    Count the number of intersections between the path and the mask with bounding box optimization.

    Args:
        mask: 2D binary mask where 255 indicates valid regions.
        explored_pixel_path: List of points [(x1, y1), (x2, y2), ...] representing the path.

    Returns:
        Total number of intersections between the path and the mask.
    """
    # Calculate the bounding box of the mask
    x, y, w, h = cv2.boundingRect(mask)
    mask_bbox = [x, y, x + w, y + h]

    total_intersections = 0

    for i in range(len(explored_pixel_path) - 1):
        start = explored_pixel_path[i]
        end = explored_pixel_path[i + 1]

        # Calculate the bounding box of the line segment
        line_bbox = [
            min(start[0], end[0]),
            min(start[1], end[1]),
            max(start[0], end[0]),
            max(start[1], end[1])
        ]

        # Skip line segments whose bounding boxes do not intersect the mask bbox
        if not is_bbox_intersect(line_bbox, mask_bbox):
            continue

        # Line drawing to get pixels along the segment
        line_pixels = cv2.line(np.zeros_like(mask, dtype=np.uint8), start, end, 255, 1)
        intersection = np.logical_and(line_pixels > 0, mask > 0)

        # Count intersections
        if np.any(intersection):
            total_intersections += 1

    return total_intersections

def is_pixel_in_mask(mask, point):
    """
    Check if a specific pixel in a mask has a value of 255.

    Args:
        mask: Input binary mask (numpy array).
        point: Tuple (x, y) representing the pixel coordinates.

    Returns:
        True if the pixel value is 255, False otherwise.
    """
    x, y = point

    # Ensure the point is within the mask's bounds
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
        return mask[y, x] == 255
    else:
        return False

def draw_directional_line(image, start, rotation_vec, length, color=(255, 0, 255, 255), thickness=5):
    """
    Draw a directional line on the image based on a rotation vector.

    Args:
        image: Input image (H, W, C).
        start: Tuple (x, y) for the starting point.
        rotation_vec: Tuple (dx, dy) for the direction vector.
        length: Length of the line.
        color: Color of the line (B, G, R, A) or (B, G, R).
        thickness: Thickness of the line.

    Returns:
        Image with the directional line drawn.
    """
    # Normalize the rotation vector
    dx, dy = rotation_vec
    magnitude = np.sqrt(dx**2 + dy**2)
    dx, dy = dx / magnitude, dy / magnitude

    # Compute the end point
    end = (int(start[0] + dx * length), int(start[1] + dy * length))

    # Draw the line
    cv2.line(image, start, end, color, thickness)

    return image


def local_to_global(position, orientation, local_point):
    """
    Transforms a local coordinate point to global coordinates based on position and quaternion orientation.

    Args:
        position (np.ndarray): The global position.
        orientation (quaternion.quaternion): The quaternion representing the rotation.
        local_point (np.ndarray): The point in local coordinates.

    Returns:
        np.ndarray: Transformed global coordinates.
    """
    rotated_point = quaternion.rotate_vectors(orientation, local_point)
    global_point = rotated_point + position
    return global_point


def global_to_local(position, orientation, global_point):
    """
    Transforms a global coordinate point to local coordinates based on position and quaternion orientation.

    Args:
        position (np.ndarray): The global position.
        orientation (quaternion.quaternion): The quaternion representing the rotation.
        global_point (np.ndarray): The point in global coordinates.

    Returns:
        np.ndarray: Transformed local coordinates.
    """
    translated_point = global_point - position
    inverse_orientation = np.quaternion.conj(orientation)
    local_point = quaternion.rotate_vectors(inverse_orientation, translated_point)
    return local_point


def calculate_focal_length(fov_degrees, image_width):
    """
    Calculates the focal length in pixels based on the field of view and image width.

    Args:
        fov_degrees (float): Field of view in degrees.
        image_width (int): The width of the image in pixels.

    Returns:
        float: The focal length in pixels.
    """
    fov_radians = np.deg2rad(fov_degrees)
    focal_length = (image_width / 2) / np.tan(fov_radians / 2)
    return focal_length


def local_to_image(local_point, resolution, focal_length):
    """
    Converts a local 3D point to image pixel coordinates.

    Args:
        local_point (np.ndarray): The point in local coordinates.
        resolution (tuple): The image resolution as (height, width).
        focal_length (float): The focal length of the camera in pixels.

    Returns:
        tuple: The pixel coordinates (x_pixel, y_pixel).
    """
    point_3d = [local_point[0], -local_point[1], -local_point[2]]  # Inconsistency between Habitat camera frame and classical convention
    if point_3d[2] == 0:
        point_3d[2] = 0.0001
    x = focal_length * point_3d[0] / point_3d[2]
    x_pixel = int(resolution[1] / 2 + x)

    y = focal_length * point_3d[1] / point_3d[2]
    y_pixel = int(resolution[0] / 2 + y)
    return x_pixel, y_pixel


def unproject_2d(x_pixel, y_pixel, depth, resolution, focal_length):
    """
    Unprojects a 2D pixel coordinate back to 3D space given depth information.

    Args:
        x_pixel (int): The x coordinate of the pixel.
        y_pixel (int): The y coordinate of the pixel.
        depth (float): The depth value at the pixel.
        resolution (tuple): The image resolution as (height, width).
        focal_length (float): The focal length of the camera in pixels.

    Returns:
        tuple: The 3D coordinates (x, y, z).
    """
    x = (x_pixel - resolution[1] / 2) * depth / focal_length
    y = (y_pixel - resolution[0] / 2) * depth / focal_length
    return x, -y, -depth


def agent_frame_to_image_coords(point, agent_state, sensor_state, resolution, focal_length):
    """
    Converts a point from agent frame to image coordinates.

    Args:
        point (np.ndarray): The point in agent frame coordinates.
        agent_state (6dof): The agent's state containing position and rotation.
        sensor_state (6dof): The sensor's state containing position and rotation.
        resolution (tuple): The image resolution as (height, width).
        focal_length (float): The focal length of the camera in pixels.

    Returns:
        tuple or None: The image coordinates (x_pixel, y_pixel), or None if the point is behind the camera.
    """
    global_p = local_to_global(agent_state.position, agent_state.rotation, point)
    camera_pt = global_to_local(sensor_state.position, sensor_state.rotation, global_p)
    temp_z = copy.deepcopy(camera_pt[2])
    camera_pt[2] = camera_pt[1]
    camera_pt[1] = temp_z
    
    if camera_pt[2] > 0:
        return None
    return local_to_image(camera_pt, resolution, focal_length)


def put_text_on_image(image, text, location, font=cv2.FONT_HERSHEY_SIMPLEX, text_size=2.7, bg_color=(255, 255, 255), 
                      text_color=(0, 0, 0), text_thickness=3, highlight=True):
    """
    Puts text on an image with optional background highlighting.

    Args:
        image (np.ndarray): The image to draw on.
        text (str): The text to put on the image.
        location (str): Position for the text ('top_left', 'top_right', 'bottom_left', etc.).
        font (int): Font to use for the text.
        text_size (float): Size of the text.
        bg_color (tuple): Background color for the text (BGR).
        text_color (tuple): Color of the text (BGR).
        text_thickness (int): Thickness of the text font.
        highlight (bool): Whether to highlight the text background.

    Returns:
        np.ndarray: Image with text added.
    """
    scale_factor = image.shape[0] / 1080
    adjusted_thickness = math.ceil(scale_factor * text_thickness)
    adjusted_size = scale_factor * text_size

    assert location in ['top_left', 'top_right', 'bottom_left', 'bottom_right', 'top_center', 'center'], \
        "Invalid location. Choose from 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'top_center', 'center'."

    img_height, img_width = image.shape[:2]
    text_size, _ = cv2.getTextSize(text, font, adjusted_size, adjusted_thickness)

    # Calculate text position
    offset = math.ceil(10 * scale_factor)
    text_x, text_y = 0, 0

    if location == 'top_left':
        text_x, text_y = offset, text_size[1] + offset
    elif location == 'top_right':
        text_x, text_y = img_width - text_size[0] - offset, text_size[1] + offset
    elif location == 'bottom_left':
        text_x, text_y = offset, img_height - offset
    elif location == 'bottom_right':
        text_x, text_y = img_width - text_size[0] - offset, img_height - offset
    elif location == 'top_center':
        text_x, text_y = (img_width - text_size[0]) // 2, text_size[1] + offset
    elif location == 'center':
        text_x, text_y = (img_width - text_size[0]) // 2, (img_height + text_size[1]) // 2

    # Draw background rectangle
    if highlight:
        cv2.rectangle(image, (text_x - offset // 2, text_y - text_size[1] - offset), 
                      (text_x + text_size[0] + offset // 2, text_y + offset), bg_color, -1)

    # Add the text
    cv2.putText(image, text, (text_x, text_y), font, adjusted_size, text_color, adjusted_thickness)
    return image

def find_intersections(x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int):
    """
    Find the intersections of a line defined by two points with the image boundaries.
    Args:
        x1 (int): The x-coordinate of the first point.
        y1 (int): The y-coordinate of the first point.
        x2 (int): The x-coordinate of the second point.
        y2 (int): The y-coordinate of the second point.
        img_width (int): The width of the image.
        img_height (int): The height of the image.

    Returns:
        list of tuple or None: A list of two tuples representing the intersection points 
        with the image boundaries, or None if there are not exactly two intersections.
    """
    if x2 != x1:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
    else:
        m = None  # Vertical line
        b = None

    intersections = []
    if m is not None and m != 0:  # Avoid division by zero for horizontal lines
        x_at_yh = int((img_height - b) / m)  # When y = img_height, x = (img_height - b) / m
        if 0 <= x_at_yh <= img_width:
            intersections.append((x_at_yh, img_height - 1))

    if m is not None:
        y_at_x0 = int(b)  # When x = 0, y = b
        if 0 <= y_at_x0 <= img_height:
            intersections.append((0, y_at_x0))

    if m is not None:
        y_at_xw = int(m * img_width + b)  # When x = img_width, y = m * img_width + b
        if 0 <= y_at_xw <= img_height:
            intersections.append((img_width - 1, y_at_xw))

    if m is not None and m != 0:  # Avoid division by zero for horizontal lines
        x_at_y0 = int(-b / m)  # When y = 0, x = -b / m
        if 0 <= x_at_y0 <= img_width:
            intersections.append((x_at_y0, 0))

    if m is None:
        intersections.append((x1, img_height - 1))  # Bottom edge
        intersections.append((x1, 0))  # Top edge

    if len(intersections) == 2:
        return intersections
    return None

def depth_to_height(depth_image, hfov, camera_position, camera_orientation):
    """
    Converts depth image to a height map using camera parameters.

    Args:
        depth_image (np.ndarray): The input depth image.
        hfov (float): Horizontal field of view in degrees.
        camera_position (np.ndarray): The global position of the camera.
        camera_orientation (quaternion.quaternion): The camera's quaternion orientation.

    Returns:
        np.ndarray: Global height map derived from depth image.
    """
    img_height, img_width = depth_image.shape
    focal_length_px = img_width / (2 * np.tan(np.radians(hfov / 2)))

    i_idx, j_idx = np.indices((img_height, img_width))
    x_prime = (j_idx - img_width / 2)
    y_prime = (i_idx - img_height / 2)

    x_local = x_prime * depth_image / focal_length_px
    y_local = y_prime * depth_image / focal_length_px
    z_local = depth_image

    local_points = np.stack((x_local, -y_local, -z_local), axis=-1)
    global_points = local_to_global(camera_position, camera_orientation, local_points)

    return global_points[:, :, 1]  # Return height map

def log_exception(e):
    """Logs an exception with traceback information."""
    tb = traceback.extract_tb(e.__traceback__)
    for frame in tb:
        logging.error(f"Exception in {frame.filename} at line {frame.lineno}")
    logging.error(f"Error: {e}")


def create_gif(image_dir, interval=600):
    """
    Creates a GIF animation from images in the specified directory.

    Args:
        image_dir (str): Path to the directory containing images.
        interval (int): Interval between frames in milliseconds.

    Returns:
        None: Saves the GIF animation in the directory.
    """
    # Create a figure that tightly matches the size of the images (1920x1080)
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    ax.set_position([0, 0, 1, 1])  # Remove all padding
    ax.axis('off')

    frames = []

    # Process up to 80 steps
    for i in range(min(len(os.listdir(image_dir)) - 1, 80)):
        try:
            img = cv2.imread(f"{image_dir}/step{i}/color_sensor.png")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = [ax.imshow(img_rgb, animated=True)]
            frames.append(frame)

            img_copy = cv2.imread(f"{image_dir}/step{i}/color_sensor_chosen.png")
            img_copy_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            frame_copy = [ax.imshow(img_copy_rgb, animated=True)]
            frames.append(frame_copy)

        except Exception as e:
            continue

    # Add a black frame at the end
    black_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    black_frame_rgb = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
    frame_black = [ax.imshow(black_frame_rgb, animated=True)]
    frames.append(frame_black)

    # Create the animation
    ani = animation.ArtistAnimation(fig, frames, interval=interval, blit=True)

    # Save the animation
    ani.save(f'{image_dir}/animation.gif', writer='imagemagick')
    logging.info('GIF animation saved successfully!')

