import numpy as np
import habitat_sim
import math


def pos_normal_to_habitat(pts):
    # +90 deg around x-axis
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))


def pos_habitat_to_normal(pts):
    # -90 deg around x-axis
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))


def pose_habitat_to_normal(pose):
    # T_normal_cam = T_normal_habitat * T_habitat_cam
    return np.dot(
        np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), pose
    )


def pose_normal_to_tsdf(pose):
    return np.dot(
        pose, np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    )


def pose_normal_to_tsdf_real(pose):
    # This one makes sense, which is making x-forward, y-left, z-up to z-forward, x-right, y-down
    return pose @ np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

from habitat_sim.utils.common import quat_from_angle_axis
import quaternion  # numpy-quaternion
from scipy.spatial.transform import Rotation as R


def rotate_vector_by_quaternion(vector, quat):
    """
    Rotates a 3D vector by a quaternion.

    Args:
        vector (np.ndarray): A 3D vector (x, y, z).
        quat (quaternion.quaternion): The rotation quaternion.

    Returns:
        np.ndarray: The rotated vector (x, y, z).
    """
    # Convert vector to a pure quaternion (0, x, y, z)
    vector_quat = quaternion.quaternion(0, *vector)
    # Apply the rotation: q * v * q^-1
    rotated_quat = quat * vector_quat * quat.conjugate()
    # Extract the vector part of the quaternion
    return np.array([rotated_quat.x, rotated_quat.y, rotated_quat.z])

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    #print(dir(sim_cfg))
    sim_cfg.load_semantic_mesh = False  # 禁用语义网格加载
    sim_cfg.enable_physics = False 
    # agent 半径
    agent_radius = 0.4
    # 初始化 sensor_specs 为一个空列表
    sensor_specs = []
    agent_cfgs = []
    num_sensors = 1  # 设置传感器的数量
    radius = agent_radius  # 设置传感器环绕的半径（相对于agent几何中心, 此时传感器位于agent碰撞模型表面）
    
    agent_cfgs = []
    for index in settings["default_agent"]:
        agent_cfg = habitat_sim.AgentConfiguration()
        agent_cfg.radius = agent_radius  # 设置 agent 的碰撞半径
        agent_cfg.height = 1.5  # 设置 agent 的高度
        #print("index", index)
        for i in range(num_sensors):
            angle = 2 * math.pi * i / num_sensors + math.pi / 2 # 计算每个传感器的角度：等夹角环绕360度
            #angle = 0 时 为正前方摄像头
            
            # RGB传感器配置
            rgb_sensor_spec = habitat_sim.CameraSensorSpec()
            rgb_sensor_spec.uuid = f"color_{index}_{i}"  # 每个传感器的唯一ID
            rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            rgb_sensor_spec.resolution = [settings["height"], settings["width"]]

            # 四元数到欧拉角的转换
            rotation = quat_from_angle_axis(angle - math.pi / 2, np.array([0, 1, 0]))  # 计算四元数
            euler_angles = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).as_euler('xyz', degrees=False)  # 转换为欧拉角
            
            rgb_sensor_spec.position = [
                radius * math.cos(angle),  # x坐标
                settings["sensor_height"],  # y坐标（高度）
                - radius * math.sin(angle)  # z坐标
            ]
            #print(f"position of sensor {i}:", rgb_sensor_spec.position)
            rgb_sensor_spec.orientation = euler_angles  # 设置欧拉角
            rgb_sensor_spec.hfov = settings["hfov"]
            sensor_specs.append(rgb_sensor_spec)
            
            # # 深度传感器配置
            # depth_sensor_spec = habitat_sim.CameraSensorSpec()
            # depth_sensor_spec.uuid = f"depth_{index}_{i}"  # 每个传感器的唯一ID
            # depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            # depth_sensor_spec.resolution = [settings["height"], settings["width"]]
            # depth_sensor_spec.position = [
            #     radius * math.cos(angle),  # x坐标
            #     settings["sensor_height"],  # y坐标（高度）
            #     - radius * math.sin(angle)   # z坐标
            # ]
            # depth_sensor_spec.orientation = euler_angles  # 设置欧拉角
            # depth_sensor_spec.hfov = settings["hfov"]
            # sensor_specs.append(depth_sensor_spec)

        agent_cfg.sensor_specifications = sensor_specs
        # agent_cfg.action_space = {
        #     "move_forward": habitat_sim.agent.ActionSpec(
        #         "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        #     ),
        #     "turn_left": habitat_sim.agent.ActionSpec(
        #         "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        #     ),
        #     "turn_right": habitat_sim.agent.ActionSpec(
        #         "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        #     ),
        # }
        agent_cfgs.append(agent_cfg)
        
    return habitat_sim.Configuration(sim_cfg, agent_cfgs)


def merge_pointcloud(
    pts_orig, pts_new, clip_grads=None, new_clip_grads=None, threshold=1e-2
):
    """Merge two pointclouds, do not add if already exists. Add clip grads if provided.
    Args:
        pts_orig: Nx3 float array of 3D points
        pts_new: Mx3 float array of 3D points
        clip_grads: NxK float array of clip grads
        new_clip_grads: MxK float array of clip grads
    Returns:
        pts_orig: Nx3 float array of merged 3D points
        clip_grads: NxK float array of merged clip grads
    """
    pts_orig = np.vstack((pts_orig, pts_new))
    # merge points that are too close
    close_point_sets = []
    visited = np.zeros(len(pts_orig), dtype=bool)
    for i in range(len(pts_orig)):
        if not visited[i]:
            close_points = np.linalg.norm(pts_orig - pts_orig[i], axis=1) < threshold
            visited[close_points] = True
            close_point_sets.append(np.where(close_points)[0].tolist())

    # get new point cloud
    pts_orig = np.array(
        [np.mean(pts_orig[point_set], axis=0) for point_set in close_point_sets]
    )

    # add clip grads, also take average
    if clip_grads is not None:
        clip_grads = np.vstack((clip_grads, new_clip_grads))
        clip_grads = np.array(
            [np.mean(clip_grads[point_set], axis=0) for point_set in close_point_sets]
        )
        return pts_orig, clip_grads
    return pts_orig


def transform_pointcloud(xyz_pts, rigid_transform):
    """Apply rigid transformation to 3D pointcloud.
    Args:
        xyz_pts: Nx3 float array of 3D points
        rigid_transform: 3x4 or 4x4 float array defining a rigid transformation (rotation and translation)
    Returns:
        xyz_pts: Nx3 float array of transformed 3D points
    """
    xyz_pts = np.dot(rigid_transform[:3, :3], xyz_pts.T)  # apply rotation
    xyz_pts = xyz_pts + np.tile(
        rigid_transform[:3, 3].reshape(3, 1), (1, xyz_pts.shape[1])
    )  # apply translation
    return xyz_pts.T


def get_pointcloud(depth, hfov, cam_pose=None):
    """Get 3D pointcloud from depth image. Calculate camera intrinsics based on image sizes and hfov."""
    H, W = depth.shape
    hfov = hfov * np.pi / 180  # deg2rad
    # vfov = 2 * np.arctan(np.tan(hfov / 2) * H / W)
    # fx = (1.0 / np.tan(hfov / 2.)) * W / 2.0
    # fy = (1.0 / np.tan(vfov / 2.)) * H / 2.0
    # cx = W // 2
    # cy = H // 2

    # # Project depth into 3D pointcloud in camera coordinates
    # pixel_x, pixel_y = np.meshgrid(np.linspace(0, img_w - 1, img_w),
    #                                np.linspace(0, img_h - 1, img_h))
    # cam_pts_x = ((pixel_x - cx) / fx) * depth
    # cam_pts_y = ((pixel_y - cy) / fy) * depth
    # cam_pts_z = -depth
    # cam_pts = (np.array([cam_pts_x, cam_pts_y,
    #                      cam_pts_z]).transpose(1, 2, 0).reshape(-1, 3))

    K = np.array(
        [
            [1 / np.tan(hfov / 2.0), 0.0, 0.0, 0.0],
            [0.0, 1 / np.tan(hfov / 2.0) * W / H, 0.0, 0.0],
            [0.0, 0.0, 1, 0],
            [0.0, 0.0, 0, 1],
        ]
    )

    # Now get an approximation for the true world coordinates -- see if they make sense
    # [-1, 1] for x and [1, -1] for y as array indexing is y-down while world is y-up
    xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, H))
    depth = depth.reshape(1, W, H)
    xs = xs.reshape(1, W, H)
    ys = ys.reshape(1, W, H)

    # Unproject
    # negate depth as the camera looks along -Z
    xys = np.vstack((xs * depth, ys * depth, -depth, np.ones(depth.shape)))
    xys = xys.reshape(4, -1)
    cam_pts = np.matmul(np.linalg.inv(K), xys)
    cam_pts = cam_pts.T[:, :3]

    # # Transform to world coordinates
    if cam_pose is not None:
        # cam_pts = transform_pointcloud(cam_pts, np.linalg.inv(cam_pose))
        cam_pts = transform_pointcloud(cam_pts, cam_pose)

    # Flip axes?
    cam_pts = np.hstack((cam_pts[:, 0:1], -cam_pts[:, 2:3], cam_pts[:, 1:2]))
    # print(np.min(cam_pts, axis=0), np.max(cam_pts, axis=0))
    return cam_pts


def rgba2rgb(rgba, background=(1, 1, 1)):
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, "RGBA image has 4 channels."
    rgb = np.zeros((row, col, 3), dtype="float32")
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    a = np.asarray(a, dtype="float32")
    R, G, B = background
    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B
    return rgb
