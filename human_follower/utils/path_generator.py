import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.optimize import minimize
import cv2
import matplotlib.pyplot as plt
#from scipy.optimize import differential_evolution #差分进化
from scipy.ndimage import distance_transform_edt

def generate_cost_field_for_floor(pathfinder, agent_location, global_resolution, agent_height, path, tile_size=3, step=2):
    """
    基于轨迹点和 tiling algorithm 思想生成局部代价场，重叠区域使用平均值合并。

    Args:
        pathfinder: Habitat 的导航网格对象。
        path (list): 机器人的路径点，每个点包含 (位置, 四元数)。
        global_resolution (int): 全局代价场的分辨率。
        tile_size (float): tile 的物理大小（世界坐标系下的边长）。
        step (int): 路径点采样步长。

    Returns:
        ndarray: 拼接后的全局代价场。
    """
    bounds = pathfinder.get_bounds()

    # 初始化全局代价场和覆盖计数矩阵
    global_cost_field = np.zeros((global_resolution, global_resolution), dtype=np.float32)
    global_coverage_count = np.zeros((global_resolution, global_resolution), dtype=np.int32)

    # tile 的分辨率
    tile_resolution = int(tile_size / ((bounds[1][0] - bounds[0][0]) / global_resolution))

    for i in range(0, len(path), step):
        center_pos = path[i][0]  # 当前采样点的中心 (x, y, z)

        # 初始化 tile 的障碍物图：0 表示障碍物，1 表示自由空间
        tile_cost_field = np.ones((tile_resolution, tile_resolution), dtype=np.float32)

        # 遍历 tile 的局部坐标系
        for x in range(tile_resolution):
            for y in range(tile_resolution):
                # 计算 tile 内的实际世界坐标
                real_coords = np.array([
                    center_pos[0] - tile_size / 2 + x * tile_size / tile_resolution,
                    center_pos[1],  # 使用当前采样点的高度
                    center_pos[2] - tile_size / 2 + y * tile_size / tile_resolution
                ])

                # 检查是否可导航
                if not pathfinder.is_navigable(real_coords):
                    tile_cost_field[x, y] = 0  # 障碍物标记为 0

        # 计算 tile 的距离场
        distance_field = distance_transform_edt(tile_cost_field)
        #TODO：是否有对数变换的必要？
        distance_field = np.log1p(distance_field)  # 对数变换 log(x+1)
        normalized_tile_cost_field = 1 - cv2.normalize(distance_field, None, 0, 1.0, cv2.NORM_MINMAX)

        # 将 tile_cost_field 映射到全局代价场
        center_x = int((center_pos[0] - bounds[0][0]) / (bounds[1][0] - bounds[0][0]) * global_resolution)
        center_y = int((center_pos[2] - bounds[0][2]) / (bounds[1][2] - bounds[0][2]) * global_resolution)

        # 计算全局代价场中的 tile 范围 （全局索引范围）
        start_x = max(0, center_x - tile_resolution // 2)
        end_x = min(global_resolution, center_x + tile_resolution // 2)
        start_y = max(0, center_y - tile_resolution // 2)
        end_y = min(global_resolution, center_y + tile_resolution // 2)

        # 局部 tile 的实际范围 （tile 内索引范围，受到全局索引范围裁剪的影响）
        local_start_x = max(0, tile_resolution // 2 - center_x)
        local_end_x = local_start_x + (end_x - start_x)
        local_start_y = max(0, tile_resolution // 2 - center_y)
        local_end_y = local_start_y + (end_y - start_y)

        # 合并到全局代价场
        global_cost_field[start_x:end_x, start_y:end_y] += normalized_tile_cost_field[local_start_x:local_end_x, local_start_y:local_end_y]
        global_coverage_count[start_x:end_x, start_y:end_y] += 1

    # 平均重叠区域
    global_coverage_count[global_coverage_count == 0] = 1  # 避免除以 0
    averaged_cost_field = global_cost_field / global_coverage_count

    ## 可视化最终全局代价场
    #visualize_output(
    #    save_path="/root/autodl-fs/combined_cost_field.png",
    #    cost_field=averaged_cost_field,
    #    bounds=bounds
    #)
    #print("Combined cost field saved at:", "/root/autodl-fs/combined_cost_field.png")

    return averaged_cost_field


# 假设 cost_field 是代价场矩阵
def analyze_cost_field(cost_field):
    # 统计数值分布
    min_value = np.min(cost_field)
    max_value = np.max(cost_field)
    mean_value = np.mean(cost_field)
    std_dev = np.std(cost_field)

    print(f"Min Value: {min_value}")
    print(f"Max Value: {max_value}")
    print(f"Mean Value: {mean_value}")
    print(f"Standard Deviation: {std_dev}")

    # 返回分布统计结果
    return {
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "std_dev": std_dev
    }

def optimize_path_with_floor_constraint(path, cost_field, bounds, resolution=1024, learning_rate=0.01, max_iters=5000, tolerance=1e-6):
    """
    使用随机梯度下降优化路径。

    Args:
        path: 初始路径点，列表形式，每个点为 [x, y, z]。
        cost_field: 代价场矩阵。
        bounds: 导航网格的边界范围。
        resolution: 地图分辨率。
        learning_rate: 学习率，控制每次更新步长。
        max_iters: 最大迭代次数。
        tolerance: 梯度变化的停止阈值。

    Returns:
        list: 优化后的路径点。
    """
    def compute_weighted_gradient(cost_field, x_idx, y_idx, resolution, steps=[1, 2, 5, 7, 9], weights=[0.5, 0.5, 0.2, 0.2, 0.1]):
        """
        使用加权多步中心差分法计算梯度。

        Args:
            cost_field (ndarray): 代价场矩阵。
            x_idx (int): X 轴索引。
            y_idx (int): Y 轴索引。
            resolution (int): 地图分辨率。
            steps (list): 多步差分范围。
            weights (list): 每个范围的权重。

        Returns:
            (float, float): dx 和 dy 的梯度值。
        """
        max_x, max_y = cost_field.shape

        # 初始化梯度值
        dx = 0.0
        dy = 0.0
        total_weight = sum(weights)

        for step, weight in zip(steps, weights):
            # 计算 X 方向的加权梯度
            x_minus = max(x_idx - step, 0)
            x_plus = min(x_idx + step, max_x - 1)
            dx += weight * (cost_field[x_plus, y_idx] - cost_field[x_minus, y_idx]) / (2 * step)

            # 计算 Y 方向的加权梯度
            y_minus = max(y_idx - step, 0)
            y_plus = min(y_idx + step, max_y - 1)
            dy += weight * (cost_field[x_idx, y_plus] - cost_field[x_idx, y_minus]) / (2 * step)

        # 归一化权重
        dx /= total_weight
        dy /= total_weight

        return dx, dy

    def map_to_grid(point):
        """将真实坐标转换为代价场网格坐标"""
        x_idx = int((point[0] - bounds[0][0]) / (bounds[1][0] - bounds[0][0]) * resolution)
        y_idx = int((point[2] - bounds[0][2]) / (bounds[1][2] - bounds[0][2]) * resolution)
        x_idx = np.clip(x_idx, 0, resolution - 1)  # 防止越界
        y_idx = np.clip(y_idx, 0, resolution - 1)  # 防止越界
        return x_idx, y_idx

    def compute_gradient(point):
        # 将路径点转换为代价场的网格坐标
        x_idx, y_idx = map_to_grid(point)
        # 调用加权多步中心差分法
        dx, dy = compute_weighted_gradient(cost_field, x_idx, y_idx, resolution, steps=[1, 2, 5, 10], weights=[1, 0.5, 0.2, 0.1])
        # 返回梯度向量，Y 轴保持为 0
        return np.array([dx, 0, dy])


    # 初始化路径点
    optimized_path = np.array([pos.tolist() for pos, _ in path], dtype=np.float32)

    for iteration in range(max_iters):
        total_gradient = 0

        for i, point in enumerate(optimized_path):
            gradient = compute_gradient(point)  # 计算梯度
            total_gradient += np.linalg.norm(gradient)  # 累计梯度范数
            # 更新路径点位置
            optimized_path[i] -= learning_rate * gradient

            # 确保路径点在边界范围内
            optimized_path[i][0] = np.clip(optimized_path[i][0], bounds[0][0], bounds[1][0])
            optimized_path[i][2] = np.clip(optimized_path[i][2], bounds[0][2], bounds[1][2])

        # 判断收敛条件
        if total_gradient < tolerance:
            print(f"Converged after {iteration} iterations.")
            break

    return optimized_path

def generate_path_with_cost_optimization(path, pathfinder, agent_location, visualize=False, resolution=1024, agent_height=1.5):
    """
    基于代价场优化路径，仅限定在当前楼层。

    Args:
        path (list): 初始路径点，每个元素为 (位置, 四元数) 的元组。
        pathfinder: Habitat 的导航网格对象。
        resolution (int): 地图分辨率。
        visualize (bool): 是否可视化路径优化结果。
        agent_height (float): 代理的高度。

    Returns:
        list: 优化后的路径点，每个元素为 (位置, 四元数) 的元组。
    """
    # 获取导航网格边界
    bounds = pathfinder.get_bounds()
    #print(path)

    # 生成特定楼层的代价场
    cost_field = generate_cost_field_for_floor(pathfinder, agent_location, resolution, agent_height, path)

    # 优化路径，仅改变位置
    optimized_positions = optimize_path_with_floor_constraint(
        path, cost_field, bounds, resolution
    )

    #print("agent_location", agent_location)
    if visualize:
        # 可视化
        save_path = f"/root/autodl-fs/loc_{agent_location[0][0]}_{agent_location[0][1]}_{agent_location[0][2]}_final_path.png"
        visualize_output(save_path, bounds, cost_field, path, optimized_positions, agent_location, resolution)
        print(f"Image saved in {save_path}")

    # 组合优化后的路径点与原始四元组
    optimized_path = [(optimized_positions[i], path[i][1]) for i in range(len(path))]

    return optimized_path

def direction_to_combined_quaternion(direction, camera_tilt = 0 * np.pi / 180):
    # 确保 direction 是单位向量
    direction = direction / np.linalg.norm(direction)
    
    # 选择 Z 轴作为参考方向
    reference = np.array([0, 0, 1])
    
    # 计算 direction 和 Z 轴之间的旋转轴和旋转角度
    angle = np.arccos(np.clip(np.dot(reference, direction), -1.0, 1.0)) + np.pi
    rotation_axis = np.cross(reference, direction)
    
    # 如果 rotation_axis 近似为零向量，则表示方向与 Z 轴平行
    if np.linalg.norm(rotation_axis) < 1e-6:
        if np.dot(reference, direction) > 0:
            # 同方向，无需旋转
            quat_y = np.array([0, 0, 0, 1])  # 单位四元数
        else:
            # 反方向，需要 180 度旋转
            quat_y = np.array([1, 0, 0, 0])  # 180 度旋转四元数
    else:
        # 归一化旋转轴
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        # 创建绕 rotation_axis 旋转 angle 的四元数
        quat_y = R.from_rotvec(angle * rotation_axis).as_quat()
    
    # 绕 X 轴旋转 camera_tilt 的四元数
    quat_x = R.from_rotvec(camera_tilt * np.array([1, 0, 0])).as_quat()
    
    # 组合旋转：先应用 X 轴的倾斜，再应用 Y 轴旋转
    combined_quat = R.from_quat(quat_y) * R.from_quat(quat_x)
    
    # 返回结果四元数 [x, y, z, w]
    return combined_quat.as_quat()

def direction_to_quaternion(direction, reference=np.array([0, 0, 1])):
    # 确保方向向量是单位向量
    direction = direction / np.linalg.norm(direction)
    
    # 计算旋转轴
    axis = np.cross(reference, direction)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-6:  # 向量几乎与参考方向平行
        if np.dot(reference, direction) > 0:
            # 同方向，无需旋转
            return np.array([0, 0, 0, 1])  # 单位四元数
        else:
            # 反方向，需要 180 度旋转
            return np.array([1, 0, 0, 0])  # 180 度旋转四元数

    # 归一化旋转轴
    axis = axis / axis_norm
    
    # 计算旋转角度
    angle = np.arccos(np.clip(np.dot(reference, direction), -1.0, 1.0))
    
    # 将旋转轴和角度转换为四元数
    quaternion = R.from_rotvec(axis * angle).as_quat()  # [x, y, z, w] 格式
    return quaternion

def generate_path(path, pathfinder, window_size=10, height_threshold=0.1, max_rotation_angle=15, visualize=False):
    """
    生成高粒度路径，先插值，再代价场优化，最后处理方向并进行旋转插值。

    Args:
        path (list): 初始路径点列表。
        pathfinder: 导航网格对象，用于检查点的可行性。
        window_size (int): 滑动窗口大小，用于计算平均高度。
        height_threshold (float): 高度差阈值。
        max_rotation_angle (float): 最大允许旋转角度。
        visualize (bool): 是否绘制轨迹图片

    Returns:
        list: 经过插值、代价场优化、方向处理和旋转插值后的路径点。
    """
    # 滑动窗口初始化
    height_window = []
    new_path = []
    num_points_between = 5  # 插值点数
    resolution = 1024  # 代价场优化: 分辨率越高优化越精细
    agent_height = 1.5

    # Step 1: 插值路径
    print(f'Path length: {len(path)}')
    for i in range(len(path) - 1):
        start = np.array(path[i])
        end = np.array(path[i + 1])

        # 插值生成中间点
        direction = end - start
        if np.linalg.norm(direction) < 1e-1:
            path[i + 1] = path[i]
            continue
        
        orientation = direction / np.linalg.norm(direction) 
        quaternion = direction_to_combined_quaternion(orientation)
        
        if np.linalg.norm(direction) < 1e-1:
            path[i + 1] = path[i]
            continue

        interpolated_points = [
            start + (end - start) * t / (num_points_between + 1)
            for t in range(num_points_between + 2)
        ]
        for point in interpolated_points:
            new_path.append((point, quaternion))

    # Step 2: 代价场优化
    agent_beginning_location = new_path[0]
    optimized_path = generate_path_with_cost_optimization(
        new_path, pathfinder, agent_beginning_location, False, resolution, agent_height
    )
    #e.g. location (array([ 7.527, -0.536, -0.45 ], dtype=float32), array([-8.578e-06, -9.904e-01,  0.000e+00, -1.385e-01]))
    optimized_path = [
    position[0]  # 丢弃 rotation
    for i, position in enumerate(optimized_path)
    ]
    #print(optimized_path)
    processed_path = []    
    for i in range(len(optimized_path) - 1):
        start = np.array(optimized_path[i])
        end = np.array(optimized_path[i + 1])
    
        # 计算朝向向量（单位向量）
        direction = end - start
        if np.linalg.norm(direction) < 1e-1:
            continue
    
        orientation = direction / np.linalg.norm(direction)
        quaternion = direction_to_combined_quaternion(orientation)
    
        ## 高度过滤逻辑
        #current_height = start[1]  # 高度为 y 坐标
        #if len(height_window) >= window_size:
        #    avg_height = np.mean(height_window)
        #    if current_height > avg_height + height_threshold:
        #        print(f"Skipping point with excessive height: {current_height}")
        #        continue
        #if len(height_window) >= window_size:
        #    height_window.pop(0)
        #height_window.append(current_height)
    
        # 加入路径点
        processed_path.append((start.tolist(), quaternion))
    
    # 添加最后一个点
    processed_path.append((optimized_path[-1], direction_to_combined_quaternion(np.array([0, 0, 1]))))

    ## Step 4: 旋转插值处理
    #smoothed_path = []
    #for i in range(len(processed_path) - 1):
    #    current_pos, current_quat = processed_path[i]
    #    next_pos, next_quat = processed_path[i + 1]
#
    #    # 计算两个四元数之间的旋转角度
    #    current_rot = R.from_quat(current_quat)
    #    next_rot = R.from_quat(next_quat)
    #    relative_rot = current_rot.inv() * next_rot
    #    angle = np.degrees(np.linalg.norm(relative_rot.as_rotvec()))  # 旋转角度
#
    #    # 如果旋转角度大于最大允许值，插入平滑的旋转点
    #    if angle > max_rotation_angle:
    #        num_rot_steps = int(np.ceil(angle / max_rotation_angle))
    #        times = np.linspace(0, 1, num_rot_steps + 1)
    #        slerp = Slerp([0, 1], R.concatenate([current_rot, next_rot]))
    #        interpolated_rots = slerp(times)
#
    #        for interpolated_rot in interpolated_rots:
    #            smoothed_path.append((current_pos, interpolated_rot.as_quat()))
    #    else:
    #        smoothed_path.append((current_pos, current_quat))
#
    ## 添加最后一个点
    #smoothed_path.append(processed_path[-1])
    
    # Step 5: 确保所有点的 rotation 机器人始终保持立起来
    final_path = []
    for pos, quat in processed_path:
        # 获取原始旋转的矩阵表示
        original_rotation = R.from_quat(quat)
        rotation_matrix = original_rotation.as_matrix()

        # 提取当前的 X 和 Z 轴方向
        x_axis = rotation_matrix[:, 0]  # 原始 X 轴
        z_axis = rotation_matrix[:, 2]  # 原始 Z 轴

        # 将 Y 轴设置为世界坐标系的 Y 轴
        y_axis = np.array([0, 1, 0])  # 保证机器人 Y 轴始终铅锤

        # 调整 X 和 Z 轴以保持正交
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)  # 归一化 Z 轴

        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)  # 归一化 X 轴

        # 构造新的旋转矩阵
        adjusted_matrix = np.column_stack((x_axis, y_axis, z_axis))

        # 将修正后的矩阵转换为四元数
        adjusted_rotation = R.from_matrix(adjusted_matrix).as_quat()

        # 保存调整后的路径点
        final_path.append((pos, adjusted_rotation))
        
    # Step 6: 删除距离目标点0.5米以内的点
    target_pos = np.array(final_path[-1][0])  # 获取目标点的位置
    filtered_path = [
        (pos, quat) for pos, quat in final_path
        if np.linalg.norm(np.array(pos) - target_pos) > 0.5 #wzj
    ]
    
    if visualize:
        import os
        import datetime
        
        bounds = pathfinder.get_bounds()
        cost_field = generate_cost_field_for_floor(
            pathfinder,
            agent_beginning_location,
            resolution,
            agent_height,
            new_path
        )

        # 创建输出文件夹
        output_folder = "trajectory_visualizations"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 生成唯一的文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"trajectory_visualization_{timestamp}.png"
        save_path = os.path.join(output_folder, file_name)
        
        visualize_output(
            save_path,
            bounds,
            cost_field,
            initial_path=new_path,
            optimized_path=filtered_path,
            agent_location=agent_beginning_location,
            target_location=target_pos,
            resolution=resolution
        )
        print(f"Visualization saved at: {save_path}")
            
    return filtered_path

def visualize_output(save_path, bounds, cost_field=None, initial_path=None, optimized_path=None, agent_location=None, target_location=None, resolution=1024):
    """
    可视化代价场、初始路径和优化路径。并确保路径点与热力图坐标对齐。

    Args:
        cost_field (ndarray): 代价场矩阵。
        initial_path (list): 初始路径点列表，每个点为 [x, y, z]。
        optimized_path (list): 优化后的路径点列表，每个点为 [x, y, z]。
        agent_location (list): 代理的初始位置 [x, y, z]。
        target_location (list): 目标位置 [x, y, z]。
        resolution (int): 代价场分辨率。
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # 可视化代价场
    if cost_field is not None:
        extent = [bounds[0][0], bounds[1][0], bounds[0][2], bounds[1][2]]  # X 和 Z 的范围
        im = ax.imshow(cost_field.T, cmap="hot_r", origin="lower", extent=extent)
        fig.colorbar(im, ax=ax, label="Cost Field Value")
        ax.set_title("Cost Field Visualization")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")

    # 可视化初始路径
    if initial_path is not None:
        initial_path = np.array([list(pos) for pos, _ in initial_path], dtype=np.float32)
        initial_x = initial_path[:, 0]
        initial_y = initial_path[:, 2]  # Z 轴
        ax.plot(initial_x, initial_y, label="Initial Path", marker="o", color="blue")

    # 可视化优化路径
    if optimized_path is not None:
        optimized_positions = np.array([pos for pos, _ in optimized_path], dtype=np.float32)
        optimized_x = optimized_positions[:, 0]
        optimized_y = optimized_positions[:, 2]  # Z 轴
        ax.plot(optimized_x, optimized_y, label="Optimized Path", marker="x", color="green")

    # 可视化代理位置
    if agent_location is not None:
        agent_x = agent_location[0][0]
        agent_y = agent_location[0][2]
        ax.scatter(agent_x, agent_y, label="Agent Location", color="red", s=100)

    # 可视化目标位置
    if target_location is not None:
        target_x = target_location[0]
        target_y = target_location[2]  # Z 轴
        ax.scatter(target_x, target_y, label="Target Location", color="green", marker="*", s=200)

    ax.legend()

    # 保存图片到指定路径
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

