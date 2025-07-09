import os
import sys
import numpy as np
import matplotlib.pyplot as plt
#import habitat
#from habitat_sim.utils.common import calculate_euclidean_distance
import habitat_sim
from matplotlib.colors import ListedColormap
import magnum as mn
# 动态设置路径 外部调用时注释掉防止循环引入
#current_dir = os.path.dirname(os.path.abspath(__file__))
#project_root = os.path.join(current_dir, "ON-MLLM")
#sys.path.append(project_root)

#from goat_data_loader import read_yaml, find_scene_path
#from utils.explore.explore_habitat import make_simple_cfg

class FrontierExploration:
    def __init__(self, sim, map_resolution=256):
        """
        这里的 map_resolution 不要过大，256 或更低（128、64）都可以。
        """
        self.sim = sim
        self.map_resolution = map_resolution
        # 简易占据网格，用来做可视化
        self.occupancy_map = np.zeros((map_resolution, map_resolution), dtype=int)
        self.agent_position = None
        self.trail = []  # 世界坐标下，机器人走过的路径

        # 先把整个场景中的可行走区域（navigable area）标记为 1
        self._populate_navigable_area()

    def _populate_navigable_area(self):
        """
        遍历整张 occupancy_map，对每个像素坐标转换成世界坐标；
        如果该点可行走（navigable），则 occupancy_map[y, x] = 1，否则保持 0。
        """
        for y in range(self.map_resolution):
            for x in range(self.map_resolution):
                world_coords = self.pixel_to_world((x, y))
                world_coords = mn.Vector3(*world_coords)
                if self.sim.pathfinder.is_navigable(world_coords):
                    self.occupancy_map[y, x] = 1  # 白色（可导航区域）
                else:
                    self.occupancy_map[y, x] = 0  # 灰色（不可行区域）
                    
    def _calculate_euclidean_distance(self, point1, point2):
        """Calculates the Euclidean distance between two points."""
        return np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))
    
    def _calculate_path_distance(self, start_position, end_position):
        """
        计算从 start_position 到 end_position 的最短路径距离（若无路径则返回 None）
        """
        path_points = self.generate_path(start_position, end_position)
        if path_points is None or len(path_points) < 2:
            return None

        total_dist = 0.0
        for i in range(len(path_points) - 1):
            segment = self._calculate_euclidean_distance(path_points[i], path_points[i + 1])
            total_dist += segment
        return total_dist

    def _calculate_tortuosity(self, path_points):
        """
        计算路径扭曲度 (tortuosity)，定义为 (导航路径长度 / 起终点欧几里得距离)。
        如果 path_points < 2 则返回 None。
        """
        if path_points is None or len(path_points) < 2:
            return None

        # 实际路径长度
        path_len = 0.0
        for i in range(len(path_points) - 1):
            path_len += self._calculate_euclidean_distance(path_points[i], path_points[i + 1])

        # 欧几里得直线距离
        start_p, end_p = path_points[0], path_points[-1]
        straight_dist = self._calculate_euclidean_distance(start_p, end_p)
        if straight_dist < 1e-6:
            # 避免除零
            return None

        return path_len / straight_dist

    def world_to_pixel(self, world_coordinates):
        """Converts world coordinates to pixel coordinates (仅用于可视化)."""
        bounds_min, bounds_max = self.sim.pathfinder.get_bounds()
        map_size_meters = [
            bounds_max[0] - bounds_min[0],
            bounds_max[2] - bounds_min[2]
        ]
        # 这里取 max(...)，保证 x 和 z 方向的分辨率一致，便于画方形图
        meters_per_pixel = max(map_size_meters) / self.map_resolution

        x, _, z = world_coordinates  # Ignore Y-axis
        px = int((x - bounds_min[0]) / meters_per_pixel)
        py = int((z - bounds_min[2]) / meters_per_pixel)
        return px, py

    def pixel_to_world(self, pixel_coordinates):
        """Converts pixel coordinates back to world coordinates (如需要)."""
        bounds_min, bounds_max = self.sim.pathfinder.get_bounds()
        map_size_meters = [
            bounds_max[0] - bounds_min[0],
            bounds_max[2] - bounds_min[2]
        ]
        meters_per_pixel = max(map_size_meters) / self.map_resolution

        px, py = pixel_coordinates
        x = bounds_min[0] + px * meters_per_pixel
        z = bounds_min[2] + py * meters_per_pixel
        return [x, 0, z]

    def sample_frontier_points(self, num_points=5, start_location=None):
        """
        随机采样一些可行走的世界坐标，伪装成“前沿点”。
        这些点需要在导航网格(navigable)上，且与 start_location 处于同一楼层。
        """
        if start_location is None:
            raise ValueError("必须提供起始位置 start_location。")

        start_y = start_location[1]  # 第二个坐标值为高度（y 轴）
        tolerance = 0.5  # 楼层高度的容差

        frontiers = []
        attempts = 0
        max_attempts = num_points * 1000  # 防止极端情况下无限循环

        while len(frontiers) < num_points and attempts < max_attempts:
            rand_point = self.sim.pathfinder.get_random_navigable_point()
            if self.sim.pathfinder.is_navigable(rand_point):
                rand_y = rand_point[1]  # 假设第二个坐标值为高度（y 轴）
                if abs(rand_y - start_y) <= tolerance:
                    frontiers.append(rand_point)
            attempts += 1

        if len(frontiers) < num_points:
            num = len(frontiers)
            print(f"[Warn]: 无法在规定尝试次数内找到足够的可导航点。实际导航点数量有所减少,实际点数：{num}")

        return frontiers

    #def sample_frontier_points(self, center_position, radius=3.0, num_points=5):
    #    """
    #    以当前 agent 的位置 center_position 为圆心，
    #    以 radius 为半径，在圆周上随机采样 num_points 个候选点，
    #    并筛选出可导航(navigable)的点作为前沿点。
    #    """
    #    frontiers = []
#
    #    # 在 [0, 2π) 区间内随机采样 num_points 个角度
    #    angles = np.random.uniform(low=0.0, high=2.0 * np.pi, size=num_points)
#
    #    for angle in angles:
    #        x = center_position[0] + radius * np.cos(angle)
    #        y = center_position[1]  # 沿用中心 Y，不做改动
    #        z = center_position[2] + radius * np.sin(angle)
#
    #        candidate = [x, y, z]
    #        # 检查该点是否在导航网格上（即可行走）
    #        if self.sim.pathfinder.is_navigable(candidate):
    #            frontiers.append(candidate)
#
    #    return frontiers


    def generate_path(self, start_position, end_position):
        """
        生成从 start_position 到 end_position 的最短路径（世界坐标）。
        如果成功，返回路径点；失败则返回 None。
        """
        shortest_path = habitat_sim.ShortestPath()
        shortest_path.requested_start = start_position
        shortest_path.requested_end = end_position

        found_path = self.sim.pathfinder.find_path(shortest_path)
        if found_path and len(shortest_path.points) > 0:
            return shortest_path.points
        else:
            return None

    def update_occupancy_map(self, path_points):
        """
        更新占据图：把路径点标记为 2（黄色）。
        """
        for p in path_points:
            px, py = self.world_to_pixel(p)
            # 判断是否在地图范围内
            if 0 <= px < self.map_resolution and 0 <= py < self.map_resolution:
                self.occupancy_map[py, px] = 2  # 轨迹（黄色）
                
    def explore_until_target(self,
                         start_position,
                         target_position,
                         num_frontiers=3,
                         distance_threshold=3):
        """
        1. 从 start_position 出发，先随机采一些“前沿点”(frontiers)。
        2. 依次导航到这些前沿点，并更新蹊迹。
        3. 如果中途发现离 target 的“导航距离”很近(可选地也可结合扭曲度判断)，就直接去 target。
        4. 最后尝试走向 target。
        """

        # (A) 初始化
        self.agent_position = start_position
        self.trail = [start_position]

        # -- Step 1：若初始时到目标的导航距离已小于阀值，则直接去目标
        initial_dist = self._calculate_path_distance(self.agent_position, target_position)
        if initial_dist is not None and initial_dist < distance_threshold:
            final_path = self.generate_path(self.agent_position, target_position)
            if final_path is not None:
                for p in final_path:
                    self.agent_position = p
                    self.trail.append(p)
                self.update_occupancy_map(final_path)
            return self.trail

        # -- Step 2：采样若干 frontiers
        frontiers = self.sample_frontier_points(num_points=num_frontiers, start_location=start_position)

        # 依次前往这些前沿点
        for frontier in frontiers:
            # 每走一个 frontier 前先检查是否已足够接近目标
            nav_dist_to_target = self._calculate_path_distance(self.agent_position, target_position)
            if nav_dist_to_target is not None and nav_dist_to_target < distance_threshold:
                print("Target is already within navigation distance threshold!")
                break

            # 找到从当前点到前沿点的路径
            path_to_frontier = self.generate_path(self.agent_position, frontier)
            if path_to_frontier is None:
                print("Cannot find path to this frontier. Skipping...")
                continue

            # 沿着此路径行走
            for p in path_to_frontier:
                self.agent_position = p
                self.trail.append(p)

            # 更新可视化用的 occupancy_map（把这条路径标记为蹊迹）
            self.update_occupancy_map(path_to_frontier)

        # -- Step 3：所有 frontiers 走完后，最终尝试从当前位置直接去目标
        final_path = self.generate_path(self.agent_position, target_position)
        if final_path is not None:
            for p in final_path:
                self.agent_position = p
                self.trail.append(p)
            self.update_occupancy_map(final_path)

        # -- Step 4：返回前，根据问题需要进行最后的处理
        for i, point in enumerate(self.trail):
            dist_to_target = self._calculate_path_distance(point, target_position)
            if dist_to_target is not None and dist_to_target < distance_threshold:
                print("A point close to the target found. Adjusting the trail...")
                self.trail = self.trail[:i + 1]  # 保留到该点为止
                path_to_target = self.generate_path(point, target_position)
                if path_to_target is not None:
                    for p in path_to_target:
                        self.trail.append(p)
                break

        # -- Step 5：最后判断是否真正到达目标
        final_dist = self._calculate_path_distance(self.agent_position, target_position)
        if final_dist is not None and final_dist < distance_threshold:
            print("Target reached!")
        else:
            print("Exploration ended without reaching the target.")

        return self.trail


    #def explore_until_target(self,
    #                         start_position,
    #                         target_position,
    #                         num_frontiers=5,
    #                         distance_threshold=5):
    #    """
    #    1. 从 start_position 出发，先随机采一些“前沿点”。
    #    2. 依次导航到这些前沿点，并更新轨迹。
    #    3. 如果中途发现离 target 很近，就直接去 target。
    #    4. 最后走向 target。
    #    """
    #    self.agent_position = start_position
    #    self.trail = [start_position]
#
    #    # 初始化时先判断是否在距离阈值内，如果是则直接前往目标点
    #    if self._calculate_euclidean_distance(self.agent_position, target_position) < distance_threshold:
    #        self.generate_path(self.agent_position, target_position)
    #        if final_path is not None:
    #            for p in final_path:
    #                self.agent_position = p
    #                self.trail.append(p)
    #            self.update_occupancy_map(final_path)
    #        return self.trail
    #    
    #    # 1. 采样前沿点
    #    frontiers = self.sample_frontier_points(num_points=num_frontiers, start_location=start_position)
#
    #    # 2. 依次走前沿点
    #    for frontier in frontiers:
    #        # 检查是否已经靠近目标
    #        if self._calculate_euclidean_distance(self.agent_position, target_position) < distance_threshold:
    #            print("Target is already within distance threshold!")
    #            break
#
    #        path = self.generate_path(self.agent_position, frontier)
    #        if path is None:
    #            print("Cannot find path to this 'frontier'. Skipping...")
    #            continue
#
    #        # 沿着 path 行走
    #        for p in path:
    #            self.agent_position = p
    #            self.trail.append(p)
#
    #        # 更新可视化用的 occupancy_map
    #        self.update_occupancy_map(path)
#
    #    # 3. 最后从当前位置直接去目标点
    #    final_path = self.generate_path(self.agent_position, target_position)
    #    if final_path is not None:
    #        for p in final_path:
    #            self.agent_position = p
    #            self.trail.append(p)
    #        self.update_occupancy_map(final_path)
#
    #    # 判断最终是否到达目标
    #    if self._calculate_euclidean_distance(self.agent_position, target_position) < distance_threshold:
    #        print("Target reached!")
    #    else:
    #        print("Exploration ended without reaching the target.")
#
    #    return self.trail
    
    #def explore_until_target(self,
    #                     start_position,
    #                     target_position,
    #                     radius=3,
    #                     num_points=1,
    #                     distance_threshold=5,
    #                     max_iterations=5,
    #                     max_frontier_retry=100):
    #    """
    #    基于“以当前agent为圆心、给定半径在圆周上采样前沿点”做多次探索
    #    其中，当一次采样不到任何 navigable frontier 时，会尝试重复采样若干次，
    #    超过 max_frontier_retry 后仍无前沿点，才会跳出循环或进行下一步处理。
#
    #    参数：
    #      - start_position, target_position: 起始与目标位置 (x, y, z)
    #      - radius: 在圆周上采样的半径
    #      - num_points: 每次均匀采样多少前沿点
    #      - distance_threshold: 判断是否到达目标的距离阈值
    #      - max_iterations: 整个探索过程最多迭代多少轮
    #      - max_frontier_retry: 当一次采样结果为空时，允许的最大重复采样次数
    #    """
#
    #    def within_threshold(pos1, pos2, thresh):
    #        return self._calculate_euclidean_distance(pos1, pos2) < thresh
#
    #    # 初始化
    #    self.agent_position = start_position
    #    self.trail = [start_position]
#
    #    # Step 1：若目标已在起始点的采样圆内，尝试直接导航
    #    if within_threshold(start_position, target_position, radius):
    #        print("[Info] Target is within the initial sampling radius. Trying to go directly...")
    #        direct_path = self.generate_path(start_position, target_position)
    #        if direct_path is not None:
    #            for p in direct_path:
    #                self.agent_position = p
    #                self.trail.append(p)
    #            self.update_occupancy_map(direct_path)
#
    #            # 判断是否到达
    #            if within_threshold(self.agent_position, target_position, distance_threshold):
    #                print("[Info] Target reached directly!")
    #                return self.trail
    #            else:
    #                print("[Warn] Cannot reach target even though it's in the initial radius.")
    #        else:
    #            print("[Warn] No valid path to target, although target is within initial radius. Continue exploring...")
#
    #    # Step 2：循环多次采样，直到到达目标或超过最大迭代次数
    #    iteration = 0
    #    while iteration < max_iterations:
    #        # 2.1 如果已经在目标附近，则终止
    #        if within_threshold(self.agent_position, target_position, distance_threshold):
    #            print("[Info] Target is already within distance threshold!")
    #            break
#
    #        # 2.2 以当前agent位置为中心，尝试采样前沿点（带重试）
    #        attempt_count = 0
    #        frontiers = []
    #        while len(frontiers) == 0 and attempt_count < max_frontier_retry:
    #            frontiers = self.sample_frontier_points(self.agent_position, radius=radius, num_points=num_points)
    #            attempt_count += 1
#
    #        # 如果反复采样后仍没有任何前沿点，就跳出
    #        if len(frontiers) == 0:
    #            print(f"[Warn] No navigable frontiers found after {attempt_count} attempts. "
    #                  "Attempting direct path to target...")
    #            break  # 离开 while 循环后，会走 Step 3 做最后一次直接尝试
#
    #        # 遍历当前这批 frontiers
    #        for i, frontier in enumerate(frontiers):
    #            # 到达某个前沿点之前，先确认下是否已在目标附近
    #            if within_threshold(self.agent_position, target_position, distance_threshold):
    #                print("[Info] Target is already within distance threshold!")
    #                break
#
    #            path_to_frontier = self.generate_path(self.agent_position, frontier)
    #            if path_to_frontier is None or len(path_to_frontier) == 0:
    #                print(f"[Info] Frontier {i} is not reachable. Skipping...")
    #                continue
#
    #            # 沿着 path 行走
    #            for p in path_to_frontier:
    #                self.agent_position = p
    #                self.trail.append(p)
#
    #            # 更新 occupancy_map
    #            self.update_occupancy_map(path_to_frontier)
#
    #            # 到达该前沿点后，再判断：目标是否落在新的采样圆内
    #            if within_threshold(self.agent_position, target_position, radius):
    #                print("[Info] Target is within the new sampling radius. Trying to go directly...")
    #                path_to_target = self.generate_path(self.agent_position, target_position)
    #                if path_to_target is not None and len(path_to_target) > 0:
    #                    for p in path_to_target:
    #                        self.agent_position = p
    #                        self.trail.append(p)
    #                    self.update_occupancy_map(path_to_target)
    #                # 不管是否成功走到目标，都结束这批 frontiers 的遍历
    #                break
#
    #        iteration += 1
    #        # 进入下一轮 while 循环时，会基于当前 agent_position 再次采样 frontiers
#
    #    # Step 3：离开 while 循环后（无论是 frontiers 耗尽、达到 max_iterations 等），
    #    #         做一次从当前点到目标的最后尝试
    #    if not within_threshold(self.agent_position, target_position, distance_threshold):
    #        print("[Info] Doing final attempt to reach target...")
    #        final_path = self.generate_path(self.agent_position, target_position)
    #        if final_path is not None and len(final_path) > 0:
    #            for p in final_path:
    #                self.agent_position = p
    #                self.trail.append(p)
    #            self.update_occupancy_map(final_path)
#
    #    # 最终检查距离
    #    dist_to_target = self._calculate_euclidean_distance(self.agent_position, target_position)
    #    if dist_to_target < distance_threshold:
    #        print(f"[Info] Target reached! (final distance = {dist_to_target:.2f})")
    #    else:
    #        print(f"[Warn] Exploration ended. Final distance to target: {dist_to_target:.2f} > threshold.")
#
    #    return self.trail


    def visualize(self,
                  save_path="./frontier_exploration.png",
                  start_position=None,
                  target_position=None):
        """
        可视化：使用不同颜色表示不可行区域、可行区域、路径轨迹。
        同时在图中标记起始点与目标点。
        
        occupancy_map 值含义：
        0 -> 灰色(不可行/未探索)
        1 -> 白色(可行区域)
        2 -> 黄色(轨迹)
        """
        plt.figure(figsize=(8, 8))

        # 自定义颜色映射表: 灰 -> 白 -> 黄
        # 如果你想让 0 是黑色，可以改成 "black"；这里只是演示灰色
        colors = ["gray", "white", "yellow"]
        cmap = ListedColormap(colors)

        # 直接使用 occupancy_map 可视化
        plt.imshow(self.occupancy_map, cmap=cmap)
        plt.title("Frontier Exploration - Occupancy Map")
        plt.xlabel("X-axis (pixels)")
        plt.ylabel("Y-axis (pixels)")

        # 如果你想额外叠加一条轨迹线（像素坐标系下）
        if len(self.trail) > 1:
            trail_pixels = np.array([self.world_to_pixel(pos) for pos in self.trail])
            plt.plot(trail_pixels[:, 0],
                     trail_pixels[:, 1],
                     color='orange',
                     linewidth=2,
                     label='Trail')

        # 在图上标记起始点和目标点（如果提供了）
        if start_position is not None:
            sx, sy = self.world_to_pixel(start_position)
            plt.scatter(sx, sy, c='green', marker='o', s=100, label='Start')
        if target_position is not None:
            tx, ty = self.world_to_pixel(target_position)
            plt.scatter(tx, ty, c='red', marker='x', s=100, label='Goal')

        plt.legend()
        plt.savefig(save_path)
        plt.close()
        print(f"Visualization saved to {save_path}")



# 测试程序
if __name__ == "__main__":
    folder = "/data2/zheyu/ON-MLLM/goat_bench_process/goat_bench/hm3d/v1/train/content"
    yaml_file_path = ""

    # 初始化目标文件列表
    target_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), folder)
            target_files.append(relative_path)

    cfg = read_yaml(yaml_file_path)
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    cfg.scenes_data_path = "/data2/zejinw/data/scene_datasets/hm3d/train"

    # 加载场景
    scene = "1S7LAXRdDqK"  # 示例场景 ID
    scene_mesh_dir = find_scene_path(cfg, scene)

    sim_settings = {
        "scene": scene_mesh_dir,
        "default_agent": [0],
        "sensor_height": 1.5,
        "width": 640,
        "height": 480,
        "hfov": 120,
    }
    sim_cfg = make_simple_cfg(sim_settings)
    simulator = habitat_sim.Simulator(sim_cfg)

    # 从 sim_cfg 中获取 agent 配置
    agent_cfg = sim_cfg.agents[0]
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.agent_radius = agent_cfg.radius
    navmesh_settings.agent_height = agent_cfg.height
    navmesh_settings.agent_max_climb = 1
    navmesh_settings.agent_max_slope = 45.0

    navmesh_success = simulator.recompute_navmesh(simulator.pathfinder, navmesh_settings)
    if not navmesh_success or not simulator.pathfinder.is_loaded:
        raise RuntimeError("Navmesh recomputation failed. Cannot proceed with pathfinding.")

    # 初始化探索类
    explorer = FrontierExploration(simulator)

    # 初始位置和目标位置
    start_position = [-1.84784, 0.0416, -1.65592]
    target_position = [-7.78058, 0.51908, -4.08003]
    start_rotation = [0, 0.58772, 0, -0.80907]

    # 模拟路径信息
    agent_path_info = [
        start_position,
        [[start_position]]
    ]

    # 执行探索
    explorer.explore_until_target(start_position=start_position,
                              target_position=target_position,
                              num_frontiers=5)

    # 可视化结果
    explorer.visualize(start_position=start_position, target_position=target_position)
