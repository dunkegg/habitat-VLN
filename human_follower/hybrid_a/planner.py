import heapq
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import magnum as mn
import numpy as np


@dataclass
class Node:
    """A search node in SE2 (x, y, yaw)."""

    x: float
    y: float
    yaw: float  # radians (−pi, pi]
    cost: float  # g(n)
    parent: Optional["Node"]

    # ------------------------------------------------------------------
    # Discrete key for hash‑set (grid index in x, y, yaw)
    # ------------------------------------------------------------------
    def key(self, dx: float, dyaw: float, ox: float, oy: float) -> Tuple[int, int, int]:
        xi = int(round((self.x - ox) / dx))
        yi = int(round((self.y - oy) / dx))  # dx == dy resolution in xy plane
        yawi = int(round(self.yaw / dyaw)) % int(2 * math.pi / dyaw)
        return xi, yi, yawi


class HybridAStar:
    """Absolute‑coordinate Hybrid A* (SE2) planner.

    Attributes
    ----------
    sim : Optional[habitat_sim.Simulator]
        Used for precise navmesh check via pathfinder.is_navigable(). If not
        provided, only occupancy_map filtering is applied.
    occupancy_map : Optional[np.ndarray]
        0/1 grid (0 = obstacle) in world frame. Resolution must match
        xy_resolution. Origin (map_origin) defines its (0,0) grid cell.
    """

    # ---------------------------------------------------------------
    # Initialise planner
    # ---------------------------------------------------------------
    def __init__(
        self,
        *,
        xy_resolution: float = 0.1,  # metres per grid
        yaw_resolution: float = math.radians(5),
        step_length: float = 0.25,  # forward progress each expansion
        # steering_angles: Tuple[float, ...] = (
        #     -0.35,
        #     0.0,
        #     0.35,
        # ),  # rad, left / straight / right
        rear_wheelbase: float = 0.5,  # L in bicycle model
        heuristic_weight: float = 2.0,
        map_origin: Tuple[float, float] = (0.0, 0.0),  # world coords of occ_map (0,0)
        occupancy_map: Optional[np.ndarray] = None,
        sim=None,
        height = None
    ) -> None:
        self.xy_res = xy_resolution
        self.yaw_res = yaw_resolution
        self.step = step_length
        self.height = height
        self.L = rear_wheelbase
        self.w_heur = heuristic_weight

        # world origin of occupancy grid lower‑left corner (x, y)
        self.origin_x, self.origin_y = map_origin
        self.map = occupancy_map  # 0/1 uint8 grid
        self.sim = sim  # habitat simulator (optional)
        R  = 0.5              # 期望最小转弯半径
        n  = 5                # 想要 5 个离散档
        delta_max = math.atan(self.L / R)   # 车辆几何极限
        steering_angles = np.linspace(-delta_max, delta_max, n)    # 
        self.steers = steering_angles
    # ---------------------------------------------------------------
    # Public entry
    # ---------------------------------------------------------------
    def plan(self, start_pose: Tuple[float, float, float], goal_pose: Tuple[float, float, float]) -> Optional[List[Tuple[float, float, float]]]:
        """Plan a path in absolute world coordinates.

        Parameters
        ----------
        start_pose, goal_pose : (x, y, yaw)
            World frame pose. yaw in radians.

        Returns
        -------
        List[(x, y, yaw)] | None
            Sequence including start & goal if success else None.
        """
        self.origin_x= goal_pose[0]
        self.origin_y= goal_pose[1]
        start = Node(*start_pose, cost=0.0, parent=None)
        goal = Node(*goal_pose, cost=0.0, parent=None)

        open_heap: List[Tuple[float, int, Node]] = []
        closed: set = set()
        heapq.heappush(open_heap, (0.0, 0, start))
        node_id = 1

        while open_heap:
            f, _, current = heapq.heappop(open_heap)
            key = current.key(self.xy_res, self.yaw_res, self.origin_x, self.origin_y)
            if key in closed:
                continue
            closed.add(key)

            # goal check
            if (
                math.hypot(goal.x - current.x, goal.y - current.y) < self.xy_res
                and abs(self._norm_ang(goal.yaw - current.yaw)) < self.yaw_res
            ):
                return self._trace_path(current)

            # expansion
            for delta in self.steers:
                child = self._propagate(current, delta)
                if not self._is_valid(child):
                    continue
                ckey = child.key(self.xy_res, self.yaw_res, self.origin_x, self.origin_y)
                if ckey in closed:
                    continue
                h = self._heuristic(child, goal) * self.w_heur
                heapq.heappush(open_heap, (child.cost + h, node_id, child))
                node_id += 1
        return None

    # ---------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------
    def _propagate(self, node: Node, delta: float) -> Node:
        x = node.x + self.step * math.cos(node.yaw)
        y = node.y + self.step * math.sin(node.yaw)
        yaw = node.yaw + self.step / self.L * math.tan(delta)
        return Node(x, y, self._norm_ang(yaw), node.cost + self.step, node)

    @staticmethod
    def _norm_ang(a: float) -> float:
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    def _heuristic(self, n: Node, g: Node) -> float:
        return math.hypot(g.x - n.x, g.y - n.y) + abs(self._norm_ang(g.yaw - n.yaw)) * self.L

    def _is_valid(self, node: Node) -> bool:

        if self.sim is not None:
            world_pt = mn.Vector3(node.x, self.height, node.y)  # 假设 Y-up
            distance = self.pathfinder.distance_to_closest_obstacle(np.array[node.x, self.height, node.y])
            if distance <0.05:
                return False
        return True

    @staticmethod
    def _trace_path(goal_node: Node) -> List[Tuple[float, float, float]]:
        rev: List[Tuple[float, float, float]] = []
        node = goal_node
        while node:
            rev.append((node.x, node.y, node.yaw))
            node = node.parent
        rev.reverse()
        return rev
