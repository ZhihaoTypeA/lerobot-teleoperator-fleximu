import numpy as np
import mujoco
from .config import JOINT_NAMES, MUJOCO_RANGES


class RRTConnectPlanner:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES
        ]

        self.limits_min = np.array([MUJOCO_RANGES[name][0] for name in JOINT_NAMES], dtype=np.float64)
        self.limits_max = np.array([MUJOCO_RANGES[name][1] for name in JOINT_NAMES], dtype=np.float64)

    def check_collision(self, q: np.ndarray) -> bool:
        old_qpos = self.data.qpos.copy()
        for i, jid in enumerate(self.joint_ids):
            self.data.qpos[jid] = q[i]

        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        is_collision = self.data.ncon > 0

        self.data.qpos[:] = old_qpos
        return bool(is_collision)

    def plan(
        self,
        start_q: np.ndarray,
        goal_q: np.ndarray,
        max_iter: int = 2000,
        step_size: float = 0.1,
    ) -> list[np.ndarray] | None:
        print(f"[Planner] Start planning: {np.round(start_q, 2)} -> {np.round(goal_q, 2)}")
        if self.check_collision(start_q):
            print("[Planner] Warning: Start position is in collision!")
            return None
        if self.check_collision(goal_q):
            print("[Planner] Warning: Goal position is in collision!")
            return None

        tree_start = [(start_q.copy(), -1)]
        tree_goal = [(goal_q.copy(), -1)]

        for i in range(max_iter):
            if np.random.rand() < 0.1:
                rand_q = goal_q if i % 2 == 0 else start_q
            else:
                rand_q = np.random.uniform(self.limits_min, self.limits_max)

            idx_near_s, q_new_s = self._extend(tree_start, rand_q, step_size)
            if q_new_s is not None:
                idx_near_g, q_connect = self._connect(tree_goal, q_new_s, step_size)
                if q_connect is not None:
                    if i % 2 == 0:
                        return self._construct_path(tree_start, tree_goal, idx_near_s, idx_near_g)
                    return self._construct_path(tree_goal, tree_start, idx_near_g, idx_near_s)

            tree_start, tree_goal = tree_goal, tree_start
        
        print("[Planner] Timeout: No path found.")
        return None

    def _extend(
        self,
        tree: list[tuple[np.ndarray, int]],
        target_q: np.ndarray,
        step_size: float,
    ) -> tuple[int | None, np.ndarray | None]:
        distances = [np.linalg.norm(node[0] - target_q) for node in tree]
        nearest_idx = int(np.argmin(distances))
        nearest_q = tree[nearest_idx][0]

        diff = target_q - nearest_q
        dist = float(np.linalg.norm(diff))
        if dist < 1e-6:
            return None, None

        scale = min(step_size, dist) / dist
        new_q = nearest_q + diff * scale

        if not self.check_collision(new_q):
            tree.append((new_q, nearest_idx))
            return len(tree) - 1, new_q
        return None, None

    def _connect(
        self,
        tree: list[tuple[np.ndarray, int]],
        target_q: np.ndarray,
        step_size: float,
    ) -> tuple[int | None, np.ndarray | None]:
        distances = [np.linalg.norm(node[0] - target_q) for node in tree]
        nearest_idx = int(np.argmin(distances))
        curr_q = tree[nearest_idx][0]

        while True:
            diff = target_q - curr_q
            dist = float(np.linalg.norm(diff))
            if dist < step_size:
                new_q = target_q
            else:
                new_q = curr_q + (diff / dist) * step_size

            if self.check_collision(new_q):
                return None, None

            tree.append((new_q, nearest_idx))
            nearest_idx = len(tree) - 1
            curr_q = new_q

            if np.linalg.norm(curr_q - target_q) < 1e-6:
                return nearest_idx, curr_q

    def _construct_path(
        self,
        tree_a: list[tuple[np.ndarray, int]],
        tree_b: list[tuple[np.ndarray, int]],
        idx_a: int,
        idx_b: int,
    ) -> list[np.ndarray]:
        path_a = []
        curr = idx_a
        while curr != -1:
            path_a.append(tree_a[curr][0])
            curr = tree_a[curr][1]
        path_a.reverse()

        path_b = []
        curr = idx_b
        while curr != -1:
            path_b.append(tree_b[curr][0])
            curr = tree_b[curr][1]

        return path_a + path_b

