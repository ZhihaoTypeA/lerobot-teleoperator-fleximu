from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import warnings

import numpy as np
try:
    import mujoco
except Exception:  # pragma: no cover - optional runtime dependency
    mujoco = None


@dataclass
class GroundSafetyFilterConfig:
    """Config for floor-collision shielding."""

    model_xml: str = "SO101/scene.xml"
    tracked_geom_root_body: Optional[str] = "wrist"
    floor_geom_name: str = "floor"
    d_safe: float = 0.04
    d_on: float = 0.06
    d_off: float = 0.07
    h_max: float = 0.20
    bisection_iters: int = 18
    distmax: float = 1.0
    alpha: float = 0.70
    track_only_collision_geoms: bool = True
    ik_q123_scale: float = -1.0


class GroundSafetyFilter:
    """
    Ground-collision safety filter (shield).

    It keeps target x/y and only raises target z by a minimal amount to keep
    tracked geoms above floor with a distance threshold.
    """

    def __init__(self, config: GroundSafetyFilterConfig | None = None) -> None:
        self.config = config if config is not None else GroundSafetyFilterConfig()
        self._validate_config()
        if mujoco is None:
            raise RuntimeError(
                "mujoco is required for GroundSafetyFilter. "
                "Install it with `pip install mujoco` or disable GSF."
            )

        try:
            self.model = mujoco.MjModel.from_xml_path(self.config.model_xml)
            self.data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)
        except Exception as exc:  # pragma: no cover - runtime environment specific
            raise RuntimeError(
                f"Failed to load MuJoCo model from '{self.config.model_xml}': {exc}"
            ) from exc

        self.floor_geom_id = self._resolve_floor_geom_id(self.config.floor_geom_name)
        self.tracked_geom_ids = self._collect_tracked_geom_ids(
            self.config.tracked_geom_root_body
        )
        if not self.tracked_geom_ids:
            raise ValueError(
                "No tracked geoms found. Please check tracked_geom_root_body or "
                "disable collision-only filtering."
            )

        self.tracked_geom_names = [self._geom_name(gid) for gid in self.tracked_geom_ids]
        self.intervention_active = False
        self.dz_exec = 0.0
        self.last_saturated = False
        self.last_warning: Optional[str] = None

    def reset(self, dz_exec: float = 0.0, intervention_active: bool = False) -> None:
        """Reset internal state of hysteresis and low-pass lift."""
        self.dz_exec = float(np.clip(dz_exec, 0.0, self.config.h_max))
        self.intervention_active = bool(intervention_active)
        self.last_saturated = False
        self.last_warning = None

    def evaluate_min_distance(self, qpos_full: Sequence[float]) -> float:
        """
        Evaluate min signed distance between tracked geoms and floor.

        Args:
            qpos_full: Full robot qpos in model joint order (length model.nq).
        """
        qpos = np.asarray(qpos_full, dtype=np.float64).reshape(-1)
        if qpos.size != self.model.nq:
            raise ValueError(
                f"qpos_full length mismatch: expected {self.model.nq}, got {qpos.size}"
            )
        if not np.all(np.isfinite(qpos)):
            raise ValueError("qpos_full contains NaN/Inf.")

        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

        d_min = np.inf
        for geom_id in self.tracked_geom_ids:
            dist = self._geom_distance(geom_id, self.floor_geom_id)
            if not np.isfinite(dist):
                raise FloatingPointError(
                    f"Invalid distance for geom '{self._geom_name(geom_id)}'."
                )
            if dist < d_min:
                d_min = dist
        return float(d_min)

    def compute_safe_target(
        self,
        target_xyz_des: Sequence[float],
        hand_quat: Sequence[float],
        q456_map: Sequence[float],
        solver,
    ) -> Tuple[np.ndarray, float, float, float]:
        """
        Compute safe target with minimal z-lift shielding.

        Returns:
            target_xyz_safe, dz_exec, d_min_nom, d_min_safe
        """
        target_des = self._as_vec(target_xyz_des, 3, "target_xyz_des")
        hand_q = self._as_vec(hand_quat, 4, "hand_quat")
        q456 = np.asarray(q456_map, dtype=np.float64).reshape(-1)

        q_nom = self._solve_full_qpos(target_des, hand_q, q456, solver)
        d_min_nom = self.evaluate_min_distance(q_nom)

        # Hysteresis.
        if self.intervention_active:
            if d_min_nom >= self.config.d_off:
                self.intervention_active = False
        else:
            if d_min_nom <= self.config.d_on:
                self.intervention_active = True

        self.last_saturated = False
        self.last_warning = None

        dz_target = 0.0
        if self.intervention_active:
            dz_target = self._search_min_lift(target_des, hand_q, q456, solver)

        self.dz_exec = self.dz_exec + self.config.alpha * (dz_target - self.dz_exec)
        self.dz_exec = float(np.clip(self.dz_exec, 0.0, self.config.h_max))

        target_safe = target_des.copy()
        target_safe[2] += self.dz_exec

        q_safe = self._solve_full_qpos(target_safe, hand_q, q456, solver)
        d_min_safe = self.evaluate_min_distance(q_safe)
        return target_safe, self.dz_exec, d_min_nom, d_min_safe

    def _search_min_lift(
        self,
        target_xyz_des: np.ndarray,
        hand_quat: np.ndarray,
        q456_map: np.ndarray,
        solver,
    ) -> float:
        """Bisection in [0, h_max] for minimal dz reaching d_safe."""

        def dmin_at(dz: float) -> float:
            target = target_xyz_des.copy()
            target[2] += dz
            try:
                q = self._solve_full_qpos(target, hand_quat, q456_map, solver)
                return self.evaluate_min_distance(q)
            except Exception as exc:
                warnings.warn(
                    f"Failed to evaluate distance at dz={dz:.4f}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return -np.inf

        if dmin_at(0.0) >= self.config.d_safe:
            return 0.0

        d_high = dmin_at(self.config.h_max)
        if d_high < self.config.d_safe:
            self.last_saturated = True
            self.last_warning = (
                f"GroundSafetyFilter saturated: dz=h_max({self.config.h_max:.4f}) "
                f"still gives d_min={d_high:.4f} < d_safe={self.config.d_safe:.4f}"
            )
            warnings.warn(self.last_warning, RuntimeWarning, stacklevel=2)
            return self.config.h_max

        lo = 0.0
        hi = float(self.config.h_max)
        for _ in range(self.config.bisection_iters):
            mid = 0.5 * (lo + hi)
            if dmin_at(mid) >= self.config.d_safe:
                hi = mid
            else:
                lo = mid
        return hi

    def _solve_full_qpos(
        self,
        target_xyz: np.ndarray,
        hand_quat: np.ndarray,
        q456_map: np.ndarray,
        solver,
    ) -> np.ndarray:
        ik_out = np.asarray(solver.robot_ik(target_xyz, hand_quat), dtype=np.float64).reshape(
            -1
        )
        if ik_out.size < 3:
            raise ValueError(f"IK output must have at least 3 entries, got {ik_out.size}.")
        if not np.all(np.isfinite(ik_out[:3])):
            raise FloatingPointError(f"IK q1~q3 contains NaN/Inf: {ik_out[:3]}")

        q123 = ik_out[:3] * float(self.config.ik_q123_scale)
        tail_needed = self.model.nq - 3
        if tail_needed <= 0:
            raise ValueError(f"Unexpected model.nq={self.model.nq}, expected at least 3.")
        if q456_map.size < tail_needed:
            raise ValueError(
                f"q456_map length too short: need {tail_needed}, got {q456_map.size}."
            )
        if not np.all(np.isfinite(q456_map[:tail_needed])):
            raise FloatingPointError("q456_map contains NaN/Inf.")

        qpos = np.concatenate([q123, q456_map[:tail_needed]], axis=0)
        if qpos.size != self.model.nq:
            raise RuntimeError(
                f"Internal qpos size mismatch: expected {self.model.nq}, got {qpos.size}"
            )
        return qpos

    def _resolve_floor_geom_id(self, floor_geom_name: str) -> int:
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name)
        if floor_id != -1:
            return int(floor_id)

        plane_type = int(mujoco.mjtGeom.mjGEOM_PLANE)
        plane_ids = [gid for gid in range(self.model.ngeom) if int(self.model.geom_type[gid]) == plane_type]
        if plane_ids:
            return int(plane_ids[0])

        raise ValueError(
            f"Floor geom '{floor_geom_name}' not found and no plane geom exists in model."
        )

    def _collect_tracked_geom_ids(self, root_body_name: Optional[str]) -> list[int]:
        root_body_id = None
        if root_body_name is not None:
            root_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, root_body_name
            )
            if root_body_id == -1:
                raise ValueError(f"Tracked root body '{root_body_name}' not found in model.")
            root_body_id = int(root_body_id)

        tracked: list[int] = []
        for geom_id in range(self.model.ngeom):
            if geom_id == self.floor_geom_id:
                continue

            geom_body = int(self.model.geom_bodyid[geom_id])
            if root_body_id is not None and not self._is_descendant(geom_body, root_body_id):
                continue

            if self.config.track_only_collision_geoms:
                contype = int(self.model.geom_contype[geom_id])
                conaffinity = int(self.model.geom_conaffinity[geom_id])
                if (contype | conaffinity) == 0:
                    continue

            tracked.append(int(geom_id))
        return tracked

    def _is_descendant(self, body_id: int, root_body_id: int) -> bool:
        current = int(body_id)
        while current != -1:
            if current == root_body_id:
                return True
            if current == 0:
                break
            current = int(self.model.body_parentid[current])
        return root_body_id == 0 and current == 0

    def _geom_distance(self, geom1: int, geom2: int) -> float:
        try:
            fromto = np.zeros(6, dtype=np.float64)
            dist = mujoco.mj_geomDistance(
                self.model,
                self.data,
                int(geom1),
                int(geom2),
                float(self.config.distmax),
                fromto,
            )
            return float(dist)
        except TypeError:
            result = mujoco.mj_geomDistance(
                self.model,
                self.data,
                int(geom1),
                int(geom2),
                float(self.config.distmax),
            )
            if isinstance(result, (tuple, list, np.ndarray)):
                return float(result[0])
            return float(result)

    def _geom_name(self, geom_id: int) -> str:
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        return name if name is not None else f"geom_{geom_id}"

    @staticmethod
    def _as_vec(v: Sequence[float], expected: int, name: str) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64).reshape(-1)
        if arr.size != expected:
            raise ValueError(f"{name} size mismatch: expected {expected}, got {arr.size}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN/Inf.")
        return arr

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.d_on > cfg.d_off:
            raise ValueError(f"d_on ({cfg.d_on}) must be <= d_off ({cfg.d_off}).")
        if cfg.d_safe < 0.0:
            raise ValueError("d_safe must be non-negative.")
        if cfg.h_max < 0.0:
            raise ValueError("h_max must be non-negative.")
        if cfg.bisection_iters <= 0:
            raise ValueError("bisection_iters must be > 0.")
        if cfg.distmax <= 0.0:
            raise ValueError("distmax must be > 0.")
        if not (0.0 <= cfg.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
