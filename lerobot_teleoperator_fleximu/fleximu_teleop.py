import time
from typing import Any

import numpy as np
from lerobot.teleoperators.teleoperator import Teleoperator

from .config_fleximu_teleop import FleximuTeleopConfig
from .common.config import ZERO_POSE, MUJOCO_RANGES, MUJOCO_RANGES_EXPECTED
from .common.filters import OneEuroFilter, DeadbandEMA
from .common.ground_safety_filter import GroundSafetyFilter, GroundSafetyFilterConfig
from .common.kinematics import GeometricSolver
from .common.sensors_node import SensorsNode

_ACTION_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

_ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

def _scale_to(value: float, src_min: float, src_max: float, exp_min: float, exp_max: float) -> float:
    if src_min == src_max:
        return float(exp_min)
    ratio = (value - src_min) / (src_max - src_min)
    return float(exp_min + ratio * (exp_max - exp_min))

def _clip_joint_ranges(joints: np.ndarray) -> np.ndarray:
    clipped = joints.copy()
    for i, j_name in enumerate(_ARM_JOINT_NAMES):
        j_min, j_max = MUJOCO_RANGES[j_name]
        clipped[i] = float(np.clip(clipped[i], j_min, j_max))
    return clipped

def _apply_virtual_wall(target: np.ndarray, ground_z: float = 0.0, clearance: float = 0.06) -> np.ndarray:
    limited = target.copy()
    min_z = ground_z + clearance
    if limited[2] < min_z:
        limited[2] = min_z
    return limited

class FleximuTeleop(Teleoperator):
    config_class = FleximuTeleopConfig
    name = "fleximu_teleop"

    def __init__(self, config: FleximuTeleopConfig):
        super().__init__(config)
        self.config = config
        self._target_filter : OneEuroFilter | None = None
        self._gripper_filter : DeadbandEMA | None = None
        self._gs_filter: GroundSafetyFilter | None = None
        self._sensors : SensorsNode | None = None
        self._solver = GeometricSolver()
        self._first_run = True

    @property
    def action_features(self) -> dict:
        return {k: float for k in _ACTION_KEYS}
    
    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool: 
        return self._sensors is not None

    def connect(self, calibrate: bool = True) -> None:
        if self._sensors is None:
            self._sensors = SensorsNode()

        time.sleep(max(self.config.warmup_seconds, 0.0))

        self._target_filter = OneEuroFilter(
            t0=time.time(),
            x0=self._solver.ua_base_offset.copy(),
            min_cutoff=1.0,
            beta=0.05,
            dcutoff=1.0,
        )
        self._gripper_filter = DeadbandEMA(alpha=0.2, deadband=0.1, x0=ZERO_POSE[5])
        self._ensure_gsf_ready()

        if calibrate and not self.is_calibrated:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self._sensors is not None and self._sensors.is_calibrated

    def calibrate(self) -> None:
        if self._sensors is None:
            self.connect(calibrate=False)
        if self._sensors is None:
            raise RuntimeError("Sensors are not available now.")
        
        self._sensors.calibrate(flex_duration=self.config.flex_duration)
        if self._gs_filter is not None:
            self._gs_filter.reset()
        self._first_run = True

    def configure(self) -> None:
        return None

    def _build_gsf_config(self) -> GroundSafetyFilterConfig:
        tracked_geom_root_body = self.config.tracked_geom_root_body
        if tracked_geom_root_body == "":
            tracked_geom_root_body = None
        return GroundSafetyFilterConfig(
            model_xml=self.config.model_xml,
            tracked_geom_root_body=tracked_geom_root_body,
            floor_geom_name=self.config.floor_geom_name,
            d_safe=self.config.d_safe,
            d_on=self.config.d_on,
            d_off=self.config.d_off,
            h_max=self.config.h_max,
            bisection_iters=self.config.bisection_iters,
            distmax=self.config.distmax,
            alpha=self.config.alpha,
            track_only_collision_geoms=self.config.track_only_collision_geoms,
            ik_q123_scale=self.config.ik_q123_scale,
        )

    def _ensure_gsf_ready(self) -> None:
        if not self.config.enable_gsf:
            self._gs_filter = None
            return
        if self._gs_filter is None:
            self._gs_filter = GroundSafetyFilter(config=self._build_gsf_config())

    def _get_robot_target_and_hand_quat(self) -> tuple[np.ndarray, np.ndarray]:
        q_upper = self._sensors.abs_rotations['upperarm'].as_quat()
        q_lower = self._sensors.abs_rotations['lowerarm'].as_quat()
        q_hand = self._sensors.abs_rotations['hand'].as_quat()

        human_wrist_pos = self._solver.human_fk(q_upper=q_upper, q_lower=q_lower)
        raw_robot_target = human_wrist_pos * self._solver.scale_factor
        raw_robot_target = raw_robot_target + self._solver.ua_base_offset

        if self._target_filter is not None:
            current_time = time.time()
            if self._first_run:
                self._target_filter.x_prev = raw_robot_target
                self._target_filter.t_prev = current_time
                robot_target = raw_robot_target
                self._first_run = False 
            else:
                robot_target = self._target_filter(t=current_time, x=raw_robot_target)
        else:
            robot_target = raw_robot_target

        if self.config.enable_virtual_wall:
            robot_target = _apply_virtual_wall(robot_target, clearance=self.config.virtual_wall_clearance)
        return robot_target, q_hand

    def _solve_arm_commands_rad(self, robot_target: np.ndarray, q_hand: np.ndarray) -> np.ndarray:
        raw_joint_angles = self._solver.robot_ik(robot_target, q_hand)
        if self.config.invert_arm:
            raw_joint_angles = raw_joint_angles * -1.0

        joint_angles = _clip_joint_ranges(raw_joint_angles)
        return joint_angles.astype(np.float64)
        
    def _get_gripper_commands_rad(self) -> float:
        raw_flex_ratio = self._sensors.get_flex_ratio()
        if self._gripper_filter is not None:
            filtered_ratio = self._gripper_filter(raw_flex_ratio)
        else:
            filtered_ratio = raw_flex_ratio
        g_min, g_max = MUJOCO_RANGES_EXPECTED["gripper"]
        gripper_angles = g_min + filtered_ratio * (g_max - g_min)

        return float(np.clip(gripper_angles, g_min, g_max))

    def _get_gsf_commands_rad(
        self,
        robot_target: np.ndarray,
        q_hand: np.ndarray,
        gripper_rad: float,
    ) -> np.ndarray:
        if self._gs_filter is None:
            raise RuntimeError("GroundSafetyFilter is enabled but not initialized.")

        joint_angles_nom = self._solver.robot_ik(robot_target, q_hand)
        if self.config.invert_arm:
            joint_angles_nom = joint_angles_nom * -1.0
        
        q456_map = np.array([joint_angles_nom[3], joint_angles_nom[4], gripper_rad], dtype=np.float64)
        robot_target_safe, _, _, _ = self._gs_filter.compute_safe_target(
            target_xyz_des=robot_target,
            hand_quat=q_hand,
            q456_map=q456_map,
            solver=self._solver
        )
        return self._solve_arm_commands_rad(robot_target_safe, q_hand)

    def _to_output(self, arm_rad: np.ndarray, gripper_rad: float) -> np.ndarray:
        arm_norm = np.zeros(5, dtype=np.float32)
        for i, j_name in enumerate(_ARM_JOINT_NAMES):
            j_min, j_max = MUJOCO_RANGES[j_name]
            arm_norm[i] = _scale_to(float(arm_rad[i]), src_min=j_min, src_max=j_max, exp_min=-100.0, exp_max=100.0)

        g_min, g_max = MUJOCO_RANGES["gripper"]
        gripper_norm = _scale_to(gripper_rad, src_min=g_min, src_max=g_max, exp_min=0.0, exp_max=100.0)

        return np.append(arm_norm, gripper_norm).astype(np.float32)

    def _action_vec_to_dict(self, action:np.ndarray) -> dict[str, Any]:
        return {k: float(v) for k, v in zip(_ACTION_KEYS, action)}

    def get_action(self):
        if self._sensors is None:
            raise RuntimeError("Sensors are not available now.")

        robot_target, q_hand = self._get_robot_target_and_hand_quat()
        gripper_rad = self._get_gripper_commands_rad()
        if self.config.enable_gsf:
            self._ensure_gsf_ready()
            arm_rad = self._get_gsf_commands_rad(
                robot_target=robot_target,
                q_hand=q_hand,
                gripper_rad=gripper_rad,
            )
        else:
            arm_rad = self._solve_arm_commands_rad(robot_target, q_hand)

        action_vec = self._to_output(arm_rad=arm_rad, gripper_rad=gripper_rad)
        
        return self._action_vec_to_dict(action=action_vec)

    def send_feedback(self, feedback: dict[str, Any]) -> dict[str, Any]:
        return feedback

    def disconnect(self) -> None:
        if self._sensors is not None:
            self._sensors.close()
        self._sensors = None
        self._target_filter = None
        self._gripper_filter = None
        self._gs_filter = None
        self._first_run = True
        
    
