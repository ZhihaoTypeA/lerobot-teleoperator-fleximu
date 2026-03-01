import argparse
import copy
import importlib
import logging
import pkgutil
import time
from dataclasses import dataclass
import draccus
import numpy as np
from pathlib import Path

from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.utils.robot_utils import precise_sleep

from .common.config import JOINT_NAMES, MUJOCO_RANGES, MUJOCO_RANGES_EXPECTED, ZERO_POSE
from .common.planner import RRTConnectPlanner

LOGGER = logging.getLogger(__name__)

_ARM_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def _joint_key(name: str) -> str:
    return f"{name}.pos"


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _scale_to(value: float, src_min: float, src_max: float, exp_min: float, exp_max: float) -> float:
    if src_min == src_max:
        return float(exp_min)
    ratio = (value - src_min) / (src_max - src_min)
    return float(exp_min + ratio * (exp_max - exp_min))


def _zero_pose_action(use_degrees: bool) -> dict[str, float]:
    action: dict[str, float] = {}
    for idx, joint_name in enumerate(_ARM_NAMES):
        q = float(ZERO_POSE[idx])
        if use_degrees:
            action[_joint_key(joint_name)] = float(np.rad2deg(q))
        else:
            j_min, j_max = MUJOCO_RANGES[joint_name]
            action[_joint_key(joint_name)] = _scale_to(q, j_min, j_max, -100.0, 100.0)

    g_min, g_max = MUJOCO_RANGES["gripper"]
    action[_joint_key("gripper")] = _scale_to(float(ZERO_POSE[5]), g_min, g_max, 0.0, 100.0)
    return action


def _action_to_q_rad(action: dict[str, float], use_degrees: bool) -> np.ndarray:
    q = np.zeros(len(JOINT_NAMES), dtype=np.float64)

    for i, joint_name in enumerate(_ARM_NAMES):
        key = _joint_key(joint_name)
        if key not in action:
            raise KeyError(f"Missing joint in action for planner mode: {key}")
        val = float(action[key])
        if use_degrees:
            q[i] = float(np.deg2rad(val))
        else:
            j_min, j_max = MUJOCO_RANGES[joint_name]
            q[i] = _scale_to(val, -100.0, 100.0, j_min, j_max)
        q[i] = float(np.clip(q[i], MUJOCO_RANGES[joint_name][0], MUJOCO_RANGES[joint_name][1]))

    g_key = _joint_key("gripper")
    if g_key not in action:
        raise KeyError(f"Missing joint in action for planner mode: {g_key}")
    g_min, g_max = MUJOCO_RANGES["gripper"]
    q[5] = _scale_to(float(action[g_key]), 0.0, 100.0, g_min, g_max)
    q[5] = float(np.clip(q[5], g_min, g_max))
    return q


def _q_rad_to_action(q: np.ndarray, use_degrees: bool) -> dict[str, float]:
    action: dict[str, float] = {}
    for i, joint_name in enumerate(_ARM_NAMES):
        q_i = float(np.clip(q[i], MUJOCO_RANGES[joint_name][0], MUJOCO_RANGES[joint_name][1]))
        if use_degrees:
            action[_joint_key(joint_name)] = float(np.rad2deg(q_i))
        else:
            j_min, j_max = MUJOCO_RANGES[joint_name]
            action[_joint_key(joint_name)] = _scale_to(q_i, j_min, j_max, -100.0, 100.0)

    g_min, g_max = MUJOCO_RANGES["gripper"]
    g_i = float(np.clip(q[5], g_min, g_max))
    action[_joint_key("gripper")] = _scale_to(g_i, g_min, g_max, 0.0, 100.0)
    return action


def _default_mujoco_model_path() -> str | None:
    candidates = [
        Path.cwd() / "SO101" / "scene.xml",
        Path(__file__).resolve().parents[2] / "SO101" / "scene.xml",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


@dataclass
class RobotOnlyConfig:
    robot: RobotConfig


@dataclass
class PreflightOptions:
    enabled: bool = True
    mode: str = "rrt"
    duration_s: float = 3.0
    hz: int = 30
    tolerance: float = 2.0
    require_enter: bool = False
    hold_s: float = 0.3
    keep_torque_on_disconnect: bool = True
    strict: bool = False
    mujoco_model_path: str | None = None
    rrt_max_iter: int = 2000
    rrt_step_size: float = 0.1
    rrt_interp_step: float = 0.08


def parse_preflight_args(args: list[str]) -> tuple[PreflightOptions, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preflight.enabled", dest="enabled", default="true")
    parser.add_argument("--preflight.mode", dest="mode", default="rrt")
    parser.add_argument("--preflight.duration_s", dest="duration_s", default="5.0")
    parser.add_argument("--preflight.hz", dest="hz", default="30")
    parser.add_argument("--preflight.tolerance", dest="tolerance", default="2.0")
    parser.add_argument("--preflight.require_enter", dest="require_enter", default="false")
    parser.add_argument("--preflight.hold_s", dest="hold_s", default="0.3")
    parser.add_argument("--preflight.keep_torque_on_disconnect", dest="keep_torque_on_disconnect", default="true")
    parser.add_argument("--preflight.strict", dest="strict", default="false")
    parser.add_argument("--mujoco_model_path", dest="mujoco_model_path", default=None)
    parser.add_argument("--preflight.rrt_max_iter", dest="rrt_max_iter", default="2000")
    parser.add_argument("--preflight.rrt_step_size", dest="rrt_step_size", default="0.1")
    parser.add_argument("--preflight.rrt_interp_step", dest="rrt_interp_step", default="0.05")
    known, remaining = parser.parse_known_args(args)

    mujoco_model_path = known.mujoco_model_path
    if mujoco_model_path is None:
        mujoco_model_path = _default_mujoco_model_path()

    options = PreflightOptions(
        enabled=_parse_bool(known.enabled),
        mode=str(known.mode).strip().lower(),
        duration_s=float(known.duration_s),
        hz=int(known.hz),
        tolerance=float(known.tolerance),
        require_enter=_parse_bool(known.require_enter),
        hold_s=float(known.hold_s),
        keep_torque_on_disconnect=_parse_bool(known.keep_torque_on_disconnect),
        strict=_parse_bool(known.strict),
        mujoco_model_path=mujoco_model_path,
        rrt_max_iter=int(known.rrt_max_iter),
        rrt_step_size=float(known.rrt_step_size),
        rrt_interp_step=float(known.rrt_interp_step),
    )
    return options, remaining


def _extract_robot_args(args: list[str]) -> list[str]:
    robot_args: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--robot.discover_packages_path":
            skip_next = True
            continue
        if arg.startswith("--robot.discover_packages_path="):
            continue
        if arg.startswith("--robot."):
            robot_args.append(arg)
    return robot_args


def _extract_kv_values(args: list[str], key: str) -> list[str]:
    values: list[str] = []
    prefix = f"--{key}="
    exact = f"--{key}"
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg.startswith(prefix):
            values.append(arg[len(prefix):])
        elif arg == exact and idx + 1 < len(args):
            values.append(args[idx + 1])
            idx += 1
        idx += 1
    return values


def _extract_robot_type(args: list[str]) -> str | None:
    values = _extract_kv_values(args, "robot.type")
    if not values:
        return None
    value = values[-1].strip()
    return value or None


def _extract_discover_packages(args: list[str]) -> list[str]:
    values = _extract_kv_values(args, "robot.discover_packages_path")
    modules: list[str] = []
    for raw in values:
        chunks = raw.replace(";", ",").split(",")
        for chunk in chunks:
            name = chunk.strip()
            if name:
                modules.append(name)
    return modules


def _import_optional_module(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except Exception:
        return


def _import_package_tree(package_name: str) -> None:
    try:
        package = importlib.import_module(package_name)
    except Exception:
        return

    package_path = getattr(package, "__path__", None)
    if not package_path:
        return

    for _, module_name, _ in pkgutil.walk_packages(package_path, prefix=f"{package_name}."):
        _import_optional_module(module_name)


def _ensure_robot_registry(args: list[str]) -> None:
    # Load any user-discovered robot packages first (if provided).
    for module_name in _extract_discover_packages(args):
        _import_optional_module(module_name)

    # Then best-effort load built-in robot modules so RobotConfig subclasses are registered.
    _import_package_tree("lerobot.robots")
    _import_package_tree("lerobot.common.robot_devices.robots")

    robot_type = _extract_robot_type(args)
    if robot_type is None:
        return

    # Last-chance targeted imports for common module naming patterns.
    candidate_modules = [
        f"lerobot.robots.{robot_type}",
        f"lerobot.robots.config_{robot_type}",
        f"lerobot.common.robot_devices.robots.{robot_type}",
        f"lerobot.common.robot_devices.robots.config_{robot_type}",
    ]
    for module_name in candidate_modules:
        _import_optional_module(module_name)


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _get_current_action(robot, keys: list[str]) -> dict[str, float]:
    obs = robot.get_observation()
    missing = [k for k in keys if k not in obs]
    if missing:
        raise KeyError(f"Missing observation keys for preflight: {missing}")
    return {k: float(obs[k]) for k in keys}


def _send_linear_trajectory(
    robot,
    start: dict[str, float],
    goal: dict[str, float],
    duration_s: float,
    hz: int,
) -> None:
    step_count = max(1, int(max(duration_s, 0.0) * max(hz, 1)))
    period_s = 1.0 / max(hz, 1)
    for i in range(1, step_count + 1):
        alpha = i / step_count
        blend = _smoothstep(alpha)
        action = {key: float(start[key] + (goal[key] - start[key]) * blend) for key in start}
        t0 = time.perf_counter()
        robot.send_action(action)
        precise_sleep(max(period_s - (time.perf_counter() - t0), 0.0))


def _densify_path(path_q: list[np.ndarray], max_step: float) -> list[np.ndarray]:
    if not path_q:
        return path_q
    dense = [path_q[0]]
    step = max(max_step, 1e-4)
    for idx in range(1, len(path_q)):
        q0 = path_q[idx - 1]
        q1 = path_q[idx]
        max_delta = float(np.max(np.abs(q1 - q0)))
        n = max(1, int(np.ceil(max_delta / step)))
        for i in range(1, n + 1):
            alpha = i / n
            dense.append(q0 + (q1 - q0) * alpha)
    return dense


def _send_rrt_trajectory(
    robot,
    start_action: dict[str, float],
    goal_action: dict[str, float],
    use_degrees: bool,
    options: PreflightOptions,
) -> bool:
    if options.mujoco_model_path is None:
        raise ValueError(
            "preflight.mode=rrt requires --mujoco_model_path=<path_to_mjcf_or_xml>."
        )

    try:
        import mujoco
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mujoco is required for preflight.mode=rrt. Install it with `pip install mujoco`.") from exc

    q_start = _action_to_q_rad(start_action, use_degrees=use_degrees)
    q_goal = _action_to_q_rad(goal_action, use_degrees=use_degrees)

    model = mujoco.MjModel.from_xml_path(options.mujoco_model_path)
    data = mujoco.MjData(model)
    planner = RRTConnectPlanner(model, data)
    path_q = planner.plan(
        start_q=q_start,
        goal_q=q_goal,
        max_iter=options.rrt_max_iter,
        step_size=options.rrt_step_size,
    )
    if not path_q:
        return False

    path_q = _densify_path(path_q, options.rrt_interp_step)
    period_s = 1.0 / max(options.hz, 1)
    for q in path_q:
        action = _q_rad_to_action(np.asarray(q, dtype=np.float64), use_degrees=use_degrees)
        action = {k: action[k] for k in start_action}
        t0 = time.perf_counter()
        robot.send_action(action)
        precise_sleep(max(period_s - (time.perf_counter() - t0), 0.0))
    return True


def _compute_max_abs_err(current: dict[str, float], target: dict[str, float]) -> float:
    return max(abs(float(current[k]) - float(target[k])) for k in target)


def run_preflight(robot_cli_args: list[str], options: PreflightOptions) -> None:
    if not options.enabled:
        LOGGER.info("Preflight disabled.")
        return

    _ensure_robot_registry(robot_cli_args)
    robot_args = _extract_robot_args(robot_cli_args)
    if not robot_args:
        raise ValueError("No --robot.* arguments found, cannot run preflight.")

    parsed = draccus.parse(config_class=RobotOnlyConfig, args=robot_args)
    robot_config = copy.deepcopy(parsed.robot)
    if hasattr(robot_config, "cameras"):
        robot_config.cameras = {}
    if options.keep_torque_on_disconnect and hasattr(robot_config, "disable_torque_on_disconnect"):
        robot_config.disable_torque_on_disconnect = False

    robot = make_robot_from_config(robot_config)
    goal_action = _zero_pose_action(use_degrees=bool(getattr(robot_config, "use_degrees", False)))
    supported_keys = [k for k in goal_action if k in robot.action_features]
    if not supported_keys:
        raise ValueError(f"Preflight cannot find expected action keys in robot: {list(robot.action_features.keys())}")
    goal_action = {k: goal_action[k] for k in supported_keys}

    LOGGER.info("Preflight connecting robot...")
    robot.connect()
    try:
        if options.require_enter:
            input(
                "Preflight will move robot to ZERO_POSE. "
                "Ensure workspace is clear, then press ENTER to continue..."
            )

        start_action = _get_current_action(robot, keys=supported_keys)
        start_err = _compute_max_abs_err(start_action, goal_action)
        LOGGER.info("Preflight start max error to ZERO_POSE: %.3f", start_err)

        used_linear_fallback = False
        mode = options.mode
        if mode in {"rrt", "planner"}:
            if not all(_joint_key(j) in start_action for j in JOINT_NAMES):
                msg = "RRT mode requires all 6 joints in robot observation/action features."
                if options.strict:
                    raise RuntimeError(msg)
                LOGGER.warning("%s Falling back to linear.", msg)
                used_linear_fallback = True
            else:
                try:
                    LOGGER.info("Preflight planning with RRT mode.")
                    planned = _send_rrt_trajectory(
                        robot=robot,
                        start_action=start_action,
                        goal_action=goal_action,
                        use_degrees=bool(getattr(robot_config, "use_degrees", False)),
                        options=options,
                    )
                    if not planned:
                        msg = "RRT planner failed to find a path."
                        if options.strict:
                            raise RuntimeError(msg)
                        LOGGER.warning("%s Falling back to linear.", msg)
                        used_linear_fallback = True
                except Exception as exc:
                    if options.strict:
                        raise
                    LOGGER.warning("RRT mode error (%s). Falling back to linear.", exc)
                    used_linear_fallback = True
        elif mode != "linear":
            msg = f"Unsupported preflight.mode='{mode}'."
            if options.strict:
                raise ValueError(msg)
            LOGGER.warning("%s Falling back to linear.", msg)
            used_linear_fallback = True

        if mode == "linear" or used_linear_fallback:
            _send_linear_trajectory(
                robot=robot,
                start=start_action,
                goal=goal_action,
                duration_s=options.duration_s,
                hz=options.hz,
            )

        if options.hold_s > 0:
            t_end = time.perf_counter() + options.hold_s
            while time.perf_counter() < t_end:
                robot.send_action(goal_action)
                precise_sleep(1.0 / max(options.hz, 1))

        end_action = _get_current_action(robot, keys=supported_keys)
        end_err = _compute_max_abs_err(end_action, goal_action)
        LOGGER.info("Preflight end max error to ZERO_POSE: %.3f", end_err)
        if options.strict and end_err > options.tolerance:
            raise RuntimeError(
                f"Preflight strict check failed: end_err={end_err:.3f} > tolerance={options.tolerance:.3f}"
            )
    finally:
        robot.disconnect()
        LOGGER.info("Preflight done.")
