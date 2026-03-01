import logging
import subprocess
import sys

from .safe_preflight import parse_preflight_args, run_preflight


def _inject_plugin_discovery_arg(args: list[str]) -> list[str]:
    if any("discover_packages_path" in arg for arg in args):
        return args
    return [*args, "--teleop.discover_packages_path=lerobot_teleoperator_fleximu"]


def _extract_teleop_type(args: list[str]) -> str | None:
    prefix = "--teleop.type="
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def _inject_mujoco_model_arg(args: list[str], mujoco_model_path: str | None) -> list[str]:
    if not mujoco_model_path:
        return args
    if any(arg.startswith("--teleop.model_xml=") for arg in args):
        return args

    teleop_type = _extract_teleop_type(args)
    if teleop_type not in (None, "fleximu_teleop"):
        return args

    return [*args, f"--teleop.model_xml={mujoco_model_path}"]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cli_args = sys.argv[1:]
    preflight_options, passthrough_args = parse_preflight_args(cli_args)
    run_preflight(robot_cli_args=passthrough_args, options=preflight_options)
    passthrough_args = _inject_plugin_discovery_arg(passthrough_args)
    passthrough_args = _inject_mujoco_model_arg(
        passthrough_args,
        preflight_options.mujoco_model_path,
    )

    cmd = ["lerobot-record", *passthrough_args]
    logging.info("Launching official command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
