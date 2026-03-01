import logging
import sys

from .safe_preflight import parse_preflight_args, run_preflight


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    preflight_options, passthrough_args = parse_preflight_args(sys.argv[1:])
    run_preflight(robot_cli_args=passthrough_args, options=preflight_options)


if __name__ == "__main__":
    main()

