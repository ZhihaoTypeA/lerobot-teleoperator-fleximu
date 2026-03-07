# lerobot_teleoperator_fleximu

![Demo](media/teleop_demo.gif)

Teleoperator plugin for LeRobot that maps 3 IMUs + 1 flex sensor signals to SO-101 6-DoF joint actions.
This package also provides preflight wrappers to reduce dangerous startup jumps after IMU/FLEX calibration.

- `fleximu-preflight`: run preflight only
- `fleximu-safe-teleoperate`: run preflight to a fully extended pose , then launch the official `lerobot-teleoperate`
- `fleximu-safe-record`: run preflight to a fully extended pose, then launch the official `lerobot-record`

## Install

```bash
pip install -e .
```

## Usage

#### Preflight Only

```bash
fleximu-preflight \
  --robot.type=so101_follower \
  --robot.port=[your_port] \
  --robot.disable_torque_on_disconnect=false \
  --teleop.type=fleximu_teleop \
  --teleop.id=fleximu_controller \
  --mujoco_model_path=path/to/robot_scene.xml \
  --display_data=true \
```

#### Safe Teleoperate

```bash
fleximu-safe-teleoperate \
  --robot.type=so101_follower \
  --robot.port=[your_port] \
  --robot.disable_torque_on_disconnect=false \
  --teleop.type=fleximu_teleop \
  --teleop.id=fleximu_controller \
  --mujoco_model_path=path/to/robot_scene.xml \
  --display_data=true \
```

#### Safe Record

```bash
fleximu-safe-record \
  --robot.type=so101_follower \
  --robot.port=[your_port] \
  --robot.disable_torque_on_disconnect=false \
  --teleop.type=fleximu_teleop \
  --teleop.id=fleximu_controller \
  --dataset.repo_id=<user>/<dataset_name> \
  --dataset.num_episodes=10 \
  --dataset.single_task="my task" \
  --display_data=true \
```

## BOM

Hardware BOM is available in [hardware/BOM.md](hardware/BOM.md).

## Notes
- `fleximu-safe-teleoperate` is a wrapper around `lerobot-teleoperate` and `fleximu-safe-record` is a wrapper around `lerobot-record`.
- All unknown arguments are forwarded to the official command as-is.
- Wrapper-only arguments are consumed by wrapper:
  - `--preflight.*`
  - `--mujoco_model_path`
- The wrapper also auto-injects:
  - `--teleop.discover_packages_path=lerobot_teleoperator_fleximu`
  - `--teleop.model_xml=<mujoco_model_path>` (unless you already set `--teleop.model_xml`)
- In preflight, `linear`: smooth interpolation in action space, `rrt`: runs RRT-connect in joint space, then executes densified trajectory (highly recommended).
- In `fleximu-safe-teleoperate` / `fleximu-safe-record`, `mujoco_model_path` is forwarded to `--teleop.model_xml` unless you set `--teleop.model_xml` explicitly.
- `rrt` mode in preflight and `Ground Safety Filter` requires `mujoco` and a valid model XML/MJCF path.

## Demo 
Results of Data Collection and Training with the Teleoperator Fleximu

**ACT**

![ACT Demo](media/ACT_demo.gif)

**SmolVLA** (task: "Pick up the marked white block and place it into the cardboard box")

![SmolVLA Demo](media/SmolVLA_demo.gif)



