## Flags
#### Preflight Flags

- `--preflight.enabled=true|false` (default `true`)
- `--preflight.mode=rrt|linear` (unsupported values fallback to linear)
- `--preflight.duration_s=3.0`
- `--preflight.hz=30`
- `--preflight.hold_s=0.3`
- `--preflight.require_enter=true|false` (default `false`)
- `--preflight.keep_torque_on_disconnect=true|false` (default `true`)
- `--preflight.tolerance=2.0`
- `--preflight.strict=true|false` (default `false`)
- `--preflight.rrt_max_iter=2000`
- `--preflight.rrt_step_size=0.1`
- `--preflight.rrt_interp_step=0.08`
- `--mujoco_model_path=...` (required for `rrt`)

#### Teleop Safety Flags

These are plugin teleop parameters. Pass them with `--teleop.<name>=...`.

**Virtual wall Related:** (enforces a minimum end-effector z height).

- `--teleop.enable_virtual_wall=true|false` (default `true`)
- `--teleop.virtual_wall_clearance=0.06` (meter)

**Ground safety filter (GSF) Related:** (automatically raises the end-effector z so the closest tracked geom stays at least `d_safe` above the ground).

- `--teleop.enable_gsf=true|false` (default `true`)
- `--teleop.model_xml=...` (MuJoCo model xml path, which is required for floor collision detection)
- `--teleop.tracked_geom_root_body=wrist` (set empty string to disable root-body filter)
- `--teleop.floor_geom_name=floor`
- `--teleop.d_safe=0.04`
- `--teleop.d_on=0.06`
- `--teleop.d_off=0.07`
- `--teleop.h_max=0.20`
- `--teleop.bisection_iters=18`
- `--teleop.distmax=1.0`
- `--teleop.alpha=0.70`
- `--teleop.track_only_collision_geoms=true|false`
- `--teleop.ik_q123_scale=-1.0`
