from dataclasses import dataclass
from typing import Optional

from lerobot.teleoperators.config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("fleximu_teleop")
@dataclass
class FleximuTeleopConfig(TeleoperatorConfig):
    flex_duration: float = 5.0
    warmup_seconds: float = 0.5

    invert_arm: bool = True

    #virtual wall related
    enable_virtual_wall: bool = True
    virtual_wall_clearance: float = 0.06

    #ground_safety_filter related
    enable_gsf: bool = True
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
