__all__ = ["FleximuTeleopConfig", "FleximuTeleop"]


def __getattr__(name: str):
    if name == "FleximuTeleopConfig":
        from .config_fleximu_teleop import FleximuTeleopConfig

        return FleximuTeleopConfig
    if name == "FleximuTeleop":
        from .fleximu_teleop import FleximuTeleop

        return FleximuTeleop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
