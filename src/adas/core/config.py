"""Configuration objects and JSON loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from adas.core.exceptions import ConfigurationError
from adas.core.logger import setup_logger
from adas.core.validation import validate_config_value

logger = setup_logger(__name__)


def _fill(cls: type, data: dict, defaults: Any) -> Any:
    """Build a dataclass from a dict, keeping defaults for missing keys."""
    kwargs = {}
    for f in fields(cls):
        if f.name in data and data[f.name] is not None:
            value = data[f.name]
            default = getattr(defaults, f.name)
            if isinstance(default, bool):
                value = bool(value)
            elif isinstance(default, int) and not isinstance(default, bool):
                value = int(value)
            elif isinstance(default, float):
                value = float(value)
            kwargs[f.name] = value
        else:
            kwargs[f.name] = getattr(defaults, f.name)
    return cls(**kwargs)


@dataclass
class DetectorConfig:
    """Object detection configuration."""

    backend: str = "mock"  # mock | tensorrt
    model_path: str = "models/yolov5n.engine"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.5
    max_detections: int = 100
    input_size: int = 640

    def __post_init__(self) -> None:
        if self.backend not in ("mock", "tensorrt"):
            raise ConfigurationError(f"Invalid detector backend: {self.backend}")
        validate_config_value("confidence_threshold", self.confidence_threshold, 0.0, 1.0)
        validate_config_value("iou_threshold", self.iou_threshold, 0.0, 1.0)
        validate_config_value("max_detections", self.max_detections, 1, 1000)
        validate_config_value("input_size", self.input_size, 32, 2048)


@dataclass
class LaneConfig:
    """Lane estimation configuration."""

    backend: str = "mock"  # mock | ufld
    model_path: str = ""
    input_width: int = 1600
    input_height: int = 320
    crop_ratio: float = 0.6
    num_row: int = 72
    num_col: int = 81

    def __post_init__(self) -> None:
        if self.backend not in ("mock", "ufld"):
            raise ConfigurationError(f"Invalid lane backend: {self.backend}")
        validate_config_value("input_width", self.input_width, 32, 4096)
        validate_config_value("input_height", self.input_height, 32, 4096)
        validate_config_value("crop_ratio", self.crop_ratio, 0.1, 1.0)


@dataclass
class TrackerConfig:
    """Multi-object tracker configuration."""

    max_missed_frames: int = 5
    association_threshold_px: float = 120.0
    focal_length_px: float = 910.0
    object_height_m: float = 1.5
    min_box_height_px: float = 1.0
    max_distance_m: float = 200.0

    def __post_init__(self) -> None:
        validate_config_value("max_missed_frames", self.max_missed_frames, 1, 100)
        validate_config_value("association_threshold_px", self.association_threshold_px, 1.0, 1000.0)
        validate_config_value("focal_length_px", self.focal_length_px, 1.0, 5000.0)
        validate_config_value("object_height_m", self.object_height_m, 0.1, 10.0)


@dataclass
class PlannerConfig:
    """Behavior planner configuration."""

    cruise_speed_mps: float = 15.0  # ~54 km/h
    min_follow_distance_m: float = 12.0
    max_steering_deg: float = 22.0
    time_gap_s: float = 2.0
    max_decel_mps2: float = 3.0
    lane_center_gain: float = 1.0
    ego_lane_half_width_frac: float = 0.0

    def __post_init__(self) -> None:
        validate_config_value("cruise_speed_mps", self.cruise_speed_mps, 0.0, 50.0)
        validate_config_value("min_follow_distance_m", self.min_follow_distance_m, 0.0, 100.0)
        validate_config_value("max_steering_deg", self.max_steering_deg, 0.0, 45.0)
        validate_config_value("time_gap_s", self.time_gap_s, 0.5, 5.0)
        validate_config_value("ego_lane_half_width_frac", self.ego_lane_half_width_frac, 0.0, 1.0)


@dataclass
class ControllerConfig:
    """Controller configuration."""

    kp_speed: float = 0.15
    max_throttle: float = 1.0
    max_brake: float = 1.0
    max_steering_angle_deg: float = 25.0
    steering_deadband_deg: float = 0.5

    def __post_init__(self) -> None:
        validate_config_value("kp_speed", self.kp_speed, 0.0, 10.0)
        validate_config_value("max_throttle", self.max_throttle, 0.0, 1.0)
        validate_config_value("max_brake", self.max_brake, 0.0, 1.0)


@dataclass
class SafetyConfig:
    """Safety limits configuration."""

    max_speed_mps: float = 33.0  # ~120 km/h
    max_acceleration_mps2: float = 3.0
    max_deceleration_mps2: float = 8.0
    max_steering_rate_rad_s: float = 0.5
    max_steering_angle_rad: float = 0.52  # ~30 degrees
    min_following_distance_m: float = 2.0
    max_lateral_offset_m: float = 1.5
    plan_horizon_s: float = 1.0

    def __post_init__(self) -> None:
        validate_config_value("max_speed_mps", self.max_speed_mps, 0.0, 100.0)
        validate_config_value("max_acceleration_mps2", self.max_acceleration_mps2, 0.0, 10.0)
        validate_config_value("max_deceleration_mps2", self.max_deceleration_mps2, 0.0, 15.0)
        validate_config_value("plan_horizon_s", self.plan_horizon_s, 0.05, 5.0)


@dataclass
class SourceConfig:
    """Frame source configuration."""

    type: str = "synthetic"  # synthetic | video | camera
    uri: str = ""
    width: int = 1280
    height: int = 720

    def __post_init__(self) -> None:
        if self.type not in ("synthetic", "video", "camera"):
            raise ConfigurationError(f"Invalid source type: {self.type}")
        validate_config_value("width", self.width, 16, 7680)
        validate_config_value("height", self.height, 16, 4320)


@dataclass
class RuntimeConfig:
    """Complete runtime configuration for ADAS system."""

    detector: DetectorConfig
    tracker: TrackerConfig
    planner: PlannerConfig
    controller: ControllerConfig
    safety: SafetyConfig
    lane: LaneConfig = field(default_factory=LaneConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    fps: int = 20
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        validate_config_value("fps", self.fps, 1, 120)
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ConfigurationError(f"Invalid log level: {self.log_level}")


DEFAULT_CONFIG = RuntimeConfig(
    detector=DetectorConfig(),
    tracker=TrackerConfig(),
    planner=PlannerConfig(),
    controller=ControllerConfig(),
    safety=SafetyConfig(),
    lane=LaneConfig(),
    source=SourceConfig(),
)


def load_config(path: str | Path | None = None) -> RuntimeConfig:
    """Load configuration from JSON file or return defaults."""
    if path is None:
        logger.info("Using default configuration")
        return DEFAULT_CONFIG

    try:
        config_path = Path(path)
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")

        logger.info(f"Loading configuration from {path}")
        payload = json.loads(config_path.read_text())

        config = RuntimeConfig(
            detector=_fill(DetectorConfig, payload.get("detector", {}), DEFAULT_CONFIG.detector),
            tracker=_fill(TrackerConfig, payload.get("tracker", {}), DEFAULT_CONFIG.tracker),
            planner=_fill(PlannerConfig, payload.get("planner", {}), DEFAULT_CONFIG.planner),
            controller=_fill(ControllerConfig, payload.get("controller", {}), DEFAULT_CONFIG.controller),
            safety=_fill(SafetyConfig, payload.get("safety", {}), DEFAULT_CONFIG.safety),
            lane=_fill(LaneConfig, payload.get("lane", {}), DEFAULT_CONFIG.lane),
            source=_fill(SourceConfig, payload.get("source", {}), DEFAULT_CONFIG.source),
            fps=int(payload.get("fps", DEFAULT_CONFIG.fps)),
            log_level=payload.get("log_level", DEFAULT_CONFIG.log_level),
        )
        logger.info("Configuration loaded and validated successfully")
        return config

    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in config file: {e}") from e
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {e}") from e
