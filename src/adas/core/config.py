"""Configuration objects and JSON loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from adas.core.exceptions import ConfigurationError
from adas.core.logger import setup_logger
from adas.core.validation import validate_config_value

logger = setup_logger(__name__)


@dataclass(slots=True)
class DetectorConfig:
    """Object detection configuration."""
    
    model_path: str = "models/yolov8n.onnx"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.5
    max_detections: int = 100
    
    def __post_init__(self) -> None:
        """Validate detector configuration."""
        validate_config_value("confidence_threshold", self.confidence_threshold, 0.0, 1.0)
        validate_config_value("iou_threshold", self.iou_threshold, 0.0, 1.0)
        validate_config_value("max_detections", self.max_detections, 1, 1000)


@dataclass(slots=True)
class TrackerConfig:
    """Multi-object tracker configuration."""
    
    max_missed_frames: int = 5
    association_threshold_px: float = 120.0
    focal_length_px: float = 35.0
    min_box_height_px: float = 1.0
    max_distance_m: float = 200.0
    
    def __post_init__(self) -> None:
        """Validate tracker configuration."""
        validate_config_value("max_missed_frames", self.max_missed_frames, 1, 100)
        validate_config_value("association_threshold_px", self.association_threshold_px, 1.0, 1000.0)
        validate_config_value("focal_length_px", self.focal_length_px, 1.0, 1000.0)


@dataclass(slots=True)
class PlannerConfig:
    """Behavior planner configuration."""
    
    cruise_speed_mps: float = 15.0  # ~54 km/h
    min_follow_distance_m: float = 12.0
    max_steering_deg: float = 22.0
    time_gap_s: float = 2.0
    max_decel_mps2: float = 3.0
    lane_center_gain: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate planner configuration."""
        validate_config_value("cruise_speed_mps", self.cruise_speed_mps, 0.0, 50.0)
        validate_config_value("min_follow_distance_m", self.min_follow_distance_m, 0.0, 100.0)
        validate_config_value("max_steering_deg", self.max_steering_deg, 0.0, 45.0)
        validate_config_value("time_gap_s", self.time_gap_s, 0.5, 5.0)


@dataclass(slots=True)
class ControllerConfig:
    """Controller configuration."""
    
    kp_speed: float = 0.15
    max_throttle: float = 1.0
    max_brake: float = 1.0
    max_steering_angle_deg: float = 25.0
    steering_deadband_deg: float = 0.5
    
    def __post_init__(self) -> None:
        """Validate controller configuration."""
        validate_config_value("kp_speed", self.kp_speed, 0.0, 10.0)
        validate_config_value("max_throttle", self.max_throttle, 0.0, 1.0)
        validate_config_value("max_brake", self.max_brake, 0.0, 1.0)


@dataclass(slots=True)
class SafetyConfig:
    """Safety limits configuration."""
    
    max_speed_mps: float = 33.0  # ~120 km/h
    max_acceleration_mps2: float = 3.0
    max_deceleration_mps2: float = 8.0
    max_steering_rate_rad_s: float = 0.5
    max_steering_angle_rad: float = 0.52  # ~30 degrees
    min_following_distance_m: float = 2.0
    max_lateral_offset_m: float = 1.5
    
    def __post_init__(self) -> None:
        """Validate safety configuration."""
        validate_config_value("max_speed_mps", self.max_speed_mps, 0.0, 100.0)
        validate_config_value("max_acceleration_mps2", self.max_acceleration_mps2, 0.0, 10.0)
        validate_config_value("max_deceleration_mps2", self.max_deceleration_mps2, 0.0, 15.0)


@dataclass(slots=True)
class RuntimeConfig:
    """Complete runtime configuration for ADAS system."""
    
    detector: DetectorConfig
    tracker: TrackerConfig
    planner: PlannerConfig
    controller: ControllerConfig
    safety: SafetyConfig
    fps: int = 20
    log_level: str = "INFO"
    
    def __post_init__(self) -> None:
        """Validate runtime configuration."""
        validate_config_value("fps", self.fps, 1, 120)
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ConfigurationError(f"Invalid log level: {self.log_level}")


DEFAULT_CONFIG = RuntimeConfig(
    detector=DetectorConfig(),
    tracker=TrackerConfig(),
    planner=PlannerConfig(),
    controller=ControllerConfig(),
    safety=SafetyConfig(),
)


def load_config(path: str | Path | None = None) -> RuntimeConfig:
    """Load configuration from JSON file or return defaults.
    
    Args:
        path: Path to JSON configuration file (None for defaults)
        
    Returns:
        Validated runtime configuration
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    if path is None:
        logger.info("Using default configuration")
        return DEFAULT_CONFIG

    try:
        config_path = Path(path)
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")
        
        logger.info(f"Loading configuration from {path}")
        payload = json.loads(config_path.read_text())
        
        # Parse each section with defaults
        detector = _parse_detector_config(payload.get("detector", {}))
        tracker = _parse_tracker_config(payload.get("tracker", {}))
        planner = _parse_planner_config(payload.get("planner", {}))
        controller = _parse_controller_config(payload.get("controller", {}))
        safety = _parse_safety_config(payload.get("safety", {}))
        
        config = RuntimeConfig(
            detector=detector,
            tracker=tracker,
            planner=planner,
            controller=controller,
            safety=safety,
            fps=int(payload.get("fps", DEFAULT_CONFIG.fps)),
            log_level=payload.get("log_level", DEFAULT_CONFIG.log_level),
        )
        
        logger.info("Configuration loaded and validated successfully")
        return config
        
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in config file: {e}") from e
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {e}") from e


def _parse_detector_config(data: dict) -> DetectorConfig:
    """Parse detector configuration from dict."""
    return DetectorConfig(
        model_path=data.get("model_path", DEFAULT_CONFIG.detector.model_path),
        confidence_threshold=float(data.get("confidence_threshold", DEFAULT_CONFIG.detector.confidence_threshold)),
        iou_threshold=float(data.get("iou_threshold", DEFAULT_CONFIG.detector.iou_threshold)),
        max_detections=int(data.get("max_detections", DEFAULT_CONFIG.detector.max_detections)),
    )


def _parse_tracker_config(data: dict) -> TrackerConfig:
    """Parse tracker configuration from dict."""
    return TrackerConfig(
        max_missed_frames=int(data.get("max_missed_frames", DEFAULT_CONFIG.tracker.max_missed_frames)),
        association_threshold_px=float(data.get("association_threshold_px", DEFAULT_CONFIG.tracker.association_threshold_px)),
        focal_length_px=float(data.get("focal_length_px", DEFAULT_CONFIG.tracker.focal_length_px)),
        min_box_height_px=float(data.get("min_box_height_px", DEFAULT_CONFIG.tracker.min_box_height_px)),
        max_distance_m=float(data.get("max_distance_m", DEFAULT_CONFIG.tracker.max_distance_m)),
    )


def _parse_planner_config(data: dict) -> PlannerConfig:
    """Parse planner configuration from dict."""
    return PlannerConfig(
        cruise_speed_mps=float(data.get("cruise_speed_mps", DEFAULT_CONFIG.planner.cruise_speed_mps)),
        min_follow_distance_m=float(data.get("min_follow_distance_m", DEFAULT_CONFIG.planner.min_follow_distance_m)),
        max_steering_deg=float(data.get("max_steering_deg", DEFAULT_CONFIG.planner.max_steering_deg)),
        time_gap_s=float(data.get("time_gap_s", DEFAULT_CONFIG.planner.time_gap_s)),
        max_decel_mps2=float(data.get("max_decel_mps2", DEFAULT_CONFIG.planner.max_decel_mps2)),
        lane_center_gain=float(data.get("lane_center_gain", DEFAULT_CONFIG.planner.lane_center_gain)),
    )


def _parse_controller_config(data: dict) -> ControllerConfig:
    """Parse controller configuration from dict."""
    return ControllerConfig(
        kp_speed=float(data.get("kp_speed", DEFAULT_CONFIG.controller.kp_speed)),
        max_throttle=float(data.get("max_throttle", DEFAULT_CONFIG.controller.max_throttle)),
        max_brake=float(data.get("max_brake", DEFAULT_CONFIG.controller.max_brake)),
        max_steering_angle_deg=float(data.get("max_steering_angle_deg", DEFAULT_CONFIG.controller.max_steering_angle_deg)),
        steering_deadband_deg=float(data.get("steering_deadband_deg", DEFAULT_CONFIG.controller.steering_deadband_deg)),
    )


def _parse_safety_config(data: dict) -> SafetyConfig:
    """Parse safety configuration from dict."""
    return SafetyConfig(
        max_speed_mps=float(data.get("max_speed_mps", DEFAULT_CONFIG.safety.max_speed_mps)),
        max_acceleration_mps2=float(data.get("max_acceleration_mps2", DEFAULT_CONFIG.safety.max_acceleration_mps2)),
        max_deceleration_mps2=float(data.get("max_deceleration_mps2", DEFAULT_CONFIG.safety.max_deceleration_mps2)),
        max_steering_rate_rad_s=float(data.get("max_steering_rate_rad_s", DEFAULT_CONFIG.safety.max_steering_rate_rad_s)),
        max_steering_angle_rad=float(data.get("max_steering_angle_rad", DEFAULT_CONFIG.safety.max_steering_angle_rad)),
        min_following_distance_m=float(data.get("min_following_distance_m", DEFAULT_CONFIG.safety.min_following_distance_m)),
        max_lateral_offset_m=float(data.get("max_lateral_offset_m", DEFAULT_CONFIG.safety.max_lateral_offset_m)),
    )
