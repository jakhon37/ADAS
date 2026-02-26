"""Core infrastructure modules for ADAS system.

This package contains fundamental building blocks:
- Configuration management
- Exception handling
- Logging utilities
- Performance metrics
- Data models
- Input validation
"""

from adas.core.config import (
    ControllerConfig,
    DetectorConfig,
    PlannerConfig,
    RuntimeConfig,
    SafetyConfig,
    TrackerConfig,
    load_config,
)
from adas.core.exceptions import (
    ADASException,
    ConfigurationError,
    ControlError,
    PerceptionError,
    PlanningError,
    SafetyViolation,
    SensorError,
    TrackingError,
    ValidationError,
)
from adas.core.logger import log_performance, log_safety_event, setup_logger
from adas.core.metrics import PerformanceMetrics, SystemHealthMonitor
from adas.core.models import (
    BoundingBox,
    ControlCommand,
    LaneModel,
    MotionPlan,
    PerceptionFrame,
    TrackedObject,
)
from adas.core.validation import (
    validate_bounding_box,
    validate_config_value,
    validate_control_command,
    validate_image_dimensions,
    validate_lane_model,
    validate_motion_plan,
)

__all__ = [
    # Config
    "ControllerConfig",
    "DetectorConfig",
    "PlannerConfig",
    "RuntimeConfig",
    "SafetyConfig",
    "TrackerConfig",
    "load_config",
    # Exceptions
    "ADASException",
    "ConfigurationError",
    "ControlError",
    "PerceptionError",
    "PlanningError",
    "SafetyViolation",
    "SensorError",
    "TrackingError",
    "ValidationError",
    # Logger
    "log_performance",
    "log_safety_event",
    "setup_logger",
    # Metrics
    "PerformanceMetrics",
    "SystemHealthMonitor",
    # Models
    "BoundingBox",
    "ControlCommand",
    "LaneModel",
    "MotionPlan",
    "PerceptionFrame",
    "TrackedObject",
    # Validation
    "validate_bounding_box",
    "validate_config_value",
    "validate_control_command",
    "validate_image_dimensions",
    "validate_lane_model",
    "validate_motion_plan",
]
