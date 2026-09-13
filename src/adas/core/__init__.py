"""Core infrastructure: configuration, exceptions, logging, metrics, models, validation.

Nothing in this package imports numpy, OpenCV or TensorRT, so it is safe to
import on any machine and is what the ops layer and the tests build against.
"""

from adas.core.config import (
    CameraConfig,
    ControllerConfig,
    DepthConfig,
    DetectorConfig,
    EgoConfig,
    EventsConfig,
    HealthConfig,
    LaneConfig,
    PlannerConfig,
    RuntimeConfig,
    SafetyConfig,
    SourceConfig,
    TrackerConfig,
    default_config,
    load_config,
)
from adas.core.exceptions import (
    ADASException,
    ConfigurationError,
    ControlError,
    EngineError,
    PerceptionError,
    PlanningError,
    SafetyViolation,
    SensorError,
    TrackingError,
    ValidationError,
)
from adas.core.logger import (
    configure_logging,
    log_event,
    log_performance,
    log_safety_event,
    setup_logger,
)
from adas.core.metrics import PerformanceMetrics, SystemHealthMonitor
from adas.core.models import (
    ArbitrationResult,
    BoundingBox,
    ControlCommand,
    DrivableArea,
    EgoState,
    LaneLine,
    LaneModel,
    MotionPlan,
    PerceptionFrame,
    PerceptionStatus,
    RangeEstimate,
    RangeSource,
    SafetyState,
    TrackedObject,
)
from adas.core.validation import (
    validate_bounding_box,
    validate_config_value,
    validate_control_command,
    validate_ego_state,
    validate_image_dimensions,
    validate_lane_model,
    validate_motion_plan,
    validate_perception_frame,
)

__all__ = [
    # Config
    "CameraConfig",
    "ControllerConfig",
    "DepthConfig",
    "DetectorConfig",
    "EgoConfig",
    "EventsConfig",
    "HealthConfig",
    "LaneConfig",
    "PlannerConfig",
    "RuntimeConfig",
    "SafetyConfig",
    "SourceConfig",
    "TrackerConfig",
    "default_config",
    "load_config",
    # Exceptions
    "ADASException",
    "ConfigurationError",
    "ControlError",
    "EngineError",
    "PerceptionError",
    "PlanningError",
    "SafetyViolation",
    "SensorError",
    "TrackingError",
    "ValidationError",
    # Logging
    "configure_logging",
    "log_event",
    "log_performance",
    "log_safety_event",
    "setup_logger",
    # Metrics
    "PerformanceMetrics",
    "SystemHealthMonitor",
    # Models
    "ArbitrationResult",
    "BoundingBox",
    "ControlCommand",
    "DrivableArea",
    "EgoState",
    "LaneLine",
    "LaneModel",
    "MotionPlan",
    "PerceptionFrame",
    "PerceptionStatus",
    "RangeEstimate",
    "RangeSource",
    "SafetyState",
    "TrackedObject",
    # Validation
    "validate_bounding_box",
    "validate_config_value",
    "validate_control_command",
    "validate_ego_state",
    "validate_image_dimensions",
    "validate_lane_model",
    "validate_motion_plan",
    "validate_perception_frame",
]
