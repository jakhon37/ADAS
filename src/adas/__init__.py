"""ADAS Core - a camera-only ADAS reference pipeline for the Jetson Xavier NX.

What this package is: a research and bring-up platform that runs real TensorRT
models (YOLOX-Nano detection, UFLD-v2 or YOLOP lane perception, MiDaS depth as an
independent range cross-check), a Kalman/Hungarian tracker in ground-plane
metres, a constant-time-gap longitudinal law, and an authoritative safety
arbiter whose command is the only one that may reach an actuator.

What this package is NOT: a production driving system.  It has never been run in
a vehicle, the camera on this board does not exist (everything is validated by
replaying clips), the shipped camera calibration is an assumption, and the
control loop has only ever been closed against a point-mass model.  ``README.md``
carries the explicit not-production-ready list; please read it before quoting
any number from here.

Layout::

    adas.core        configuration, models, validation, logging, metrics
    adas.perception  detection, lane, camera geometry, depth
    adas.tracking    Kalman filters, association, the multi-object tracker
    adas.planning    longitudinal (ACC/AEB) and lateral (LKA) laws
    adas.control     the controller and the authoritative safety arbiter
    adas.runtime     frame sources, the frame loop, the pipeline
    adas.io          health endpoint, event log, systemd integration
    adas.ros2        an OPTIONAL ROS 2 bridge; rclpy is not installed on this board
"""

__version__ = "0.3.0"

from adas.core import (
    ADASException,
    ArbitrationResult,
    BoundingBox,
    ControlCommand,
    EgoState,
    LaneModel,
    MotionPlan,
    PerceptionFrame,
    PerceptionStatus,
    RuntimeConfig,
    SafetyState,
    SafetyViolation,
    TrackedObject,
    ValidationError,
    load_config,
)
from adas.runtime import ADASPipeline, PipelineRunner

__all__ = [
    "__version__",
    "ADASException",
    "ADASPipeline",
    "ArbitrationResult",
    "BoundingBox",
    "ControlCommand",
    "EgoState",
    "LaneModel",
    "MotionPlan",
    "PerceptionFrame",
    "PerceptionStatus",
    "PipelineRunner",
    "RuntimeConfig",
    "SafetyState",
    "SafetyViolation",
    "TrackedObject",
    "ValidationError",
    "load_config",
]
