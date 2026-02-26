"""ADAS Core - Production-grade Advanced Driver Assistance System.

This package provides a modular ADAS implementation with:
- Core infrastructure (config, logging, validation, models)
- Perception (object detection, lane estimation)
- Tracking (multi-object tracking)
- Planning (behavior planning, ACC, LKA)
- Control (PID controller, safety monitoring)
- Runtime (pipeline orchestration)
"""

__version__ = "0.1.0"

# Re-export commonly used components for convenience
from adas.core import (
    # Models
    BoundingBox,
    ControlCommand,
    LaneModel,
    MotionPlan,
    PerceptionFrame,
    TrackedObject,
    # Config
    RuntimeConfig,
    load_config,
    # Exceptions
    ADASException,
    SafetyViolation,
    ValidationError,
)
from adas.runtime import ADASPipeline, PipelineRunner

__all__ = [
    "__version__",
    # Core models
    "BoundingBox",
    "ControlCommand",
    "LaneModel",
    "MotionPlan",
    "PerceptionFrame",
    "TrackedObject",
    # Config
    "RuntimeConfig",
    "load_config",
    # Exceptions
    "ADASException",
    "SafetyViolation",
    "ValidationError",
    # Runtime
    "ADASPipeline",
    "PipelineRunner",
]
