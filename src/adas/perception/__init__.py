"""Perception: detection, lane estimation, camera geometry and depth.

Import cost matters here.  Every backend module pulls in numpy and, on first
construction, TensorRT; this package therefore exports only the lightweight
entry points (the factory, the mock backends, the camera model and the shared
lane helpers) and leaves the engine-backed classes to be imported from their own
modules by the factory, so ``import adas`` on a machine with no GPU still works.
"""

from adas.perception.detection import ObjectDetector
from adas.perception.factory import (
    build_camera,
    build_depth_channel,
    build_detector,
    build_lane_estimator,
    resolve_model_path,
)
from adas.perception.geometry import CameraConfig, default_camera, estimate_range
from adas.perception.lane import (
    LaneBackend,
    LaneEstimator,
    MockLaneEstimator,
    lane_geometry_from_model,
)

__all__ = [
    "CameraConfig",
    "LaneBackend",
    "LaneEstimator",
    "MockLaneEstimator",
    "ObjectDetector",
    "build_camera",
    "build_depth_channel",
    "build_detector",
    "build_lane_estimator",
    "default_camera",
    "estimate_range",
    "lane_geometry_from_model",
    "resolve_model_path",
]
