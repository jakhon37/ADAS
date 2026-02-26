"""Core domain models for the ADAS pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass(slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str


@dataclass(slots=True)
class LaneModel:
    left_coeffs: tuple[float, float, float]
    right_coeffs: tuple[float, float, float]
    lane_center_px: float
    curvature_m: float


@dataclass(slots=True)
class TrackedObject:
    track_id: int
    box: BoundingBox
    velocity_mps: float
    distance_m: float


@dataclass(slots=True)
class PerceptionFrame:
    frame_id: int
    timestamp_s: float
    rgb: Any
    width: int
    height: int
    detections: List[BoundingBox] = field(default_factory=list)
    lane: LaneModel | None = None


@dataclass(slots=True)
class MotionPlan:
    target_speed_mps: float
    steering_angle_deg: float
    reason: str


@dataclass(slots=True)
class ControlCommand:
    throttle: float
    brake: float
    steering: float
