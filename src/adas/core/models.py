"""Core domain models for the ADAS pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str


@dataclass
class LaneModel:
    left_coeffs: tuple[float, float, float]
    right_coeffs: tuple[float, float, float]
    lane_center_px: float
    curvature_m: float
    confidence: float = 0.0
    lines: List["LaneLine"] = field(default_factory=list)
    is_mock: bool = True


@dataclass
class TrackedObject:
    track_id: int
    box: BoundingBox
    velocity_mps: float
    distance_m: float
    age_frames: int = 0
    hits: int = 0
    time_since_update: int = 0
    lateral_offset_m: float = 0.0
    ttc_s: float = float("inf")
    range_estimate: "RangeEstimate | None" = None
    in_ego_lane: bool = False


@dataclass
class PerceptionFrame:
    frame_id: int
    timestamp_s: float
    rgb: Any
    width: int
    height: int
    detections: List[BoundingBox] = field(default_factory=list)
    lane: LaneModel | None = None
    status: "PerceptionStatus | None" = None
    drivable: "DrivableArea | None" = None
    ego: "EgoState | None" = None


@dataclass
class MotionPlan:
    """What the behaviour planner decided this frame.

    Two longitudinal outputs, and they are not interchangeable.
    ``target_speed_mps`` is a COMFORT REQUEST the controller may use throttle to
    reach.  ``decel_demand_mps2`` is the AUTHORITATIVE braking figure in m/s^2,
    already jerk shaped by the planner, and it is the only source of brake in the
    primary path when it is present.

    The split exists because a target speed cannot express a deceleration: a
    rate-limited target falls 0.15 m/s per frame for a 3 m/s^2 request, a
    proportional speed law needs a 20 m/s error to answer that, and the brake the
    primary path actually produced while trailing a comfort ramp was an eighth of
    what the planner asked for.  The safety arbiter was then the only component
    in the vehicle that really braked -- a backstop carrying the primary duty,
    which is the single point of failure this field removes.

    ``None`` means the planner is not stating a deceleration, and the controller
    falls back to its own speed-error law.  Every pre-existing caller therefore
    behaves exactly as before.
    """

    target_speed_mps: float
    steering_angle_deg: float
    reason: str
    decel_demand_mps2: float | None = None


@dataclass
class ControlCommand:
    throttle: float
    brake: float
    steering: float


class SafetyState(str, Enum):
    """Authoritative arbitration outcome. The pipeline actuates according to this, not the plan."""

    NOMINAL = "nominal"
    LIMITED = "limited"
    MIN_RISK_MANEUVER = "min_risk_maneuver"
    DISENGAGE = "disengage"


class RangeSource(str, Enum):
    PINHOLE = "pinhole"
    HOMOGRAPHY = "homography"
    DEPTH_MODEL = "depth_model"
    FUSED = "fused"
    UNAVAILABLE = "unavailable"


@dataclass
class EgoState:
    """Ego vehicle state. Without this the planner cannot compute a time-gap law."""

    speed_mps: float = 0.0
    yaw_rate_dps: float = 0.0
    accel_mps2: float = 0.0
    timestamp_s: float = 0.0
    valid: bool = False


@dataclass
class PerceptionStatus:
    """Distinguishes nothing is there from the sensor failed. Never conflate the two."""

    ok: bool = True
    consecutive_failures: int = 0
    last_good_timestamp_s: float = 0.0
    detector_ok: bool = True
    lane_ok: bool = True
    reason: str = ""


@dataclass
class RangeEstimate:
    distance_m: float = 0.0
    confidence: float = 0.0
    source: RangeSource = RangeSource.UNAVAILABLE
    truncated: bool = False


@dataclass
class LaneLine:
    """One detected lane boundary in image space plus its ego-frame fit."""

    points_px: List[Any] = field(default_factory=list)
    coeffs: Any = None
    confidence: float = 0.0
    index: int = -1


@dataclass
class DrivableArea:
    """Binary free-space mask from a segmentation head, downsampled for cheap queries."""

    mask: Any = None
    width: int = 0
    height: int = 0
    confidence: float = 0.0

    def is_free(self, x_frac: float, y_frac: float) -> bool:
        if self.mask is None or self.width == 0 or self.height == 0:
            return True
        xi = min(self.width - 1, max(0, int(x_frac * self.width)))
        yi = min(self.height - 1, max(0, int(y_frac * self.height)))
        return bool(self.mask[yi][xi])


@dataclass
class ArbitrationResult:
    """What SafetyMonitor.arbitrate returns. `command` is what actually reaches the actuators."""

    command: "ControlCommand"
    state: SafetyState = SafetyState.NOMINAL
    violations: List[str] = field(default_factory=list)
    reason: str = ""
