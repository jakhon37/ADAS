"""Behavior planning modules.

* :class:`BehaviorPlanner` -- composes the two laws into a ``MotionPlan``.
* :class:`LongitudinalPlanner` -- constant-time-gap ACC plus an AEB stage.
* :class:`LateralPlanner` -- speed-scheduled lane centring with a lateral
  acceleration cap.  :class:`CameraGeometry` switches it from the non-metric
  pixel law to a metric Stanley law when extrinsics are available.
"""

from adas.planning.behavior_planner import BehaviorPlanner
from adas.planning.lateral import CameraGeometry, LateralLimits, LateralPlanner, SteeringDecision
from adas.planning.longitudinal import (
    LeadVehicle,
    LongitudinalLimits,
    LongitudinalPlanner,
    SpeedDecision,
)

__all__ = [
    "BehaviorPlanner",
    "CameraGeometry",
    "LateralLimits",
    "LateralPlanner",
    "LeadVehicle",
    "LongitudinalLimits",
    "LongitudinalPlanner",
    "SpeedDecision",
    "SteeringDecision",
]
