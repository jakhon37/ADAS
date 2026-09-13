"""Multi-object tracking for the ADAS stack.

Three layers, each usable and testable on its own:

* :mod:`adas.tracking.kalman` -- the motion models. A constant-acceleration
  filter over ground-plane range (metres, m/s, m/s^2) for the numbers the
  planner and the safety arbiter consume, and a constant-velocity filter over
  the image-plane box that exists to give data association a prediction and a
  covariance.
* :mod:`adas.tracking.association` -- pure functions: IoU, class compatibility,
  a gated cost matrix, and a self-contained Hungarian solve so a global optimum
  replaces order-dependent greedy matching.
* :mod:`adas.tracking.tracker` -- :class:`MultiObjectTracker`, which owns track
  lifecycle (M-of-N confirmation before a track is ever reported, coast, death),
  range filtering, time-to-collision and the ego-lane decision.

Units are metres, seconds and pixels; ``TrackedObject.velocity_mps`` keeps this
codebase's positive-when-closing convention. See the module docstrings for the
failure behaviour of each piece.
"""

from adas.tracking.association import (
    AssociationParams,
    Assignment,
    build_cost_matrix,
    class_compatible,
    class_group,
    hungarian,
    iou_xyxy,
    solve_assignment,
)
from adas.tracking.kalman import (
    CHI2_95,
    CHI2_99,
    BoxFilter,
    FilterError,
    KalmanFilter,
    RangeFilter,
    range_measurement_sigma_m,
)
from adas.tracking.tracker import (
    NOMINAL_OBJECT_HEIGHT_M,
    MultiObjectTracker,
    TrackStatus,
)

__all__ = [
    "AssociationParams",
    "Assignment",
    "BoxFilter",
    "CHI2_95",
    "CHI2_99",
    "FilterError",
    "KalmanFilter",
    "MultiObjectTracker",
    "NOMINAL_OBJECT_HEIGHT_M",
    "RangeFilter",
    "TrackStatus",
    "build_cost_matrix",
    "class_compatible",
    "class_group",
    "hungarian",
    "iou_xyxy",
    "range_measurement_sigma_m",
    "solve_assignment",
]
