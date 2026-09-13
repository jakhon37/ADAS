"""Lane estimation: the backend interface, the honest mock, and the shared fit.

Every lane backend in this package (UFLD-v2, YOLOP, TwinLiteNet, mock) satisfies
:class:`LaneBackend` so the choice is pure configuration. They all funnel their
detected boundary polylines through :func:`lane_model_from_boundaries`, so the
pixel fit, the metric ground-plane fit, the confidence and the ``is_mock`` flag
are computed in exactly one place.

Units
-----
* ``LaneModel.left_coeffs`` / ``right_coeffs`` stay in PIXELS as
  ``x = a*y^2 + b*y + c`` (source-frame pixels), because that is what
  ``BehaviorPlanner`` has always consumed.
* ``LaneModel.lane_center_px`` is a source-frame column, evaluated at
  :data:`PIXEL_EVAL_ROW_FRAC` of the frame height.
* ``LaneModel.curvature_m`` is a RADIUS in metres, computed on the road plane.
  A straight lane reports :data:`~adas.perception.geometry.MAX_CURVATURE_RADIUS_M`
  (not infinity, which ``validate_lane_model`` rejects).
* ``LaneModel.lines[i].coeffs`` is the METRIC ground-plane fit
  ``X = a*Z^2 + b*Z + c`` (metres) when a camera model was supplied, else
  ``None``. ``LaneModel.lines[i].points_px`` is always the source-frame polyline.

Lane indices in ``LaneModel.lines`` follow the UFLD-v2 convention:
``1`` = ego-lane left boundary, ``2`` = ego-lane right boundary, ``0``/``3`` =
adjacent lanes (informational only -- they never vote on the ego boundary), and
:data:`EGO_CENTRELINE_INDEX` = the derived ego centreline.

Failure behaviour
-----------------
An estimator that cannot measure a lane returns ``None``. It never returns a
plausible-looking default; the only backend that fabricates geometry is
:class:`MockLaneEstimator`, which marks it ``is_mock=True`` with
``confidence=0.0`` and logs a NOT-FOR-VEHICLE-USE warning on construction.
"""

from __future__ import annotations

import abc
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from adas.core.logger import setup_logger
from adas.core.models import DrivableArea, LaneLine, LaneModel
from adas.perception.geometry import (
    MAX_CURVATURE_RADIUS_M,
    NOMINAL_LANE_WIDTH_M,
    CameraConfig,
    GroundPolyline,
    LaneGeometry,
    curvature_radius_m,
    fit_ground_polyline,
    lane_geometry,
)

logger = setup_logger(__name__)

#: Row (as a fraction of frame height) at which ``lane_center_px`` is evaluated.
#: Close to the bottom of the frame, where both boundaries have real support and
#: the quadratic is not extrapolating.
PIXEL_EVAL_ROW_FRAC = 0.92

#: Range (m) at which the metric lateral offset and heading error are evaluated.
#: ``None`` means "the nearest range both boundaries actually observed", which
#: for a bonnet-mounted camera is a few metres ahead. A fixed 0.0 extrapolates
#: the ground fit below its support and inflates the reported lane width.
METRIC_EVAL_RANGE_M = None

#: ``LaneLine.index`` used for the derived ego centreline.
EGO_CENTRELINE_INDEX = -1

#: Support window assumed when a GroundPolyline is reconstructed from a stored
#: ``LaneLine.coeffs`` (which does not record the fitted range). Only affects
#: where :func:`lane_geometry_from_model` is allowed to evaluate.
RECONSTRUCTED_SUPPORT_M = (3.0, 40.0)

#: Fallback half-width, as a fraction of frame width, used to synthesise a
#: missing boundary when no camera model is available. Historical value.
FALLBACK_HALF_LANE_FRAC = 0.14

#: A boundary is fitted as ``x = f(y)``, which requires it to be predominantly
#: VERTICAL in image space. A polyline whose row span is less than this fraction
#: of its column span is rejected: ``np.polyfit`` will happily return
#: coefficients for it (slope 100 px/px, intercept -50000 px in one observed
#: case) whose extrapolation to the bottom of the frame is arbitrary, and those
#: arbitrary values would then vote on the ego lane centre.
MIN_VERTICAL_ASPECT = 0.25


class LaneBackend(abc.ABC):
    """Interface every lane estimator implements.

    ``estimate`` must never raise for ordinary "no lane visible" conditions --
    it returns ``None``. It may raise :class:`~adas.core.exceptions.PerceptionError`
    for genuine faults (engine failure, malformed input), which the pipeline
    latches as a perception failure rather than silently reading as clear road.
    """

    #: Short backend name, used in logs and in ``LaneLine`` provenance.
    name = "lane"

    @abc.abstractmethod
    def estimate(self, frame: object, width: int, height: int) -> Optional[LaneModel]:
        """Return a lane model for one frame, or ``None`` if no lane is measurable."""

    def drivable_area(self) -> Optional[DrivableArea]:
        """Free-space mask from the most recent :meth:`estimate`, if the backend has one.

        Returns ``None`` for backends without a segmentation head. The mask is
        an independent signal from the lane fit and the safety layer may use it
        on its own.
        """
        return None

    @property
    def is_mock(self) -> bool:
        """``True`` only for backends that fabricate geometry."""
        return False

    def close(self) -> None:
        """Release engine resources. Safe to call more than once."""


# --------------------------------------------------------------------------- #
# Shared fitting
# --------------------------------------------------------------------------- #


def fit_pixel_quadratic(
    points_px: Sequence[Tuple[float, float]],
) -> Optional[Tuple[float, float, float]]:
    """Fit ``x = a*y^2 + b*y + c`` in source-frame pixels.

    Returns ``None`` for fewer than 3 points, for a near-horizontal polyline
    (row span under 2 px, or under :data:`MIN_VERTICAL_ASPECT` of the column
    span -- see that constant for why), or for a fit that comes back
    non-finite.
    """
    if points_px is None or len(points_px) < 3:
        return None
    arr = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    xs = arr[:, 0]
    ys = arr[:, 1]
    y_span = float(ys.max() - ys.min())
    x_span = float(xs.max() - xs.min())
    if y_span < 2.0 or y_span < MIN_VERTICAL_ASPECT * x_span:
        return None
    degree = 2 if len(arr) >= 4 else 1
    try:
        coeff = np.polyfit(ys, xs, degree)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if degree == 1:
        coeff = np.array([0.0, coeff[0], coeff[1]], dtype=np.float64)
    if not np.all(np.isfinite(coeff)):
        return None
    return (float(coeff[0]), float(coeff[1]), float(coeff[2]))


def _eval_pixel(coeffs: Tuple[float, float, float], y_px: float) -> float:
    a, b, c = coeffs
    return a * y_px * y_px + b * y_px + c


def _clamped_eval_row(
    frame_height: int,
    *point_sets: Optional[Sequence[Tuple[float, float]]],
) -> float:
    """Row at which to evaluate the pixel fits, clamped to the fitted support."""
    y_eval = float(frame_height) * PIXEL_EVAL_ROW_FRAC
    lo = None
    hi = None
    for pts in point_sets:
        if not pts:
            continue
        ys = [float(p[1]) for p in pts]
        lo = min(ys) if lo is None else min(lo, min(ys))
        hi = max(ys) if hi is None else max(hi, max(ys))
    if lo is None or hi is None:
        return y_eval
    return float(min(max(y_eval, lo), hi))


def lane_model_from_boundaries(
    left_points_px: Optional[Sequence[Tuple[float, float]]],
    right_points_px: Optional[Sequence[Tuple[float, float]]],
    frame_width: int,
    frame_height: int,
    camera: Optional[CameraConfig] = None,
    left_confidence: float = 0.0,
    right_confidence: float = 0.0,
    extra_lines: Optional[Sequence[LaneLine]] = None,
    metric_eval_range_m: Optional[float] = METRIC_EVAL_RANGE_M,
) -> Optional[Tuple[LaneModel, Optional[LaneGeometry]]]:
    """Turn ego-lane boundary polylines into a :class:`LaneModel`.

    ``left_points_px``/``right_points_px`` are source-frame ``(x, y)`` polylines
    for the ego lane's left and right boundaries. At least one must fit; if
    neither does, the function returns ``None`` rather than inventing a lane.

    When ``camera`` is supplied the boundaries are additionally projected onto
    the road plane and fitted in metres; that fit populates the real
    ``curvature_m`` and the returned :class:`LaneGeometry`. Without a camera the
    curvature is reported as :data:`MAX_CURVATURE_RADIUS_M` (straight) and the
    geometry is ``None`` -- an uncalibrated system honestly has no metric lane.

    ``lane_center_px`` is always computed in pixel space, because it is
    calibration-free and it is what the behaviour planner consumes. The metric
    fit is never allowed to move it.

    Returns ``(model, geometry)``; ``geometry`` is ``None`` when no camera was
    supplied or when neither boundary survived projection to the road plane.
    """
    left_pix = fit_pixel_quadratic(left_points_px)
    right_pix = fit_pixel_quadratic(right_points_px)
    if left_pix is None and right_pix is None:
        return None

    y_eval = _clamped_eval_row(frame_height, left_points_px, right_points_px)
    fallback_half = float(frame_width) * FALLBACK_HALF_LANE_FRAC

    if left_pix is not None and right_pix is not None:
        x_left = _eval_pixel(left_pix, y_eval)
        x_right = _eval_pixel(right_pix, y_eval)
        both = True
    elif left_pix is not None:
        x_left = _eval_pixel(left_pix, y_eval)
        x_right = x_left + 2.0 * fallback_half
        right_pix = (0.0, 0.0, x_right)
        both = False
    else:
        x_right = _eval_pixel(right_pix, y_eval)
        x_left = x_right - 2.0 * fallback_half
        left_pix = (0.0, 0.0, x_left)
        both = False

    lane_center_px = (x_left + x_right) / 2.0
    if not math.isfinite(lane_center_px):
        return None

    left_ground = None
    right_ground = None
    if camera is not None:
        if left_points_px:
            left_ground = fit_ground_polyline(left_points_px, camera)
        if right_points_px:
            right_ground = fit_ground_polyline(right_points_px, camera)

    geometry = lane_geometry(left_ground, right_ground, eval_range_m=metric_eval_range_m)
    if geometry is not None:
        curvature = geometry.curvature_radius_m
    else:
        curvature = MAX_CURVATURE_RADIUS_M

    # Confidence: the weaker of the two boundaries dominates, a synthesised
    # boundary halves it, and an implausible metric lane width caps it hard.
    if both:
        confidence = min(float(left_confidence), float(right_confidence))
    else:
        confidence = 0.5 * max(float(left_confidence), float(right_confidence))
    if geometry is not None and geometry.both_boundaries and not geometry.plausible:
        logger.warning(
            "lane width %.2f m is outside the plausible band; capping lane confidence",
            geometry.lane_width_m,
        )
        confidence = min(confidence, 0.1)
    confidence = max(0.0, min(1.0, confidence))

    lines: List[LaneLine] = []
    if left_points_px:
        lines.append(
            LaneLine(
                points_px=[(float(x), float(y)) for x, y in left_points_px],
                coeffs=left_ground.coeffs if left_ground is not None else None,
                confidence=float(max(0.0, min(1.0, left_confidence))),
                index=1,
            )
        )
    if right_points_px:
        lines.append(
            LaneLine(
                points_px=[(float(x), float(y)) for x, y in right_points_px],
                coeffs=right_ground.coeffs if right_ground is not None else None,
                confidence=float(max(0.0, min(1.0, right_confidence))),
                index=2,
            )
        )
    if left_ground is not None or right_ground is not None:
        centre_coeffs = _centreline_coeffs(left_ground, right_ground)
        lines.append(
            LaneLine(
                points_px=[],
                coeffs=centre_coeffs,
                confidence=confidence,
                index=EGO_CENTRELINE_INDEX,
            )
        )
    if extra_lines:
        lines.extend(extra_lines)

    model = LaneModel(
        left_coeffs=left_pix,
        right_coeffs=right_pix,
        lane_center_px=float(lane_center_px),
        curvature_m=float(curvature),
        confidence=float(confidence),
        lines=lines,
        is_mock=False,
    )
    return model, geometry


def _centreline_coeffs(
    left: Optional[GroundPolyline],
    right: Optional[GroundPolyline],
    nominal_lane_width_m: float = NOMINAL_LANE_WIDTH_M,
) -> Optional[Tuple[float, float, float]]:
    """Ground-plane quadratic of the ego centreline, in metres."""
    if left is not None and right is not None:
        return tuple((float(a) + float(b)) / 2.0 for a, b in zip(left.coeffs, right.coeffs))
    if left is not None:
        a, b, c = left.coeffs
        return (a, b, c + nominal_lane_width_m / 2.0)
    if right is not None:
        a, b, c = right.coeffs
        return (a, b, c - nominal_lane_width_m / 2.0)
    return None


def lane_geometry_from_model(
    model: Optional[LaneModel],
    eval_range_m: Optional[float] = METRIC_EVAL_RANGE_M,
) -> Optional[LaneGeometry]:
    """Recompute the metric ego-lane geometry from a :class:`LaneModel`.

    Reads the ground-plane coefficients stored on ``model.lines`` (indices 1 and
    2). Returns ``None`` for a mock model, for a model produced without a camera
    (no metric fit exists), or when neither ego boundary carries ground
    coefficients. This is the accessor the safety and planning layers should use
    for ``lateral_offset_m`` -- it never re-derives metres from pixels.
    """
    if model is None or model.is_mock or not model.lines:
        return None
    left = None
    right = None
    for line in model.lines:
        if line.coeffs is None:
            continue
        if line.index == 1:
            left = _as_polyline(line)
        elif line.index == 2:
            right = _as_polyline(line)
    if left is None and right is None:
        return None
    return lane_geometry(left, right, eval_range_m=eval_range_m)


def _as_polyline(line: LaneLine) -> Optional[GroundPolyline]:
    coeffs = line.coeffs
    try:
        a, b, c = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
    except (TypeError, ValueError, IndexError):
        return None
    # LaneLine carries no support window, so reconstruct a permissive one:
    # RECONSTRUCTED_SUPPORT_M lets lane_geometry evaluate anywhere the original
    # fit plausibly covered. Callers that need the exact window should keep the
    # GroundPolyline the estimator produced (``estimator.last_geometry``).
    return GroundPolyline(
        coeffs=(a, b, c),
        z_min_m=RECONSTRUCTED_SUPPORT_M[0],
        z_max_m=RECONSTRUCTED_SUPPORT_M[1],
        num_points=len(line.points_px),
        rms_residual_m=0.0,
    )


def lane_curvature_from_pixels(
    points_px: Sequence[Tuple[float, float]],
    camera: CameraConfig,
) -> float:
    """Radius of curvature (m) of a single pixel polyline, via the road plane.

    Returns :data:`MAX_CURVATURE_RADIUS_M` when the polyline cannot be fitted on
    the ground plane -- never a fabricated finite radius.
    """
    fit = fit_ground_polyline(points_px, camera)
    if fit is None:
        return MAX_CURVATURE_RADIUS_M
    return curvature_radius_m(fit.coeffs, 0.0)


# --------------------------------------------------------------------------- #
# Mock backend
# --------------------------------------------------------------------------- #


class MockLaneEstimator(LaneBackend):
    """Fixed 36%/64% image-fraction lane. NOT A MEASUREMENT.

    Exists so the pipeline can run without an engine (unit tests, CI, wiring
    smoke tests). Every model it emits carries ``is_mock=True`` and
    ``confidence=0.0`` so nothing downstream can mistake it for perception, and
    ``curvature_m`` is :data:`MAX_CURVATURE_RADIUS_M` (straight) rather than the
    invented 220 m radius it used to report.

    Failure behaviour: none -- it always succeeds, which is exactly why it must
    never be the default in a vehicle configuration.
    """

    name = "mock"

    def __init__(self, left_frac: float = 0.36, right_frac: float = 0.64) -> None:
        self.left_frac = float(left_frac)
        self.right_frac = float(right_frac)
        logger.warning(
            "MockLaneEstimator active: lane geometry is FABRICATED (is_mock=True, "
            "confidence=0.0). NOT FOR VEHICLE USE."
        )

    @property
    def is_mock(self) -> bool:
        return True

    def estimate(self, frame: object, width: int, height: int) -> Optional[LaneModel]:
        left_base = float(width) * self.left_frac
        right_base = float(width) * self.right_frac
        return LaneModel(
            left_coeffs=(0.0, 0.0, left_base),
            right_coeffs=(0.0, 0.0, right_base),
            lane_center_px=(left_base + right_base) / 2.0,
            curvature_m=MAX_CURVATURE_RADIUS_M,
            confidence=0.0,
            lines=[],
            is_mock=True,
        )


#: Backwards-compatible alias. ``adas.perception.factory`` and
#: ``adas.perception.__init__`` import ``LaneEstimator``.
LaneEstimator = MockLaneEstimator
