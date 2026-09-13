"""Ground-plane camera geometry: the metric layer the ADAS stack was missing.

Coordinate conventions
----------------------
* Pixel frame: ``u`` right, ``v`` down, origin at the top-left of the *source*
  frame (not of any letterboxed network input).
* Camera frame: ``x`` right, ``y`` down, ``z`` forward along the optical axis.
* Ground/vehicle frame: ``X`` right, ``Y`` down, ``Z`` forward and horizontal.
  The origin is the camera optical centre, so the road surface is the plane
  ``Y = +mount_height_m``.
* ``pitch_deg`` is positive when the camera is nose-**down** (the usual dash
  mount). Zero means the optical axis is horizontal.

Units
-----
Every length is METRES unless the identifier ends in ``_px``. Every angle is
radians unless the identifier ends in ``_deg``. Curvature is reported as a
*radius* in metres; a straight lane reports :data:`MAX_CURVATURE_RADIUS_M`
rather than infinity, because ``adas.core.validation.validate_lane_model``
requires a finite value.

Failure behaviour
-----------------
* Invalid camera parameters raise :class:`~adas.core.exceptions.ConfigurationError`
  at construction time, never mid-frame.
* Every projection that can fail returns ``None`` -- a pixel at or above the
  horizon, or a ground point behind the camera, has no valid counterpart and
  this module refuses to invent one. Callers must treat ``None`` as
  "unmeasurable" and degrade; nothing here ever substitutes a plausible
  constant.
* :class:`~adas.core.models.RangeEstimate` results always carry the
  :class:`~adas.core.models.RangeSource` that produced them and a confidence in
  ``[0, 1]``. A camera that has not been calibrated
  (``CameraConfig.calibrated is False``) has its range confidence capped at
  :data:`UNCALIBRATED_CONFIDENCE_CAP`, because an assumed focal length is a
  guess and must not present as a measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from adas.core.exceptions import ConfigurationError
from adas.core.logger import setup_logger
from adas.core.models import BoundingBox, RangeEstimate, RangeSource

logger = setup_logger(__name__)

#: Radius reported for a lane that is straight to within numerical noise (m).
MAX_CURVATURE_RADIUS_M = 10000.0

#: Ranges beyond this are not trusted from a single monocular camera (m).
MAX_TRUSTED_RANGE_M = 200.0

#: Confidence ceiling applied to any estimate derived from an uncalibrated camera.
UNCALIBRATED_CONFIDENCE_CAP = 0.6

#: Nominal object heights (m) used by the pinhole range model, keyed by the COCO
#: label strings that ``adas.perception.yolo`` emits. ``_default`` is the
#: fallback for an unknown label.
NOMINAL_OBJECT_HEIGHT_M = {
    "person": 1.70,
    "bicycle": 1.70,
    "motorcycle": 1.60,
    "car": 1.50,
    "bus": 3.20,
    "truck": 3.60,
    "train": 4.00,
    "vehicle": 1.50,
    "_default": 1.50,
}

#: Plausibility band for a highway/urban lane, used to reject nonsense fits (m).
MIN_PLAUSIBLE_LANE_WIDTH_M = 2.0
MAX_PLAUSIBLE_LANE_WIDTH_M = 5.0

#: Default lane width assumed when only one boundary was detected (m).
NOMINAL_LANE_WIDTH_M = 3.5


# --------------------------------------------------------------------------- #
# Camera model
# --------------------------------------------------------------------------- #


@dataclass
class CameraConfig:
    """Pinhole intrinsics plus the extrinsics needed for a road-plane homography.

    Attributes
    ----------
    image_width, image_height:
        Resolution the intrinsics were measured at, in pixels. Use
        :meth:`scaled_to` when the live stream differs -- an ``fx`` measured at
        1280 wide is 1.5x too small at 1920 wide, and nothing else in the stack
        notices.
    fx, fy:
        Focal length in pixels along each axis.
    cx, cy:
        Principal point in pixels.
    mount_height_m:
        Height of the optical centre above the road surface. Must be > 0; a
        zero height makes the road-plane homography singular.
    pitch_deg:
        Downward tilt of the optical axis, degrees. Positive is nose-down.
    calibrated:
        ``False`` (the default) means these numbers are an assumption, not a
        measurement. Every :class:`RangeEstimate` derived from an uncalibrated
        camera has its confidence capped at
        :data:`UNCALIBRATED_CONFIDENCE_CAP`, and the geometry helpers log once
        at WARNING.
    label:
        Free-form provenance string, e.g. the calibration file name.
    """

    image_width: int = 1280
    image_height: int = 720
    fx: float = 910.0
    fy: float = 910.0
    cx: float = 640.0
    cy: float = 360.0
    mount_height_m: float = 1.30
    pitch_deg: float = 2.0
    calibrated: bool = False
    label: str = "uncalibrated-default"

    # Derived, never passed to __init__.
    pitch_rad: float = field(init=False, repr=False, compare=False, default=0.0)
    ground_to_image_h: np.ndarray = field(init=False, repr=False, compare=False, default=None)
    image_to_ground_h: np.ndarray = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        for name in ("fx", "fy"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ConfigurationError("CameraConfig.%s must be finite and > 0, got %r" % (name, value))
            setattr(self, name, value)
        for name in ("cx", "cy"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ConfigurationError("CameraConfig.%s must be finite, got %r" % (name, value))
            setattr(self, name, value)
        self.image_width = int(self.image_width)
        self.image_height = int(self.image_height)
        if self.image_width <= 0 or self.image_height <= 0:
            raise ConfigurationError(
                "CameraConfig image size must be positive, got %dx%d" % (self.image_width, self.image_height)
            )
        self.mount_height_m = float(self.mount_height_m)
        if not math.isfinite(self.mount_height_m) or self.mount_height_m <= 0.0:
            raise ConfigurationError(
                "CameraConfig.mount_height_m must be finite and > 0 (a camera on the road "
                "plane has a singular homography), got %r" % (self.mount_height_m,)
            )
        self.pitch_deg = float(self.pitch_deg)
        if not math.isfinite(self.pitch_deg) or abs(self.pitch_deg) >= 89.0:
            raise ConfigurationError(
                "CameraConfig.pitch_deg must be finite and within +/-89 degrees, got %r" % (self.pitch_deg,)
            )
        self.pitch_rad = math.radians(self.pitch_deg)

        c = math.cos(self.pitch_rad)
        s = math.sin(self.pitch_rad)
        h = self.mount_height_m
        # Ground point (X, Z) on the plane Y = h, expressed in camera coordinates:
        #   p_cam = (X, h*c - Z*s, h*s + Z*c)
        plane_to_cam = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -s, h * c],
                [0.0, c, h * s],
            ],
            dtype=np.float64,
        )
        k = np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.ground_to_image_h = k @ plane_to_cam
        try:
            self.image_to_ground_h = np.linalg.inv(self.ground_to_image_h)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded by validation above
            raise ConfigurationError("degenerate camera geometry: %s" % exc) from exc

        if not self.calibrated:
            logger.warning(
                "CameraConfig %r is NOT calibrated (fx=%.1f fy=%.1f h=%.2fm pitch=%.1fdeg). "
                "Metric ranges and lane widths are assumptions, not measurements; "
                "confidence is capped at %.2f.",
                self.label,
                self.fx,
                self.fy,
                self.mount_height_m,
                self.pitch_deg,
                UNCALIBRATED_CONFIDENCE_CAP,
            )

    # -- constructors ------------------------------------------------------ #

    @classmethod
    def from_fov(
        cls,
        image_width: int,
        image_height: int,
        horizontal_fov_deg: float,
        mount_height_m: float = 1.30,
        pitch_deg: float = 2.0,
        label: str = "from_fov",
    ) -> "CameraConfig":
        """Build an *approximate* camera from a datasheet field of view.

        Assumes square pixels and a centred principal point. The result is
        marked ``calibrated=False`` because a datasheet FOV is not a
        calibration.
        """
        if not (1.0 < float(horizontal_fov_deg) < 179.0):
            raise ConfigurationError("horizontal_fov_deg must be in (1, 179), got %r" % (horizontal_fov_deg,))
        fx = (float(image_width) / 2.0) / math.tan(math.radians(float(horizontal_fov_deg)) / 2.0)
        return cls(
            image_width=int(image_width),
            image_height=int(image_height),
            fx=fx,
            fy=fx,
            cx=float(image_width) / 2.0,
            cy=float(image_height) / 2.0,
            mount_height_m=float(mount_height_m),
            pitch_deg=float(pitch_deg),
            calibrated=False,
            label=label,
        )

    def scaled_to(self, image_width: int, image_height: int) -> "CameraConfig":
        """Return the same camera expressed in a different frame resolution.

        Intrinsics live in pixels of the source frame, so they must be rescaled
        whenever the capture resolution changes. Extrinsics and the
        ``calibrated`` flag are preserved.
        """
        image_width = int(image_width)
        image_height = int(image_height)
        if image_width == self.image_width and image_height == self.image_height:
            return self
        sx = float(image_width) / float(self.image_width)
        sy = float(image_height) / float(self.image_height)
        return CameraConfig(
            image_width=image_width,
            image_height=image_height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            mount_height_m=self.mount_height_m,
            pitch_deg=self.pitch_deg,
            calibrated=self.calibrated,
            label="%s@%dx%d" % (self.label, image_width, image_height),
        )

    def with_horizon_px(
        self,
        horizon_y_px: float,
        calibrated: Optional[bool] = None,
        label: Optional[str] = None,
    ) -> "CameraConfig":
        """Return the same camera re-pitched so its horizon lands on ``horizon_y_px``.

        Inverts :attr:`horizon_y_px`: ``pitch = atan((cy - v_horizon) / fy)``.
        This is how a monocular camera is pitch-calibrated in the field -- the
        vanishing point of the lane markings IS the road-plane horizon, so
        :func:`estimate_vanishing_point` feeds straight into this. Pitch is the
        single parameter that range is most sensitive to: it sets where ``Z``
        goes to infinity, so a few pixels of error is a large far-field range
        error.

        Raises :class:`~adas.core.exceptions.ConfigurationError` for a horizon
        that implies a pitch beyond +/-89 degrees.
        """
        v = float(horizon_y_px)
        if not math.isfinite(v):
            raise ConfigurationError("horizon_y_px must be finite, got %r" % (horizon_y_px,))
        pitch_deg = math.degrees(math.atan((self.cy - v) / self.fy))
        return CameraConfig(
            image_width=self.image_width,
            image_height=self.image_height,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            mount_height_m=self.mount_height_m,
            pitch_deg=pitch_deg,
            calibrated=self.calibrated if calibrated is None else bool(calibrated),
            label=label if label is not None else "%s+horizon%.1f" % (self.label, v),
        )

    def with_mount_height(
        self,
        mount_height_m: float,
        calibrated: Optional[bool] = None,
        label: Optional[str] = None,
    ) -> "CameraConfig":
        """Return the same camera with a different mount height.

        Mount height is the pure scale factor of the road-plane homography:
        every ``X`` and ``Z`` is proportional to it, so it can be solved for
        separately once the pitch is right (see
        :func:`calibrate_mount_height_from_lane`).
        """
        return CameraConfig(
            image_width=self.image_width,
            image_height=self.image_height,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            mount_height_m=float(mount_height_m),
            pitch_deg=self.pitch_deg,
            calibrated=self.calibrated if calibrated is None else bool(calibrated),
            label=label if label is not None else "%s+h%.2f" % (self.label, float(mount_height_m)),
        )

    # -- derived quantities ------------------------------------------------ #

    @property
    def horizon_y_px(self) -> float:
        """Image row where the road plane recedes to infinity.

        Derived analytically from the pitch: ``v = cy - fy * tan(pitch)``. Rows
        at or above this value never intersect the road in front of the
        vehicle. May legitimately fall outside ``[0, image_height)`` for a
        steeply pitched camera.
        """
        return self.cy - self.fy * math.tan(self.pitch_rad)

    @property
    def focal_length_px(self) -> float:
        """Vertical focal length, the one the pinhole height model needs."""
        return self.fy

    def is_below_horizon(self, v_px: float, margin_px: float = 1.0) -> bool:
        """True when image row ``v_px`` can intersect the road plane ahead."""
        return float(v_px) > self.horizon_y_px + float(margin_px)

    # -- projections ------------------------------------------------------- #

    def image_to_ground(self, u_px: float, v_px: float) -> Optional[Tuple[float, float]]:
        """Project a pixel onto the road plane.

        Returns ``(X_m, Z_m)`` -- lateral offset (right positive) and forward
        range -- or ``None`` when the pixel is at/above the horizon, projects
        behind the camera, or lands beyond :data:`MAX_TRUSTED_RANGE_M`.
        """
        vec = self.image_to_ground_h @ np.array([float(u_px), float(v_px), 1.0], dtype=np.float64)
        w = float(vec[2])
        if not math.isfinite(w) or w <= 1e-12:
            return None
        x_m = float(vec[0]) / w
        z_m = float(vec[1]) / w
        if not (math.isfinite(x_m) and math.isfinite(z_m)):
            return None
        if z_m <= 0.0 or z_m > MAX_TRUSTED_RANGE_M:
            return None
        return (x_m, z_m)

    def ground_to_image(self, x_m: float, z_m: float) -> Optional[Tuple[float, float]]:
        """Project a road-plane point back to a pixel.

        Returns ``(u_px, v_px)``, or ``None`` when the point is behind the
        image plane.
        """
        vec = self.ground_to_image_h @ np.array([float(x_m), float(z_m), 1.0], dtype=np.float64)
        w = float(vec[2])
        if not math.isfinite(w) or w <= 1e-12:
            return None
        return (float(vec[0]) / w, float(vec[1]) / w)

    @property
    def ground_to_image_matrix(self) -> np.ndarray:
        """3x3 homography mapping ``[X, Z, 1]`` on the road plane to ``[u, v, 1]``."""
        return self.ground_to_image_h

    @property
    def image_to_ground_matrix(self) -> np.ndarray:
        """3x3 homography mapping ``[u, v, 1]`` to ``[X, Z, 1]`` on the road plane."""
        return self.image_to_ground_h

    def project_points(self, points_px: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Vectorised :meth:`image_to_ground` over a polyline.

        Pixels that do not intersect the road plane are dropped, so the result
        may be shorter than the input (possibly empty).
        """
        if not points_px:
            return []
        arr = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
        homo = np.concatenate([arr, np.ones((arr.shape[0], 1), dtype=np.float64)], axis=1)
        ground = homo @ self.image_to_ground_h.T
        w = ground[:, 2]
        ok = np.isfinite(w) & (w > 1e-12)
        out: List[Tuple[float, float]] = []
        if not ok.any():
            return out
        xs = np.where(ok, ground[:, 0] / np.where(ok, w, 1.0), np.nan)
        zs = np.where(ok, ground[:, 1] / np.where(ok, w, 1.0), np.nan)
        good = ok & np.isfinite(xs) & np.isfinite(zs) & (zs > 0.0) & (zs <= MAX_TRUSTED_RANGE_M)
        for x_m, z_m in zip(xs[good], zs[good]):
            out.append((float(x_m), float(z_m)))
        return out


def default_camera(image_width: int = 1280, image_height: int = 720) -> CameraConfig:
    """Explicitly-uncalibrated stand-in camera.

    Matches the historical ``focal_length_px = 910`` that ``TrackerConfig`` has
    always assumed, so ranges are unchanged until an integrator supplies a real
    calibration -- but ``calibrated`` is ``False``, so every derived estimate is
    confidence-capped and the construction logs a warning. This is a function,
    not a module constant, so importing :mod:`adas.perception.geometry` does not
    emit that warning on its own.
    """
    return CameraConfig(
        image_width=int(image_width),
        image_height=int(image_height),
        fx=910.0,
        fy=910.0,
        cx=float(image_width) / 2.0,
        cy=float(image_height) / 2.0,
        mount_height_m=1.30,
        pitch_deg=2.0,
        calibrated=False,
        label="uncalibrated-default",
    )


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #


@dataclass
class BoxTruncation:
    """Which image borders a detection box is clipped against.

    A vertically truncated box breaks the pinhole height model in the *unsafe*
    direction: the visible box is shorter than the object, so ``H*f/h`` reports
    the vehicle as further away than it is, and it does so precisely when the
    vehicle is closest.
    """

    left: bool = False
    right: bool = False
    top: bool = False
    bottom: bool = False

    @property
    def any_edge(self) -> bool:
        return self.left or self.right or self.top or self.bottom

    @property
    def vertical(self) -> bool:
        """True when the height of the box is unreliable."""
        return self.top or self.bottom


def truncation_flags(
    box: BoundingBox,
    frame_width: int,
    frame_height: int,
    margin_px: float = 2.0,
) -> BoxTruncation:
    """Detect which frame borders ``box`` touches, within ``margin_px``."""
    margin = float(margin_px)
    return BoxTruncation(
        left=float(box.x1) <= margin,
        right=float(box.x2) >= float(frame_width) - 1.0 - margin,
        top=float(box.y1) <= margin,
        bottom=float(box.y2) >= float(frame_height) - 1.0 - margin,
    )


# --------------------------------------------------------------------------- #
# Range estimation
# --------------------------------------------------------------------------- #


def _cap_confidence(value: float, camera: CameraConfig) -> float:
    value = max(0.0, min(1.0, float(value)))
    if not camera.calibrated:
        value = min(value, UNCALIBRATED_CONFIDENCE_CAP)
    return value


def pinhole_range(
    box: BoundingBox,
    camera: CameraConfig,
    object_height_m: Optional[float] = None,
    truncation: Optional[BoxTruncation] = None,
    min_box_height_px: float = 1.0,
) -> RangeEstimate:
    """Range from apparent object height: ``Z = H_object * fy / h_px``.

    ``object_height_m`` defaults to the per-class nominal height looked up from
    :data:`NOMINAL_OBJECT_HEIGHT_M` using ``box.label``. Using one height for
    every class is the single largest systematic error in this model: a 3.6 m
    truck scored as a 1.5 m car reads 2.4x too close.

    Returns ``source=PINHOLE``. When the box is vertically truncated the
    returned distance is an **over**-estimate (the true object is nearer);
    ``truncated`` is set and confidence is halved. Confidence 0 with
    ``source=UNAVAILABLE`` is returned for a degenerate box.
    """
    trunc = truncation or BoxTruncation()
    height_px = float(box.y2) - float(box.y1)
    if not math.isfinite(height_px) or height_px < float(min_box_height_px):
        return RangeEstimate(
            distance_m=0.0,
            confidence=0.0,
            source=RangeSource.UNAVAILABLE,
            truncated=trunc.vertical,
        )
    if object_height_m is None:
        object_height_m = NOMINAL_OBJECT_HEIGHT_M.get(
            str(box.label).lower(), NOMINAL_OBJECT_HEIGHT_M["_default"]
        )
    distance_m = (float(object_height_m) * camera.focal_length_px) / height_px
    if not math.isfinite(distance_m) or distance_m <= 0.0:
        return RangeEstimate(0.0, 0.0, RangeSource.UNAVAILABLE, trunc.vertical)

    # One pixel of height jitter is worth d^2 / (H*f) metres; use that to turn
    # a nominal 1.5 px detector jitter into a confidence.
    hf = max(1e-6, float(object_height_m) * camera.focal_length_px)
    sigma_m = 1.5 * distance_m * distance_m / hf
    confidence = 1.0 / (1.0 + (sigma_m / max(1.0, 0.1 * distance_m)) ** 2)
    if trunc.vertical:
        confidence *= 0.5
    if distance_m > MAX_TRUSTED_RANGE_M:
        confidence *= 0.25
    return RangeEstimate(
        distance_m=float(min(distance_m, MAX_TRUSTED_RANGE_M)),
        confidence=_cap_confidence(confidence, camera),
        source=RangeSource.PINHOLE,
        truncated=trunc.vertical,
    )


def ground_plane_range(
    box: BoundingBox,
    camera: CameraConfig,
    frame_width: int,
    frame_height: int,
    truncation: Optional[BoxTruncation] = None,
) -> RangeEstimate:
    """Range from the road-contact point: where the box bottom meets the ground.

    This is the better monocular estimator because it depends only on the
    *bottom edge* of the box and on the camera extrinsics, not on an assumed
    object height -- so it is immune to the class-height error that dominates
    :func:`pinhole_range`, and it survives a box whose top is clipped.

    Assumes the object stands on the same flat plane as the camera. On a crest,
    a dip or a banked curve that assumption breaks and the error is unbounded,
    which is why the result is fused rather than trusted alone.

    Returns ``source=HOMOGRAPHY``. If the box bottom is clipped by the image
    edge the true contact point is *below* the frame, so the returned distance
    is an **upper bound** (the object is at most this far); ``truncated`` is set
    and confidence is reduced. Returns ``source=UNAVAILABLE`` with confidence 0
    when the contact point is at or above the horizon.
    """
    trunc = truncation or truncation_flags(box, frame_width, frame_height)
    u_px = (float(box.x1) + float(box.x2)) / 2.0
    v_px = float(box.y2)
    horizon = camera.horizon_y_px
    if not camera.is_below_horizon(v_px):
        return RangeEstimate(
            distance_m=0.0,
            confidence=0.0,
            source=RangeSource.UNAVAILABLE,
            truncated=trunc.bottom,
        )
    ground = camera.image_to_ground(u_px, v_px)
    if ground is None:
        return RangeEstimate(0.0, 0.0, RangeSource.UNAVAILABLE, trunc.bottom)
    _x_m, z_m = ground

    # Sensitivity: dZ/dv ~ -Z^2 / (fy * h). Convert a nominal 2 px contact-point
    # error into a relative uncertainty.
    sigma_m = 2.0 * z_m * z_m / max(1e-6, camera.fy * camera.mount_height_m)
    confidence = 1.0 / (1.0 + (sigma_m / max(1.0, 0.1 * z_m)) ** 2)
    # A contact point only a few rows below the horizon is numerically hopeless.
    rows_below = v_px - horizon
    confidence *= min(1.0, rows_below / max(1.0, 0.05 * float(frame_height)))
    if trunc.bottom:
        confidence *= 0.35
    return RangeEstimate(
        distance_m=float(z_m),
        confidence=_cap_confidence(confidence, camera),
        source=RangeSource.HOMOGRAPHY,
        truncated=trunc.bottom,
    )


def fuse_range(*estimates: RangeEstimate) -> RangeEstimate:
    """Confidence-weighted fusion of range estimates, averaged in inverse range.

    Both monocular estimators are linear in ``1/Z`` with respect to their pixel
    measurement (box height for pinhole, contact-point row for the homography),
    so uniform pixel noise is uniform noise in ``1/Z``. Averaging there rather
    than in ``Z`` keeps far-field noise from dominating.

    Estimates with zero confidence or ``UNAVAILABLE`` source are ignored. With
    no usable input the result is ``UNAVAILABLE`` with confidence 0. With one
    usable input that input is returned unchanged. ``truncated`` is sticky: if
    any contributing estimate was truncated the fused result says so.
    """
    usable = [
        e
        for e in estimates
        if e is not None
        and e.source is not RangeSource.UNAVAILABLE
        and e.confidence > 0.0
        and math.isfinite(e.distance_m)
        and e.distance_m > 0.0
    ]
    if not usable:
        truncated = any(e is not None and e.truncated for e in estimates)
        return RangeEstimate(0.0, 0.0, RangeSource.UNAVAILABLE, truncated)
    if len(usable) == 1:
        return usable[0]
    weight_sum = sum(e.confidence for e in usable)
    inv = sum(e.confidence / e.distance_m for e in usable) / weight_sum
    if inv <= 1e-12:
        return usable[0]
    # Independent-ish agreement raises confidence, disagreement lowers it.
    distances = [e.distance_m for e in usable]
    spread = (max(distances) - min(distances)) / max(1e-6, sum(distances) / len(distances))
    agreement = 1.0 / (1.0 + spread)
    confidence = min(1.0, (weight_sum / len(usable)) * (1.0 + 0.5 * agreement))
    return RangeEstimate(
        distance_m=float(1.0 / inv),
        confidence=float(max(0.0, min(1.0, confidence))),
        source=RangeSource.FUSED,
        truncated=any(e.truncated for e in usable),
    )


def estimate_range(
    box: BoundingBox,
    camera: CameraConfig,
    frame_width: int,
    frame_height: int,
    object_height_m: Optional[float] = None,
    min_box_height_px: float = 1.0,
) -> RangeEstimate:
    """Best available monocular range for one detection.

    Runs both the pinhole-height and ground-plane models, then fuses them in
    inverse-range space.

    When the box is vertically truncated the pinhole model is **discarded**
    rather than down-weighted:

    * Top clipped (the common occlusion / frame-edge case): the visible box is
      shorter than the object, so ``H*f/h`` reads too FAR by an unbounded
      factor, exactly when the object is closest. The road-contact point is
      still visible, so the ground-plane estimate is unaffected and correct.
    * Bottom clipped (the object is so close its wheels leave the frame): both
      models over-read, but the ground-plane error is BOUNDED -- the answer can
      be no larger than the range of the bottom image row -- while the pinhole
      error is not. The result is published as that bound with
      ``truncated=True`` and a reduced confidence; the caller must treat it as
      "no further than this", never as a fix.

    If the contact point is also unusable (above the horizon) the result is
    ``UNAVAILABLE`` with confidence 0 rather than a guess.
    """
    trunc = truncation_flags(box, frame_width, frame_height)
    ground = ground_plane_range(box, camera, frame_width, frame_height, truncation=trunc)
    if trunc.vertical:
        if ground.source is RangeSource.UNAVAILABLE:
            # Nothing trustworthy left. Say so rather than guessing.
            return RangeEstimate(0.0, 0.0, RangeSource.UNAVAILABLE, True)
        return RangeEstimate(ground.distance_m, ground.confidence, ground.source, True)
    pinhole = pinhole_range(
        box,
        camera,
        object_height_m=object_height_m,
        truncation=trunc,
        min_box_height_px=min_box_height_px,
    )
    return fuse_range(ground, pinhole)


# --------------------------------------------------------------------------- #
# Lane geometry on the ground plane
# --------------------------------------------------------------------------- #


@dataclass
class GroundPolyline:
    """A lane boundary fitted on the road plane as ``X = a*Z^2 + b*Z + c``.

    ``coeffs`` is ``(a, b, c)`` with ``X`` and ``Z`` in metres, so ``c`` is the
    lateral offset of the boundary abeam the camera and ``b`` is its heading
    relative to the vehicle axis (radians, small-angle). ``z_min_m``/``z_max_m``
    bound the fitted support: evaluating far outside it is extrapolation and
    the caller should not.
    """

    coeffs: Tuple[float, float, float]
    z_min_m: float
    z_max_m: float
    num_points: int
    rms_residual_m: float

    def x_at(self, z_m: float) -> float:
        a, b, c = self.coeffs
        return a * z_m * z_m + b * z_m + c

    def dx_dz_at(self, z_m: float) -> float:
        a, b, _c = self.coeffs
        return 2.0 * a * z_m + b


#: Ground-fit window. Beyond this range the road-plane back-projection of a
#: monocular pixel is dominated by pitch error -- one pixel near the horizon is
#: tens of metres -- so including those points lets the far field dictate a fit
#: that is only ever evaluated in the near field.
DEFAULT_FIT_RANGE_M = 60.0


def fit_ground_polyline(
    points_px: Sequence[Tuple[float, float]],
    camera: CameraConfig,
    min_points: int = 4,
    max_rms_residual_m: float = 1.0,
    max_range_m: float = DEFAULT_FIT_RANGE_M,
) -> Optional[GroundPolyline]:
    """Project a pixel polyline to the road plane and fit ``X = f(Z)`` in metres.

    Points beyond ``max_range_m`` are dropped before fitting; see
    :data:`DEFAULT_FIT_RANGE_M` for why.

    Returns ``None`` when fewer than ``min_points`` points survive projection,
    when the surviving points span less than 1 m of range (the fit would be
    numerically meaningless), or when the residual exceeds
    ``max_rms_residual_m`` -- a fit that bad is not a lane boundary.
    """
    ground = [p for p in camera.project_points(points_px) if p[1] <= float(max_range_m)]
    if len(ground) < int(min_points):
        return None
    arr = np.asarray(ground, dtype=np.float64)
    xs = arr[:, 0]
    zs = arr[:, 1]
    if float(zs.max() - zs.min()) < 1.0:
        return None
    degree = 2 if len(ground) >= 5 else 1
    try:
        coeff = np.polyfit(zs, xs, degree)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if degree == 1:
        coeff = np.array([0.0, coeff[0], coeff[1]], dtype=np.float64)
    if not np.all(np.isfinite(coeff)):
        return None
    residual = xs - np.polyval(coeff, zs)
    rms = float(np.sqrt(float(np.mean(residual * residual))))
    if not math.isfinite(rms) or rms > float(max_rms_residual_m):
        return None
    return GroundPolyline(
        coeffs=(float(coeff[0]), float(coeff[1]), float(coeff[2])),
        z_min_m=float(zs.min()),
        z_max_m=float(zs.max()),
        num_points=len(ground),
        rms_residual_m=rms,
    )


def curvature_radius_m(coeffs: Sequence[float], z_m: float = 0.0) -> float:
    """Signed-magnitude radius of curvature (m) of ``X = a Z^2 + b Z + c`` at ``Z``.

    ``R = (1 + (dX/dZ)^2)^1.5 / |d2X/dZ2|``. A straight lane has ``a == 0`` and
    an infinite radius; :data:`MAX_CURVATURE_RADIUS_M` is returned instead
    because the validation layer requires a finite number.
    """
    a, b, _c = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
    second = 2.0 * a
    if abs(second) < 1e-9:
        return MAX_CURVATURE_RADIUS_M
    first = 2.0 * a * float(z_m) + b
    radius = (1.0 + first * first) ** 1.5 / abs(second)
    if not math.isfinite(radius):
        return MAX_CURVATURE_RADIUS_M
    return float(min(radius, MAX_CURVATURE_RADIUS_M))


@dataclass
class LaneGeometry:
    """Metric description of the ego lane, computed on the road plane.

    Attributes
    ----------
    lateral_offset_m:
        Signed distance from the vehicle centreline to the lane centreline
        abeam the camera. **Positive means the vehicle is to the RIGHT of the
        lane centre** and must steer left. This is the quantity
        ``SafetyConfig.max_lateral_offset_m`` was always meant to bound.
    heading_error_rad:
        Vehicle heading relative to the lane tangent. Positive means the
        vehicle is pointing to the RIGHT of the lane direction.
    curvature_radius_m:
        Radius of the lane centreline at the vehicle, capped at
        :data:`MAX_CURVATURE_RADIUS_M` for a straight lane.
    lane_width_m:
        Separation of the two boundaries abeam the camera. ``0.0`` when only one
        boundary was measured and the other was assumed.
    both_boundaries:
        ``False`` when one side was synthesised from
        :data:`NOMINAL_LANE_WIDTH_M`; treat the offset as much weaker evidence.
    plausible:
        ``False`` when ``lane_width_m`` falls outside
        ``[MIN_PLAUSIBLE_LANE_WIDTH_M, MAX_PLAUSIBLE_LANE_WIDTH_M]``. A false
        value means the geometry is almost certainly a mis-association of lane
        boundaries and should not steer anything.
    """

    lateral_offset_m: float = 0.0
    heading_error_rad: float = 0.0
    curvature_radius_m: float = MAX_CURVATURE_RADIUS_M
    lane_width_m: float = 0.0
    both_boundaries: bool = False
    plausible: bool = False
    eval_range_m: float = 0.0


def lane_geometry(
    left: Optional[GroundPolyline],
    right: Optional[GroundPolyline],
    eval_range_m: Optional[float] = None,
    nominal_lane_width_m: float = NOMINAL_LANE_WIDTH_M,
) -> Optional[LaneGeometry]:
    """Combine two fitted boundaries into the metric ego-lane description.

    ``eval_range_m`` is the look-ahead at which the offset, heading and
    curvature are evaluated. ``None`` (the default) means "the nearest range
    both boundaries actually observed". A supplied value is CLAMPED into the
    fitted support: a quadratic fitted over 3-12 m says nothing about 0 m or
    40 m, and evaluating it there produced 5 m "lane widths" from a perfectly
    good 3.5 m lane. The value actually used is reported back in
    :attr:`LaneGeometry.eval_range_m`.

    Returns ``None`` when neither boundary is available. With only one boundary
    the other is placed ``nominal_lane_width_m`` away in the appropriate
    direction and ``both_boundaries`` is ``False``.
    """
    if left is None and right is None:
        return None
    supports = [p for p in (left, right) if p is not None]
    z_lo = max(p.z_min_m for p in supports)
    z_hi = min(p.z_max_m for p in supports)
    if z_hi < z_lo:  # disjoint supports; fall back to the nearer boundary's window
        z_lo, z_hi = min(p.z_min_m for p in supports), max(p.z_max_m for p in supports)
    if eval_range_m is None:
        z = float(z_lo)
    else:
        z = float(min(max(float(eval_range_m), z_lo), z_hi))
    if left is not None and right is not None:
        x_left = left.x_at(z)
        x_right = right.x_at(z)
        slope = (left.dx_dz_at(z) + right.dx_dz_at(z)) / 2.0
        centre_coeffs = tuple(
            (float(l) + float(r)) / 2.0 for l, r in zip(left.coeffs, right.coeffs)
        )
        width = x_right - x_left
        both = True
    elif left is not None:
        x_left = left.x_at(z)
        x_right = x_left + nominal_lane_width_m
        slope = left.dx_dz_at(z)
        centre_coeffs = (left.coeffs[0], left.coeffs[1], left.coeffs[2] + nominal_lane_width_m / 2.0)
        width = 0.0
        both = False
    else:
        x_right = right.x_at(z)
        x_left = x_right - nominal_lane_width_m
        slope = right.dx_dz_at(z)
        centre_coeffs = (right.coeffs[0], right.coeffs[1], right.coeffs[2] - nominal_lane_width_m / 2.0)
        width = 0.0
        both = False

    centre_x = (x_left + x_right) / 2.0
    plausible = (not both) or (
        MIN_PLAUSIBLE_LANE_WIDTH_M <= width <= MAX_PLAUSIBLE_LANE_WIDTH_M
    )
    return LaneGeometry(
        # The vehicle sits at X = 0; if the lane centre is at X = +1 the vehicle
        # is 1 m to the LEFT of it, hence the sign flip.
        lateral_offset_m=float(-centre_x),
        heading_error_rad=float(-math.atan(slope)),
        curvature_radius_m=curvature_radius_m(centre_coeffs, z),
        lane_width_m=float(width),
        both_boundaries=both,
        plausible=bool(plausible),
        eval_range_m=z,
    )


def ground_centreline_to_pixel(
    left: Optional[GroundPolyline],
    right: Optional[GroundPolyline],
    camera: CameraConfig,
    eval_range_m: float = 10.0,
    nominal_lane_width_m: float = NOMINAL_LANE_WIDTH_M,
) -> Optional[float]:
    """Column of the lane centre at ``eval_range_m``, in source-frame pixels.

    The behaviour planner still steers on a pixel error, so the metric fit has
    to come back to pixels for it. ``eval_range_m`` must be strictly positive:
    a road-plane point abeam the camera (Z = 0) projects far below the image.
    Returns ``None`` when ``eval_range_m <= 0`` or when the centre point does
    not project into the image.
    """
    if not (float(eval_range_m) > 0.0):
        return None
    geom_left = left.x_at(eval_range_m) if left is not None else None
    geom_right = right.x_at(eval_range_m) if right is not None else None
    if geom_left is None and geom_right is None:
        return None
    if geom_left is None:
        geom_left = geom_right - nominal_lane_width_m
    if geom_right is None:
        geom_right = geom_left + nominal_lane_width_m
    centre_x = (geom_left + geom_right) / 2.0
    projected = camera.ground_to_image(centre_x, float(eval_range_m))
    if projected is None:
        return None
    return float(projected[0])


# --------------------------------------------------------------------------- #
# Monocular self-calibration from lane geometry
# --------------------------------------------------------------------------- #


def estimate_vanishing_point(
    left_points_px: Sequence[Tuple[float, float]],
    right_points_px: Sequence[Tuple[float, float]],
    min_points: int = 4,
) -> Optional[Tuple[float, float]]:
    """Intersect two lane boundaries in image space to find the road vanishing point.

    Both boundaries are fitted as straight lines ``x = m*y + b`` (a lane is
    locally straight, and the straight-line fit is what makes the intersection
    well defined) and intersected. The result is the horizon row of the ROAD
    plane, which is what :meth:`CameraConfig.with_horizon_px` needs -- not the
    visual skyline, which sits somewhere else entirely on a road that crests or
    dips.

    Returns ``(u_px, v_px)``, or ``None`` when either side has fewer than
    ``min_points`` points, when either fit is degenerate, or when the two lines
    are near-parallel (``|m_left - m_right| < 1e-6``) and the intersection is
    numerically meaningless.
    """
    if left_points_px is None or right_points_px is None:
        return None
    if len(left_points_px) < int(min_points) or len(right_points_px) < int(min_points):
        return None

    def _line(points):
        arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        ys = arr[:, 1]
        if float(ys.max() - ys.min()) < 2.0:
            return None
        try:
            m, b = np.polyfit(ys, arr[:, 0], 1)
        except (np.linalg.LinAlgError, ValueError):
            return None
        if not (math.isfinite(m) and math.isfinite(b)):
            return None
        return float(m), float(b)

    left = _line(left_points_px)
    right = _line(right_points_px)
    if left is None or right is None:
        return None
    ml, bl = left
    mr, br = right
    denom = ml - mr
    if abs(denom) < 1e-6:
        return None
    v = (br - bl) / denom
    u = ml * v + bl
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return (float(u), float(v))


def calibrate_mount_height_from_lane(
    camera: CameraConfig,
    left_points_px: Sequence[Tuple[float, float]],
    right_points_px: Sequence[Tuple[float, float]],
    known_lane_width_m: float = 3.65,
    max_range_m: float = DEFAULT_FIT_RANGE_M,
) -> Optional[float]:
    """Solve for the mount height that makes the observed lane the known width.

    Once the pitch is right, mount height is the only remaining scale freedom:
    every ground coordinate is exactly proportional to it. So measuring the lane
    at the current height and scaling by ``known_lane_width_m / measured`` gives
    the height directly.

    ``left_points_px`` and ``right_points_px`` must be sampled at the SAME image
    rows (as UFLD's row head produces them); pairs whose rows differ by more
    than half a pixel are skipped. Returns the height in metres, or ``None``
    when fewer than 3 usable pairs project onto the road plane.

    This is a coarse field calibration, not a substitute for a chequerboard: it
    inherits every error in ``fy`` and in the lane-width assumption.
    """
    pairs = []
    for (lx, ly), (rx, ry) in zip(left_points_px or [], right_points_px or []):
        if abs(float(ly) - float(ry)) > 0.5:
            continue
        gl = camera.image_to_ground(float(lx), float(ly))
        gr = camera.image_to_ground(float(rx), float(ry))
        if gl is None or gr is None:
            continue
        if gl[1] > float(max_range_m):
            continue
        width = gr[0] - gl[0]
        if width > 0.0:
            pairs.append(width)
    if len(pairs) < 3:
        return None
    measured = float(np.median(pairs))
    if measured <= 1e-6:
        return None
    height = camera.mount_height_m * float(known_lane_width_m) / measured
    if not math.isfinite(height) or height <= 0.0:
        return None
    return float(height)


def calibrate_from_lane_observations(
    camera: CameraConfig,
    left_points_px: Sequence[Tuple[float, float]],
    right_points_px: Sequence[Tuple[float, float]],
    known_lane_width_m: float = 3.65,
    label: str = "lane-calibrated",
) -> Optional[CameraConfig]:
    """Two-step monocular extrinsic calibration from one pair of lane boundaries.

    1. Pitch from the lane vanishing point (:func:`estimate_vanishing_point`).
    2. Mount height from the known lane width
       (:func:`calibrate_mount_height_from_lane`).

    Returns a camera marked ``calibrated=True``, or ``None`` when either step
    fails. Intrinsics are NOT touched -- ``fx``/``fy``/``cx``/``cy`` still come
    from wherever they came from, and a wrong ``fy`` propagates straight into
    the recovered height.

    Single-frame calibration is noisy. Average the vanishing point over many
    straight-road frames before trusting it; a hard brake, a crest or a banked
    curve each move it by tens of pixels.
    """
    vp = estimate_vanishing_point(left_points_px, right_points_px)
    if vp is None:
        return None
    repitched = camera.with_horizon_px(vp[1], calibrated=False, label=label)
    height = calibrate_mount_height_from_lane(
        repitched, left_points_px, right_points_px, known_lane_width_m
    )
    if height is None:
        return None
    return repitched.with_mount_height(height, calibrated=True, label=label)


class LaneCalibrator:
    """Accumulate lane observations over frames, then emit a calibrated camera.

    Single-frame :func:`calibrate_from_lane_observations` is noisy: on this
    board's replay clip the per-frame vanishing point scatters by about +/-7 px
    vertically, which is +/-0.5 degrees of pitch. Feeding many straight-road
    frames in and taking the median collapses that.

    Usage::

        cal = LaneCalibrator(base_camera)
        for frame in clip:
            model = estimator.estimate(frame, w, h)
            cal.observe_lane_model(model)
        camera = cal.camera()          # None until min_samples is reached

    ``camera()`` returns ``None`` until at least ``min_samples`` usable
    observations have arrived, and never returns a camera whose implied mount
    height is outside ``height_bounds_m`` -- a solved height of 4 m means the
    lane-width assumption or ``fy`` is wrong, and shipping it would silently
    scale every range.

    Failure behaviour: observations that do not yield a vanishing point inside
    the image, or that yield no height solution, are counted in
    :attr:`rejected` and otherwise ignored.
    """

    def __init__(
        self,
        base_camera: CameraConfig,
        known_lane_width_m: float = 3.65,
        min_samples: int = 30,
        height_bounds_m: Tuple[float, float] = (0.5, 3.0),
    ) -> None:
        self.base_camera = base_camera
        self.known_lane_width_m = float(known_lane_width_m)
        self.min_samples = int(min_samples)
        self.height_bounds_m = (float(height_bounds_m[0]), float(height_bounds_m[1]))
        self._vanishing_v: List[float] = []
        self._observations: List[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]] = []
        self.rejected = 0

    def observe(
        self,
        left_points_px: Sequence[Tuple[float, float]],
        right_points_px: Sequence[Tuple[float, float]],
    ) -> bool:
        """Add one frame's ego-lane boundaries. Returns True if it was usable."""
        vp = estimate_vanishing_point(left_points_px, right_points_px)
        if vp is None:
            self.rejected += 1
            return False
        u, v = vp
        cam = self.base_camera
        if not (0.0 <= u < cam.image_width and 0.0 <= v < cam.image_height):
            self.rejected += 1
            return False
        self._vanishing_v.append(float(v))
        self._observations.append(
            ([(float(x), float(y)) for x, y in left_points_px],
             [(float(x), float(y)) for x, y in right_points_px])
        )
        return True

    def observe_lane_model(self, model) -> bool:
        """Convenience wrapper: pull the ego boundaries out of a ``LaneModel``.

        Ignores mock models and models without both ego boundaries (indices 1
        and 2).
        """
        if model is None or getattr(model, "is_mock", True):
            self.rejected += 1
            return False
        left = None
        right = None
        for line in getattr(model, "lines", ()):
            if line.index == 1:
                left = line.points_px
            elif line.index == 2:
                right = line.points_px
        if not left or not right:
            self.rejected += 1
            return False
        return self.observe(left, right)

    @property
    def num_samples(self) -> int:
        return len(self._vanishing_v)

    def camera(self, label: str = "lane-calibrated") -> Optional[CameraConfig]:
        """Solve for pitch then mount height. ``None`` until enough samples."""
        if self.num_samples < self.min_samples:
            return None
        horizon = float(np.median(np.asarray(self._vanishing_v, dtype=np.float64)))
        repitched = self.base_camera.with_horizon_px(horizon, calibrated=False, label=label)
        heights = []
        for left, right in self._observations:
            h = calibrate_mount_height_from_lane(
                repitched, left, right, self.known_lane_width_m
            )
            if h is not None:
                heights.append(h)
        if len(heights) < self.min_samples // 2:
            return None
        height = float(np.median(np.asarray(heights, dtype=np.float64)))
        lo, hi = self.height_bounds_m
        if not (lo <= height <= hi):
            logger.warning(
                "LaneCalibrator solved mount height %.2f m, outside the sanity band "
                "[%.2f, %.2f]; refusing to publish a calibration. Check fy and the "
                "assumed lane width (%.2f m).",
                height,
                lo,
                hi,
                self.known_lane_width_m,
            )
            return None
        camera = repitched.with_mount_height(height, calibrated=True, label=label)
        logger.info(
            "LaneCalibrator: %d samples -> horizon %.1f px (pitch %.2f deg), "
            "mount height %.3f m",
            self.num_samples,
            horizon,
            camera.pitch_deg,
            height,
        )
        return camera
