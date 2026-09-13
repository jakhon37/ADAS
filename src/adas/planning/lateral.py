"""Lateral (steering) planning: speed-scheduled lane centring with a hard a_lat cap.

Units
-----
``lane_center_px``       pixels, image column of the lane centre.
``lateral_error_m``      metres, signed offset of the ego from the lane centre
                         (positive = ego is left of centre, i.e. the lane centre is
                         to the RIGHT, i.e. steer right).  Only populated on the
                         metric path.
``steering_angle_deg``   degrees of ROAD-WHEEL angle requested.
``ego_speed_mps``        metres per second.
``max_lateral_accel``    m/s^2.

Two laws, one interface
-----------------------
1. METRIC (preferred).  When a :class:`CameraGeometry` is configured AND the image
   row of the lane-centre sample is known, the pixel column is projected onto the
   flat ground plane and a Stanley law is used::

       delta = atan(k * e_lat / (v + v_soft))

   ``SteeringDecision.is_metric`` is True and ``lateral_error_m`` is a real metric
   quantity.

2. NON-METRIC fallback (what ships today).  Without camera extrinsics there is no
   honest pixel-to-metre conversion, so the law regulates the NORMALISED pixel
   error instead::

       e_frac  = (lane_center_px - width/2) / (width/2)      in [-1, 1]
       delta   = clamp(gain(v) * e_frac, -1, 1) * max_steering_deg

   ``SteeringDecision.is_metric`` is False and ``lateral_error_m`` is None.  The
   number is a *pseudo*-angle and is labelled as such; it is NEVER presented as a
   measured lateral offset.  ``RuntimeConfig`` currently has no camera mount height
   or pitch, which is why this is the shipped path.

Both laws are speed-scheduled and both are subject to the same hard cap:

    a_lat = v^2 * tan(delta) / L  <=  max_lateral_accel_mps2

which, solved for delta, gives ``delta_max = atan(a_lat_max * L / v^2)``.  At
33 m/s with a 3.0 m/s^2 cap and a 2.8 m wheelbase that is 0.44 deg of road wheel --
small, and correct: anything larger is a rollover/spin command.  The cap is
enforced here AND re-checked independently by
:class:`adas.control.arbiter.SafetyArbiter`, which does not trust this module.

Slew rate
---------
This planner emits a steering SETPOINT and deliberately does NOT rate-limit it.
Slew-rate limiting is enforced authoritatively by the safety arbiter against the
last command that actually reached the actuators, which is the only place that
knows what the actuators last did.

Failure behaviour
-----------------
* ``lane_center_px is None``      -> 0 deg, reason ``no_lane``.  Straight ahead;
  the longitudinal side is what keeps the vehicle safe when the lane is lost.
* lane centre outside the frame   -> 0 deg, reason ``invalid_lane``.  A lane
  estimator that returns an out-of-frame column tends to keep doing it, so the
  WARNING is gated (once on entry, at most once per gate period, once on
  recovery); ``SteeringDecision.reason`` carries it on every frame.
* non-finite ego speed            -> treated as unknown; the gain schedule uses the
  most restrictive assumption (the configured maximum speed) rather than 0.
* ``frame_width_px <= 0``         -> :class:`ValidationError`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from adas.core.exceptions import ValidationError
from adas.core.logger import setup_logger
from adas.core.models import LaneModel
from adas.planning.longitudinal import LogGate

logger = setup_logger(__name__)


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


@dataclass
class CameraGeometry:
    """Pinhole + flat-ground extrinsics needed to turn pixels into metres.

    The camera frame is X right, Y down, Z forward.  ``pitch_rad`` is positive when
    the camera looks DOWN.  ``mount_height_m`` is the height of the optical centre
    above the road surface.

    An instance with the defaults is *not configured*: :meth:`is_configured`
    returns False and every consumer must fall back to the non-metric path rather
    than invent a conversion.
    """

    focal_length_px: float = 0.0
    principal_x_px: float = 0.0
    principal_y_px: float = 0.0
    mount_height_m: float = 0.0
    pitch_rad: float = 0.0

    @classmethod
    def from_camera_config(cls, config, allow_uncalibrated: bool = False) -> "CameraGeometry":
        """Adapt an ``adas.perception.geometry.CameraConfig``.

        Duck-typed on the attribute names (``fx``, ``cx``, ``cy``,
        ``mount_height_m``, ``pitch_deg``, ``calibrated``) so this module does not
        import the perception package and cannot be broken by a change there.

        An UNCALIBRATED config yields an UNCONFIGURED geometry unless
        ``allow_uncalibrated`` is set explicitly. An assumed mount height would
        turn the Stanley law's ``lateral_error_m`` into a fabricated measurement
        wearing metric units, which is exactly what ``is_metric`` exists to
        prevent. Set the flag only on a bench, never in a vehicle profile.

        Args:
            config: Any object exposing the attribute names above.
            allow_uncalibrated: Opt in to using assumed extrinsics.

        Returns:
            A :class:`CameraGeometry`; check :meth:`is_configured` before relying
            on it.
        """
        calibrated = bool(getattr(config, "calibrated", False))
        if not calibrated and not allow_uncalibrated:
            logger.warning(
                "CameraConfig %r is not calibrated; the lateral law stays on its "
                "non-metric pixel path",
                getattr(config, "label", "<unnamed>"),
            )
            return cls()
        return cls(
            focal_length_px=float(getattr(config, "fx", 0.0)),
            principal_x_px=float(getattr(config, "cx", 0.0)),
            principal_y_px=float(getattr(config, "cy", 0.0)),
            mount_height_m=float(getattr(config, "mount_height_m", 0.0)),
            pitch_rad=math.radians(float(getattr(config, "pitch_deg", 0.0))),
        )

    def is_configured(self) -> bool:
        """True only when a real focal length and mount height are present."""

        return (
            math.isfinite(self.focal_length_px)
            and self.focal_length_px > 1.0
            and math.isfinite(self.mount_height_m)
            and self.mount_height_m > 0.1
        )

    def ground_point_m(self, u_px: float, v_px: float) -> tuple[float, float] | None:
        """Project an image point onto the flat ground plane.

        Args:
            u_px: image column.
            v_px: image row.

        Returns:
            ``(lateral_m, forward_m)`` with lateral positive to the right of the
            camera axis and forward strictly positive, or None when the ray does
            not intersect the ground ahead of the vehicle (at or above the
            horizon).
        """
        if not self.is_configured():
            return None
        x = (u_px - self.principal_x_px) / self.focal_length_px
        y = (v_px - self.principal_y_px) / self.focal_length_px
        cos_p = math.cos(self.pitch_rad)
        sin_p = math.sin(self.pitch_rad)
        # Rotate the camera-frame ray into the (gravity-aligned) vehicle frame.
        down = y * cos_p + sin_p
        forward_dir = cos_p - y * sin_p
        if down <= 1e-6 or forward_dir <= 1e-6:
            return None
        scale = self.mount_height_m / down
        forward_m = scale * forward_dir
        if not math.isfinite(forward_m) or forward_m <= 0.0:
            return None
        return scale * x, forward_m


@dataclass
class LateralLimits:
    """Tuning and hard limits for :class:`LateralPlanner`."""

    max_steering_deg: float = 22.0
    """Largest road-wheel angle the planner will ever request."""
    wheelbase_m: float = 2.8
    max_lateral_accel_mps2: float = 3.0
    lane_center_gain: float = 1.0
    speed_ref_mps: float = 12.0
    """Gain schedule reference: the pixel gain is halved at this speed."""
    stanley_gain: float = 1.2
    """Metric path only, units 1/s."""
    stanley_softening_mps: float = 2.0
    """Metric path only; keeps the law finite at standstill."""
    min_scheduling_speed_mps: float = 1.0
    """Below this the a_lat cap is not applied (v^2 makes it vacuous anyway)."""
    assumed_speed_when_unknown_mps: float = 33.0
    """Used for the gain schedule and a_lat cap when ego speed is unavailable."""

    def __post_init__(self) -> None:
        if self.max_steering_deg <= 0:
            raise ValidationError(f"max_steering_deg must be positive, got {self.max_steering_deg}")
        if self.wheelbase_m <= 0:
            raise ValidationError(f"wheelbase_m must be positive, got {self.wheelbase_m}")
        if self.max_lateral_accel_mps2 <= 0:
            raise ValidationError(
                f"max_lateral_accel_mps2 must be positive, got {self.max_lateral_accel_mps2}"
            )
        if self.speed_ref_mps <= 0:
            raise ValidationError(f"speed_ref_mps must be positive, got {self.speed_ref_mps}")
        if self.lane_center_gain < 0:
            raise ValidationError("lane_center_gain must be non-negative")


@dataclass
class SteeringDecision:
    """Result of one lateral planning step.

    ``steering_angle_deg`` is a road-wheel angle on the metric path and a
    *pseudo*-angle (normalised pixel error scaled by ``max_steering_deg``) on the
    non-metric path.  ``is_metric`` says which, and no consumer may treat the
    non-metric value as a physical measurement.
    """

    steering_angle_deg: float
    reason: str
    is_metric: bool = False
    lateral_error_m: float | None = None
    lateral_accel_mps2: float = 0.0
    limited_by: str = ""


class LateralPlanner:
    """Lane-centring steering setpoint with speed scheduling and an a_lat cap."""

    def __init__(
        self,
        limits: LateralLimits | None = None,
        camera: CameraGeometry | None = None,
        degraded_log_period_s: float = 60.0,
    ) -> None:
        self.limits = limits or LateralLimits()
        self.camera = camera
        self._invalid_lane_gate = LogGate(degraded_log_period_s)

    def reset(self) -> None:
        """Clear the log latch. The steering law itself carries no state."""
        self._invalid_lane_gate.reset()

    def max_steering_deg_at(self, ego_speed_mps: float | None) -> float:
        """Largest road-wheel angle allowed at this speed by the a_lat cap.

        ``delta_max = atan(a_lat_max * L / v^2)``, capped at ``max_steering_deg``.
        """
        lim = self.limits
        speed = self._effective_speed(ego_speed_mps)
        if speed <= lim.min_scheduling_speed_mps:
            return lim.max_steering_deg
        delta_max_rad = math.atan(lim.max_lateral_accel_mps2 * lim.wheelbase_m / (speed * speed))
        return min(lim.max_steering_deg, math.degrees(delta_max_rad))

    def lateral_accel_mps2(self, steering_deg: float, ego_speed_mps: float | None) -> float:
        """Bicycle-model lateral acceleration for this road-wheel angle and speed."""
        speed = self._effective_speed(ego_speed_mps)
        return (speed * speed) * math.tan(math.radians(steering_deg)) / self.limits.wheelbase_m

    def plan(
        self,
        frame_width_px: int,
        lane_center_px: float | None,
        ego_speed_mps: float | None = None,
        lane: LaneModel | None = None,
        frame_height_px: int | None = None,
    ) -> SteeringDecision:
        """Produce one steering setpoint.

        Args:
            frame_width_px: Image width. Must be > 0.
            lane_center_px: Lane-centre column, or None when the lane is unknown.
            ego_speed_mps: Measured ego speed, or None when unavailable (the most
                restrictive schedule is then used).
            lane: Optional lane model, used only for the confidence gate today.
            frame_height_px: Image height. Required, together with a configured
                camera, to take the metric path.

        Returns:
            A :class:`SteeringDecision` whose ``steering_angle_deg`` always
            satisfies the lateral-acceleration cap at the given speed.
        """
        if frame_width_px <= 0:
            raise ValidationError(f"Invalid frame width: {frame_width_px}")

        if lane_center_px is None or not math.isfinite(lane_center_px):
            return SteeringDecision(0.0, "no_lane", limited_by="no_lane")

        if not (0.0 <= lane_center_px <= float(frame_width_px)):
            emit, dropped = self._invalid_lane_gate.mark()
            if emit:
                logger.warning(
                    "Lane centre %.1f outside frame [0, %d]; steering straight. "
                    "Condition is latched: repeats at most every %.0f s (%d frames "
                    "suppressed since the last line).",
                    lane_center_px,
                    frame_width_px,
                    self._invalid_lane_gate.period_s,
                    dropped,
                )
            return SteeringDecision(0.0, "invalid_lane", limited_by="invalid_lane")
        self._invalid_lane_gate.clear()

        metric = self._metric_law(
            frame_width_px, frame_height_px, lane_center_px, ego_speed_mps
        )
        if metric is not None:
            raw_deg, lateral_error_m, reason = metric
            is_metric = True
        else:
            raw_deg, reason = self._pixel_law(frame_width_px, lane_center_px, ego_speed_mps)
            lateral_error_m = None
            is_metric = False

        limit_deg = self.max_steering_deg_at(ego_speed_mps)
        limited_by = ""
        if abs(raw_deg) > limit_deg:
            limited_by = "lateral_accel_cap" if limit_deg < self.limits.max_steering_deg else "max_steering"
            logger.debug(
                "Steering %.2f deg capped to %.2f deg by %s at %s m/s",
                raw_deg,
                limit_deg,
                limited_by,
                ego_speed_mps,
            )
        steering_deg = _clamp(raw_deg, -limit_deg, limit_deg)

        if lane is not None and not lane.is_mock and lane.confidence <= 0.0:
            # A real lane model that reports zero confidence is not a lane.
            return SteeringDecision(0.0, "lane_confidence_zero", is_metric, None, 0.0, "lane_confidence")

        return SteeringDecision(
            steering_angle_deg=steering_deg,
            reason=reason,
            is_metric=is_metric,
            lateral_error_m=lateral_error_m,
            lateral_accel_mps2=self.lateral_accel_mps2(steering_deg, ego_speed_mps),
            limited_by=limited_by,
        )

    # --------------------------------------------------------------- helpers

    def _effective_speed(self, ego_speed_mps: float | None) -> float:
        """Speed used for scheduling: the measured one, else the worst case."""
        if ego_speed_mps is None or not math.isfinite(ego_speed_mps) or ego_speed_mps < 0.0:
            return self.limits.assumed_speed_when_unknown_mps
        return ego_speed_mps

    def _pixel_law(
        self, frame_width_px: int, lane_center_px: float, ego_speed_mps: float | None
    ) -> tuple[float, str]:
        """Non-metric fallback: speed-scheduled proportional control on pixel error."""
        lim = self.limits
        img_center = frame_width_px / 2.0
        error_frac = (lane_center_px - img_center) / img_center
        speed = self._effective_speed(ego_speed_mps)
        gain = lim.lane_center_gain / (1.0 + speed / lim.speed_ref_mps)
        normalised = _clamp(gain * error_frac, -1.0, 1.0)
        return normalised * lim.max_steering_deg, "lane_center_err_%.2f" % error_frac

    def _metric_law(
        self,
        frame_width_px: int,
        frame_height_px: int | None,
        lane_center_px: float,
        ego_speed_mps: float | None,
    ) -> tuple[float, float, str] | None:
        """Stanley law on a ground-plane lateral offset, or None if not available."""
        if self.camera is None or not self.camera.is_configured():
            return None
        if frame_height_px is None or frame_height_px <= 0:
            return None
        # Sample the lane centre at the bottom of the image, the closest ground row
        # the camera can see, then project it.
        ground = self.camera.ground_point_m(lane_center_px, float(frame_height_px - 1))
        if ground is None:
            return None
        lateral_m, forward_m = ground
        lim = self.limits
        speed = max(0.0, self._effective_speed(ego_speed_mps))
        delta_rad = math.atan(
            lim.stanley_gain * lateral_m / (speed + lim.stanley_softening_mps)
        )
        reason = "lane_offset_%.2fm_at_%.1fm" % (lateral_m, forward_m)
        return math.degrees(delta_rad), lateral_m, reason
