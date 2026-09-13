"""Input validation for the ADAS pipeline.

Every function here either returns ``None`` or raises
:class:`~adas.core.exceptions.ValidationError`.  None of them repair a value --
repair is a policy decision that belongs to the controller and the safety
arbiter, both of which validate their own output rather than trusting a caller
to have validated its input.

Failure behaviour: a raise means "this value must not be used".  Callers on the
actuation path must translate that into their own fail-safe (a zero-throttle,
rate-shaped brake), never into a clamp of the offending number -- clamping a NaN
brake to 1.0 is how ``sanitize_control_command`` used to ship a full ABS stop on
corrupt input.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from adas.core.exceptions import ValidationError
from adas.core.models import (
    BoundingBox,
    ControlCommand,
    EgoState,
    LaneModel,
    MotionPlan,
    PerceptionFrame,
)


def validate_bounding_box(box: BoundingBox) -> None:
    """Validate one detection box, in source-image pixels.

    Args:
        box: box to validate.

    Raises:
        ValidationError: negative coordinates, non-positive width or height,
            a non-finite coordinate, or a confidence outside ``[0, 1]``.
    """
    for name in ("x1", "y1", "x2", "y2"):
        value = getattr(box, name)
        if not math.isfinite(value):
            raise ValidationError("Invalid box coordinate %s=%r" % (name, value))

    if box.x1 < 0 or box.y1 < 0:
        raise ValidationError("Negative coordinates not allowed: (%s, %s)" % (box.x1, box.y1))

    if box.x2 <= box.x1:
        raise ValidationError("Invalid box width: x2=%s <= x1=%s" % (box.x2, box.x1))

    if box.y2 <= box.y1:
        raise ValidationError("Invalid box height: y2=%s <= y1=%s" % (box.y2, box.y1))

    if not math.isfinite(box.confidence) or not (0.0 <= box.confidence <= 1.0):
        raise ValidationError("Confidence must be in [0, 1], got %s" % (box.confidence,))


def validate_lane_model(lane: LaneModel) -> None:
    """Validate a lane model.

    Args:
        lane: lane model to validate.

    Raises:
        ValidationError: non-finite lane centre or curvature, or boundary
            coefficients that are not quadratic triples.

    Note:
        A curvature of ``inf`` is rejected: "straight" is represented by a large
        finite radius (``MAX_CURVATURE_RADIUS_M``), because ``inf`` propagates
        into every downstream arithmetic expression as ``nan``.
    """
    if not math.isfinite(lane.lane_center_px):
        raise ValidationError("Invalid lane center: %s" % (lane.lane_center_px,))
    if not math.isfinite(lane.curvature_m):
        raise ValidationError("Invalid curvature: %s" % (lane.curvature_m,))
    if len(lane.left_coeffs) != 3 or len(lane.right_coeffs) != 3:
        raise ValidationError("Lane coefficients must be quadratic (3 terms)")
    if not math.isfinite(lane.confidence) or not (0.0 <= lane.confidence <= 1.0):
        raise ValidationError("Lane confidence must be in [0, 1], got %s" % (lane.confidence,))


def validate_motion_plan(plan: MotionPlan) -> None:
    """Validate a motion plan.

    Args:
        plan: plan to validate.

    Raises:
        ValidationError: non-finite or negative target speed, or a non-finite
            steering angle.  The *magnitude* of the steering angle is a policy
            limit checked by the safety arbiter, not a validity question.
    """
    if not math.isfinite(plan.target_speed_mps):
        raise ValidationError("Invalid target speed: %s" % (plan.target_speed_mps,))

    if plan.target_speed_mps < 0:
        raise ValidationError("Negative speed not allowed: %s" % (plan.target_speed_mps,))

    if not math.isfinite(plan.steering_angle_deg):
        raise ValidationError("Invalid steering angle: %s" % (plan.steering_angle_deg,))


def validate_control_command(cmd: ControlCommand) -> None:
    """Validate an actuator command.

    This is the last gate before a command reaches an actuator interface, so it
    checks every field.  The previous version checked only ``throttle`` (and
    permitted it to be negative) and never looked at ``brake`` at all, which let
    a NaN brake through to a sanitiser whose ``max(0, min(1, nan))`` evaluates to
    ``1.0`` -- a full-authority stop synthesised out of corrupt data.

    Args:
        cmd: command to validate.

    Raises:
        ValidationError: any field non-finite, ``throttle`` outside ``[0, 1]``,
            ``brake`` outside ``[0, 1]``, ``steering`` outside ``[-1, 1]``, or
            throttle and brake both commanded at once.
    """
    if not math.isfinite(cmd.throttle):
        raise ValidationError("Invalid throttle: %s" % (cmd.throttle,))
    if not (0.0 <= cmd.throttle <= 1.0):
        raise ValidationError("Throttle must be in [0, 1], got %s" % (cmd.throttle,))

    if not math.isfinite(cmd.brake):
        raise ValidationError("Invalid brake: %s" % (cmd.brake,))
    if not (0.0 <= cmd.brake <= 1.0):
        raise ValidationError("Brake must be in [0, 1], got %s" % (cmd.brake,))

    if not math.isfinite(cmd.steering):
        raise ValidationError("Invalid steering: %s" % (cmd.steering,))
    if not (-1.0 <= cmd.steering <= 1.0):
        raise ValidationError("Steering must be in [-1, 1], got %s" % (cmd.steering,))

    if cmd.throttle > 0.0 and cmd.brake > 0.0:
        raise ValidationError(
            "Throttle and brake must not be commanded together, got throttle=%s brake=%s"
            % (cmd.throttle, cmd.brake)
        )


def validate_ego_state(ego: Optional[EgoState]) -> None:
    """Validate an ego state that claims to be valid.

    An ``EgoState`` with ``valid=False`` is always acceptable: it is the honest
    representation of "no measurement", and the planner and arbiter both have a
    defined degraded behaviour for it.  A state that claims ``valid=True`` must
    carry usable numbers.

    Args:
        ego: state to validate, or ``None``.

    Raises:
        ValidationError: ``valid`` is True but the speed is non-finite, negative
            or implausibly large, or the yaw rate / acceleration is non-finite.
    """
    if ego is None or not ego.valid:
        return
    if not math.isfinite(ego.speed_mps):
        raise ValidationError("Ego speed must be finite when valid, got %s" % (ego.speed_mps,))
    if ego.speed_mps < 0.0:
        raise ValidationError("Ego speed must be non-negative, got %s" % (ego.speed_mps,))
    if ego.speed_mps > 120.0:
        raise ValidationError("Ego speed is implausible: %s m/s" % (ego.speed_mps,))
    if not math.isfinite(ego.yaw_rate_dps):
        raise ValidationError("Ego yaw rate must be finite, got %s" % (ego.yaw_rate_dps,))
    if not math.isfinite(ego.accel_mps2):
        raise ValidationError("Ego acceleration must be finite, got %s" % (ego.accel_mps2,))


def validate_image_dimensions(width: int, height: int) -> None:
    """Validate image dimensions in pixels.

    Raises:
        ValidationError: non-positive or absurdly large dimensions.
    """
    if width <= 0 or height <= 0:
        raise ValidationError("Invalid image dimensions: %sx%s" % (width, height))

    if width > 10000 or height > 10000:
        raise ValidationError("Image dimensions too large: %sx%s" % (width, height))


def validate_perception_frame(frame: PerceptionFrame) -> None:
    """Validate the frame envelope before any model touches it.

    Checks the dimensions, the timestamp and -- when the payload is a numpy
    array -- that its shape agrees with the declared width and height.  A frame
    whose declared size disagrees with its buffer is the failure that makes every
    downstream geometric result quietly wrong (ADAS-PERC-26).

    Raises:
        ValidationError: bad dimensions, a non-finite timestamp, or an
            array shape that contradicts ``width``/``height``.
    """
    validate_image_dimensions(frame.width, frame.height)
    if not math.isfinite(frame.timestamp_s):
        raise ValidationError("Frame timestamp must be finite, got %s" % (frame.timestamp_s,))
    shape = getattr(frame.rgb, "shape", None)
    if shape is not None and len(shape) >= 2:
        rows, cols = int(shape[0]), int(shape[1])
        if rows != int(frame.height) or cols != int(frame.width):
            raise ValidationError(
                "Frame buffer is %dx%d but the frame declares %dx%d"
                % (cols, rows, frame.width, frame.height)
            )


def validate_config_value(
    name: str,
    value: Any,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> None:
    """Validate one numeric configuration value.

    Args:
        name: dotted parameter name, used verbatim in the error message so an
            operator can find the key in their JSON file.
        value: value to validate.
        min_val: inclusive lower bound, or ``None``.
        max_val: inclusive upper bound, or ``None``.

    Raises:
        ValidationError: non-numeric, non-finite, or out of range.  Booleans are
            rejected as non-numeric: ``"fps": true`` is a mistake, not 1 Hz.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("%s must be numeric, got %r" % (name, value))

    if not math.isfinite(value):
        raise ValidationError("%s must be finite, got %s" % (name, value))

    if min_val is not None and value < min_val:
        raise ValidationError("%s must be >= %s, got %s" % (name, min_val, value))

    if max_val is not None and value > max_val:
        raise ValidationError("%s must be <= %s, got %s" % (name, max_val, value))


__all__ = [
    "validate_bounding_box",
    "validate_config_value",
    "validate_control_command",
    "validate_ego_state",
    "validate_image_dimensions",
    "validate_lane_model",
    "validate_motion_plan",
    "validate_perception_frame",
]
