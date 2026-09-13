"""Tests for :mod:`adas.core.validation`.

The command-validation tests are the ones that matter: every gate here exists
because its absence let a corrupt value reach an actuator interface.
"""

from __future__ import annotations

import math

import pytest

from adas.core.exceptions import ValidationError
from adas.core.models import (
    BoundingBox,
    ControlCommand,
    EgoState,
    LaneLine,
    LaneModel,
    MotionPlan,
    PerceptionFrame,
)
from adas.core.validation import (
    validate_bounding_box,
    validate_config_value,
    validate_control_command,
    validate_ego_state,
    validate_image_dimensions,
    validate_lane_model,
    validate_motion_plan,
    validate_perception_frame,
)

# --------------------------------------------------------------------------- #
# Bounding boxes
# --------------------------------------------------------------------------- #


def test_validate_bounding_box_valid():
    validate_bounding_box(BoundingBox(10, 20, 30, 40, 0.8, "car"))


def test_validate_bounding_box_negative_coords():
    with pytest.raises(ValidationError, match="Negative coordinates"):
        validate_bounding_box(BoundingBox(-10, 20, 30, 40, 0.8, "car"))


def test_validate_bounding_box_invalid_width():
    with pytest.raises(ValidationError, match="Invalid box width"):
        validate_bounding_box(BoundingBox(30, 20, 10, 40, 0.8, "car"))


def test_validate_bounding_box_invalid_height():
    with pytest.raises(ValidationError, match="Invalid box height"):
        validate_bounding_box(BoundingBox(10, 40, 30, 20, 0.8, "car"))


def test_validate_bounding_box_invalid_confidence():
    with pytest.raises(ValidationError, match="Confidence must be"):
        validate_bounding_box(BoundingBox(10, 20, 30, 40, 1.5, "car"))


def test_validate_bounding_box_rejects_nan_coordinates():
    with pytest.raises(ValidationError, match="Invalid box coordinate"):
        validate_bounding_box(BoundingBox(float("nan"), 20, 30, 40, 0.8, "car"))


# --------------------------------------------------------------------------- #
# Motion plans
# --------------------------------------------------------------------------- #


def test_validate_motion_plan_negative_speed():
    with pytest.raises(ValidationError, match="Negative speed"):
        validate_motion_plan(MotionPlan(-5.0, 0.0, "test"))


def test_validate_motion_plan_nan_steering():
    with pytest.raises(ValidationError, match="Invalid steering angle"):
        validate_motion_plan(MotionPlan(10.0, float("nan"), "test"))


def test_validate_motion_plan_accepts_a_large_steering_angle():
    """Magnitude is a policy limit for the arbiter, not a validity question."""
    validate_motion_plan(MotionPlan(10.0, 120.0, "test"))


# --------------------------------------------------------------------------- #
# Control commands - each of these was a real hole
# --------------------------------------------------------------------------- #


def test_validate_control_command_valid():
    validate_control_command(ControlCommand(throttle=0.5, brake=0.0, steering=0.2))


def test_validate_control_command_invalid_throttle():
    with pytest.raises(ValidationError, match="Throttle must be"):
        validate_control_command(ControlCommand(2.0, 0.0, 0.0))


def test_negative_throttle_is_rejected():
    """The old bound was [-1, 1]: a negative throttle passed every gate."""
    with pytest.raises(ValidationError, match="Throttle must be"):
        validate_control_command(ControlCommand(-0.5, 0.0, 0.0))


def test_nan_brake_is_rejected():
    """A NaN brake used to reach a sanitiser whose max(0, min(1, nan)) is 1.0."""
    with pytest.raises(ValidationError, match="Invalid brake"):
        validate_control_command(ControlCommand(0.0, float("nan"), 0.0))


def test_out_of_range_brake_is_rejected():
    with pytest.raises(ValidationError, match="Brake must be"):
        validate_control_command(ControlCommand(0.0, 1.5, 0.0))


def test_out_of_range_steering_is_rejected():
    with pytest.raises(ValidationError, match="Steering must be"):
        validate_control_command(ControlCommand(0.0, 0.0, 2.0))


def test_throttle_and_brake_together_is_rejected():
    with pytest.raises(ValidationError, match="must not be commanded together"):
        validate_control_command(ControlCommand(0.4, 0.4, 0.0))


# --------------------------------------------------------------------------- #
# Ego state
# --------------------------------------------------------------------------- #


def test_an_invalid_ego_state_is_always_acceptable():
    """valid=False is the honest representation of 'no measurement'."""
    validate_ego_state(EgoState(speed_mps=float("nan"), valid=False))
    validate_ego_state(None)


def test_a_valid_ego_state_must_carry_usable_numbers():
    with pytest.raises(ValidationError, match="Ego speed must be finite"):
        validate_ego_state(EgoState(speed_mps=float("nan"), valid=True))
    with pytest.raises(ValidationError, match="non-negative"):
        validate_ego_state(EgoState(speed_mps=-1.0, valid=True))
    with pytest.raises(ValidationError, match="implausible"):
        validate_ego_state(EgoState(speed_mps=500.0, valid=True))


# --------------------------------------------------------------------------- #
# Lane models
# --------------------------------------------------------------------------- #


def _lane(**kwargs):
    base = dict(
        left_coeffs=(0.0, 0.0, 540.0),
        right_coeffs=(0.0, 0.0, 740.0),
        lane_center_px=640.0,
        curvature_m=10000.0,
        confidence=0.5,
        lines=[LaneLine(index=1)],
        is_mock=False,
    )
    base.update(kwargs)
    return LaneModel(**base)


def test_validate_lane_model_valid():
    validate_lane_model(_lane())


def test_infinite_curvature_is_rejected():
    """'Straight' must be a large finite radius; inf becomes nan downstream."""
    with pytest.raises(ValidationError, match="Invalid curvature"):
        validate_lane_model(_lane(curvature_m=math.inf))


def test_non_quadratic_coefficients_are_rejected():
    with pytest.raises(ValidationError, match="quadratic"):
        validate_lane_model(_lane(left_coeffs=(0.0, 1.0)))


def test_out_of_range_lane_confidence_is_rejected():
    with pytest.raises(ValidationError, match="confidence"):
        validate_lane_model(_lane(confidence=1.7))


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #


def test_validate_image_dimensions():
    validate_image_dimensions(1280, 720)
    with pytest.raises(ValidationError, match="Invalid image dimensions"):
        validate_image_dimensions(0, 720)
    with pytest.raises(ValidationError, match="too large"):
        validate_image_dimensions(20000, 720)


def test_validate_perception_frame_accepts_a_dict_payload():
    validate_perception_frame(
        PerceptionFrame(0, 1.0, {"width": 1280, "height": 720}, 1280, 720)
    )


def test_a_frame_whose_buffer_disagrees_with_its_size_is_rejected():
    """The failure that makes every geometric result quietly wrong."""
    numpy = pytest.importorskip("numpy")
    buffer = numpy.zeros((480, 640, 3), dtype=numpy.uint8)
    with pytest.raises(ValidationError, match="declares"):
        validate_perception_frame(PerceptionFrame(0, 1.0, buffer, 1280, 720))


def test_a_frame_whose_buffer_matches_is_accepted():
    numpy = pytest.importorskip("numpy")
    buffer = numpy.zeros((720, 1280, 3), dtype=numpy.uint8)
    validate_perception_frame(PerceptionFrame(0, 1.0, buffer, 1280, 720))


# --------------------------------------------------------------------------- #
# Config values
# --------------------------------------------------------------------------- #


def test_validate_config_value_range():
    validate_config_value("fps", 20, 1, 120)
    with pytest.raises(ValidationError, match=">="):
        validate_config_value("fps", 0, 1, 120)
    with pytest.raises(ValidationError, match="<="):
        validate_config_value("fps", 200, 1, 120)


def test_a_boolean_is_not_a_number():
    """'fps': true is a mistake, not 1 Hz."""
    with pytest.raises(ValidationError, match="must be numeric"):
        validate_config_value("fps", True, 1, 120)


def test_non_finite_config_values_are_rejected():
    with pytest.raises(ValidationError, match="finite"):
        validate_config_value("gain", float("inf"), 0.0, 10.0)
