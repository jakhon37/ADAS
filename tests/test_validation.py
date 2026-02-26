"""Tests for validation module."""

import pytest

from adas.core.exceptions import ValidationError
from adas.core.models import BoundingBox, ControlCommand, LaneModel, MotionPlan
from adas.core.validation import (
    validate_bounding_box,
    validate_control_command,
    validate_lane_model,
    validate_motion_plan,
)


def test_validate_bounding_box_valid():
    """Test validation of valid bounding box."""
    box = BoundingBox(x1=10, y1=20, x2=30, y2=40, confidence=0.8, label="car")
    validate_bounding_box(box)  # Should not raise


def test_validate_bounding_box_negative_coords():
    """Test that negative coordinates are rejected."""
    box = BoundingBox(x1=-10, y1=20, x2=30, y2=40, confidence=0.8, label="car")
    
    with pytest.raises(ValidationError, match="Negative coordinates"):
        validate_bounding_box(box)


def test_validate_bounding_box_invalid_width():
    """Test that invalid box dimensions are rejected."""
    box = BoundingBox(x1=30, y1=20, x2=10, y2=40, confidence=0.8, label="car")
    
    with pytest.raises(ValidationError, match="Invalid box width"):
        validate_bounding_box(box)


def test_validate_bounding_box_invalid_confidence():
    """Test that invalid confidence is rejected."""
    box = BoundingBox(x1=10, y1=20, x2=30, y2=40, confidence=1.5, label="car")
    
    with pytest.raises(ValidationError, match="Confidence must be"):
        validate_bounding_box(box)


def test_validate_motion_plan_negative_speed():
    """Test that negative speed is rejected."""
    plan = MotionPlan(target_speed_mps=-5.0, steering_angle_deg=0.0, reason="test")
    
    with pytest.raises(ValidationError, match="Negative speed"):
        validate_motion_plan(plan)


def test_validate_control_command_valid():
    """Test validation of valid control command."""
    cmd = ControlCommand(throttle=0.5, brake=0.0, steering=0.2)
    validate_control_command(cmd)  # Should not raise


def test_validate_control_command_invalid_throttle():
    """Test that invalid throttle is rejected."""
    cmd = ControlCommand(throttle=2.0, brake=0.0, steering=0.0)
    
    with pytest.raises(ValidationError, match="Throttle must be"):
        validate_control_command(cmd)
