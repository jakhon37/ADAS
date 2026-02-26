"""Tests for safety monitor."""

import pytest

from adas.control import SafetyMonitor
from adas.core.exceptions import SafetyViolation
from adas.core.models import ControlCommand, MotionPlan, TrackedObject, BoundingBox


def test_safety_monitor_speed_limit():
    """Test that excessive speed is detected."""
    monitor = SafetyMonitor()
    
    # Plan exceeding speed limit
    plan = MotionPlan(
        target_speed_mps=50.0,  # Way over limit
        steering_angle_deg=0.0,
        reason="test"
    )
    
    with pytest.raises(SafetyViolation, match="exceeds limit"):
        monitor.check_motion_plan(plan, current_speed_mps=10.0)


def test_safety_monitor_steering_limit():
    """Test that excessive steering is detected."""
    monitor = SafetyMonitor()
    
    # Plan with excessive steering (>30 degrees = 0.52 rad)
    plan = MotionPlan(
        target_speed_mps=10.0,
        steering_angle_deg=40.0,  # Too much
        reason="test"
    )
    
    with pytest.raises(SafetyViolation, match="Steering angle"):
        monitor.check_motion_plan(plan, current_speed_mps=10.0)


def test_safety_monitor_following_distance():
    """Test minimum following distance check."""
    monitor = SafetyMonitor()
    
    lead = TrackedObject(
        track_id=1,
        box=BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.9, label="car"),
        velocity_mps=0.0,
        distance_m=1.0  # Too close!
    )
    
    with pytest.raises(SafetyViolation, match="Following distance"):
        monitor.check_following_distance(lead, ego_speed_mps=15.0)


def test_safety_command_sanitization():
    """Test that control commands are sanitized."""
    monitor = SafetyMonitor()
    
    # Command with out-of-bounds values
    cmd = ControlCommand(
        throttle=1.5,  # > 1.0
        brake=-0.5,    # < 0.0
        steering=2.0   # > 1.0
    )
    
    sanitized = monitor.sanitize_control_command(cmd)
    
    assert sanitized.throttle == 1.0
    assert sanitized.brake == 0.0
    assert sanitized.steering == 1.0


def test_safety_valid_plan():
    """Test that valid plans pass safety checks."""
    monitor = SafetyMonitor()
    
    plan = MotionPlan(
        target_speed_mps=15.0,
        steering_angle_deg=10.0,
        reason="test"
    )
    
    # Should not raise
    monitor.check_motion_plan(plan, current_speed_mps=10.0)
