"""Tests for controller."""


from adas.control import PIDLikeLongitudinalController
from adas.core.models import MotionPlan


def test_controller_accelerates():
    """Test controller generates throttle when target speed > current."""
    controller = PIDLikeLongitudinalController(kp_speed=0.15)
    
    plan = MotionPlan(target_speed_mps=20.0, steering_angle_deg=0.0, reason="test")
    cmd = controller.to_command(plan, current_speed_mps=10.0)
    
    assert cmd.throttle > 0
    assert cmd.brake == 0
    

def test_controller_brakes():
    """Test controller generates brake when target speed < current."""
    controller = PIDLikeLongitudinalController(kp_speed=0.15)
    
    plan = MotionPlan(target_speed_mps=5.0, steering_angle_deg=0.0, reason="test")
    cmd = controller.to_command(plan, current_speed_mps=20.0)
    
    assert cmd.throttle == 0
    assert cmd.brake > 0


def test_controller_maintains_speed():
    """Test controller when at target speed."""
    controller = PIDLikeLongitudinalController(kp_speed=0.15)
    
    plan = MotionPlan(target_speed_mps=15.0, steering_angle_deg=0.0, reason="test")
    cmd = controller.to_command(plan, current_speed_mps=15.0)
    
    # Should have minimal throttle/brake
    assert cmd.throttle < 0.1
    assert cmd.brake < 0.1


def test_controller_steering():
    """Test steering command generation."""
    controller = PIDLikeLongitudinalController(max_steering_angle_deg=25.0)
    
    # Right turn
    plan = MotionPlan(target_speed_mps=15.0, steering_angle_deg=10.0, reason="test")
    cmd = controller.to_command(plan, current_speed_mps=15.0)
    
    assert cmd.steering > 0
    assert cmd.steering <= 1.0
    
    # Left turn
    plan = MotionPlan(target_speed_mps=15.0, steering_angle_deg=-10.0, reason="test")
    cmd = controller.to_command(plan, current_speed_mps=15.0)
    
    assert cmd.steering < 0
    assert cmd.steering >= -1.0


def test_controller_steering_deadband():
    """Test that small steering commands are ignored."""
    controller = PIDLikeLongitudinalController(steering_deadband_deg=0.5)
    
    plan = MotionPlan(target_speed_mps=15.0, steering_angle_deg=0.3, reason="test")
    cmd = controller.to_command(plan, current_speed_mps=15.0)
    
    # Should be zero due to deadband
    assert cmd.steering == 0.0


def test_controller_limits_throttle():
    """Test that throttle is clamped to [0, 1]."""
    controller = PIDLikeLongitudinalController(kp_speed=10.0)  # High gain
    
    plan = MotionPlan(target_speed_mps=100.0, steering_angle_deg=0.0, reason="test")
    cmd = controller.to_command(plan, current_speed_mps=0.0)
    
    assert cmd.throttle <= 1.0
    assert cmd.throttle >= 0.0
