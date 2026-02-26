"""Tests for behavior planner."""


from adas.core.models import BoundingBox, TrackedObject
from adas.planning import BehaviorPlanner


def test_planner_cruise_speed():
    """Test cruise speed when no obstacles."""
    planner = BehaviorPlanner(cruise_speed_mps=15.0)
    
    plan = planner.plan(frame_width_px=1280, lane_center_px=640.0, objects=[])
    
    assert plan.target_speed_mps == 15.0
    assert "cruise" in plan.reason


def test_planner_slows_for_close_vehicle():
    """Test that planner reduces speed for close vehicles."""
    planner = BehaviorPlanner(cruise_speed_mps=15.0, min_follow_distance_m=12.0)
    
    # Vehicle at 6m (half min distance)
    close_vehicle = TrackedObject(
        track_id=1,
        box=BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.9, label="car"),
        velocity_mps=0.0,
        distance_m=6.0
    )
    
    plan = planner.plan(frame_width_px=1280, lane_center_px=640.0, objects=[close_vehicle])
    
    # Should be about half cruise speed
    assert plan.target_speed_mps < 15.0
    assert plan.target_speed_mps > 0.0
    assert "follow" in plan.reason


def test_planner_stops_for_very_close_vehicle():
    """Test that planner nearly stops for very close vehicles."""
    planner = BehaviorPlanner(cruise_speed_mps=15.0, min_follow_distance_m=12.0)
    
    # Vehicle at 1m (very close)
    very_close = TrackedObject(
        track_id=1,
        box=BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.9, label="car"),
        velocity_mps=0.0,
        distance_m=1.0
    )
    
    plan = planner.plan(frame_width_px=1280, lane_center_px=640.0, objects=[very_close])
    
    # Should be very low speed
    assert plan.target_speed_mps < 2.0
    assert plan.target_speed_mps >= 0.0


def test_planner_lane_centering():
    """Test lane centering steering."""
    planner = BehaviorPlanner(max_steering_deg=22.0)
    
    # Lane center to the right
    plan = planner.plan(frame_width_px=1280, lane_center_px=800.0, objects=[])
    
    # Should steer right (positive)
    assert plan.steering_angle_deg > 0
    
    # Lane center to the left
    plan = planner.plan(frame_width_px=1280, lane_center_px=400.0, objects=[])
    
    # Should steer left (negative)
    assert plan.steering_angle_deg < 0


def test_planner_no_lane_goes_straight():
    """Test that missing lane info results in straight steering."""
    planner = BehaviorPlanner()
    
    plan = planner.plan(frame_width_px=1280, lane_center_px=None, objects=[])
    
    assert plan.steering_angle_deg == 0.0
    assert "no_lane" in plan.reason


def test_planner_steering_limits():
    """Test that steering doesn't exceed limits."""
    planner = BehaviorPlanner(max_steering_deg=22.0)
    
    # Extreme lane offset
    plan = planner.plan(frame_width_px=1280, lane_center_px=1200.0, objects=[])
    
    # Should be clamped to max
    assert abs(plan.steering_angle_deg) <= 22.0
