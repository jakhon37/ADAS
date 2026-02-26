"""Input validation utilities for ADAS system.

This module provides validation functions for sensor data and system parameters.
"""

from __future__ import annotations

import math
from typing import Any

from adas.core.exceptions import ValidationError
from adas.core.models import BoundingBox, ControlCommand, LaneModel, MotionPlan


def validate_bounding_box(box: BoundingBox) -> None:
    """Validate bounding box parameters.
    
    Args:
        box: Bounding box to validate
        
    Raises:
        ValidationError: If box parameters are invalid
    """
    if box.x1 < 0 or box.y1 < 0:
        raise ValidationError(f"Negative coordinates not allowed: ({box.x1}, {box.y1})")
    
    if box.x2 <= box.x1:
        raise ValidationError(f"Invalid box width: x2={box.x2} <= x1={box.x1}")
    
    if box.y2 <= box.y1:
        raise ValidationError(f"Invalid box height: y2={box.y2} <= y1={box.y1}")
    
    if not (0.0 <= box.confidence <= 1.0):
        raise ValidationError(f"Confidence must be in [0, 1], got {box.confidence}")


def validate_lane_model(lane: LaneModel) -> None:
    """Validate lane model parameters.
    
    Args:
        lane: Lane model to validate
        
    Raises:
        ValidationError: If lane parameters are invalid
    """
    if not math.isfinite(lane.lateral_offset_m):
        raise ValidationError(f"Invalid lateral offset: {lane.lateral_offset_m}")
    
    if not math.isfinite(lane.heading_error_rad):
        raise ValidationError(f"Invalid heading error: {lane.heading_error_rad}")
    
    if abs(lane.heading_error_rad) > math.pi:
        raise ValidationError(f"Heading error too large: {lane.heading_error_rad} rad")


def validate_motion_plan(plan: MotionPlan) -> None:
    """Validate motion plan parameters.
    
    Args:
        plan: Motion plan to validate
        
    Raises:
        ValidationError: If plan parameters are invalid
    """
    if not math.isfinite(plan.target_speed_mps):
        raise ValidationError(f"Invalid target speed: {plan.target_speed_mps}")
    
    if plan.target_speed_mps < 0:
        raise ValidationError(f"Negative speed not allowed: {plan.target_speed_mps}")
    
    if not math.isfinite(plan.steering_angle_deg):
        raise ValidationError(f"Invalid steering angle: {plan.steering_angle_deg}")


def validate_control_command(cmd: ControlCommand) -> None:
    """Validate control command parameters.
    
    Args:
        cmd: Control command to validate
        
    Raises:
        ValidationError: If command parameters are invalid
    """
    if not math.isfinite(cmd.throttle):
        raise ValidationError(f"Invalid throttle: {cmd.throttle}")
    
    if not (-1.0 <= cmd.throttle <= 1.0):
        raise ValidationError(f"Throttle must be in [-1, 1], got {cmd.throttle}")
    
    if not math.isfinite(cmd.steering):
        raise ValidationError(f"Invalid steering: {cmd.steering}")


def validate_image_dimensions(width: int, height: int) -> None:
    """Validate image dimensions.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        
    Raises:
        ValidationError: If dimensions are invalid
    """
    if width <= 0 or height <= 0:
        raise ValidationError(f"Invalid image dimensions: {width}x{height}")
    
    if width > 10000 or height > 10000:
        raise ValidationError(f"Image dimensions too large: {width}x{height}")


def validate_config_value(name: str, value: Any, min_val: float | None = None, 
                         max_val: float | None = None) -> None:
    """Validate a configuration value.
    
    Args:
        name: Parameter name for error messages
        value: Value to validate
        min_val: Minimum allowed value (optional)
        max_val: Maximum allowed value (optional)
        
    Raises:
        ValidationError: If value is invalid
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(value)}")
    
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value}")
    
    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} must be >= {min_val}, got {value}")
    
    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} must be <= {max_val}, got {value}")
