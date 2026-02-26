"""Behavior planning: lane centering and adaptive speed policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

from adas.core.exceptions import PlanningError, ValidationError
from adas.core.logger import setup_logger
from adas.core.models import MotionPlan, TrackedObject
from adas.core.validation import validate_motion_plan

logger = setup_logger(__name__)


@dataclass(slots=True)
class BehaviorPlanner:
    """Behavior planner for lane keeping and adaptive cruise control."""
    
    cruise_speed_mps: float = 15.0  # ~54 km/h default cruise
    min_follow_distance_m: float = 12.0  # Minimum safe following distance
    max_steering_deg: float = 22.0  # Maximum steering angle
    time_gap_s: float = 2.0  # Desired time gap to lead vehicle
    max_decel_mps2: float = 3.0  # Maximum comfortable deceleration
    
    # Steering control gains
    lane_center_gain: float = 1.0  # Proportional gain for centering
    
    def __post_init__(self) -> None:
        """Validate planner configuration."""
        if self.cruise_speed_mps <= 0:
            raise ValidationError(f"Cruise speed must be positive, got {self.cruise_speed_mps}")
        if self.min_follow_distance_m < 0:
            raise ValidationError(f"Min follow distance must be non-negative, got {self.min_follow_distance_m}")
        if self.max_steering_deg <= 0:
            raise ValidationError(f"Max steering must be positive, got {self.max_steering_deg}")
        
        logger.info(
            f"BehaviorPlanner initialized: cruise={self.cruise_speed_mps:.1f} m/s, "
            f"min_distance={self.min_follow_distance_m:.1f} m"
        )

    def plan(
        self,
        frame_width_px: int,
        lane_center_px: float | None,
        objects: list[TrackedObject],
    ) -> MotionPlan:
        """Generate motion plan based on lane and object information.
        
        Args:
            frame_width_px: Image width in pixels
            lane_center_px: Detected lane center position (None if unavailable)
            objects: List of tracked objects
            
        Returns:
            Motion plan with target speed and steering
            
        Raises:
            PlanningError: If planning fails
        """
        try:
            # Input validation
            if frame_width_px <= 0:
                raise ValidationError(f"Invalid frame width: {frame_width_px}")
            
            # Longitudinal planning (speed control)
            target_speed, speed_reason = self._plan_speed(objects)
            
            # Lateral planning (steering control)
            steering_deg, steer_reason = self._plan_steering(frame_width_px, lane_center_px)
            
            # Combine reasoning
            reason = f"{speed_reason}|{steer_reason}"
            
            # Create motion plan
            plan = MotionPlan(
                target_speed_mps=target_speed,
                steering_angle_deg=steering_deg,
                reason=reason
            )
            
            # Validate output
            validate_motion_plan(plan)
            
            logger.debug(
                f"Plan: speed={target_speed:.1f} m/s, steering={steering_deg:.1f}°, reason={reason}"
            )
            
            return plan
            
        except ValidationError as e:
            raise PlanningError(f"Planning validation failed: {e}") from e
        except Exception as e:
            raise PlanningError(f"Planning failed: {e}") from e
    
    def _plan_speed(self, objects: list[TrackedObject]) -> tuple[float, str]:
        """Plan target speed based on detected objects.
        
        Args:
            objects: List of tracked objects
            
        Returns:
            Tuple of (target_speed_mps, reason_string)
        """
        target_speed = self.cruise_speed_mps
        reason = "cruise"
        
        if not objects:
            return target_speed, reason
        
        # Find nearest object ahead
        nearest_distance = min((obj.distance_m for obj in objects), default=float("inf"))
        
        if not math.isfinite(nearest_distance):
            logger.warning("Non-finite distance in objects, using cruise speed")
            return target_speed, "invalid_distance"
        
        # Adaptive cruise control logic
        if nearest_distance < self.min_follow_distance_m:
            # Too close - reduce speed proportionally
            ratio = max(0.0, nearest_distance / self.min_follow_distance_m)
            target_speed = self.cruise_speed_mps * ratio
            reason = f"follow_close_{nearest_distance:.1f}m"
            logger.debug(f"Following at {nearest_distance:.1f}m, reducing speed to {target_speed:.1f} m/s")
        elif nearest_distance < self.cruise_speed_mps * self.time_gap_s:
            # Within time gap - moderate speed
            desired_distance = self.cruise_speed_mps * self.time_gap_s
            ratio = nearest_distance / desired_distance
            target_speed = self.cruise_speed_mps * ratio
            reason = f"follow_{nearest_distance:.1f}m"
        else:
            # Clear ahead - cruise
            reason = f"cruise_clear_{nearest_distance:.1f}m"
        
        # Ensure non-negative speed
        target_speed = max(0.0, target_speed)
        
        return target_speed, reason
    
    def _plan_steering(self, frame_width_px: int, lane_center_px: float | None) -> tuple[float, str]:
        """Plan steering angle based on lane detection.
        
        Args:
            frame_width_px: Image width in pixels
            lane_center_px: Detected lane center (None if unavailable)
            
        Returns:
            Tuple of (steering_angle_deg, reason_string)
        """
        if lane_center_px is None:
            # No lane detected - go straight
            return 0.0, "no_lane"
        
        # Validate lane position
        if not (0 <= lane_center_px <= frame_width_px):
            logger.warning(
                f"Lane center {lane_center_px} outside frame [0, {frame_width_px}], going straight"
            )
            return 0.0, "invalid_lane"
        
        # Calculate lateral error (normalized to [-1, 1])
        img_center = frame_width_px / 2.0
        lateral_error = (lane_center_px - img_center) / img_center
        
        # Proportional steering control
        steering_normalized = self.lane_center_gain * lateral_error
        
        # Clamp to [-1, 1] and scale to degrees
        steering_normalized = max(-1.0, min(1.0, steering_normalized))
        steering_deg = steering_normalized * self.max_steering_deg
        
        reason = f"lane_center_err_{lateral_error:.2f}"
        
        return steering_deg, reason
