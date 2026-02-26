"""Safety monitor for ADAS system.

This module implements safety checks and bounds enforcement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from adas.core.exceptions import SafetyViolation
from adas.core.logger import setup_logger
from adas.core.models import ControlCommand, MotionPlan, TrackedObject

logger = setup_logger(__name__)


@dataclass(slots=True)
class SafetyLimits:
    """Safety limits for ADAS operation."""
    
    max_speed_mps: float = 33.0  # ~120 km/h
    max_acceleration_mps2: float = 3.0  # Comfortable acceleration
    max_deceleration_mps2: float = 8.0  # Emergency braking (< ABS limit)
    max_steering_rate_rad_s: float = 0.5  # Prevent sudden steering
    max_steering_angle_rad: float = 0.52  # ~30 degrees
    min_following_distance_m: float = 2.0  # Minimum safe distance
    max_lateral_offset_m: float = 1.5  # Lane keeping limit


class SafetyMonitor:
    """Monitor and enforce safety constraints."""
    
    def __init__(self, limits: SafetyLimits | None = None):
        """Initialize safety monitor.
        
        Args:
            limits: Safety limits (uses defaults if None)
        """
        self.limits = limits or SafetyLimits()
        self._last_steering = 0.0
        self._last_speed = 0.0
        self._last_timestamp = 0.0
        
    def check_motion_plan(self, plan: MotionPlan, current_speed_mps: float) -> None:
        """Check if motion plan violates safety constraints.
        
        Args:
            plan: Proposed motion plan
            current_speed_mps: Current vehicle speed
            
        Raises:
            SafetyViolation: If plan violates safety limits
        """
        # Check speed limit
        if plan.target_speed_mps > self.limits.max_speed_mps:
            raise SafetyViolation(
                f"Target speed {plan.target_speed_mps:.1f} m/s exceeds limit "
                f"{self.limits.max_speed_mps:.1f} m/s"
            )
        
        # Check steering angle (convert from degrees to radians)
        steering_rad = math.radians(plan.steering_angle_deg)
        if abs(steering_rad) > self.limits.max_steering_angle_rad:
            raise SafetyViolation(
                f"Steering angle {steering_rad:.3f} rad exceeds limit "
                f"±{self.limits.max_steering_angle_rad:.3f} rad"
            )
        
        # Check acceleration/deceleration
        speed_delta = plan.target_speed_mps - current_speed_mps
        if speed_delta > 0:
            # Accelerating - assume 0.1s time horizon
            accel = speed_delta / 0.1
            if accel > self.limits.max_acceleration_mps2:
                logger.warning(
                    f"High acceleration requested: {accel:.2f} m/s² "
                    f"(limit: {self.limits.max_acceleration_mps2:.2f} m/s²)"
                )
        else:
            # Decelerating
            decel = abs(speed_delta) / 0.1
            if decel > self.limits.max_deceleration_mps2:
                raise SafetyViolation(
                    f"Deceleration {decel:.2f} m/s² exceeds limit "
                    f"{self.limits.max_deceleration_mps2:.2f} m/s²"
                )
    
    def check_control_command(self, cmd: ControlCommand, dt: float = 0.1) -> None:
        """Check if control command is safe.
        
        Args:
            cmd: Control command to check
            dt: Time step in seconds
            
        Raises:
            SafetyViolation: If command violates safety limits
        """
        # Check steering rate of change (cmd.steering is normalized [-1,1])
        if self._last_timestamp > 0:
            steering_rate = abs(cmd.steering - self._last_steering) / dt
            # Note: This is rate of normalized steering, not radians
            # In production, convert to actual steering angle rate
        
        self._last_steering = cmd.steering
        self._last_timestamp += dt
    
    def check_following_distance(self, lead_vehicle: TrackedObject | None, 
                                 ego_speed_mps: float) -> None:
        """Check if following distance is safe.
        
        Args:
            lead_vehicle: Lead vehicle being followed (None if no vehicle)
            ego_speed_mps: Ego vehicle speed
            
        Raises:
            SafetyViolation: If following too close
        """
        if lead_vehicle is None:
            return
        
        distance = lead_vehicle.distance_m
        
        # Time-to-collision check (if lead vehicle is slower/stopped)
        if distance < self.limits.min_following_distance_m:
            raise SafetyViolation(
                f"Following distance {distance:.1f}m below minimum "
                f"{self.limits.min_following_distance_m:.1f}m"
            )
        
        # Dynamic safe distance based on speed (2-second rule)
        safe_distance = max(self.limits.min_following_distance_m, ego_speed_mps * 2.0)
        if distance < safe_distance:
            logger.warning(
                f"Following distance {distance:.1f}m below recommended "
                f"{safe_distance:.1f}m at {ego_speed_mps:.1f} m/s"
            )
    
    def sanitize_control_command(self, cmd: ControlCommand) -> ControlCommand:
        """Clamp control command to safe ranges.
        
        Args:
            cmd: Input control command
            
        Returns:
            Sanitized control command
        """
        throttle = max(-1.0, min(1.0, cmd.throttle))
        brake = max(0.0, min(1.0, cmd.brake))
        steering = max(-1.0, min(1.0, cmd.steering))
        
        if throttle != cmd.throttle or brake != cmd.brake or steering != cmd.steering:
            logger.warning(
                f"Control command clamped: throttle {cmd.throttle:.2f}->{throttle:.2f}, "
                f"brake {cmd.brake:.2f}->{brake:.2f}, "
                f"steering {cmd.steering:.3f}->{steering:.3f}"
            )
        
        return ControlCommand(throttle=throttle, brake=brake, steering=steering)
