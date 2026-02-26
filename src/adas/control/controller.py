"""Low-level control conversion from motion plan to actuator command."""

from __future__ import annotations

import math
from dataclasses import dataclass

from adas.core.exceptions import ControlError, ValidationError
from adas.core.logger import setup_logger
from adas.core.models import ControlCommand, MotionPlan
from adas.core.validation import validate_control_command, validate_motion_plan

logger = setup_logger(__name__)


@dataclass(slots=True)
class PIDLikeLongitudinalController:
    """Proportional controller for speed and steering commands.
    
    This controller is stateless - current speed is passed as an argument
    to maintain functional purity and thread safety.
    """
    
    # Speed control gains
    kp_speed: float = 0.15  # Proportional gain for speed error
    max_throttle: float = 1.0  # Maximum throttle command
    max_brake: float = 1.0  # Maximum brake command
    
    # Steering control parameters
    max_steering_angle_deg: float = 25.0  # Maximum physical steering angle
    steering_deadband_deg: float = 0.5  # Ignore small steering commands
    
    def __post_init__(self) -> None:
        """Validate controller configuration."""
        if self.kp_speed <= 0:
            raise ValidationError(f"Speed gain must be positive, got {self.kp_speed}")
        if self.max_steering_angle_deg <= 0:
            raise ValidationError(f"Max steering must be positive, got {self.max_steering_angle_deg}")
        
        logger.info(
            f"PIDController initialized: kp={self.kp_speed}, "
            f"max_steering={self.max_steering_angle_deg}°"
        )

    def to_command(self, plan: MotionPlan, current_speed_mps: float) -> ControlCommand:
        """Convert motion plan to control command.
        
        Args:
            plan: Motion plan with target speed and steering
            current_speed_mps: Current vehicle speed in m/s
            
        Returns:
            Control command for vehicle actuators
            
        Raises:
            ControlError: If command generation fails
        """
        try:
            # Validate inputs
            validate_motion_plan(plan)
            
            if current_speed_mps < 0:
                raise ValidationError(f"Current speed must be non-negative, got {current_speed_mps}")
            
            if not math.isfinite(current_speed_mps):
                raise ValidationError(f"Current speed must be finite, got {current_speed_mps}")
            
            # Longitudinal control (speed)
            throttle, brake = self._compute_longitudinal_control(
                plan.target_speed_mps, current_speed_mps
            )
            
            # Lateral control (steering)
            steering = self._compute_steering_control(plan.steering_angle_deg)
            
            # Create control command
            cmd = ControlCommand(throttle=throttle, brake=brake, steering=steering)
            
            # Validate output
            validate_control_command(cmd)
            
            logger.debug(
                f"Control: throttle={throttle:.2f}, brake={brake:.2f}, "
                f"steering={steering:.3f} (target={plan.target_speed_mps:.1f} m/s, "
                f"current={current_speed_mps:.1f} m/s)"
            )
            
            return cmd
            
        except ValidationError as e:
            raise ControlError(f"Control validation failed: {e}") from e
        except Exception as e:
            raise ControlError(f"Control command generation failed: {e}") from e
    
    def _compute_longitudinal_control(
        self, target_speed_mps: float, current_speed_mps: float
    ) -> tuple[float, float]:
        """Compute throttle and brake commands.
        
        Args:
            target_speed_mps: Target speed
            current_speed_mps: Current speed
            
        Returns:
            Tuple of (throttle, brake) in range [0, 1]
        """
        speed_error = target_speed_mps - current_speed_mps
        
        if speed_error >= 0:
            # Accelerate
            throttle = min(self.max_throttle, self.kp_speed * speed_error)
            brake = 0.0
        else:
            # Decelerate
            throttle = 0.0
            brake = min(self.max_brake, self.kp_speed * abs(speed_error))
        
        return throttle, brake
    
    def _compute_steering_control(self, target_steering_deg: float) -> float:
        """Compute normalized steering command.
        
        Args:
            target_steering_deg: Target steering angle in degrees
            
        Returns:
            Normalized steering in range [-1, 1]
        """
        # Apply deadband to avoid jitter
        if abs(target_steering_deg) < self.steering_deadband_deg:
            return 0.0
        
        # Normalize to [-1, 1]
        steering_normalized = target_steering_deg / self.max_steering_angle_deg
        
        # Clamp to valid range
        steering_normalized = max(-1.0, min(1.0, steering_normalized))
        
        return steering_normalized
