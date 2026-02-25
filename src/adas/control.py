"""Low-level control conversion from motion plan to actuator command."""

from __future__ import annotations

from dataclasses import dataclass

from adas.types import ControlCommand, MotionPlan


@dataclass(slots=True)
class PIDLikeLongitudinalController:
    kp_speed: float = 0.15
    current_speed_mps: float = 0.0

    def to_command(self, plan: MotionPlan) -> ControlCommand:
        speed_error = plan.target_speed_mps - self.current_speed_mps
        if speed_error >= 0:
            throttle = min(1.0, self.kp_speed * speed_error)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(1.0, self.kp_speed * abs(speed_error))

        steering = max(-1.0, min(1.0, plan.steering_angle_deg / 25.0))
        return ControlCommand(throttle=throttle, brake=brake, steering=steering)
