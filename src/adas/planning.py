"""Behavior planning: lane centering and adaptive speed policy."""

from __future__ import annotations

from dataclasses import dataclass

from adas.types import MotionPlan, TrackedObject


@dataclass(slots=True)
class BehaviorPlanner:
    cruise_speed_mps: float = 15.0
    min_follow_distance_m: float = 12.0
    max_steering_deg: float = 22.0

    def plan(
        self,
        frame_width_px: int,
        lane_center_px: float | None,
        objects: list[TrackedObject],
    ) -> MotionPlan:
        target_speed = self.cruise_speed_mps
        reason = "cruise"

        nearest = min((obj.distance_m for obj in objects), default=999.0)
        if nearest < self.min_follow_distance_m:
            ratio = max(0.0, nearest / self.min_follow_distance_m)
            target_speed = self.cruise_speed_mps * ratio
            reason = f"lead_vehicle_{nearest:.1f}m"

        if lane_center_px is None:
            steering = 0.0
            reason = f"{reason}_lane_unavailable"
        else:
            img_center = frame_width_px / 2.0
            error = (lane_center_px - img_center) / max(1.0, img_center)
            steering = max(-1.0, min(1.0, error)) * self.max_steering_deg

        return MotionPlan(target_speed_mps=target_speed, steering_angle_deg=steering, reason=reason)
