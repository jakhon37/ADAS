"""End-to-end ADAS inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adas.control import PIDLikeLongitudinalController, SafetyMonitor
from adas.core.exceptions import ADASException
from adas.core.logger import setup_logger
from adas.core.metrics import PerformanceMetrics
from adas.core.models import ControlCommand, MotionPlan, PerceptionFrame
from adas.planning import BehaviorPlanner
from adas.tracking import MultiObjectTracker

logger = setup_logger(__name__)


@dataclass
class ADASPipeline:
    """Main ADAS pipeline coordinating perception, planning, and control."""

    detector: Any
    lane_estimator: Any
    tracker: MultiObjectTracker
    planner: BehaviorPlanner
    controller: PIDLikeLongitudinalController
    safety_monitor: SafetyMonitor = field(default_factory=SafetyMonitor)
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    _current_speed_mps: float = field(default=0.0, init=False)
    _frame_count: int = field(default=0, init=False)

    def step(
        self,
        frame: PerceptionFrame,
        current_speed_mps: float = 0.0,
        dt_s: float = 0.05,
    ) -> tuple[MotionPlan, ControlCommand]:
        """Execute one pipeline step."""
        try:
            self._current_speed_mps = current_speed_mps
            self._frame_count += 1
            logger.debug(f"Processing frame {frame.frame_id} (count={self._frame_count})")

            try:
                frame.detections = self.detector.infer(frame.rgb, frame.width, frame.height)
                frame.lane = self.lane_estimator.estimate(frame.rgb, frame.width, frame.height)
            except Exception as e:
                logger.error(f"Perception failed: {e}")
                frame.detections = []
                frame.lane = None

            tracked = self.tracker.update(frame.detections, dt_s=dt_s)

            lane_center = frame.lane.lane_center_px if frame.lane else None
            plan = self.planner.plan(
                frame_width_px=frame.width,
                lane_center_px=lane_center,
                objects=tracked,
            )

            try:
                self.safety_monitor.check_motion_plan(plan, current_speed_mps)
                if tracked:
                    lead_vehicle = min(tracked, key=lambda obj: obj.distance_m)
                    self.safety_monitor.check_following_distance(lead_vehicle, current_speed_mps)
            except Exception as e:
                logger.warning(f"Safety check: {e}")
                self.metrics.record_safety_event(is_violation=True)

            command = self.controller.to_command(plan, current_speed_mps)
            command = self.safety_monitor.sanitize_control_command(command)
            self.safety_monitor.check_control_command(command, dt=dt_s)

            self.metrics.update_frame(
                frame_time=dt_s,
                num_detections=len(frame.detections),
                num_tracks=len(tracked),
                has_lane=frame.lane is not None,
            )

            range_s = f"range={tracked[0].distance_m:.1f}m, " if tracked else ""
            logger.info(
                f"Frame {frame.frame_id}: detections={len(frame.detections)}, "
                f"tracks={len(tracked)}, lane={'✓' if frame.lane else '✗'}, "
                f"{range_s}plan={plan.reason}, "
                f"cmd=t{command.throttle:.2f}/b{command.brake:.2f}/s{command.steering:.2f}"
            )

            return plan, command

        except ADASException:
            raise
        except Exception as e:
            logger.error(f"Pipeline step failed: {e}", exc_info=True)
            raise ADASException(f"Pipeline execution failed: {e}") from e

    def reset(self) -> None:
        """Reset pipeline state."""
        logger.info("Pipeline reset")
        self.tracker.reset()
        self._current_speed_mps = 0.0
        self._frame_count = 0
