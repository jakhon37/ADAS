"""End-to-end ADAS inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from adas.control import PIDLikeLongitudinalController, SafetyMonitor
from adas.core.exceptions import ADASException
from adas.core.logger import setup_logger
from adas.core.models import ControlCommand, MotionPlan, PerceptionFrame
from adas.perception.detection import ObjectDetector
from adas.perception.lane import LaneEstimator
from adas.planning import BehaviorPlanner
from adas.tracking import MultiObjectTracker

logger = setup_logger(__name__)


@dataclass(slots=True)
class ADASPipeline:
    """Main ADAS pipeline coordinating perception, planning, and control."""
    
    detector: ObjectDetector
    lane_estimator: LaneEstimator
    tracker: MultiObjectTracker
    planner: BehaviorPlanner
    controller: PIDLikeLongitudinalController
    safety_monitor: SafetyMonitor = field(default_factory=SafetyMonitor)
    
    # State tracking
    _current_speed_mps: float = field(default=0.0, init=False)
    _frame_count: int = field(default=0, init=False)

    def step(self, frame: PerceptionFrame, current_speed_mps: float = 0.0) -> tuple[MotionPlan, ControlCommand]:
        """Execute one pipeline step.
        
        Args:
            frame: Perception frame with sensor data
            current_speed_mps: Current vehicle speed (from odometry/CAN)
            
        Returns:
            Tuple of (motion_plan, control_command)
            
        Raises:
            ADASException: If pipeline step fails
        """
        try:
            self._current_speed_mps = current_speed_mps
            self._frame_count += 1
            
            logger.debug(f"Processing frame {frame.frame_id} (count={self._frame_count})")
            
            # Perception stage
            try:
                frame.detections = self.detector.infer(frame.rgb, frame.width, frame.height)
                frame.lane = self.lane_estimator.estimate(frame.rgb, frame.width, frame.height)
            except Exception as e:
                logger.error(f"Perception failed: {e}")
                # Continue with empty perception - fail gracefully
                frame.detections = []
                frame.lane = None
            
            # Tracking stage
            tracked = self.tracker.update(frame.detections)
            
            # Planning stage
            lane_center = frame.lane.lane_center_px if frame.lane else None
            plan = self.planner.plan(
                frame_width_px=frame.width,
                lane_center_px=lane_center,
                objects=tracked,
            )
            
            # Safety check on plan
            try:
                self.safety_monitor.check_motion_plan(plan, current_speed_mps)
                
                # Check following distance if tracking objects
                if tracked:
                    lead_vehicle = min(tracked, key=lambda obj: obj.distance_m)
                    self.safety_monitor.check_following_distance(lead_vehicle, current_speed_mps)
            except Exception as e:
                logger.warning(f"Safety check: {e}")
            
            # Control stage
            command = self.controller.to_command(plan, current_speed_mps)
            
            # Safety sanitization of command
            command = self.safety_monitor.sanitize_control_command(command)
            
            # Final safety check
            self.safety_monitor.check_control_command(command)
            
            logger.info(
                f"Frame {frame.frame_id}: detections={len(frame.detections)}, "
                f"tracks={len(tracked)}, lane={'✓' if frame.lane else '✗'}, "
                f"plan={plan.reason}, cmd=t{command.throttle:.2f}/b{command.brake:.2f}/s{command.steering:.2f}"
            )
            
            return plan, command
            
        except ADASException:
            # Re-raise ADAS exceptions
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
