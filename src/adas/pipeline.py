"""End-to-end ADAS inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from adas.control import PIDLikeLongitudinalController
from adas.perception.detection import ObjectDetector
from adas.perception.lane import LaneEstimator
from adas.planning import BehaviorPlanner
from adas.tracking import MultiObjectTracker
from adas.types import ControlCommand, MotionPlan, PerceptionFrame


@dataclass(slots=True)
class ADASPipeline:
    detector: ObjectDetector
    lane_estimator: LaneEstimator
    tracker: MultiObjectTracker
    planner: BehaviorPlanner
    controller: PIDLikeLongitudinalController

    def step(self, frame: PerceptionFrame) -> tuple[MotionPlan, ControlCommand]:
        frame.detections = self.detector.infer(frame.rgb, frame.width, frame.height)
        frame.lane = self.lane_estimator.estimate(frame.rgb, frame.width, frame.height)
        tracked = self.tracker.update(frame.detections)
        lane_center = frame.lane.lane_center_px if frame.lane else None

        plan = self.planner.plan(
            frame_width_px=frame.width,
            lane_center_px=lane_center,
            objects=tracked,
        )
        command = self.controller.to_command(plan)
        return plan, command
