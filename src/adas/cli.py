"""CLI entrypoint for running the ADAS reference pipeline."""

from __future__ import annotations

import argparse
import time

from adas.config import load_config
from adas.control import PIDLikeLongitudinalController
from adas.perception.detection import ObjectDetector
from adas.perception.lane import LaneEstimator
from adas.pipeline import ADASPipeline
from adas.planning import BehaviorPlanner
from adas.runtime import PipelineRunner, synthetic_frame
from adas.tracking import MultiObjectTracker
from adas.types import PerceptionFrame


def build_pipeline(config_path: str | None = None) -> ADASPipeline:
    config = load_config(config_path)
    return ADASPipeline(
        detector=ObjectDetector(confidence_threshold=config.detector.confidence_threshold),
        lane_estimator=LaneEstimator(),
        tracker=MultiObjectTracker(),
        planner=BehaviorPlanner(
            cruise_speed_mps=config.planner.cruise_speed_mps,
            min_follow_distance_m=config.planner.min_follow_distance_m,
            max_steering_deg=config.planner.max_steering_deg,
        ),
        controller=PIDLikeLongitudinalController(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ADAS reference runtime")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config")
    parser.add_argument("--synthetic", action="store_true", help="Run on synthetic frame for smoke testing")
    parser.add_argument("--frames", type=int, default=60, help="Frames for synthetic run")
    args = parser.parse_args()

    pipeline = build_pipeline(args.config)
    if args.synthetic:
        frame = synthetic_frame()
        perception = PerceptionFrame(
            frame_id=0,
            timestamp_s=time.time(),
            rgb=frame,
            width=frame["width"],
            height=frame["height"],
        )
        plan, cmd = pipeline.step(perception)
        print(f"synthetic: speed={plan.target_speed_mps:.2f}, steer={plan.steering_angle_deg:.2f}, cmd={cmd}")
        return

    PipelineRunner(pipeline).run_synthetic(max_frames=args.frames)


if __name__ == "__main__":
    main()
