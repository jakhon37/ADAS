from __future__ import annotations

import time

from adas.cli import build_pipeline
from adas.core.models import BoundingBox, PerceptionFrame, TrackedObject
from adas.runtime import synthetic_frame


def test_pipeline_synthetic_smoke() -> None:
    pipeline, fps = build_pipeline()
    frame = synthetic_frame()
    perception = PerceptionFrame(
        frame_id=1,
        timestamp_s=time.time(),
        rgb=frame,
        width=frame["width"],
        height=frame["height"],
    )

    plan, command = pipeline.step(perception, current_speed_mps=10.0)

    assert plan.target_speed_mps >= 0.0
    assert -22.0 <= plan.steering_angle_deg <= 22.0
    assert 0.0 <= command.throttle <= 1.0
    assert 0.0 <= command.brake <= 1.0
    assert -1.0 <= command.steering <= 1.0


def test_planner_slows_for_close_vehicle() -> None:
    pipeline, fps = build_pipeline()
    close_obj = TrackedObject(
        track_id=1,
        box=BoundingBox(0, 0, 100, 100, 0.9, "vehicle"),
        velocity_mps=0.0,
        distance_m=5.0,
    )

    plan = pipeline.planner.plan(frame_width_px=1280, lane_center_px=640.0, objects=[close_obj])
    assert plan.target_speed_mps < pipeline.planner.cruise_speed_mps
    assert "follow" in plan.reason
