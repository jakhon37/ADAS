"""Runtime utilities for local simulation and stream processing."""

from __future__ import annotations

import time
from dataclasses import dataclass

from adas.pipeline import ADASPipeline
from adas.types import PerceptionFrame


@dataclass(slots=True)
class PipelineRunner:
    pipeline: ADASPipeline

    def run_synthetic(self, max_frames: int = 60) -> None:
        for frame_id in range(max_frames):
            frame = synthetic_frame()
            perception = PerceptionFrame(
                frame_id=frame_id,
                timestamp_s=time.time(),
                rgb=frame,
                width=frame["width"],
                height=frame["height"],
            )
            plan, cmd = self.pipeline.step(perception)
            print(
                f"frame={frame_id:04d} speed={plan.target_speed_mps:>5.2f}m/s "
                f"steer={plan.steering_angle_deg:>6.2f}deg "
                f"cmd=[t:{cmd.throttle:.2f},b:{cmd.brake:.2f},s:{cmd.steering:.2f}] "
                f"reason={plan.reason}"
            )


def synthetic_frame(width: int = 1280, height: int = 720) -> dict[str, int]:
    return {"width": width, "height": height}
