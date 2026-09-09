"""Runtime utilities for local simulation and stream processing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from adas.core.logger import log_performance, setup_logger
from adas.core.models import PerceptionFrame
from adas.runtime.capture import FrameSource, open_source
from adas.runtime.pipeline import ADASPipeline

logger = setup_logger(__name__)


@dataclass
class PipelineRunner:
    """Runtime wrapper for executing the ADAS pipeline."""

    pipeline: ADASPipeline
    target_fps: float = 30.0

    def run_synthetic(self, max_frames: int = 60) -> None:
        """Run pipeline on synthetic data for testing."""
        self.run(source_type="synthetic", max_frames=max_frames)

    def run(
        self,
        source_type: str = "synthetic",
        uri: str = "",
        max_frames: int = 60,
        width: int = 1280,
        height: int = 720,
        as_image: bool = False,
        source: Optional[FrameSource] = None,
    ) -> None:
        """Run the pipeline on a live or synthetic source."""
        logger.info(
            "Starting run: %s frames at %.1f FPS source=%s",
            max_frames,
            self.target_fps,
            source_type,
        )
        frame_time_s = 1.0 / self.target_fps if self.target_fps > 0 else 0.05
        simulated_speed_mps = 10.0
        own_source = source is None
        if source is None:
            source = open_source(
                source_type,
                uri=uri,
                width=width,
                height=height,
                as_image=as_image,
            )
        try:
            for frame_id in range(max_frames):
                start_time = time.time()
                captured = source.read()
                if captured is None:
                    logger.info("End of source at frame %s", frame_id)
                    break
                perception = PerceptionFrame(
                    frame_id=frame_id,
                    timestamp_s=time.time(),
                    rgb=captured.image,
                    width=captured.width,
                    height=captured.height,
                )
                try:
                    _plan, cmd = self.pipeline.step(
                        perception,
                        current_speed_mps=simulated_speed_mps,
                        dt_s=frame_time_s,
                    )
                    speed_delta = (cmd.throttle - cmd.brake) * frame_time_s * 5.0
                    simulated_speed_mps = max(0.0, simulated_speed_mps + speed_delta)
                except Exception as e:
                    logger.error(f"Pipeline failed on frame {frame_id}: {e}")
                    continue

                elapsed_ms = (time.time() - start_time) * 1000.0
                log_performance(logger, f"frame_{frame_id}", elapsed_ms)
                sleep_time = frame_time_s - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            if own_source:
                source.close()

        logger.info("Run completed: up to %s frames processed", max_frames)


def synthetic_frame(width: int = 1280, height: int = 720) -> dict:
    """Generate a synthetic frame for testing (metadata only)."""
    return {"width": width, "height": height}
