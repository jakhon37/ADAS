"""Runtime utilities for local simulation and stream processing."""

from __future__ import annotations

import time
from dataclasses import dataclass

from adas.core.logger import log_performance, setup_logger
from adas.core.models import PerceptionFrame
from adas.runtime.pipeline import ADASPipeline

logger = setup_logger(__name__)


@dataclass(slots=True)
class PipelineRunner:
    """Runtime wrapper for executing the ADAS pipeline."""
    
    pipeline: ADASPipeline
    target_fps: float = 30.0  # Target processing rate
    
    def run_synthetic(self, max_frames: int = 60) -> None:
        """Run pipeline on synthetic data for testing.
        
        Args:
            max_frames: Number of frames to process
        """
        logger.info(f"Starting synthetic run: {max_frames} frames at {self.target_fps} FPS")
        
        frame_time_s = 1.0 / self.target_fps
        simulated_speed_mps = 10.0  # Simulated vehicle speed
        
        for frame_id in range(max_frames):
            start_time = time.time()
            
            # Generate synthetic frame
            frame = synthetic_frame()
            perception = PerceptionFrame(
                frame_id=frame_id,
                timestamp_s=time.time(),
                rgb=frame,
                width=frame["width"],
                height=frame["height"],
            )
            
            # Run pipeline
            try:
                plan, cmd = self.pipeline.step(perception, current_speed_mps=simulated_speed_mps)
                
                # Update simulated speed based on control
                # Simple integration: speed += (throttle - brake) * dt * gain
                speed_delta = (cmd.throttle - cmd.brake) * frame_time_s * 5.0
                simulated_speed_mps = max(0.0, simulated_speed_mps + speed_delta)
                
            except Exception as e:
                logger.error(f"Pipeline failed on frame {frame_id}: {e}")
                continue
            
            # Log performance
            elapsed_ms = (time.time() - start_time) * 1000.0
            log_performance(logger, f"frame_{frame_id}", elapsed_ms)
            
            # Sleep to maintain target FPS (if processing was faster)
            sleep_time = frame_time_s - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        logger.info(f"Synthetic run completed: {max_frames} frames processed")


def synthetic_frame(width: int = 1280, height: int = 720) -> dict[str, int]:
    """Generate a synthetic frame for testing.
    
    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        
    Returns:
        Dictionary with frame metadata
    """
    return {"width": width, "height": height}
