"""Data recorder for ADAS pipeline debugging.

This module provides functionality to record pipeline inputs and outputs
for offline analysis and debugging.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from adas.core.logger import setup_logger
from adas.core.models import ControlCommand, MotionPlan, PerceptionFrame

logger = setup_logger(__name__)


@dataclass
class RecordingConfig:
    """Configuration for data recording."""
    
    output_dir: str = "recordings"
    recording_name: Optional[str] = None
    record_images: bool = False  # Images can be large
    record_detections: bool = True
    record_plans: bool = True
    record_commands: bool = True
    compression: bool = True
    format: str = "json"  # json or pickle
    
    def __post_init__(self):
        """Generate recording name if not provided."""
        if self.recording_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.recording_name = f"adas_recording_{timestamp}"


class DataRecorder:
    """Records ADAS pipeline data for debugging and analysis.
    
    Records:
    - Input frames (optional: images)
    - Perception outputs (detections, lanes)
    - Planning outputs (motion plans)
    - Control outputs (commands)
    - Timing information
    - Metadata
    """
    
    def __init__(self, config: Optional[RecordingConfig] = None):
        """Initialize data recorder.
        
        Args:
            config: Recording configuration
        """
        self.config = config or RecordingConfig()
        self.recording_dir = Path(self.config.output_dir) / self.config.recording_name
        self.recording_dir.mkdir(parents=True, exist_ok=True)
        
        self.frame_data = []
        self.metadata = {
            "recording_name": self.config.recording_name,
            "start_time": time.time(),
            "config": asdict(self.config),
        }
        
        self.is_recording = False
        logger.info(f"DataRecorder initialized: {self.recording_dir}")
    
    def start_recording(self) -> None:
        """Start recording data."""
        self.is_recording = True
        self.metadata["start_time"] = time.time()
        logger.info("Recording started")
    
    def stop_recording(self) -> None:
        """Stop recording and save data."""
        self.is_recording = False
        self.metadata["end_time"] = time.time()
        self.metadata["total_frames"] = len(self.frame_data)
        self._save_recording()
        logger.info(f"Recording stopped. Saved {len(self.frame_data)} frames")
    
    def record_frame(
        self,
        frame: PerceptionFrame,
        plan: Optional[MotionPlan] = None,
        command: Optional[ControlCommand] = None,
        timing_info: Optional[dict] = None,
    ) -> None:
        """Record a single frame's data.
        
        Args:
            frame: Perception frame
            plan: Motion plan (optional)
            command: Control command (optional)
            timing_info: Timing information (optional)
        """
        if not self.is_recording:
            return
        
        frame_record = {
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp_s,
            "width": frame.width,
            "height": frame.height,
        }
        
        # Record image data if enabled
        if self.config.record_images and frame.rgb is not None:
            if isinstance(frame.rgb, dict):
                frame_record["image"] = frame.rgb
            else:
                frame_record["image"] = {"data": "binary"}  # Placeholder
        
        # Record detections
        if self.config.record_detections and frame.detections:
            frame_record["detections"] = [
                {
                    "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                    "confidence": d.confidence,
                    "label": d.label,
                }
                for d in frame.detections
            ]
        
        # Record lane
        if frame.lane:
            frame_record["lane"] = {
                "lane_center_px": frame.lane.lane_center_px,
                "coefficients": frame.lane.coefficients,
                "lateral_offset_m": frame.lane.lateral_offset_m,
                "heading_error_rad": frame.lane.heading_error_rad,
            }
        
        # Record plan
        if self.config.record_plans and plan:
            frame_record["plan"] = {
                "target_speed_mps": plan.target_speed_mps,
                "steering_angle_deg": plan.steering_angle_deg,
                "reason": plan.reason,
            }
        
        # Record command
        if self.config.record_commands and command:
            frame_record["command"] = {
                "throttle": command.throttle,
                "brake": command.brake,
                "steering": command.steering,
            }
        
        # Record timing
        if timing_info:
            frame_record["timing"] = timing_info
        
        self.frame_data.append(frame_record)
    
    def _save_recording(self) -> None:
        """Save recorded data to disk."""
        # Save metadata
        metadata_path = self.recording_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save frame data
        if self.config.format == "json":
            data_path = self.recording_dir / "frames.json"
            with open(data_path, "w") as f:
                json.dump(self.frame_data, f, indent=2)
        elif self.config.format == "pickle":
            data_path = self.recording_dir / "frames.pkl"
            with open(data_path, "wb") as f:
                pickle.dump(self.frame_data, f)
        
        logger.info(f"Recording saved to {self.recording_dir}")
        
        # Save summary
        summary_path = self.recording_dir / "summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"ADAS Recording Summary\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Recording: {self.config.recording_name}\n")
            f.write(f"Total Frames: {len(self.frame_data)}\n")
            f.write(f"Duration: {self.metadata.get('end_time', 0) - self.metadata['start_time']:.2f}s\n")
            f.write(f"Format: {self.config.format}\n")
            f.write(f"Images Recorded: {self.config.record_images}\n")
    
    def get_stats(self) -> dict:
        """Get recording statistics.
        
        Returns:
            Dictionary with recording stats
        """
        return {
            "total_frames": len(self.frame_data),
            "recording_time": time.time() - self.metadata["start_time"],
            "is_recording": self.is_recording,
            "output_dir": str(self.recording_dir),
        }


class RecordingPipeline:
    """Wrapper that records pipeline execution.
    
    This wrapper intercepts pipeline calls and records all data.
    """
    
    def __init__(self, pipeline, recorder: DataRecorder):
        """Initialize recording pipeline wrapper.
        
        Args:
            pipeline: ADAS pipeline to wrap
            recorder: Data recorder instance
        """
        self.pipeline = pipeline
        self.recorder = recorder
    
    def step(self, frame: PerceptionFrame, current_speed_mps: float = 0.0):
        """Execute pipeline step and record data.
        
        Args:
            frame: Perception frame
            current_speed_mps: Current vehicle speed
            
        Returns:
            Tuple of (plan, command)
        """
        start_time = time.time()
        
        # Execute pipeline
        plan, command = self.pipeline.step(frame, current_speed_mps)
        
        elapsed_ms = (time.time() - start_time) * 1000.0
        
        # Record data
        self.recorder.record_frame(
            frame=frame,
            plan=plan,
            command=command,
            timing_info={"total_ms": elapsed_ms}
        )
        
        return plan, command
    
    def __getattr__(self, name):
        """Delegate other attributes to wrapped pipeline."""
        return getattr(self.pipeline, name)
