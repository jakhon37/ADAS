"""Data replayer for ADAS pipeline debugging.

This module provides functionality to replay recorded pipeline data
for offline analysis, debugging, and testing.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from adas.core.logger import setup_logger
from adas.core.models import BoundingBox, LaneModel, PerceptionFrame

logger = setup_logger(__name__)


@dataclass
class ReplayConfig:
    """Configuration for data replay."""
    
    recording_dir: str
    playback_speed: float = 1.0  # 1.0 = real-time, 0.0 = as fast as possible
    loop: bool = False
    start_frame: int = 0
    end_frame: Optional[int] = None


class DataReplayer:
    """Replays recorded ADAS pipeline data.
    
    Allows offline analysis and debugging by replaying
    previously recorded pipeline executions.
    """
    
    def __init__(self, config: ReplayConfig):
        """Initialize data replayer.
        
        Args:
            config: Replay configuration
        """
        self.config = config
        self.recording_dir = Path(config.recording_dir)
        
        if not self.recording_dir.exists():
            raise FileNotFoundError(f"Recording directory not found: {self.recording_dir}")
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Load frame data
        self.frames = self._load_frames()
        
        self.current_frame_idx = config.start_frame
        
        logger.info(f"DataReplayer initialized: {self.recording_dir}")
        logger.info(f"Loaded {len(self.frames)} frames")
    
    def _load_metadata(self) -> dict:
        """Load recording metadata.
        
        Returns:
            Metadata dictionary
        """
        metadata_path = self.recording_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return {}
    
    def _load_frames(self) -> list:
        """Load recorded frames.
        
        Returns:
            List of frame data dictionaries
        """
        # Try JSON first
        json_path = self.recording_dir / "frames.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                frames = json.load(f)
                logger.info(f"Loaded {len(frames)} frames from JSON")
                return frames
        
        # Try pickle
        pkl_path = self.recording_dir / "frames.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                frames = pickle.load(f)
                logger.info(f"Loaded {len(frames)} frames from pickle")
                return frames
        
        raise FileNotFoundError("No frame data found (frames.json or frames.pkl)")
    
    def get_frame(self, frame_idx: int) -> Optional[dict]:
        """Get a specific frame by index.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Frame data dictionary or None
        """
        if 0 <= frame_idx < len(self.frames):
            return self.frames[frame_idx]
        return None
    
    def get_perception_frame(self, frame_idx: int) -> Optional[PerceptionFrame]:
        """Reconstruct PerceptionFrame from recorded data.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            PerceptionFrame instance or None
        """
        frame_data = self.get_frame(frame_idx)
        if frame_data is None:
            return None
        
        # Reconstruct perception frame
        perception_frame = PerceptionFrame(
            frame_id=frame_data["frame_id"],
            timestamp_s=frame_data["timestamp"],
            rgb=frame_data.get("image", {"width": frame_data["width"], "height": frame_data["height"]}),
            width=frame_data["width"],
            height=frame_data["height"],
        )
        
        # Reconstruct detections
        if "detections" in frame_data:
            perception_frame.detections = [
                BoundingBox(
                    x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"],
                    confidence=d["confidence"],
                    label=d["label"]
                )
                for d in frame_data["detections"]
            ]
        
        # Reconstruct lane
        if "lane" in frame_data:
            lane_data = frame_data["lane"]
            perception_frame.lane = LaneModel(
                lane_center_px=lane_data["lane_center_px"],
                coefficients=lane_data.get("coefficients", [0.0, 0.0, 0.0]),
                lateral_offset_m=lane_data.get("lateral_offset_m", 0.0),
                heading_error_rad=lane_data.get("heading_error_rad", 0.0),
            )
        
        return perception_frame
    
    def replay_iterator(self) -> Iterator[dict]:
        """Create iterator for replaying frames.
        
        Yields:
            Frame data dictionaries
        """
        end_frame = self.config.end_frame or len(self.frames)
        
        while True:
            for idx in range(self.config.start_frame, end_frame):
                frame_data = self.frames[idx]
                
                # Handle playback speed
                if self.config.playback_speed > 0:
                    if idx > self.config.start_frame:
                        # Calculate time delta from recording
                        prev_timestamp = self.frames[idx - 1]["timestamp"]
                        curr_timestamp = frame_data["timestamp"]
                        dt = curr_timestamp - prev_timestamp
                        
                        # Sleep to maintain playback speed
                        sleep_time = dt / self.config.playback_speed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                
                self.current_frame_idx = idx
                yield frame_data
            
            if not self.config.loop:
                break
            
            logger.info("Looping replay...")
    
    def get_stats(self) -> dict:
        """Get replay statistics.
        
        Returns:
            Dictionary with replay stats
        """
        return {
            "total_frames": len(self.frames),
            "current_frame": self.current_frame_idx,
            "recording_name": self.metadata.get("recording_name", "unknown"),
            "playback_speed": self.config.playback_speed,
            "loop": self.config.loop,
        }
    
    def get_frame_summary(self, frame_idx: int) -> str:
        """Get human-readable summary of a frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Summary string
        """
        frame_data = self.get_frame(frame_idx)
        if frame_data is None:
            return "Frame not found"
        
        lines = [
            f"Frame {frame_data['frame_id']}:",
            f"  Timestamp: {frame_data['timestamp']:.3f}s",
            f"  Resolution: {frame_data['width']}x{frame_data['height']}",
        ]
        
        if "detections" in frame_data:
            lines.append(f"  Detections: {len(frame_data['detections'])}")
        
        if "lane" in frame_data:
            lane = frame_data["lane"]
            lines.append(f"  Lane: center={lane['lane_center_px']:.1f}px")
        
        if "plan" in frame_data:
            plan = frame_data["plan"]
            lines.append(f"  Plan: speed={plan['target_speed_mps']:.1f}m/s, reason={plan['reason']}")
        
        if "command" in frame_data:
            cmd = frame_data["command"]
            lines.append(f"  Command: t={cmd['throttle']:.2f}, b={cmd['brake']:.2f}, s={cmd['steering']:.2f}")
        
        if "timing" in frame_data:
            timing = frame_data["timing"]
            lines.append(f"  Timing: {timing['total_ms']:.2f}ms")
        
        return "\n".join(lines)
    
    def export_summary(self, output_path: Optional[str] = None) -> None:
        """Export replay summary to file.
        
        Args:
            output_path: Output file path (default: recording_dir/replay_summary.txt)
        """
        if output_path is None:
            output_path = self.recording_dir / "replay_summary.txt"
        
        with open(output_path, "w") as f:
            f.write("ADAS Replay Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Recording: {self.metadata.get('recording_name', 'unknown')}\n")
            f.write(f"Total Frames: {len(self.frames)}\n\n")
            
            # Sample some frames
            sample_indices = [0, len(self.frames) // 2, len(self.frames) - 1]
            for idx in sample_indices:
                f.write(self.get_frame_summary(idx))
                f.write("\n\n")
        
        logger.info(f"Summary exported to {output_path}")


def replay_with_pipeline(replayer: DataReplayer, pipeline):
    """Replay recorded data through pipeline for comparison.
    
    Args:
        replayer: DataReplayer instance
        pipeline: ADAS pipeline to run
        
    Yields:
        Tuple of (recorded_data, new_results)
    """
    for frame_data in replayer.replay_iterator():
        # Reconstruct perception frame
        perception_frame = replayer.get_perception_frame(replayer.current_frame_idx)
        
        if perception_frame is None:
            continue
        
        # Run through pipeline
        plan, command = pipeline.step(perception_frame, current_speed_mps=0.0)
        
        # Yield both for comparison
        yield frame_data, (plan, command)
