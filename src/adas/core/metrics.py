"""Metrics collection and monitoring for ADAS system.

This module provides performance metrics and system health monitoring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict

from adas.core.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for ADAS pipeline."""
    
    total_frames: int = 0
    total_detections: int = 0
    total_tracks: int = 0
    frames_with_lane: int = 0
    
    # Timing metrics (in seconds)
    total_processing_time: float = 0.0
    min_frame_time: float = float('inf')
    max_frame_time: float = 0.0
    
    # Safety metrics
    safety_warnings: int = 0
    safety_violations: int = 0
    
    # Component timing
    perception_time: float = 0.0
    tracking_time: float = 0.0
    planning_time: float = 0.0
    control_time: float = 0.0
    
    def update_frame(
        self,
        frame_time: float,
        num_detections: int,
        num_tracks: int,
        has_lane: bool,
    ) -> None:
        """Update metrics for a processed frame.
        
        Args:
            frame_time: Frame processing time in seconds
            num_detections: Number of detections
            num_tracks: Number of tracks
            has_lane: Whether lane was detected
        """
        self.total_frames += 1
        self.total_detections += num_detections
        self.total_tracks += num_tracks
        if has_lane:
            self.frames_with_lane += 1
        
        self.total_processing_time += frame_time
        self.min_frame_time = min(self.min_frame_time, frame_time)
        self.max_frame_time = max(self.max_frame_time, frame_time)
    
    def record_safety_event(self, is_violation: bool = False) -> None:
        """Record a safety warning or violation.
        
        Args:
            is_violation: True if violation, False if warning
        """
        if is_violation:
            self.safety_violations += 1
        else:
            self.safety_warnings += 1
    
    @property
    def avg_frame_time(self) -> float:
        """Average frame processing time in seconds."""
        if self.total_frames == 0:
            return 0.0
        return self.total_processing_time / self.total_frames
    
    @property
    def avg_fps(self) -> float:
        """Average frames per second."""
        if self.total_processing_time == 0:
            return 0.0
        return self.total_frames / self.total_processing_time
    
    @property
    def lane_detection_rate(self) -> float:
        """Percentage of frames with lane detection."""
        if self.total_frames == 0:
            return 0.0
        return (self.frames_with_lane / self.total_frames) * 100.0
    
    def summary(self) -> str:
        """Generate metrics summary string.
        
        Returns:
            Formatted metrics summary
        """
        return (
            f"Performance Metrics:\n"
            f"  Frames: {self.total_frames}\n"
            f"  Detections: {self.total_detections} (avg: {self.total_detections/max(1, self.total_frames):.1f}/frame)\n"
            f"  Tracks: {self.total_tracks} (avg: {self.total_tracks/max(1, self.total_frames):.1f}/frame)\n"
            f"  Lane Detection: {self.lane_detection_rate:.1f}%\n"
            f"  Avg FPS: {self.avg_fps:.1f}\n"
            f"  Avg Frame Time: {self.avg_frame_time*1000:.2f}ms\n"
            f"  Min Frame Time: {self.min_frame_time*1000:.2f}ms\n"
            f"  Max Frame Time: {self.max_frame_time*1000:.2f}ms\n"
            f"  Safety Warnings: {self.safety_warnings}\n"
            f"  Safety Violations: {self.safety_violations}\n"
        )
    
    def log_summary(self) -> None:
        """Log metrics summary."""
        logger.info(f"\n{self.summary()}")


@dataclass
class SystemHealthMonitor:
    """Monitor system health and resource usage."""
    
    start_time: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    heartbeat_interval: float = 5.0  # seconds
    
    def heartbeat(self) -> None:
        """Record heartbeat and check system health."""
        now = time.time()
        elapsed = now - self.last_heartbeat
        
        if elapsed >= self.heartbeat_interval:
            uptime = now - self.start_time
            logger.info(f"System heartbeat: uptime={uptime:.1f}s")
            self.last_heartbeat = now
    
    def check_watchdog(self, timeout: float = 10.0) -> bool:
        """Check if system is responsive.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            True if system is responsive, False otherwise
        """
        elapsed = time.time() - self.last_heartbeat
        return elapsed < timeout
