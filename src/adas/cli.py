"""CLI entrypoint for running the ADAS reference pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

from adas.control import PIDLikeLongitudinalController, SafetyLimits, SafetyMonitor
from adas.core.config import load_config
from adas.core.logger import setup_logger
from adas.perception.detection import ObjectDetector
from adas.perception.lane import LaneEstimator
from adas.planning import BehaviorPlanner
from adas.runtime import ADASPipeline, PipelineRunner
from adas.tracking import MultiObjectTracker

logger = setup_logger(__name__)


def build_pipeline(config_path: str | None = None) -> tuple[ADASPipeline, int]:
    """Build ADAS pipeline from configuration.
    
    Args:
        config_path: Path to configuration file (None for defaults)
        
    Returns:
        Tuple of (pipeline, target_fps)
    """
    config = load_config(config_path)
    
    # Configure logging level
    log_level = getattr(logging, config.log_level)
    logging.getLogger().setLevel(log_level)
    
    logger.info("Building ADAS pipeline from configuration")
    
    # Build components from configuration
    detector = ObjectDetector(
        confidence_threshold=config.detector.confidence_threshold,
    )
    
    lane_estimator = LaneEstimator()
    
    tracker = MultiObjectTracker(
        max_missed=config.tracker.max_missed_frames,
        association_threshold_px=config.tracker.association_threshold_px,
        focal_length_px=config.tracker.focal_length_px,
        min_box_height_px=config.tracker.min_box_height_px,
        max_distance_m=config.tracker.max_distance_m,
    )
    
    planner = BehaviorPlanner(
        cruise_speed_mps=config.planner.cruise_speed_mps,
        min_follow_distance_m=config.planner.min_follow_distance_m,
        max_steering_deg=config.planner.max_steering_deg,
        time_gap_s=config.planner.time_gap_s,
        max_decel_mps2=config.planner.max_decel_mps2,
        lane_center_gain=config.planner.lane_center_gain,
    )
    
    controller = PIDLikeLongitudinalController(
        kp_speed=config.controller.kp_speed,
        max_throttle=config.controller.max_throttle,
        max_brake=config.controller.max_brake,
        max_steering_angle_deg=config.controller.max_steering_angle_deg,
        steering_deadband_deg=config.controller.steering_deadband_deg,
    )
    
    # Build safety monitor from config
    safety_limits = SafetyLimits(
        max_speed_mps=config.safety.max_speed_mps,
        max_acceleration_mps2=config.safety.max_acceleration_mps2,
        max_deceleration_mps2=config.safety.max_deceleration_mps2,
        max_steering_rate_rad_s=config.safety.max_steering_rate_rad_s,
        max_steering_angle_rad=config.safety.max_steering_angle_rad,
        min_following_distance_m=config.safety.min_following_distance_m,
        max_lateral_offset_m=config.safety.max_lateral_offset_m,
    )
    safety_monitor = SafetyMonitor(limits=safety_limits)
    
    # Build pipeline
    pipeline = ADASPipeline(
        detector=detector,
        lane_estimator=lane_estimator,
        tracker=tracker,
        planner=planner,
        controller=controller,
        safety_monitor=safety_monitor,
    )
    
    logger.info("Pipeline built successfully")
    return pipeline, config.fps


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ADAS Core - Advanced Driver Assistance System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=None, 
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--synthetic", 
        action="store_true", 
        help="Run synthetic test (default mode)"
    )
    parser.add_argument(
        "--frames", 
        type=int, 
        default=60, 
        help="Number of frames to process in synthetic mode"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override log level from config"
    )
    
    args = parser.parse_args()
    
    try:
        # Build pipeline
        pipeline, fps = build_pipeline(args.config)
        
        # Override log level if specified
        if args.log_level:
            log_level = getattr(logging, args.log_level)
            logging.getLogger().setLevel(log_level)
        
        # Run synthetic test
        logger.info(f"Starting ADAS system (FPS={fps})")
        runner = PipelineRunner(pipeline, target_fps=float(fps))
        runner.run_synthetic(max_frames=args.frames)
        
        logger.info("ADAS system shutdown complete")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
