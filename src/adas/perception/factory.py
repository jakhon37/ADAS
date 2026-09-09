"""Build perception backends from configuration."""

from __future__ import annotations

from adas.core.config import DetectorConfig, LaneConfig
from adas.core.exceptions import ConfigurationError
from adas.core.logger import setup_logger
from adas.perception.detection import ObjectDetector
from adas.perception.lane import LaneEstimator

logger = setup_logger(__name__)


def build_detector(config: DetectorConfig):
    """Return a detector implementing infer(frame, width, height)."""
    backend = (config.backend or "mock").lower()
    if backend == "mock":
        logger.info("Using mock object detector")
        return ObjectDetector(confidence_threshold=config.confidence_threshold)
    if backend == "tensorrt":
        from adas.perception.yolo import YoloTensorRTDetector

        logger.info("Using TensorRT YOLO detector: %s", config.model_path)
        return YoloTensorRTDetector(
            engine_path=config.model_path,
            confidence_threshold=config.confidence_threshold,
            iou_threshold=config.iou_threshold,
            max_detections=config.max_detections,
            input_size=config.input_size,
        )
    raise ConfigurationError("Unknown detector backend: %s" % backend)


def build_lane_estimator(config: LaneConfig):
    """Return a lane estimator implementing estimate(frame, width, height)."""
    backend = (config.backend or "mock").lower()
    if backend == "mock":
        logger.info("Using mock lane estimator")
        return LaneEstimator()
    if backend == "ufld":
        from adas.perception.ufld import UFLDLaneEstimator

        logger.info("Using TensorRT UFLD lane estimator: %s", config.model_path)
        return UFLDLaneEstimator(
            engine_path=config.model_path,
            input_width=config.input_width,
            input_height=config.input_height,
            crop_ratio=config.crop_ratio,
            num_row=config.num_row,
            num_col=config.num_col,
        )
    raise ConfigurationError("Unknown lane backend: %s" % backend)
