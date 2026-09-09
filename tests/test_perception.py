"""Tests for perception helpers that do not need a GPU."""

from adas.core.config import DetectorConfig, LaneConfig, load_config
from adas.perception.factory import build_detector, build_lane_estimator
from adas.perception.yolo import nms_xyxy


def test_nms_suppresses_overlap():
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
    scores = [0.9, 0.8, 0.7]
    keep = nms_xyxy(boxes, scores, 0.5)
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep


def test_factory_mock_detector():
    det = build_detector(DetectorConfig(backend="mock"))
    boxes = det.infer({"width": 1280, "height": 720}, 1280, 720)
    assert len(boxes) == 1
    assert boxes[0].label == "vehicle"


def test_factory_mock_lane():
    lane = build_lane_estimator(LaneConfig(backend="mock"))
    model = lane.estimate(None, 1280, 720)
    assert model is not None
    assert abs(model.lane_center_px - 640.0) < 1.0


def test_example_config_loads():
    config = load_config("config.example.json")
    assert config.detector.backend == "mock"
    assert config.tracker.object_height_m == 1.5
    assert config.tracker.focal_length_px == 910.0
    assert config.source.type == "synthetic"
