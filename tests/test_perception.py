"""Perception tests that do not need a GPU, plus engine-gated replay checks.

The pure-numpy decode coverage lives in ``tests/test_yolo_decode.py`` and the
depth channel in ``tests/test_depth.py``. This file covers the factory, the
mock backends' honesty, and -- when the built engines and the replay clip are
present on the board -- a real end-to-end inference.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from adas.core.config import DetectorConfig, LaneConfig, load_config
from adas.core.exceptions import PerceptionError
from adas.perception import detection as D
from adas.perception.factory import build_detector, build_lane_estimator
from adas.perception.yolo import YoloTensorRTDetector, nms_xyxy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLOX_ENGINE = os.path.join(REPO_ROOT, "models", "yolox_nano.engine")
YOLOV5_ENGINE = os.path.join(REPO_ROOT, "models", "yolov5n.engine")
MIDAS_ENGINE = os.path.join(REPO_ROOT, "models", "midas_v21_small_256.engine")
CLIP = os.path.join(REPO_ROOT, "Ultra-Fast-Lane-Detection-v2", "example.mp4")


# --------------------------------------------------------------------------- #
# Original coverage, kept
# --------------------------------------------------------------------------- #


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
    config = load_config(os.path.join(REPO_ROOT, "config.example.json"))
    assert config.detector.backend == "mock"
    assert config.tracker.object_height_m == 1.5
    assert config.tracker.focal_length_px == 910.0
    assert config.source.type == "synthetic"


# --------------------------------------------------------------------------- #
# The mock must be impossible to mistake for perception
# --------------------------------------------------------------------------- #


def test_mock_detector_declares_itself(caplog):
    with caplog.at_level("WARNING"):
        detector = D.ObjectDetector()
    assert detector.is_mock is True
    assert any("NOT FOR VEHICLE USE" in record.message for record in caplog.records)


def test_mock_label_is_not_a_coco_class():
    """A real detector can never emit "vehicle"; that is the tell."""
    assert D.MOCK_LABEL not in D.COCO_CLASSES


def test_real_detector_reports_is_mock_false():
    from adas.infer.trt_engine import FakeTrtEngine

    engine = FakeTrtEngine({"images": (1, 3, 416, 416)},
                           {"output": np.zeros((1, 3549, 85), np.float32)})
    detector = YoloTensorRTDetector(engine=engine)
    assert detector.is_mock is False
    detector.close()


def test_missing_engine_is_a_clear_error():
    with pytest.raises(PerceptionError):
        YoloTensorRTDetector("models/there-is-no-such.engine")


# --------------------------------------------------------------------------- #
# Engine-backed replay (skipped when the artifacts are absent)
# --------------------------------------------------------------------------- #

requires_yolox = pytest.mark.skipif(
    not os.path.exists(YOLOX_ENGINE), reason="yolox_nano.engine not built on this host"
)
requires_clip = pytest.mark.skipif(
    not os.path.exists(CLIP), reason="example.mp4 replay clip not present"
)


_FRAME_CACHE = {}


def _clip_frame(index: int = 150):
    """Decode and cache one frame of the replay clip by absolute index."""
    if index in _FRAME_CACHE:
        return _FRAME_CACHE[index]
    import cv2

    capture = cv2.VideoCapture(CLIP)
    try:
        frame = None
        for _ in range(index + 1):
            ok, frame = capture.read()
            if not ok:
                pytest.skip("replay clip ended before frame %d" % index)
    finally:
        capture.release()
    _FRAME_CACHE[index] = frame
    return frame


@requires_yolox
def test_yolox_engine_contract_is_what_the_manifest_says():
    detector = YoloTensorRTDetector(YOLOX_ENGINE)
    try:
        assert detector.net_size == 416
        assert detector.contract.layout == "yolox"
        assert detector.contract.rows == 3549
        assert detector.contract.num_classes == 80
        assert detector.preproc is D.PREPROC_YOLOX
        assert 0 in detector.class_ids  # pedestrians
    finally:
        detector.close()


@requires_yolox
@requires_clip
def test_yolox_detects_the_lead_vehicle_on_the_replay_clip():
    """Frame 150 of example.mp4 has a car in the right-hand lane at ~0.64.

    Verified against YOLOX's own ``demo_postprocess`` + ``multiclass_nms``:
    identical boxes to 0.1 px and identical scores.
    """
    frame = _clip_frame(150)
    detector = YoloTensorRTDetector(YOLOX_ENGINE, confidence_threshold=0.30,
                                    iou_threshold=0.45, class_ids="all")
    try:
        height, width = frame.shape[:2]
        boxes = detector.infer(frame, width, height)
        cars = [b for b in boxes if b.label == "car"]
        assert cars, "expected at least one car on frame 150"
        best = max(cars, key=lambda b: b.confidence)
        assert best.confidence == pytest.approx(0.64, abs=0.05)
        assert 1100 < best.x1 < 1220
        assert 380 < best.y1 < 430
        # Every box must lie inside the source frame.
        for box in boxes:
            assert 0.0 <= box.x1 < box.x2 <= width - 1
            assert 0.0 <= box.y1 < box.y2 <= height - 1
        assert detector.timing_summary()["frames"] == 1
    finally:
        detector.close()


@requires_yolox
def test_engine_sha256_is_verified_when_asked():
    """Opt-in integrity check against the digest in models/MANIFEST.json."""
    import json

    from adas.infer.trt_engine import EngineError, TrtEngine, sha256_file

    manifest_path = os.path.join(REPO_ROOT, "models", "MANIFEST.json")
    if not os.path.exists(manifest_path):
        pytest.skip("models/MANIFEST.json not present")
    manifest = json.loads(open(manifest_path).read())
    expected = manifest.get("models", {}).get("yolox_nano", {}).get("engine_sha256")
    if not expected:
        pytest.skip("manifest has no engine_sha256 for yolox_nano")

    assert sha256_file(YOLOX_ENGINE) == expected
    engine = TrtEngine(YOLOX_ENGINE, expected_sha256=expected)
    try:
        assert engine.sha256 == expected
    finally:
        engine.close()

    with pytest.raises(EngineError):
        TrtEngine(YOLOX_ENGINE, expected_sha256="0" * 64)


@requires_yolox
@requires_clip
def test_engine_and_fake_engine_decode_identically():
    """The CPU double must reproduce the real engine's decode exactly."""
    from adas.infer.trt_engine import FakeTrtEngine

    frame = _clip_frame(150)
    height, width = frame.shape[:2]
    real = YoloTensorRTDetector(YOLOX_ENGINE, confidence_threshold=0.30, class_ids="all")
    try:
        boxes = real.infer(frame, width, height)
        blob, _transform = real.preprocess(frame)
        head = real.engine.infer({real.engine.input_name: blob})[real.output_name]
    finally:
        real.close()

    fake = FakeTrtEngine({"images": (1, 3, 416, 416)}, {"output": head})
    replay = YoloTensorRTDetector(engine=fake, confidence_threshold=0.30, class_ids="all")
    try:
        again = replay.infer(frame, width, height)
    finally:
        replay.close()

    assert len(again) == len(boxes)
    for a, b in zip(again, boxes):
        assert (a.x1, a.y1, a.x2, a.y2) == pytest.approx((b.x1, b.y1, b.x2, b.y2), abs=1e-4)
        assert a.label == b.label


@pytest.mark.skipif(
    not os.path.exists(MIDAS_ENGINE), reason="midas_v21_small_256.engine not built on this host"
)
@requires_clip
def test_depth_channel_runs_on_the_replay_clip():
    from adas.perception.depth import DepthRangeChannel
    from adas.perception.geometry import default_camera

    frame = _clip_frame(150)
    height, width = frame.shape[:2]
    camera = default_camera(width, height)
    channel = DepthRangeChannel(MIDAS_ENGINE, cadence_frames=5)
    try:
        assert channel.available and not channel.is_mock
        results = channel.update(frame, 0, [], width, height, camera=camera)
        assert results == []
        assert channel.scale_mode == "road_plane"
        assert channel.scale.valid
        assert channel.scale.a > 0.0
        assert channel.scale.anchors >= channel.min_road_anchors
    finally:
        channel.close()


# --------------------------------------------------------------------------- #
# Engine integrity against models/MANIFEST.json
#
# Regression cover for: no production caller verified any engine digest, so a
# swapped or corrupted .engine loaded silently into the detector that feeds AEB.
# --------------------------------------------------------------------------- #

MANIFEST = os.path.join(REPO_ROOT, "models", "MANIFEST.json")


def _local_manifest(tmp_path, engine_name, digest, nbytes, trt_version="8.5.2.2"):
    """Write a one-entry manifest and return its path."""
    import json as _json

    path = tmp_path / "MANIFEST.json"
    path.write_text(_json.dumps({
        "schema": "adas/models/manifest@1",
        "trt_version": trt_version,
        "models": {
            "unit": {
                "engine_file": engine_name,
                "engine_sha256": digest,
                "engine_bytes": nbytes,
            }
        },
    }))
    return str(path)


def test_verify_engine_file_accepts_the_recorded_digest(tmp_path):
    from adas.infer.trt_engine import sha256_file, verify_engine_file

    blob = tmp_path / "unit.engine"
    blob.write_bytes(b"the validated bytes")
    digest = sha256_file(str(blob))
    manifest = _local_manifest(tmp_path, "unit.engine", digest, blob.stat().st_size)

    assert verify_engine_file(str(blob), manifest, check_trt_version=False) == digest


def test_verify_engine_file_refuses_a_swapped_engine(tmp_path):
    """The message must name the file, the expected digest and the actual one."""
    from adas.infer.trt_engine import EngineIntegrityError, sha256_file, verify_engine_file

    blob = tmp_path / "unit.engine"
    blob.write_bytes(b"the validated bytes")
    good = sha256_file(str(blob))
    manifest = _local_manifest(tmp_path, "unit.engine", good, blob.stat().st_size)

    blob.write_bytes(b"a different model entirely")
    swapped = sha256_file(str(blob))
    assert swapped != good

    with pytest.raises(EngineIntegrityError) as excinfo:
        verify_engine_file(str(blob), manifest, check_trt_version=False)
    message = str(excinfo.value)
    assert str(blob) in message
    assert good in message
    assert swapped in message


def test_verify_engine_file_refuses_an_engine_built_for_another_tensorrt(tmp_path, monkeypatch):
    from adas.infer import trt_engine as T

    blob = tmp_path / "unit.engine"
    blob.write_bytes(b"the validated bytes")
    digest = T.sha256_file(str(blob))
    manifest = _local_manifest(tmp_path, "unit.engine", digest, blob.stat().st_size,
                               trt_version="8.5.2.2")

    monkeypatch.setattr(T, "trt_version", lambda: "10.0.1")
    with pytest.raises(T.EngineIntegrityError) as excinfo:
        T.verify_engine_file(str(blob), manifest)
    assert "8.5.2.2" in str(excinfo.value) and "10.0.1" in str(excinfo.value)

    monkeypatch.setattr(T, "trt_version", lambda: "8.5.2.2")
    assert T.verify_engine_file(str(blob), manifest) == digest


def test_an_engine_the_manifest_does_not_list_is_unverifiable_not_verified(tmp_path, caplog):
    """No digest to check against must read as 'unverified', never as 'verified'."""
    import logging

    from adas.infer.trt_engine import VERIFIED_ENGINES, sha256_file, verify_engine_file

    blob = tmp_path / "handbuilt.engine"
    blob.write_bytes(b"freshly built, not in the inventory")
    manifest = _local_manifest(tmp_path, "unit.engine", sha256_file(str(blob)), 1)

    with caplog.at_level(logging.WARNING):
        assert verify_engine_file(str(blob), manifest, check_trt_version=False) == ""
    assert "cannot be verified" in caplog.text
    assert os.path.abspath(str(blob)) not in VERIFIED_ENGINES


@pytest.mark.skipif(not os.path.exists(MANIFEST), reason="models/MANIFEST.json not present")
def test_build_detector_refuses_an_engine_that_is_not_the_manifest_one(tmp_path):
    """The production path -- factory, not a hand-written TrtEngine call -- must refuse."""
    import json as _json

    recorded = _json.loads(open(MANIFEST).read())["models"]["yolox_nano"]["engine_sha256"]
    impostor = tmp_path / "yolox_nano.engine"
    impostor.write_bytes(b"this is not the engine that was validated")

    config = DetectorConfig(backend="yolox", model_path=str(impostor))
    with pytest.raises(PerceptionError) as excinfo:
        build_detector(config, allow_mock=False, allow_fallback=False)
    message = str(excinfo.value)
    assert "MANIFEST.json" in message
    assert recorded in message
    assert str(impostor) in message


@pytest.mark.skipif(not os.path.exists(MANIFEST), reason="models/MANIFEST.json not present")
def test_a_bad_digest_downgrades_to_the_mock_only_with_allow_fallback(tmp_path):
    """Bench mode may run on a stub; it may not run on an unknown model silently."""
    impostor = tmp_path / "yolox_nano.engine"
    impostor.write_bytes(b"this is not the engine that was validated")

    config = DetectorConfig(backend="yolox", model_path=str(impostor))
    detector = build_detector(config, allow_mock=True, allow_fallback=True)
    assert getattr(detector, "is_mock", False) is True


@pytest.mark.skipif(not os.path.exists(MANIFEST), reason="models/MANIFEST.json not present")
def test_the_manifest_matches_the_engines_present_on_this_board():
    """Every built engine must be the one the manifest records, or the board lies."""
    import json as _json

    from adas.infer.trt_engine import sha256_file

    manifest = _json.loads(open(MANIFEST).read())
    checked = 0
    for name, entry in sorted(manifest.get("models", {}).items()):
        path = os.path.join(REPO_ROOT, "models", entry.get("engine_file", ""))
        if not os.path.exists(path):
            continue
        checked += 1
        assert sha256_file(path) == entry["engine_sha256"], name
    if checked == 0:
        pytest.skip("no engines built on this host")


# --------------------------------------------------------------------------- #
# The startup failure path an operator actually reads
# --------------------------------------------------------------------------- #


def test_the_missing_engine_message_names_each_search_root_once(monkeypatch):
    """It used to print the repo root twice whenever cwd was the checkout."""
    from adas.perception.factory import _REPO_ROOT

    monkeypatch.chdir(_REPO_ROOT)
    with pytest.raises(PerceptionError) as excinfo:
        build_detector(
            DetectorConfig(backend="yolox", model_path="models/definitely_absent.engine"),
            allow_mock=False, allow_fallback=False,
        )
    message = str(excinfo.value)
    tried = message.split("(tried ", 1)[1].split(")", 1)[0].split(", ")
    assert len(tried) == len(set(tried)), message
    assert tried == [os.path.join(str(_REPO_ROOT), "models", "definitely_absent.engine")]
    assert "build_engines.py" in message


def test_the_search_roots_are_listed_when_they_really_differ(tmp_path, monkeypatch):
    from adas.perception.factory import _REPO_ROOT, search_paths

    monkeypatch.chdir(tmp_path)
    paths = [str(p) for p in search_paths("models/x.engine")]
    assert paths == [
        os.path.join(os.path.realpath(str(tmp_path)), "models", "x.engine"),
        os.path.join(str(_REPO_ROOT), "models", "x.engine"),
    ]
    assert [str(p) for p in search_paths("/abs/x.engine")] == ["/abs/x.engine"]
