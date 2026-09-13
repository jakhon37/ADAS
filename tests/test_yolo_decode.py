"""CPU-only tests for the detector decode path.

Every function under test here is pure numpy, so none of this needs a Jetson, a
GPU, TensorRT or an engine file -- the engine-backed cases use
:class:`adas.infer.trt_engine.FakeTrtEngine`. That matters because the decode
path is what decides whether the ADAS sees anything, and it previously had zero
coverage.

Coordinate convention: source-frame pixels, origin top-left, ``u`` right,
``v`` down.
"""

from __future__ import annotations

import numpy as np
import pytest

from adas.core.exceptions import PerceptionError
from adas.infer.trt_engine import FakeTrtEngine
from adas.perception import detection as D
from adas.perception.yolo import YoloTensorRTDetector, decode_yolo, letterbox, nms_xyxy
from adas.perception.yolox import (
    YoloXTensorRTDetector,
    decode_yolox,
    decode_yolox_boxes,
    preprocess_yolox,
    yolox_grids,
)

SRC_W, SRC_H = 1280, 720


# --------------------------------------------------------------------------- #
# Letterbox
# --------------------------------------------------------------------------- #


def test_letterbox_centred_geometry():
    image = np.zeros((SRC_H, SRC_W, 3), np.uint8)
    canvas, transform = D.letterbox_canvas(image, 640, pad_mode="center")
    assert canvas.shape == (640, 640, 3)
    assert transform.scale == pytest.approx(0.5)
    assert transform.pad_x == 0.0
    assert transform.pad_y == pytest.approx((640 - 360) // 2)


def test_letterbox_topleft_has_no_pad_offset():
    image = np.zeros((SRC_H, SRC_W, 3), np.uint8)
    _canvas, transform = D.letterbox_canvas(image, 416, pad_mode="topleft")
    assert transform.pad_x == 0.0 and transform.pad_y == 0.0
    assert transform.scale == pytest.approx(416.0 / SRC_W)


@pytest.mark.parametrize("pad_mode", ["center", "topleft"])
@pytest.mark.parametrize("net_size", [416, 640])
@pytest.mark.parametrize("point", [(0.0, 0.0), (640.0, 360.0), (1279.0, 719.0), (37.5, 611.25)])
def test_letterbox_inverse_round_trips_exactly(pad_mode, net_size, point):
    transform = D.letterbox_transform(SRC_W, SRC_H, net_size, pad_mode=pad_mode)
    u_net, v_net = transform.to_net(*point)
    u_src, v_src = transform.to_source(u_net, v_net)
    assert u_src == pytest.approx(point[0], abs=1e-6)
    assert v_src == pytest.approx(point[1], abs=1e-6)


def test_letterbox_places_a_known_pixel_where_the_transform_says():
    """The forward transform must agree with what cv2.resize actually did."""
    image = np.zeros((SRC_H, SRC_W, 3), np.uint8)
    image[300:340, 600:640] = 255  # a solid white block
    canvas, transform = D.letterbox_canvas(image, 640, pad_mode="center")
    u_net, v_net = transform.to_net(620.0, 320.0)  # block centre
    assert canvas[int(round(v_net)), int(round(u_net))].tolist() == [255, 255, 255]
    # And the padding really is the pad value, not image content.
    assert canvas[0, 0].tolist() == [114, 114, 114]


def test_letterbox_rejects_non_uint8():
    float_frame = np.zeros((SRC_H, SRC_W, 3), np.float32)
    with pytest.raises(PerceptionError):
        letterbox(float_frame, 640)


def test_letterbox_rejects_wrong_rank():
    with pytest.raises(PerceptionError):
        letterbox(np.zeros((SRC_H, SRC_W), np.uint8), 640)


def test_letterbox_legacy_tuple_still_works():
    canvas, scale, pad_x, pad_y = letterbox(np.zeros((SRC_H, SRC_W, 3), np.uint8), 640)
    assert canvas.shape == (640, 640, 3)
    assert (scale, pad_x, pad_y) == (0.5, 0, 140)


# --------------------------------------------------------------------------- #
# Preprocessing contracts
# --------------------------------------------------------------------------- #


def test_yolov5_blob_is_rgb_scaled_to_unit_range():
    canvas = np.zeros((64, 64, 3), np.uint8)
    canvas[0, 0] = (10, 20, 30)  # BGR
    blob = D.blob_from_canvas(canvas, D.PREPROC_YOLOV5)
    assert blob.shape == (1, 3, 64, 64)
    assert blob[0, :, 0, 0] == pytest.approx([30 / 255.0, 20 / 255.0, 10 / 255.0], abs=1e-6)


def test_yolox_blob_is_bgr_and_raw():
    canvas = np.zeros((64, 64, 3), np.uint8)
    canvas[0, 0] = (10, 20, 30)
    blob = D.blob_from_canvas(canvas, D.PREPROC_YOLOX)
    assert blob[0, :, 0, 0] == pytest.approx([10.0, 20.0, 30.0])


def test_float16_blob_matches_float32_blob():
    """Guards the cv2.convertFp16 CV_16F-as-int16 trap.

    ``cv2.convertFp16`` hands back an int16 array carrying the half bit
    pattern. Using it without ``.view(np.float16)`` produces values around
    15000 that a network consumes without error and finds nothing in.
    """
    rng = np.random.default_rng(0)
    canvas = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    blob32 = D.blob_from_canvas(canvas, D.PREPROC_YOLOV5, out_dtype=np.float32)
    blob16 = D.blob_from_canvas(canvas, D.PREPROC_YOLOV5, out_dtype=np.float16)
    assert blob16.dtype == np.float16
    assert np.max(np.abs(blob32 - blob16.astype(np.float32))) < 1e-3


def test_midas_spec_applies_imagenet_normalisation():
    canvas = np.full((8, 8, 3), 128, np.uint8)
    blob = D.blob_from_canvas(canvas, D.PREPROC_MIDAS)
    expected = (128 / 255.0 - np.asarray(D.IMAGENET_MEAN)) / np.asarray(D.IMAGENET_STD)
    assert blob[0, :, 0, 0] == pytest.approx(expected, abs=1e-5)


# --------------------------------------------------------------------------- #
# Head contract
# --------------------------------------------------------------------------- #


def test_head_contract_recognises_each_layout():
    v5 = D.analyse_detection_head((1, 25200, 85), 640)
    assert (v5.layout, v5.num_classes, v5.has_objectness, v5.grid_decode) == ("yolov5", 80, True, False)
    x = D.analyse_detection_head((1, 3549, 85), 416)
    assert (x.layout, x.grid_decode) == ("yolox", True)
    v8 = D.analyse_detection_head((1, 84, 8400), 640)
    assert (v8.layout, v8.has_objectness, v8.transposed) == ("yolov8", False, True)


def test_head_contract_rejects_a_shape_that_matches_nothing():
    with pytest.raises(PerceptionError):
        D.analyse_detection_head((1, 1234, 85), 640)


def test_head_contract_rejects_a_wrong_explicit_layout():
    with pytest.raises(PerceptionError):
        D.analyse_detection_head((1, 25200, 85), 640, layout_hint="yolox")


def test_infer_net_size_recovers_the_input_size():
    assert D.infer_net_size((1, 25200, 85))[0] == 640
    assert D.infer_net_size((1, 3549, 85))[0] == 416
    with pytest.raises(PerceptionError):
        D.infer_net_size((1, 999, 85))


def test_select_detection_output_prefers_a_known_name():
    shapes = {"output0": (1, 25200, 85), "onnx::Sigmoid_1": (1, 3, 80, 80, 85)}
    assert D.select_detection_output(shapes, 640) == "output0"


def test_select_detection_output_refuses_an_efficientnms_engine():
    """The biggest tensor of an EfficientNMS engine is ``boxes``, not a head."""
    shapes = {
        "num_dets": (1, 1),
        "boxes": (1, 100, 4),
        "scores": (1, 100),
        "classes": (1, 100),
    }
    with pytest.raises(PerceptionError):
        D.select_detection_output(shapes, 640)


# --------------------------------------------------------------------------- #
# Class selection
# --------------------------------------------------------------------------- #


def test_default_class_set_contains_pedestrians():
    assert 0 in D.DEFAULT_CLASS_IDS
    assert D.class_name(0) == "person"


def test_resolve_class_ids_named_sets_and_errors():
    assert D.resolve_class_ids("vehicles") == (1, 2, 3, 5, 7)
    assert D.resolve_class_ids([7, 2, 2]) == (2, 7)
    with pytest.raises(PerceptionError):
        D.resolve_class_ids("no_such_set")
    with pytest.raises(PerceptionError):
        D.resolve_class_ids([2, 900])
    with pytest.raises(PerceptionError):
        D.resolve_class_ids([])


# --------------------------------------------------------------------------- #
# Decode: planted boxes
# --------------------------------------------------------------------------- #


def _plant_v5(head, row, transform, xyxy, obj, class_scores):
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    ncx, ncy = transform.to_net(cx, cy)
    head[0, row, :4] = [ncx, ncy, (x2 - x1) * transform.scale, (y2 - y1) * transform.scale]
    head[0, row, 4] = obj
    for class_id, score in class_scores.items():
        head[0, row, 5 + class_id] = score


def test_decode_yolov5_round_trips_a_planted_box_to_subpixel():
    transform = D.letterbox_transform(SRC_W, SRC_H, 640, "center")
    head = np.zeros((1, 25200, 85), np.float32)
    _plant_v5(head, 1234, transform, (400.0, 300.5, 560.0, 460.25), 0.9, {2: 0.8})
    boxes = decode_yolo(head, transform.scale, transform.pad_x, transform.pad_y,
                        SRC_W, SRC_H, 0.35, 0.5, 100, net_size=640)
    assert len(boxes) == 1
    box = boxes[0]
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((400.0, 300.5, 560.0, 460.25), abs=1e-3)
    assert box.label == "car"
    assert box.confidence == pytest.approx(0.72, abs=1e-4)


def test_decode_yolov8_transposed_layout():
    transform = D.letterbox_transform(SRC_W, SRC_H, 640, "center")
    head = np.zeros((1, 8400, 84), np.float32)
    x1, y1, x2, y2 = 200.0, 400.0, 300.0, 500.0
    ncx, ncy = transform.to_net((x1 + x2) / 2, (y1 + y2) / 2)
    head[0, 77, :4] = [ncx, ncy, (x2 - x1) * transform.scale, (y2 - y1) * transform.scale]
    head[0, 77, 4 + 0] = 0.77  # person
    boxes = decode_yolo(np.transpose(head, (0, 2, 1)), transform.scale, transform.pad_x,
                        transform.pad_y, SRC_W, SRC_H, 0.35, 0.5, 100, net_size=640)
    assert len(boxes) == 1
    assert boxes[0].label == "person"
    assert (boxes[0].x1, boxes[0].y2) == pytest.approx((x1, y2), abs=1e-3)


def test_decode_yolox_grid_decode_round_trips():
    transform = D.letterbox_transform(SRC_W, SRC_H, 416, "topleft")
    grid, stride = yolox_grids(416)
    row = 2000
    gx, gy = grid[row]
    s = float(stride[row, 0])
    x1, y1, x2, y2 = 400.0, 300.0, 560.0, 460.0
    ncx, ncy = transform.to_net((x1 + x2) / 2, (y1 + y2) / 2)
    head = np.zeros((1, 3549, 85), np.float32)
    head[0, row, 0] = ncx / s - gx
    head[0, row, 1] = ncy / s - gy
    head[0, row, 2] = np.log((x2 - x1) * transform.scale / s)
    head[0, row, 3] = np.log((y2 - y1) * transform.scale / s)
    head[0, row, 4] = 0.9
    head[0, row, 5 + 7] = 0.7
    boxes = decode_yolox(head, transform, 0.35, 0.5, 100)
    assert len(boxes) == 1
    assert boxes[0].label == "truck"
    assert (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == pytest.approx(
        (x1, y1, x2, y2), abs=0.05
    )


def test_yolox_grid_order_matches_the_reference_construction():
    """Level-major, then row-major with x fastest -- YOLOX's demo_postprocess."""
    grid, stride = yolox_grids(416)
    assert grid.shape == (3549, 2) and stride.shape == (3549, 1)
    assert grid[0].tolist() == [0.0, 0.0] and stride[0, 0] == 8.0
    assert grid[1].tolist() == [1.0, 0.0]        # x varies fastest
    assert grid[52].tolist() == [0.0, 1.0]       # next row of the stride-8 level
    assert stride[52 * 52, 0] == 16.0
    assert stride[52 * 52 + 26 * 26, 0] == 32.0


def test_decode_yolox_boxes_rejects_a_bad_row_index():
    with pytest.raises(PerceptionError):
        decode_yolox_boxes(np.zeros((1, 4), np.float32), 416, row_index=np.array([99999]))


# --------------------------------------------------------------------------- #
# Class filtering and NMS
# --------------------------------------------------------------------------- #


def test_allowed_class_survives_a_higher_scoring_disallowed_class():
    """ADAS-PERC-10: argmax over all 80 classes silently drops valid vehicles."""
    transform = D.letterbox_transform(SRC_W, SRC_H, 640, "center")
    head = np.zeros((1, 25200, 85), np.float32)
    # bench (13) outranks car (2) on this row; car must still be reported.
    _plant_v5(head, 10, transform, (100.0, 400.0, 200.0, 500.0), 1.0, {13: 0.9, 2: 0.5})
    boxes = decode_yolo(head, transform.scale, transform.pad_x, transform.pad_y,
                        SRC_W, SRC_H, 0.35, 0.5, 100, class_ids="road_users", net_size=640)
    assert [b.label for b in boxes] == ["car"]
    assert boxes[0].confidence == pytest.approx(0.5, abs=1e-4)


def test_pedestrian_is_detected_by_default():
    transform = D.letterbox_transform(SRC_W, SRC_H, 640, "center")
    head = np.zeros((1, 25200, 85), np.float32)
    _plant_v5(head, 11, transform, (600.0, 350.0, 640.0, 500.0), 0.9, {0: 0.8})
    boxes = decode_yolo(head, transform.scale, transform.pad_x, transform.pad_y,
                        SRC_W, SRC_H, 0.35, 0.5, 100, net_size=640)
    assert [b.label for b in boxes] == ["person"]


def test_nms_is_class_aware():
    """A car and a truck at the same place must not suppress each other."""
    transform = D.letterbox_transform(SRC_W, SRC_H, 640, "center")
    head = np.zeros((1, 25200, 85), np.float32)
    _plant_v5(head, 20, transform, (400.0, 300.0, 600.0, 500.0), 0.95, {2: 0.9})
    _plant_v5(head, 21, transform, (402.0, 302.0, 602.0, 502.0), 0.90, {7: 0.85})
    _plant_v5(head, 22, transform, (404.0, 304.0, 604.0, 504.0), 0.80, {2: 0.75})
    boxes = decode_yolo(head, transform.scale, transform.pad_x, transform.pad_y,
                        SRC_W, SRC_H, 0.35, 0.5, 100, net_size=640)
    labels = sorted(b.label for b in boxes)
    assert labels == ["car", "truck"]  # the second car is suppressed, the truck is not


def test_class_aware_nms_keeps_distant_boxes_of_the_same_class():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [500, 500, 560, 560]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    classes = np.array([2, 2, 2], np.int32)
    keep = D.class_aware_nms(boxes, scores, classes, 0.5)
    assert sorted(keep.tolist()) == [0, 2]


def test_nms_xyxy_backwards_compatible():
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
    keep = nms_xyxy(boxes, [0.9, 0.8, 0.7], 0.5)
    assert 0 in keep and 2 in keep and 1 not in keep


def test_decode_is_bounded_by_pre_nms_topk():
    """ADAS-PERC-M5: a noisy frame must not put thousands of boxes into NMS."""
    transform = D.letterbox_transform(SRC_W, SRC_H, 640, "center")
    rng = np.random.default_rng(1)
    head = np.zeros((1, 25200, 85), np.float32)
    head[0, :, 0] = rng.uniform(0, 640, 25200)
    head[0, :, 1] = rng.uniform(0, 640, 25200)
    head[0, :, 2] = rng.uniform(5, 40, 25200)
    head[0, :, 3] = rng.uniform(5, 40, 25200)
    head[0, :, 4] = 0.99
    head[0, :, 5 + 2] = 0.99
    boxes = decode_yolo(head, transform.scale, transform.pad_x, transform.pad_y,
                        SRC_W, SRC_H, 0.05, 0.5, 25, net_size=640,
                        pre_nms_topk=100)
    assert len(boxes) <= 25


def test_decode_returns_empty_when_nothing_clears_the_threshold():
    transform = D.letterbox_transform(SRC_W, SRC_H, 640, "center")
    head = np.zeros((1, 25200, 85), np.float32)
    head[0, :, 4] = 0.05
    assert decode_yolo(head, transform.scale, transform.pad_x, transform.pad_y,
                       SRC_W, SRC_H, 0.35, 0.5, 100, net_size=640) == []


def test_boxes_are_clipped_to_the_frame_so_truncation_is_detectable():
    transform = D.letterbox_transform(SRC_W, SRC_H, 640, "center")
    head = np.zeros((1, 25200, 85), np.float32)
    _plant_v5(head, 30, transform, (-200.0, 400.0, 300.0, 1200.0), 0.9, {2: 0.9})
    boxes = decode_yolo(head, transform.scale, transform.pad_x, transform.pad_y,
                        SRC_W, SRC_H, 0.35, 0.5, 100, net_size=640)
    assert len(boxes) == 1
    assert boxes[0].x1 == pytest.approx(0.0)
    assert boxes[0].y2 == pytest.approx(SRC_H - 1)

    from adas.perception.geometry import truncation_flags

    flags = truncation_flags(boxes[0], SRC_W, SRC_H)
    assert flags.left and flags.bottom and flags.vertical


# --------------------------------------------------------------------------- #
# Detector wiring, via the CPU engine double
# --------------------------------------------------------------------------- #


def _fake_v5_engine(head):
    return FakeTrtEngine(
        {"images": (1, 3, 640, 640)}, {"output0": head}, engine_path="<fake-yolov5n>"
    )


def _fake_yolox_engine(head):
    return FakeTrtEngine(
        {"images": (1, 3, 416, 416)}, {"output": head}, engine_path="<fake-yolox>"
    )


def test_detector_end_to_end_with_a_fake_engine():
    transform = D.letterbox_transform(SRC_W, SRC_H, 640, "center")
    head = np.zeros((1, 25200, 85), np.float32)
    _plant_v5(head, 5, transform, (500.0, 320.0, 700.0, 480.0), 0.9, {2: 0.9})
    detector = YoloTensorRTDetector(engine=_fake_v5_engine(head), confidence_threshold=0.35)
    assert detector.contract.layout == "yolov5"
    assert detector.preproc is D.PREPROC_YOLOV5
    assert detector.net_size == 640
    assert detector.is_mock is False

    frame = np.zeros((SRC_H, SRC_W, 3), np.uint8)
    boxes = detector.infer(frame, SRC_W, SRC_H)
    assert len(boxes) == 1
    assert (boxes[0].x1, boxes[0].y2) == pytest.approx((500.0, 480.0), abs=1e-3)
    summary = detector.timing_summary()
    assert summary["frames"] == 1 and summary["total_ms"] >= 0.0
    detector.close()


def test_detector_picks_yolox_preprocessing_from_the_engine():
    head = np.zeros((1, 3549, 85), np.float32)
    detector = YoloTensorRTDetector(engine=_fake_yolox_engine(head))
    assert detector.contract.layout == "yolox"
    assert detector.preproc is D.PREPROC_YOLOX
    assert detector.net_size == 416
    frame = np.full((SRC_H, SRC_W, 3), 200, np.uint8)
    detector.infer(frame, SRC_W, SRC_H)
    blob = detector.engine.last_inputs["images"]
    # raw 0..255 and BGR: the interior of the letterbox is the source value.
    assert blob.max() == pytest.approx(200.0)
    detector.close()


def test_detector_rejects_a_non_square_input():
    engine = FakeTrtEngine({"images": (1, 3, 640, 480)}, {"output0": np.zeros((1, 10, 85), np.float32)})
    with pytest.raises(PerceptionError):
        YoloTensorRTDetector(engine=engine)


def test_detector_rejects_an_unrecognisable_head():
    engine = FakeTrtEngine({"images": (1, 3, 640, 640)}, {"output0": np.zeros((1, 77, 85), np.float32)})
    with pytest.raises(PerceptionError):
        YoloTensorRTDetector(engine=engine)


def test_yolox_detector_refuses_a_yolov5_engine():
    head = np.zeros((1, 25200, 85), np.float32)
    with pytest.raises(PerceptionError):
        YoloXTensorRTDetector("", engine=_fake_v5_engine(head))


def test_yolox_detector_accepts_a_yolox_engine():
    detector = YoloXTensorRTDetector("", engine=_fake_yolox_engine(np.zeros((1, 3549, 85), np.float32)))
    assert detector.contract.layout == "yolox"
    assert isinstance(detector, YoloTensorRTDetector)
    detector.close()


def test_detector_rejects_a_float_frame():
    detector = YoloTensorRTDetector(engine=_fake_v5_engine(np.zeros((1, 25200, 85), np.float32)))
    with pytest.raises(PerceptionError):
        detector.infer(np.zeros((SRC_H, SRC_W, 3), np.float32), SRC_W, SRC_H)
    detector.close()


def test_preprocess_yolox_matches_the_reference_recipe():
    """YOLOX's own preproc: int-truncated resize, top-left paste, 114 pad, BGR raw."""
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 256, (SRC_H, SRC_W, 3), dtype=np.uint8)
    blob, transform = preprocess_yolox(frame, 416)

    import cv2

    padded = np.ones((416, 416, 3), np.uint8) * 114
    ratio = min(416 / frame.shape[0], 416 / frame.shape[1])
    resized = cv2.resize(
        frame,
        (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded[: int(frame.shape[0] * ratio), : int(frame.shape[1] * ratio)] = resized
    reference = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)

    assert transform.scale == pytest.approx(ratio)
    assert np.array_equal(blob[0], reference)
