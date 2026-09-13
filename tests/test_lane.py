"""Tests for the lane backend interface, the shared fit, YOLOP and TwinLiteNet.

All CPU-only: the segmentation backends are driven with hand-built logit
volumes rather than an engine, so the decode paths are covered without a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from adas.core.exceptions import ConfigurationError, PerceptionError
from adas.core.models import DrivableArea, LaneLine
from adas.core.validation import validate_lane_model
from adas.perception import twinlite as TL
from adas.perception import yolop as YP
from adas.perception.geometry import MAX_CURVATURE_RADIUS_M, CameraConfig
from adas.perception.lane import (
    EGO_CENTRELINE_INDEX,
    FALLBACK_HALF_LANE_FRAC,
    MIN_VERTICAL_ASPECT,
    LaneBackend,
    LaneEstimator,
    MockLaneEstimator,
    fit_pixel_quadratic,
    lane_curvature_from_pixels,
    lane_geometry_from_model,
    lane_model_from_boundaries,
)

FRAME_W = 1280
FRAME_H = 720


def make_camera(pitch_deg: float = 2.0) -> CameraConfig:
    return CameraConfig(
        FRAME_W, FRAME_H, 910.0, 910.0, 640.0, 360.0, 1.30, pitch_deg, True, "test"
    )


def project_lane(cam: CameraConfig, x_m: float, z_range=(4.0, 40.0), n: int = 25):
    out = []
    for z in np.linspace(z_range[0], z_range[1], n):
        pixel = cam.ground_to_image(x_m, float(z))
        if pixel is not None:
            out.append(pixel)
    return out


# --------------------------------------------------------------------------- #
# Backend interface and the mock
# --------------------------------------------------------------------------- #


def test_lane_backend_cannot_be_instantiated():
    with pytest.raises(TypeError):
        LaneBackend()


def test_lane_estimator_alias_is_the_mock():
    assert LaneEstimator is MockLaneEstimator


def test_mock_is_flagged_and_carries_no_confidence():
    est = MockLaneEstimator()
    assert est.is_mock is True
    model = est.estimate(None, FRAME_W, FRAME_H)
    assert model is not None
    assert model.is_mock is True
    assert model.confidence == 0.0
    assert model.lines == []


def test_mock_reports_a_straight_lane_not_an_invented_220_m_radius():
    model = MockLaneEstimator().estimate(None, FRAME_W, FRAME_H)
    assert model.curvature_m == MAX_CURVATURE_RADIUS_M


def test_mock_still_centres_on_the_image_and_validates():
    model = MockLaneEstimator().estimate(None, FRAME_W, FRAME_H)
    assert model.lane_center_px == pytest.approx(FRAME_W / 2.0)
    validate_lane_model(model)  # must not raise


def test_mock_has_no_drivable_area():
    assert MockLaneEstimator().drivable_area() is None


# --------------------------------------------------------------------------- #
# Pixel fitting
# --------------------------------------------------------------------------- #


def test_fit_pixel_quadratic_recovers_planted_coefficients():
    truth = (0.0005, -0.8, 900.0)
    pts = [(truth[0] * y * y + truth[1] * y + truth[2], float(y)) for y in range(400, 720, 20)]
    fit = fit_pixel_quadratic(pts)
    assert fit is not None
    for got, want in zip(fit, truth):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-9)


def test_fit_pixel_quadratic_rejects_a_near_horizontal_polyline():
    """M2: a column-head lane spans many x at nearly one y; x = f(y) is garbage."""
    pts = [(float(x), 500.0 + 0.01 * x) for x in range(100, 900, 40)]
    assert fit_pixel_quadratic(pts) is None
    # A predominantly vertical boundary with the same point count is accepted.
    steep = [(float(x), 500.0 + 4.0 * x) for x in range(100, 900, 40)]
    assert MIN_VERTICAL_ASPECT < 4.0
    assert fit_pixel_quadratic(steep) is not None


def test_fit_pixel_quadratic_needs_three_points():
    assert fit_pixel_quadratic([(1.0, 2.0), (3.0, 4.0)]) is None
    assert fit_pixel_quadratic(None) is None


# --------------------------------------------------------------------------- #
# lane_model_from_boundaries
# --------------------------------------------------------------------------- #


def test_both_boundaries_give_a_centred_model_marked_real():
    cam = make_camera()
    left = project_lane(cam, -1.825)
    right = project_lane(cam, 1.825)
    built = lane_model_from_boundaries(
        left, right, FRAME_W, FRAME_H, camera=cam, left_confidence=0.8, right_confidence=0.7
    )
    assert built is not None
    model, geom = built
    assert model.is_mock is False
    assert model.confidence == pytest.approx(0.7)  # the weaker side dominates
    assert model.lane_center_px == pytest.approx(cam.cx, abs=1.0)
    assert geom is not None
    assert geom.lane_width_m == pytest.approx(3.65, abs=0.05)
    validate_lane_model(model)


def test_lines_carry_ego_indices_and_metric_coefficients():
    cam = make_camera()
    built = lane_model_from_boundaries(
        project_lane(cam, -1.825), project_lane(cam, 1.825), FRAME_W, FRAME_H, camera=cam
    )
    assert built is not None
    model, _ = built
    by_index = {line.index: line for line in model.lines}
    assert set(by_index) == {1, 2, EGO_CENTRELINE_INDEX}
    assert by_index[1].coeffs is not None
    assert by_index[1].coeffs[2] == pytest.approx(-1.825, abs=0.02)
    assert by_index[2].coeffs[2] == pytest.approx(1.825, abs=0.02)
    assert by_index[EGO_CENTRELINE_INDEX].coeffs[2] == pytest.approx(0.0, abs=0.02)
    assert by_index[1].points_px


def test_without_a_camera_there_is_no_metric_fit_and_no_invented_curvature():
    cam = make_camera()
    built = lane_model_from_boundaries(
        project_lane(cam, -1.825), project_lane(cam, 1.825), FRAME_W, FRAME_H, camera=None
    )
    assert built is not None
    model, geom = built
    assert geom is None
    assert model.curvature_m == MAX_CURVATURE_RADIUS_M
    assert all(line.coeffs is None for line in model.lines)


def test_curvature_comes_from_the_ground_fit():
    cam = make_camera()
    left = []
    right = []
    for z in np.linspace(4.0, 40.0, 30):
        bend = 0.002 * z * z
        pl = cam.ground_to_image(float(bend - 1.825), float(z))
        pr = cam.ground_to_image(float(bend + 1.825), float(z))
        if pl and pr:
            left.append(pl)
            right.append(pr)
    built = lane_model_from_boundaries(left, right, FRAME_W, FRAME_H, camera=cam)
    assert built is not None
    model, _ = built
    # R = 1 / |2a| = 250 m for a = 0.002 (plus a small slope term at the eval range).
    assert 150.0 < model.curvature_m < 400.0


def test_single_boundary_synthesises_the_other_and_halves_confidence():
    cam = make_camera()
    built = lane_model_from_boundaries(
        project_lane(cam, -1.825), None, FRAME_W, FRAME_H, camera=cam, left_confidence=0.8
    )
    assert built is not None
    model, geom = built
    assert model.confidence == pytest.approx(0.4)
    assert geom is not None
    assert geom.both_boundaries is False
    assert geom.lane_width_m == 0.0
    # The synthesised pixel boundary is a constant column.
    assert model.right_coeffs[0] == 0.0 and model.right_coeffs[1] == 0.0


def test_single_boundary_pixel_fallback_uses_the_documented_fraction():
    pts = [(400.0, float(y)) for y in range(400, 720, 20)]
    built = lane_model_from_boundaries(pts, None, FRAME_W, FRAME_H, camera=None)
    assert built is not None
    model, _ = built
    assert model.lane_center_px == pytest.approx(400.0 + FRAME_W * FALLBACK_HALF_LANE_FRAC)


def test_no_fittable_boundary_returns_none():
    assert lane_model_from_boundaries(None, None, FRAME_W, FRAME_H) is None
    assert lane_model_from_boundaries([(1.0, 2.0)], None, FRAME_W, FRAME_H) is None


def test_implausible_lane_width_caps_confidence():
    cam = make_camera()
    built = lane_model_from_boundaries(
        project_lane(cam, -5.0),
        project_lane(cam, 5.0),
        FRAME_W,
        FRAME_H,
        camera=cam,
        left_confidence=0.9,
        right_confidence=0.9,
    )
    assert built is not None
    model, geom = built
    assert geom is not None and geom.plausible is False
    assert model.confidence <= 0.1


def test_extra_lines_are_appended_untouched():
    cam = make_camera()
    extra = [LaneLine(points_px=[(1.0, 2.0)], coeffs=None, confidence=0.3, index=0)]
    built = lane_model_from_boundaries(
        project_lane(cam, -1.825),
        project_lane(cam, 1.825),
        FRAME_W,
        FRAME_H,
        camera=cam,
        extra_lines=extra,
    )
    assert built is not None
    model, _ = built
    assert any(line.index == 0 for line in model.lines)


# --------------------------------------------------------------------------- #
# lane_geometry_from_model
# --------------------------------------------------------------------------- #


def test_geometry_from_model_round_trips_the_metric_fit():
    cam = make_camera()
    built = lane_model_from_boundaries(
        project_lane(cam, -2.3), project_lane(cam, 1.3), FRAME_W, FRAME_H, camera=cam
    )
    assert built is not None
    model, direct = built
    recovered = lane_geometry_from_model(model, eval_range_m=direct.eval_range_m)
    assert recovered is not None
    assert recovered.lane_width_m == pytest.approx(direct.lane_width_m, abs=1e-6)
    assert recovered.lateral_offset_m == pytest.approx(direct.lateral_offset_m, abs=1e-6)
    assert recovered.lateral_offset_m == pytest.approx(0.5, abs=0.05)


def test_geometry_from_a_mock_model_is_none():
    assert lane_geometry_from_model(MockLaneEstimator().estimate(None, FRAME_W, FRAME_H)) is None


def test_geometry_from_none_is_none():
    assert lane_geometry_from_model(None) is None


def test_geometry_from_a_pixel_only_model_is_none():
    cam = make_camera()
    built = lane_model_from_boundaries(
        project_lane(cam, -1.825), project_lane(cam, 1.825), FRAME_W, FRAME_H, camera=None
    )
    assert built is not None
    assert lane_geometry_from_model(built[0]) is None


def test_lane_curvature_from_pixels_falls_back_to_straight():
    cam = make_camera()
    assert lane_curvature_from_pixels([(1.0, 2.0)], cam) == MAX_CURVATURE_RADIUS_M


# --------------------------------------------------------------------------- #
# YOLOP: preprocessing
# --------------------------------------------------------------------------- #


def test_letterbox_is_centred_and_keeps_dtype():
    frame = np.full((720, 1280, 3), 200, dtype=np.uint8)
    canvas, scale, pad_x, pad_y = YP.letterbox_centred(frame, 640)
    assert canvas.shape == (640, 640, 3)
    assert canvas.dtype == np.uint8
    assert scale == pytest.approx(0.5)
    assert pad_x == 0
    assert pad_y == 140  # (640 - 360) // 2
    assert canvas[0, 0, 0] == YP.LETTERBOX_PAD_VALUE
    assert canvas[320, 320, 0] == 200


def test_letterbox_mapping_round_trips_a_known_corner():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    _canvas, scale, pad_x, pad_y = YP.letterbox_centred(frame, 640)
    for x, y in ((0.0, 0.0), (1279.0, 719.0), (640.0, 360.0)):
        nx = x * scale + pad_x
        ny = y * scale + pad_y
        assert (nx - pad_x) / scale == pytest.approx(x)
        assert (ny - pad_y) / scale == pytest.approx(y)


def test_yolop_preprocess_matches_the_naive_transform():
    import cv2

    frame = np.random.default_rng(3).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    blob, scale, pad_x, pad_y = YP.preprocess(frame, 640)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    canvas, _s, _px, _py = YP.letterbox_centred(rgb, 640)
    naive = canvas.astype(np.float32) / 255.0
    naive = (naive - YP.IMAGENET_MEAN) / YP.IMAGENET_STD
    naive = np.ascontiguousarray(naive.transpose(2, 0, 1)[None, ...])
    assert blob.shape == (1, 3, 640, 640)
    assert np.allclose(blob, naive, atol=1e-5)
    assert (scale, pad_x, pad_y) == (pytest.approx(0.5), 0, 140)


def test_yolop_preprocess_rejects_float_frames():
    with pytest.raises(PerceptionError):
        YP.preprocess(np.zeros((720, 1280, 3), dtype=np.float32), 640)


# --------------------------------------------------------------------------- #
# YOLOP: segmentation decode
# --------------------------------------------------------------------------- #


def test_class1_margin_sign_is_the_argmax():
    logits = np.zeros((1, 2, 4, 4), dtype=np.float32)
    logits[0, 1, 1, 1] = 3.0
    margin = YP.class1_margin(logits)
    assert margin.shape == (4, 4)
    assert margin[1, 1] == pytest.approx(3.0)
    assert margin[0, 0] == pytest.approx(0.0)


def test_class1_margin_rejects_a_non_binary_head():
    with pytest.raises(PerceptionError):
        YP.class1_margin(np.zeros((1, 3, 4, 4), dtype=np.float32))


def test_crop_letterbox_removes_exactly_the_padding():
    plane = np.zeros((640, 640), dtype=np.float32)
    plane[140:500, :] = 1.0
    cropped = YP.crop_letterbox(plane, 0.5, 0, 140, 1280, 720)
    assert cropped.shape == (360, 640)
    assert float(cropped.min()) == 1.0


def test_drivable_area_grid_marks_the_painted_region_free():
    margin = np.full((360, 640), -5.0, dtype=np.float32)
    margin[180:, :] = 5.0  # bottom half drivable
    area = YP.drivable_area_from_margin(margin)
    assert isinstance(area, DrivableArea)
    assert (area.width, area.height) == (YP.DRIVABLE_GRID_W, YP.DRIVABLE_GRID_H)
    assert area.confidence > 0.9
    assert area.is_free(0.5, 0.9) is True
    assert area.is_free(0.5, 0.1) is False


def test_drivable_area_with_nothing_free_is_confidence_zero():
    margin = np.full((360, 640), -5.0, dtype=np.float32)
    area = YP.drivable_area_from_margin(margin)
    assert area.confidence == 0.0
    assert not np.any(area.mask)


def test_drivable_area_from_probability_agrees_with_the_margin_form():
    margin = np.full((180, 320), -4.0, dtype=np.float32)
    margin[90:, :] = 4.0
    prob = 1.0 / (1.0 + np.exp(-margin))
    a = YP.drivable_area_from_margin(margin)
    b = YP.drivable_area_from_probability(prob)
    assert np.array_equal(a.mask, b.mask)
    assert a.confidence == pytest.approx(b.confidence, abs=1e-4)


def paint_lane_mask(height=360, width=640, left_bottom=200, right_bottom=440, apex=320, top=120):
    """Two straight lines converging on ``apex`` at row ``top``."""
    mask = np.zeros((height, width), dtype=bool)
    for y in range(top, height):
        t = (y - top) / float(height - 1 - top)
        lx = int(round(apex + t * (left_bottom - apex)))
        rx = int(round(apex + t * (right_bottom - apex)))
        mask[y, max(0, lx - 1) : lx + 2] = True
        mask[y, max(0, rx - 1) : rx + 2] = True
    return mask


def test_lane_points_from_mask_recovers_two_painted_lines():
    mask = paint_lane_mask()
    left, right = YP.lane_points_from_mask(mask, ego_x_px=320.0)
    assert len(left) >= 8 and len(right) >= 8
    for (lx, ly), (rx, ry) in zip(left, right):
        assert lx < 320.0 < rx
    # Bottom-most accepted point should sit on the painted bottom position.
    lowest_left = max(left, key=lambda p: p[1])
    assert lowest_left[0] == pytest.approx(200.0, abs=12.0)


def test_lane_points_track_a_line_that_slants():
    """Without slope momentum the tracker loses a slanting line and stops early."""
    mask = paint_lane_mask(left_bottom=60, right_bottom=580, apex=320, top=100)
    left, right = YP.lane_points_from_mask(mask, ego_x_px=320.0)
    assert len(left) >= 10
    assert len(right) >= 10
    assert min(p[1] for p in left) < 200.0  # followed it well up the frame


def test_lane_points_from_an_empty_mask_are_empty():
    left, right = YP.lane_points_from_mask(np.zeros((360, 640), dtype=bool))
    assert left == [] and right == []


def test_lane_points_reject_a_side_with_too_few_points():
    mask = np.zeros((360, 640), dtype=bool)
    mask[350:355, 200:210] = True  # a single blob, one band only
    left, right = YP.lane_points_from_mask(mask, ego_x_px=320.0, min_points=4)
    assert left == []
    assert right == []


def test_lane_points_require_a_two_dimensional_mask():
    with pytest.raises(PerceptionError):
        YP.lane_points_from_mask(np.zeros((2, 360, 640), dtype=bool))


def test_run_centroids_are_intensity_weighted():
    counts = np.zeros(20, dtype=np.int64)
    counts[5:9] = [1, 3, 3, 1]
    lit = counts >= 1
    centroids = YP._run_centroids(lit, counts, 2)
    assert len(centroids) == 1
    assert centroids[0] == pytest.approx(6.5)


def test_run_centroids_drop_runs_shorter_than_the_minimum():
    counts = np.zeros(20, dtype=np.int64)
    counts[5] = 4
    assert YP._run_centroids(counts >= 1, counts, 3) == []


def test_sample_margin_confidence_is_zero_for_no_points():
    assert YP.sample_margin_confidence(np.zeros((10, 10), dtype=np.float32), []) == 0.0


def test_sample_margin_confidence_is_a_probability():
    margin = np.full((10, 10), 4.0, dtype=np.float32)
    value = YP.sample_margin_confidence(margin, [(1.0, 1.0), (2.0, 2.0)])
    assert 0.9 < value <= 1.0


# --------------------------------------------------------------------------- #
# YOLOP: detection decode
# --------------------------------------------------------------------------- #


def test_decode_detections_recovers_a_planted_box():
    det = np.zeros((1, 100, 6), dtype=np.float32)
    # Source box (400, 300)-(500, 420) -> network box at scale 0.5, pad (0, 140).
    cx = (450.0 * 0.5) + 0.0
    cy = (360.0 * 0.5) + 140.0
    det[0, 7] = [cx, cy, 100.0 * 0.5, 120.0 * 0.5, 0.9, 0.9]
    boxes = YP.decode_detections(det, 0.5, 0, 140, 1280, 720, conf_threshold=0.3)
    assert len(boxes) == 1
    box = boxes[0]
    assert box.x1 == pytest.approx(400.0, abs=0.5)
    assert box.y1 == pytest.approx(300.0, abs=0.5)
    assert box.x2 == pytest.approx(500.0, abs=0.5)
    assert box.y2 == pytest.approx(420.0, abs=0.5)
    assert box.confidence == pytest.approx(0.81, abs=1e-4)
    assert box.label == "car"


def test_decode_detections_applies_the_confidence_gate():
    det = np.zeros((1, 10, 6), dtype=np.float32)
    det[0, 0] = [320.0, 320.0, 40.0, 40.0, 0.5, 0.5]  # score 0.25
    assert YP.decode_detections(det, 0.5, 0, 140, 1280, 720, conf_threshold=0.3) == []
    assert len(YP.decode_detections(det, 0.5, 0, 140, 1280, 720, conf_threshold=0.2)) == 1


def test_decode_detections_rejects_a_wrong_layout():
    with pytest.raises(PerceptionError):
        YP.decode_detections(np.zeros((1, 10, 85), dtype=np.float32), 0.5, 0, 140, 1280, 720)


def test_nms_suppresses_overlapping_boxes():
    boxes = np.array(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = YP.nms_single_class(boxes, scores, 0.5)
    assert keep == [0, 2]


def test_nms_respects_max_detections():
    boxes = np.array([[i * 100.0, 0.0, i * 100.0 + 10.0, 10.0] for i in range(10)])
    scores = np.linspace(0.9, 0.1, 10)
    assert len(YP.nms_single_class(boxes, scores, 0.5, max_detections=3)) == 3


def test_nms_on_no_boxes_is_empty():
    assert YP.nms_single_class(np.zeros((0, 4)), np.zeros((0,)), 0.5) == []


# --------------------------------------------------------------------------- #
# YOLOP: engine-free end-to-end decode
# --------------------------------------------------------------------------- #


class _FakeYolop(YP.YolopLaneEstimator):
    """YolopLaneEstimator with the TensorRT constructor bypassed."""

    def __init__(self, camera=None):  # noqa: D107 - test double
        self.engine = None
        self.input_size = 640
        self.camera = camera
        self.conf_threshold = 0.35
        self.iou_threshold = 0.45
        self.max_detections = 100
        self.min_confidence = 0.0
        self._drivable = None
        self._detections = []
        self.last_geometry = None
        self.last_lane_mask = None
        self.last_lane_mask_scale = 1.0


def build_yolop_outputs():
    lane = np.full((1, 2, 640, 640), 0.0, dtype=np.float32)
    lane[0, 0] = 5.0
    mask = paint_lane_mask(height=360, width=640, left_bottom=200, right_bottom=440)
    lane[0, 0, 140:500, :][mask] = -5.0
    lane[0, 1, 140:500, :][mask] = 5.0

    drive = np.full((1, 2, 640, 640), 0.0, dtype=np.float32)
    drive[0, 0] = 5.0
    drive[0, 0, 320:500, 100:540] = -5.0
    drive[0, 1, 320:500, 100:540] = 5.0

    det = np.zeros((1, 50, 6), dtype=np.float32)
    det[0, 3] = [320.0, 320.0, 60.0, 60.0, 0.9, 0.9]
    return {"lane_line_seg": lane, "drive_area_seg": drive, "det_out": det}


def test_yolop_decode_produces_a_real_lane_model():
    est = _FakeYolop(camera=make_camera())
    model = est.decode(build_yolop_outputs(), 0.5, 0, 140, FRAME_W, FRAME_H)
    assert model is not None
    assert model.is_mock is False
    assert 0.0 < model.confidence <= 1.0
    # Painted lines at network x 200/440 -> source 400/880, centre 640.
    assert model.lane_center_px == pytest.approx(640.0, abs=60.0)
    validate_lane_model(model)


def test_yolop_decode_publishes_free_space_and_detections():
    est = _FakeYolop(camera=make_camera())
    est.decode(build_yolop_outputs(), 0.5, 0, 140, FRAME_W, FRAME_H)
    area = est.drivable_area()
    assert area is not None
    assert area.confidence > 0.9
    assert np.any(area.mask)
    assert len(est.detections()) == 1


def test_yolop_decode_with_no_lane_pixels_returns_none():
    est = _FakeYolop(camera=make_camera())
    outputs = build_yolop_outputs()
    outputs["lane_line_seg"][:] = 0.0
    outputs["lane_line_seg"][0, 0] = 5.0
    assert est.decode(outputs, 0.5, 0, 140, FRAME_W, FRAME_H) is None
    assert est.last_geometry is None


def test_yolop_lane_mask_is_network_resolution_and_says_so():
    est = _FakeYolop(camera=make_camera())
    est.decode(build_yolop_outputs(), 0.5, 0, 140, FRAME_W, FRAME_H)
    assert est.last_lane_mask.shape == (360, 640)
    assert est.last_lane_mask_scale == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# TwinLiteNet
# --------------------------------------------------------------------------- #


def test_twinlite_preprocess_shape_and_stretch():
    frame = np.random.default_rng(4).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    blob = TL.preprocess(frame)
    assert blob.shape == (1, 3, 360, 640)
    assert blob.dtype == np.float32
    assert 0.0 <= float(blob.min()) and float(blob.max()) <= 1.0


def test_twinlite_preprocess_rejects_float_frames():
    with pytest.raises(PerceptionError):
        TL.preprocess(np.zeros((720, 1280, 3), dtype=np.float32))


def test_twinlite_head_classification():
    assert TL._classify_heads(["da_seg_out", "ll_seg_out"]) == ("da_seg_out", "ll_seg_out")


def test_twinlite_ambiguous_heads_raise_rather_than_guess():
    with pytest.raises(ConfigurationError):
        TL._classify_heads(["output0", "output1"])
    with pytest.raises(ConfigurationError):
        TL._classify_heads(["da_seg", "da_seg2"])


def test_twinlite_missing_engine_raises_by_default():
    with pytest.raises(PerceptionError):
        TL.build_twinlite_backend("models/definitely-not-here.engine")


def test_twinlite_unavailable_stub_is_honest():
    backend = TL.build_twinlite_backend(
        "models/definitely-not-here.engine", allow_unavailable=True
    )
    assert isinstance(backend, TL.TwinLiteNetUnavailable)
    assert backend.is_mock is True
    assert backend.estimate(None, FRAME_W, FRAME_H) is None
    area = backend.drivable_area()
    assert area is not None
    assert area.mask is None
    assert area.confidence == 0.0


def test_twinlite_estimator_refuses_to_construct_without_an_engine():
    with pytest.raises(PerceptionError):
        TL.TwinLiteNetLaneEstimator("models/definitely-not-here.engine")


class _FakeTwinLite(TL.TwinLiteNetLaneEstimator):
    """TwinLiteNetLaneEstimator with the TensorRT constructor bypassed."""

    def __init__(self, camera=None):  # noqa: D107 - test double
        self.engine = None
        self.input_height = 360
        self.input_width = 640
        self.drivable_name = "da_seg_out"
        self.lane_name = "ll_seg_out"
        self.camera = camera
        self.min_confidence = 0.0
        self._drivable = None
        self.last_geometry = None
        self.last_lane_mask = None


def test_twinlite_decode_matches_the_shared_lane_semantics():
    lane = np.zeros((1, 2, 360, 640), dtype=np.float32)
    lane[0, 0] = 5.0
    mask = paint_lane_mask()
    lane[0, 0][mask] = -5.0
    lane[0, 1][mask] = 5.0
    drive = np.zeros((1, 2, 360, 640), dtype=np.float32)
    drive[0, 0] = 5.0
    drive[0, 0, 180:, :] = -5.0
    drive[0, 1, 180:, :] = 5.0

    est = _FakeTwinLite(camera=make_camera())
    model = est.decode(
        {"da_seg_out": drive, "ll_seg_out": lane}, FRAME_W, FRAME_H
    )
    assert model is not None
    assert model.is_mock is False
    assert model.lane_center_px == pytest.approx(640.0, abs=60.0)
    assert est.drivable_area().confidence > 0.9
    validate_lane_model(model)
