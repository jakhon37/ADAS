"""Tests for the Ultra-Fast-Lane-Detection-v2 decoder.

Everything here is CPU-only. The synthetic tests plant a known peak in a
hand-built logit tensor and assert the decoder recovers it; the fixture tests
replay tensors captured from a REAL inference of
``models/ufldv2_culane_res18.engine`` on frame 150 of
``Ultra-Fast-Lane-Detection-v2/example.mp4`` (saved as float16 to keep the
fixture at 166 kB), so the decode path is exercised against the actual model
output without needing a GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adas.core.exceptions import PerceptionError
from adas.core.models import LaneModel
from adas.perception.geometry import CameraConfig, estimate_vanishing_point
from adas.perception.lane import lane_geometry_from_model
from adas.perception.ufld import (
    CULANE,
    CURVELANES,
    TUSIMPLE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    coords_to_lane_model,
    pred_to_coords,
    preprocess,
    resolve_output_tensors,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ufld_culane_example_frame150.npz"

FRAME_W = 1280
FRAME_H = 720


# --------------------------------------------------------------------------- #
# Dataset specs
# --------------------------------------------------------------------------- #


def test_culane_spec_matches_the_upstream_config():
    """configs/culane_res18.py of the vendored reference."""
    assert (CULANE.train_width, CULANE.train_height) == (1600, 320)
    assert CULANE.crop_ratio == 0.6
    assert (CULANE.num_row, CULANE.num_col) == (72, 81)
    assert (CULANE.num_cell_row, CULANE.num_cell_col) == (200, 100)
    assert CULANE.resize_height == 533  # int(320 / 0.6), truncated like torchvision
    assert CULANE.row_lane_idx == (1, 2)
    assert CULANE.col_lane_idx == (0, 3)


def test_culane_row_anchor_spans_the_bottom_58_percent():
    anchors = CULANE.row_anchor
    assert anchors.shape == (72,)
    assert anchors[0] == pytest.approx(0.42)
    assert anchors[-1] == pytest.approx(1.0)


def test_tusimple_anchors_are_not_the_culane_anchors():
    """utils/common.py: Tusimple uses linspace(160, 710, n) / 720."""
    assert TUSIMPLE.row_anchor[0] == pytest.approx(160.0 / 720.0)
    assert TUSIMPLE.row_anchor[-1] == pytest.approx(710.0 / 720.0)
    assert CURVELANES.row_anchor[0] == pytest.approx(0.4)


def test_expected_shapes_match_the_built_engine():
    shapes = CULANE.expected_shapes()
    assert shapes["loc_row"] == (200, 72, 4)
    assert shapes["loc_col"] == (100, 81, 4)
    assert shapes["exist_row"] == (2, 72, 4)
    assert shapes["exist_col"] == (2, 81, 4)


# --------------------------------------------------------------------------- #
# Synthetic decode
# --------------------------------------------------------------------------- #


def build_synthetic(
    row_peaks=None,
    col_peaks=None,
    present_rows=None,
    present_cols=None,
    peak_logit: float = 12.0,
    noise: float = 0.0,
    seed: int = 0,
):
    """Hand-built UFLD outputs with a planted argmax cell per anchor.

    ``row_peaks[lane]`` is the cell index planted at every row anchor of that
    lane. All four tensors come back batched, exactly as the engine emits them.
    """
    rng = np.random.default_rng(seed)
    loc_row = rng.normal(0.0, noise, (1, 200, 72, 4)).astype(np.float32)
    loc_col = rng.normal(0.0, noise, (1, 100, 81, 4)).astype(np.float32)
    exist_row = np.zeros((1, 2, 72, 4), dtype=np.float32)
    exist_col = np.zeros((1, 2, 81, 4), dtype=np.float32)
    exist_row[0, 0] = 5.0  # default: absent
    exist_col[0, 0] = 5.0

    for lane, cell in (row_peaks or {}).items():
        loc_row[0, cell, :, lane] = peak_logit
        rows = range(72) if present_rows is None else present_rows
        for k in rows:
            exist_row[0, 0, k, lane] = -5.0
            exist_row[0, 1, k, lane] = 5.0
    for lane, cell in (col_peaks or {}).items():
        loc_col[0, cell, :, lane] = peak_logit
        cols = range(81) if present_cols is None else present_cols
        for k in cols:
            exist_col[0, 0, k, lane] = -5.0
            exist_col[0, 1, k, lane] = 5.0
    return {
        "loc_row": loc_row,
        "loc_col": loc_col,
        "exist_row": exist_row,
        "exist_col": exist_col,
    }


def decode(outputs, **kwargs):
    return pred_to_coords(
        outputs["loc_row"],
        outputs["exist_row"],
        outputs["loc_col"],
        outputs["exist_col"],
        spec=CULANE,
        original_width=FRAME_W,
        original_height=FRAME_H,
        **kwargs,
    )


def test_decoder_recovers_a_planted_row_peak():
    out = build_synthetic(row_peaks={1: 50, 2: 150})
    lanes = decode(out)
    assert set(lanes) == {1, 2}
    # cell 50 of 200 -> x = (50 + 0.5) / 199 * 1280
    expected_left = (50 + 0.5) / 199.0 * FRAME_W
    expected_right = (150 + 0.5) / 199.0 * FRAME_W
    assert all(abs(x - expected_left) < 0.5 for x, _y in lanes[1].points)
    assert all(abs(x - expected_right) < 0.5 for x, _y in lanes[2].points)


def test_wide_local_window_biases_the_result_toward_the_image_centre():
    """This is the ADAS-PERC-03 regression.

    Softmaxing all 200 cells instead of the reference's +/-1 window drags every
    recovered x toward the grid mean, i.e. toward the horizontal image centre.
    For a lane-centering controller that is a silent under-steer, so the
    difference must be large enough for a test to see.
    """
    out = build_synthetic(row_peaks={1: 20}, peak_logit=3.0, noise=1.0, seed=7)
    correct = decode(out, local_width=1)[1].points[0][0]
    broken = decode(out, local_width=200)[1].points[0][0]
    centre = FRAME_W / 2.0
    assert correct == pytest.approx((20 + 0.5) / 199.0 * FRAME_W, abs=20.0)
    assert abs(broken - centre) < abs(correct - centre)
    # The whole-grid softmax pulls the point most of the way to the image centre.
    assert broken - correct > 0.25 * FRAME_W


def test_sub_cell_interpolation_lands_between_neighbouring_cells():
    """Two equal neighbouring logits must give the midpoint, not either cell."""
    out = build_synthetic(row_peaks={1: 50})
    out["loc_row"][0, 51, :, 1] = out["loc_row"][0, 50, :, 1]
    x = decode(out)[1].points[0][0]
    expected = (50.5 + 0.5) / 199.0 * FRAME_W
    assert x == pytest.approx(expected, abs=0.5)


def test_row_anchor_maps_to_original_frame_rows_not_crop_rows():
    """ADAS-PERC-01 / M1: UFLD coordinates are fractions of the SOURCE frame.

    The reference draws pred2coords output straight onto the un-cropped image;
    row_anchor spans 0.42..1.0 of the original height, which is exactly why the
    crop keeps the bottom 60%. Mapping through the 320-row crop instead pins
    every point into the bottom fifth of the frame.
    """
    lanes = decode(build_synthetic(row_peaks={1: 100}))
    ys = [y for _x, y in lanes[1].points]
    assert min(ys) == pytest.approx(0.42 * FRAME_H, abs=0.5)
    assert max(ys) == pytest.approx(1.0 * FRAME_H, abs=0.5)


def test_column_head_maps_x_from_the_anchor_and_y_from_the_grid():
    lanes = decode(build_synthetic(col_peaks={0: 30}))
    assert 0 in lanes
    assert lanes[0].head == "col"
    xs = sorted(x for x, _y in lanes[0].points)
    assert xs[0] == pytest.approx(0.0, abs=0.5)
    assert xs[-1] == pytest.approx(float(FRAME_W), abs=0.5)
    expected_y = (30 + 0.5) / 99.0 * FRAME_H
    assert all(abs(y - expected_y) < 0.5 for _x, y in lanes[0].points)


def test_existence_gate_drops_a_row_lane_below_half_the_anchors():
    """demo.py: a row lane needs valid_row.sum() > num_cls_row / 2."""
    out = build_synthetic(row_peaks={1: 50}, present_rows=range(30))
    assert 1 not in decode(out)
    out = build_synthetic(row_peaks={1: 50}, present_rows=range(40))
    assert 1 in decode(out)


def test_existence_gate_drops_a_column_lane_below_a_quarter_of_the_anchors():
    """demo.py: a column lane needs valid_col.sum() > num_cls_col / 4."""
    out = build_synthetic(col_peaks={0: 30}, present_cols=range(15))
    assert 0 not in decode(out)
    out = build_synthetic(col_peaks={0: 30}, present_cols=range(30))
    assert 0 in decode(out)


def test_only_present_anchors_produce_points():
    out = build_synthetic(row_peaks={1: 50}, present_rows=range(60))
    assert len(decode(out)[1].points) == 60


def test_confidence_scales_with_anchor_coverage():
    full = decode(build_synthetic(row_peaks={1: 50}))[1].confidence
    partial = decode(build_synthetic(row_peaks={1: 50}, present_rows=range(50)))[1].confidence
    assert 0.0 < partial < full <= 1.0


def test_points_are_floats_not_rounded_to_integers():
    out = build_synthetic(row_peaks={1: 50})
    out["loc_row"][0, 51, :, 1] = out["loc_row"][0, 50, :, 1] - 0.3
    x = decode(out)[1].points[0][0]
    assert isinstance(x, float)
    assert abs(x - round(x)) > 1e-6


def test_no_detected_lane_is_an_empty_dict_not_an_error():
    assert decode(build_synthetic()) == {}


def test_batch_greater_than_one_is_rejected():
    out = build_synthetic(row_peaks={1: 50})
    out["loc_row"] = np.concatenate([out["loc_row"], out["loc_row"]], axis=0)
    with pytest.raises(PerceptionError):
        decode(out)


def test_existence_head_with_wrong_channel_count_is_rejected():
    out = build_synthetic(row_peaks={1: 50})
    out["exist_row"] = np.zeros((1, 3, 72, 4), dtype=np.float32)
    with pytest.raises(PerceptionError):
        decode(out)


def test_anchor_length_mismatch_raises_instead_of_indexing_out_of_bounds():
    """M3: a Tusimple engine behind a CULane config must fail loudly."""
    out = build_synthetic(row_peaks={1: 50})
    with pytest.raises(PerceptionError):
        pred_to_coords(
            out["loc_row"],
            out["exist_row"],
            out["loc_col"],
            out["exist_col"],
            spec=TUSIMPLE,
            original_width=FRAME_W,
            original_height=FRAME_H,
        )


# --------------------------------------------------------------------------- #
# Output binding resolution
# --------------------------------------------------------------------------- #


def make_binding_dict():
    shapes = CULANE.expected_shapes()
    return {name: np.zeros((1,) + shape, dtype=np.float32) for name, shape in shapes.items()}


def test_resolve_by_name():
    resolved = resolve_output_tensors(make_binding_dict(), CULANE)
    assert set(resolved) == {"loc_row", "loc_col", "exist_row", "exist_col"}
    assert resolved["exist_row"].shape == (1, 2, 72, 4)


def test_resolve_by_shape_when_names_are_generic():
    """ADAS-PERC-15: the old positional fallback bound exist_row to loc_col.

    Generic names, deliberately ordered so a positional unpack of
    ``loc_row, exist_row, loc_col, exist_col`` would be wrong.
    """
    shapes = CULANE.expected_shapes()
    generic = {
        "443": np.zeros((1,) + shapes["loc_row"], dtype=np.float32),
        "446": np.zeros((1,) + shapes["loc_col"], dtype=np.float32),
        "449": np.zeros((1,) + shapes["exist_row"], dtype=np.float32),
        "452": np.zeros((1,) + shapes["exist_col"], dtype=np.float32),
    }
    resolved = resolve_output_tensors(generic, CULANE)
    assert resolved["loc_row"].shape[1:] == shapes["loc_row"]
    assert resolved["exist_row"].shape[1:] == shapes["exist_row"]
    assert resolved["loc_col"].shape[1:] == shapes["loc_col"]
    assert resolved["exist_col"].shape[1:] == shapes["exist_col"]


def test_resolve_prefers_shape_over_a_misleading_name():
    bindings = make_binding_dict()
    bindings["loc_row"] = np.zeros((1, 2, 72, 4), dtype=np.float32)  # wrong shape for the name
    with pytest.raises(PerceptionError):
        resolve_output_tensors(bindings, CULANE)


def test_resolve_reports_every_observed_shape_when_it_fails():
    with pytest.raises(PerceptionError) as excinfo:
        resolve_output_tensors({"x": np.zeros((1, 5, 5), dtype=np.float32)}, CULANE)
    message = str(excinfo.value)
    assert "loc_row" in message and "[1, 5, 5]" in message


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def test_preprocess_shape_and_range():
    frame = np.random.default_rng(0).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    blob = preprocess(frame, CULANE)
    assert blob.shape == (1, 3, 320, 1600)
    assert blob.dtype == np.float32
    assert blob.flags["C_CONTIGUOUS"]


def test_preprocess_matches_the_reference_transform_exactly():
    """The fused path must equal BGR->RGB, stretch, /255, ImageNet, bottom crop."""
    import cv2

    frame = np.random.default_rng(1).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    stretched = cv2.resize(rgb, (1600, 533), interpolation=cv2.INTER_LINEAR)
    naive = stretched.astype(np.float32) / 255.0
    naive = (naive - IMAGENET_MEAN) / IMAGENET_STD
    naive = np.ascontiguousarray(naive[-320:, :, :].transpose(2, 0, 1)[None, ...])
    assert np.allclose(preprocess(frame, CULANE), naive, atol=1e-5)


def test_preprocess_is_a_stretch_not_an_aspect_preserving_resize():
    """ADAS-PERC-01: the whole frame goes to 1600x533; aspect is NOT preserved.

    A frame with a single lit row near the top must still appear in the blob at
    the row the stretch puts it at, not be cropped away by an aspect-preserving
    resize that keeps only the bottom 36% of the source.
    """
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[400, :, :] = 255  # 55.6% down the source frame
    blob = preprocess(frame, CULANE)
    lit_rows = np.nonzero(blob[0, 0].max(axis=1) > blob[0, 0].min() + 0.1)[0]
    assert lit_rows.size > 0
    # source row 400 -> stretched row 400 * 533/720 = 296 -> crop row 296 - 213 = 83
    assert abs(int(lit_rows.mean()) - 83) <= 1


def test_preprocess_rejects_float_frames():
    frame = np.zeros((720, 1280, 3), dtype=np.float32)
    with pytest.raises(PerceptionError):
        preprocess(frame, CULANE)


def test_preprocess_rejects_grayscale():
    frame = np.zeros((720, 1280), dtype=np.uint8)
    with pytest.raises(PerceptionError):
        preprocess(frame, CULANE)


def test_preprocess_rejects_non_arrays():
    with pytest.raises(PerceptionError):
        preprocess({"width": 1280}, CULANE)


# --------------------------------------------------------------------------- #
# Lane model assembly
# --------------------------------------------------------------------------- #


def test_ego_boundaries_come_from_the_lane_index_not_from_geometry():
    """ADAS-PERC-21 / M2: lanes 1 and 2 are the ego boundaries, by definition.

    Here the adjacent column lane 0 is planted to the RIGHT of the ego-left row
    lane. A geometric "nearest to the image centre" rule would pick it; the
    index rule must not.
    """
    out = build_synthetic(row_peaks={1: 40, 2: 150}, col_peaks={0: 30})
    lanes = decode(out)
    built = coords_to_lane_model(lanes, FRAME_W, FRAME_H)
    assert built is not None
    model, _geometry = built
    ego = {line.index: line for line in model.lines}
    assert set(ego) >= {1, 2, 0}
    left_x = (40 + 0.5) / 199.0 * FRAME_W
    right_x = (150 + 0.5) / 199.0 * FRAME_W
    assert model.lane_center_px == pytest.approx((left_x + right_x) / 2.0, abs=2.0)


def test_adjacent_column_lanes_get_no_pixel_quadratic():
    """A near-horizontal polyline has no meaningful x = f(y) fit; store none."""
    out = build_synthetic(row_peaks={1: 40, 2: 150}, col_peaks={0: 30, 3: 60})
    built = coords_to_lane_model(decode(out), FRAME_W, FRAME_H)
    assert built is not None
    model, _ = built
    for line in model.lines:
        if line.index in (0, 3):
            assert line.coeffs is None
            assert line.points_px


def test_no_ego_lane_returns_none_even_when_adjacent_lanes_exist():
    built = coords_to_lane_model(decode(build_synthetic(col_peaks={0: 30})), FRAME_W, FRAME_H)
    assert built is None


def test_single_ego_boundary_halves_the_confidence():
    both = coords_to_lane_model(
        decode(build_synthetic(row_peaks={1: 40, 2: 150})), FRAME_W, FRAME_H
    )
    one = coords_to_lane_model(decode(build_synthetic(row_peaks={1: 40})), FRAME_W, FRAME_H)
    assert both is not None and one is not None
    assert one[0].confidence == pytest.approx(both[0].confidence / 2.0, rel=1e-6)
    assert one[0].is_mock is False


# --------------------------------------------------------------------------- #
# Real-inference fixture
# --------------------------------------------------------------------------- #


needs_fixture = pytest.mark.skipif(
    not FIXTURE.exists(), reason="UFLD fixture tensors not present"
)


@pytest.fixture(scope="module")
def real_outputs():
    data = np.load(str(FIXTURE))
    return {key: data[key].astype(np.float32) for key in data.files}


@needs_fixture
def test_fixture_shapes_match_the_engine_contract(real_outputs):
    assert real_outputs["loc_row"].shape == (1, 200, 72, 4)
    assert real_outputs["loc_col"].shape == (1, 100, 81, 4)
    assert real_outputs["exist_row"].shape == (1, 2, 72, 4)
    assert real_outputs["exist_col"].shape == (1, 2, 81, 4)


@needs_fixture
def test_real_frame_yields_both_ego_boundaries(real_outputs):
    lanes = decode(real_outputs)
    assert 1 in lanes and 2 in lanes
    assert len(lanes[1].points) >= 30
    assert len(lanes[2].points) >= 30
    assert lanes[1].confidence > 0.5 and lanes[2].confidence > 0.5


@needs_fixture
def test_real_frame_boundaries_are_ordered_and_converge_upward(real_outputs):
    """Left must stay left of right, and the two must converge with distance."""
    lanes = decode(real_outputs)
    pairs = [
        (lx, rx, ly)
        for (lx, ly), (rx, ry) in zip(lanes[1].points, lanes[2].points)
        if abs(ly - ry) < 0.5
    ]
    assert len(pairs) >= 30
    assert all(lx < rx for lx, rx, _ly in pairs)
    widths = [(rx - lx, ly) for lx, rx, ly in pairs]
    near = max(widths, key=lambda w: w[1])[0]
    far = min(widths, key=lambda w: w[1])[0]
    assert near > far * 5.0


@needs_fixture
def test_real_frame_lane_centre_is_in_the_lower_middle_of_the_frame(real_outputs):
    built = coords_to_lane_model(decode(real_outputs), FRAME_W, FRAME_H)
    assert built is not None
    model, _ = built
    assert isinstance(model, LaneModel)
    assert model.is_mock is False
    assert 0.4 * FRAME_W < model.lane_center_px < 0.75 * FRAME_W
    assert 0.0 < model.confidence <= 1.0


@needs_fixture
def test_real_frame_vanishing_point_calibration_gives_a_physical_lane(real_outputs):
    """End-to-end metric check on real model output.

    With the assumed extrinsics (pitch 2 deg, height 1.30 m) the ground-plane
    lane width is not consistent with range. Calibrating the pitch from the lane
    vanishing point -- which IS the road-plane horizon -- makes it consistent,
    and the solved mount height lands at a physically sensible dashcam height.
    """
    lanes = decode(real_outputs)
    guess = CameraConfig(
        FRAME_W, FRAME_H, 910.0, 910.0, 640.0, 360.0, 1.30, 2.0, False, "assumed"
    )
    vp = estimate_vanishing_point(lanes[1].points, lanes[2].points)
    assert vp is not None
    assert 0.0 < vp[0] < FRAME_W
    assert 350.0 < vp[1] < 520.0  # well below the visual skyline of this clip

    camera = guess.with_horizon_px(vp[1], calibrated=True, label="vp")
    from adas.perception.geometry import calibrate_mount_height_from_lane

    height = calibrate_mount_height_from_lane(
        camera, lanes[1].points, lanes[2].points, known_lane_width_m=3.65
    )
    assert height is not None
    assert 0.7 < height < 2.0  # a plausible dashcam / bonnet mount

    camera = camera.with_mount_height(height, calibrated=True, label="vp+width")
    built = coords_to_lane_model(lanes, FRAME_W, FRAME_H, camera=camera)
    assert built is not None
    model, geometry = built
    assert geometry is not None
    assert geometry.both_boundaries and geometry.plausible
    assert 3.0 < geometry.lane_width_m < 4.3
    assert abs(geometry.lateral_offset_m) < 1.5
    assert model.curvature_m > 100.0  # example.mp4 is a near-straight highway
    assert lane_geometry_from_model(model) is not None
