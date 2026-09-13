"""CPU-only tests for the MiDaS relative-depth channel.

The engine is replaced by :class:`adas.infer.trt_engine.FakeTrtEngine` returning
a *synthetic* disparity map built from a known affine law,

    ``disparity = A / Z + B``

over a known camera geometry. That gives a ground truth the channel must
recover, so these tests check the actual numbers, not just that something came
back.

Two things to know before editing these tests.

First, the channel does **not** publish metric range by default -- on the shipped
MiDaS v2.1 small engine its per-object rank correlation measured 0.14-0.25
against a 0.80 bar, so it was demoted to a unitless relative signal. See the
:mod:`adas.perception.depth` module docstring for the measurements. The tests
that exercise the metric path therefore have to opt in with ``publish_metric``
*and* feed the channel enough correctly-ordered reference ranges to satisfy its
self-audit. That is not test scaffolding working around the guard; it is the
guard's contract, and ``test_metric_range_is_not_published_by_default`` and
friends below pin the refusing side of it.

Second, the synthetic map is *perfect*: objects sit exactly on the affine law, so
the audit sees correlation 1.0 and the gate opens. Real MiDaS does not, which is
the entire point of the gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from adas.core.exceptions import PerceptionError
from adas.core.models import BoundingBox, RangeEstimate, RangeSource
from adas.infer.trt_engine import FakeTrtEngine
from adas.perception.depth import (
    MEASURED_OBJECT_SPEARMAN_MIDAS_SMALL_256,
    MIN_ORDERING_SPEARMAN,
    DepthField,
    DepthRangeChannel,
    DepthScale,
    MiDaSDepthEstimator,
    OrdinalDepth,
    fit_depth_scale,
    fit_depth_scale_robust,
    road_plane_anchors,
    spearman_rho,
)
from adas.perception.geometry import CameraConfig, ground_plane_range

SRC_W, SRC_H = 1280, 720
MAP = 256

TRUE_A = 900.0
TRUE_B = 4.0


def _camera() -> CameraConfig:
    return CameraConfig(
        image_width=SRC_W,
        image_height=SRC_H,
        fx=910.0,
        fy=910.0,
        cx=SRC_W / 2.0,
        cy=SRC_H / 2.0,
        mount_height_m=1.3,
        pitch_deg=2.0,
        calibrated=True,
        label="synthetic",
    )


def _road_disparity_map(camera: CameraConfig) -> np.ndarray:
    """A 256x256 disparity map of an empty, perfectly flat road.

    On a plane with no roll the forward range depends only on the image row, so
    one projection per map row is enough.
    """
    rows = np.empty(MAP, dtype=np.float32)
    for y in range(MAP):
        v_px = (y + 0.5) / MAP * SRC_H
        ground = camera.image_to_ground(camera.cx, v_px)
        if ground is None:
            rows[y] = TRUE_B  # at or above the horizon: infinitely far
        else:
            rows[y] = TRUE_A / max(1e-3, ground[1]) + TRUE_B
    return np.repeat(rows[:, None], MAP, axis=1)


def _paint_box(disparity: np.ndarray, box: BoundingBox, range_m: float) -> None:
    x0 = int(box.x1 / SRC_W * MAP)
    x1 = int(np.ceil(box.x2 / SRC_W * MAP))
    y0 = int(box.y1 / SRC_H * MAP)
    y1 = int(np.ceil(box.y2 / SRC_H * MAP))
    disparity[max(0, y0) : min(MAP, y1), max(0, x0) : min(MAP, x1)] = TRUE_A / range_m + TRUE_B


def _fake_midas(disparity: np.ndarray) -> FakeTrtEngine:
    return FakeTrtEngine(
        {"0": (1, 3, MAP, MAP)},
        {"797": disparity.reshape(1, MAP, MAP).astype(np.float32)},
        engine_path="<fake-midas>",
    )


def _box_at(camera: CameraConfig, range_m: float, height_m: float = 1.5, width_m: float = 1.8):
    """A detection box for an object standing on the road at ``range_m``."""
    bottom = camera.ground_to_image(0.0, range_m)
    top = camera.ground_to_image(0.0, range_m)
    assert bottom is not None and top is not None
    u_c, v_bottom = bottom
    v_top = v_bottom - height_m * camera.fy / range_m
    half_w = 0.5 * width_m * camera.fx / range_m
    return BoundingBox(
        x1=float(u_c - half_w),
        y1=float(v_top),
        x2=float(u_c + half_w),
        y2=float(v_bottom),
        confidence=0.9,
        label="car",
    )


def _scene(camera: CameraConfig, ranges):
    """Boxes at ``ranges`` plus a disparity map with each one painted in.

    ``_box_at`` puts every box on the optical axis, so a far box is nested
    *inside* a near one in image space. Painting therefore goes nearest-first:
    the small far rectangles are laid over the big near one, the way occlusion
    actually works. Their lower-central sample regions do not overlap (at 12/30/
    60 m they are rows 375-421, 347-365 and 338-347), so each box still samples
    its own disparity. Painting in the caller's order instead would let a near
    box erase a far one that was painted first.

    ``boxes`` comes back in the caller's order, which need not be sorted.
    """
    boxes = [_box_at(camera, z) for z in ranges]
    disparity = _road_disparity_map(camera)
    for box, z in sorted(zip(boxes, ranges), key=lambda pair: pair[1]):
        _paint_box(disparity, box, z)
    return boxes, disparity


def _refs(boxes, camera):
    return [ground_plane_range(b, camera, SRC_W, SRC_H) for b in boxes]


def _metric_channel(disparity, **kwargs):
    """A channel allowed to publish metres once its self-audit is satisfied.

    ``min_ordering_pairs=3`` is the floor of what a rank correlation can be
    computed from at all, so a single frame of a three-vehicle scene is enough
    evidence here. Production keeps the much larger default.
    """
    kwargs.setdefault("cadence_frames", 1)
    return DepthRangeChannel(
        engine=_fake_midas(disparity),
        publish_metric=True,
        min_ordering_pairs=3,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Affine fit
# --------------------------------------------------------------------------- #


def test_fit_recovers_a_known_affine_law():
    ranges = np.array([6.0, 10.0, 18.0, 30.0, 55.0, 90.0])
    disparities = TRUE_A / ranges + TRUE_B
    scale = fit_depth_scale(disparities, ranges, min_anchors=3)
    assert scale.valid
    assert scale.a == pytest.approx(TRUE_A, rel=1e-6)
    assert scale.b == pytest.approx(TRUE_B, abs=1e-6)
    assert scale.range_m(float(TRUE_A / 22.0 + TRUE_B)) == pytest.approx(22.0, rel=1e-6)


def test_fit_rejects_an_inverted_relationship():
    """Disparity that *grows* with range is not depth; refuse it."""
    ranges = np.array([5.0, 20.0, 50.0])
    disparities = ranges * 2.0
    assert not fit_depth_scale(disparities, ranges, min_anchors=3).valid


def test_fit_with_one_anchor_is_degraded_not_confident():
    scale = fit_depth_scale([TRUE_A / 20.0], [20.0], min_anchors=3)
    assert scale.valid and scale.anchors == 1
    assert scale.rel_rmse == 1.0  # drives the confidence to zero


def test_robust_fit_rejects_an_outlier():
    ranges = np.array([6.0, 10.0, 18.0, 30.0, 55.0, 90.0, 25.0])
    disparities = TRUE_A / ranges + TRUE_B
    disparities[-1] = 500.0  # a probe that landed on a bridge
    naive = fit_depth_scale(disparities, ranges, min_anchors=3)
    robust = fit_depth_scale_robust(disparities, ranges, min_anchors=3)
    assert abs(robust.a - TRUE_A) < abs(naive.a - TRUE_A)
    assert robust.a == pytest.approx(TRUE_A, rel=0.05)


def test_invalid_scale_returns_no_range():
    assert DepthScale().range_m(10.0) is None


def test_fit_length_mismatch_raises():
    with pytest.raises(PerceptionError):
        fit_depth_scale([1.0, 2.0], [3.0])


# --------------------------------------------------------------------------- #
# Disparity field sampling
# --------------------------------------------------------------------------- #


def test_sample_points_is_vectorised_and_matches_the_scalar_form():
    disparity = np.arange(MAP * MAP, dtype=np.float32).reshape(MAP, MAP)
    field = DepthField(disparity=disparity, frame_id=0)
    us = np.array([10.0, 640.0, 1200.0])
    vs = np.array([20.0, 360.0, 700.0])
    batch = field.sample_points(us, vs, SRC_W, SRC_H)
    scalar = [field.sample_point(u, v, SRC_W, SRC_H) for u, v in zip(us, vs)]
    assert batch == pytest.approx(scalar)


def test_sample_points_marks_out_of_frame_as_nan():
    field = DepthField(disparity=np.ones((MAP, MAP), np.float32), frame_id=0)
    out = field.sample_points([-5.0, 640.0], [100.0, 100.0], SRC_W, SRC_H)
    assert np.isnan(out[0]) and out[1] == pytest.approx(1.0)


def test_sample_box_uses_the_lower_central_region():
    disparity = np.zeros((MAP, MAP), np.float32)
    disparity[MAP // 2 :, :] = 7.0  # bottom half only
    field = DepthField(disparity=disparity, frame_id=0)
    box = BoundingBox(400.0, 360.0, 600.0, 700.0, 0.9, "car")
    assert field.sample_box(box, SRC_W, SRC_H) == pytest.approx(7.0)


def test_invalid_field_samples_nothing():
    field = DepthField()
    assert not field.valid
    assert field.sample_point(10.0, 10.0, SRC_W, SRC_H) is None
    assert field.sample_box(BoundingBox(0, 0, 10, 10, 0.5, "car"), SRC_W, SRC_H) is None


# --------------------------------------------------------------------------- #
# Road-plane anchors
# --------------------------------------------------------------------------- #


def test_road_anchors_recover_the_true_law():
    camera = _camera()
    field = DepthField(disparity=_road_disparity_map(camera), frame_id=0)
    disparities, distances = road_plane_anchors(field, camera, SRC_W, SRC_H, [])
    assert len(disparities) >= 6
    scale = fit_depth_scale_robust(disparities, distances, min_anchors=6)
    assert scale.valid
    assert scale.a == pytest.approx(TRUE_A, rel=0.10)
    assert scale.b == pytest.approx(TRUE_B, abs=0.10 * TRUE_A / 50.0 + 1.0)


def test_road_anchors_skip_pixels_a_vehicle_is_standing_on():
    camera = _camera()
    field = DepthField(disparity=_road_disparity_map(camera), frame_id=0)
    unblocked, _ = road_plane_anchors(field, camera, SRC_W, SRC_H, [])
    covering = BoundingBox(0.0, 0.0, float(SRC_W - 1), float(SRC_H - 1), 0.9, "truck")
    blocked, _ = road_plane_anchors(field, camera, SRC_W, SRC_H, [covering])
    assert len(unblocked) > 0
    assert blocked == []


def test_road_anchors_need_a_camera_and_a_field():
    camera = _camera()
    assert road_plane_anchors(DepthField(), camera, SRC_W, SRC_H, []) == ([], [])
    field = DepthField(disparity=_road_disparity_map(camera), frame_id=0)
    assert road_plane_anchors(field, None, SRC_W, SRC_H, []) == ([], [])


# --------------------------------------------------------------------------- #
# The channel
# --------------------------------------------------------------------------- #


def test_missing_engine_degrades_to_an_honest_stub():
    channel = DepthRangeChannel("models/definitely-not-here.engine")
    assert channel.available is False
    assert channel.is_mock is True
    boxes = [BoundingBox(0, 0, 10, 10, 0.9, "car")]
    results = channel.update(np.zeros((SRC_H, SRC_W, 3), np.uint8), 0, boxes, SRC_W, SRC_H)
    assert len(results) == 1
    assert results[0].source is RangeSource.UNAVAILABLE
    assert results[0].confidence == 0.0
    assert results[0].distance_m == 0.0
    channel.close()


def test_missing_engine_can_be_made_fatal():
    with pytest.raises(PerceptionError):
        DepthRangeChannel("models/definitely-not-here.engine", require_engine=True)


def test_channel_recovers_object_ranges_from_the_road_plane():
    """With a model that actually resolves range, the metric path still works."""
    camera = _camera()
    truth = [12.0, 30.0, 60.0]
    boxes, disparity = _scene(camera, truth)

    channel = _metric_channel(disparity, cadence_frames=5)
    assert channel.available and not channel.is_mock
    results = channel.update(
        np.zeros((SRC_H, SRC_W, 3), np.uint8),
        0,
        boxes,
        SRC_W,
        SRC_H,
        reference=_refs(boxes, camera),
        camera=camera,
    )
    assert channel.scale_mode == "road_plane"
    assert channel.metric_gate_open, channel.stats()
    assert len(results) == 3
    for result, z in zip(results, truth):
        assert result.source is RangeSource.DEPTH_MODEL
        assert result.confidence > 0.0
        assert result.distance_m == pytest.approx(z, rel=0.15)
    channel.close()


def test_channel_range_is_independent_of_the_box_geometry():
    """Shrinking a box must not move the depth range: different measurement.

    Three boxes rather than one, because the self-audit needs a spread of
    reference ranges before it will let any metres out at all.
    """
    camera = _camera()
    ranges = [12.0, 30.0, 60.0]
    boxes, disparity = _scene(camera, ranges)
    box = boxes[1]
    channel = _metric_channel(disparity)
    frame = np.zeros((SRC_H, SRC_W, 3), np.uint8)

    honest = channel.update(
        frame, 0, boxes, SRC_W, SRC_H, reference=_refs(boxes, camera), camera=camera
    )[1]
    # A box whose top is clipped: the pinhole range would double, the depth
    # range must not, because it reads the same surface pixels.
    clipped = BoundingBox(box.x1, box.y1 + (box.y2 - box.y1) * 0.5, box.x2, box.y2, 0.9, "car")
    mangled = [boxes[0], clipped, boxes[2]]
    degraded = channel.update(
        frame, 1, mangled, SRC_W, SRC_H, reference=_refs(mangled, camera), camera=camera
    )[1]

    assert honest.source is RangeSource.DEPTH_MODEL
    assert degraded.source is RangeSource.DEPTH_MODEL
    assert honest.distance_m == pytest.approx(30.0, rel=0.15)
    assert degraded.distance_m == pytest.approx(honest.distance_m, rel=0.10)
    assert ground_plane_range(box, camera, SRC_W, SRC_H).distance_m > 0.0
    channel.close()


def test_channel_respects_its_cadence():
    camera = _camera()
    disparity = _road_disparity_map(camera)
    engine = _fake_midas(disparity)
    channel = DepthRangeChannel(engine=engine, cadence_frames=5)
    frame = np.zeros((SRC_H, SRC_W, 3), np.uint8)
    boxes = [_box_at(camera, 25.0)]
    for frame_id in range(10):
        channel.update(frame, frame_id, boxes, SRC_W, SRC_H, camera=camera)
    assert engine.call_count == 2  # frames 0 and 5
    assert channel.stats()["duty_cycle"] == pytest.approx(0.2)
    channel.close()


def test_channel_rejects_a_frame_id_that_went_backwards():
    """A replay restart must not have the previous run's map applied to it."""
    camera = _camera()
    channel = DepthRangeChannel(engine=_fake_midas(_road_disparity_map(camera)), cadence_frames=3)
    frame = np.zeros((SRC_H, SRC_W, 3), np.uint8)
    boxes = [_box_at(camera, 25.0)]
    channel.update(frame, 500, boxes, SRC_W, SRC_H, camera=camera)
    results = channel.update(frame, 0, boxes, SRC_W, SRC_H, camera=camera)
    assert results[0].source is RangeSource.UNAVAILABLE
    channel.close()


def test_confidence_decays_across_the_reuse_window():
    """A map sampled at a box that has since moved must be trusted less."""
    camera = _camera()
    boxes, disparity = _scene(camera, [12.0, 25.0, 60.0])
    engine = _fake_midas(disparity)
    channel = DepthRangeChannel(
        engine=engine, cadence_frames=4, publish_metric=True, min_ordering_pairs=3
    )
    frame = np.zeros((SRC_H, SRC_W, 3), np.uint8)
    reference = _refs(boxes, camera)
    confidences = []
    for frame_id in range(4):
        result = channel.update(
            frame, frame_id, boxes, SRC_W, SRC_H, reference=reference, camera=camera
        )[1]
        assert result.source is RangeSource.DEPTH_MODEL
        confidences.append(result.confidence)
    assert engine.call_count == 1
    assert confidences == sorted(confidences, reverse=True)
    assert confidences[-1] < confidences[0]
    channel.close()


def test_channel_refuses_to_let_one_object_validate_itself():
    """Box anchoring with too few detections is circular; it must say so.

    ``publish_metric=True`` so the refusal is the leave-one-out guard doing its
    job, not merely the default demotion. One box also gives the self-audit no
    rank spread, so both guards independently refuse -- which is the intent.
    """
    camera = _camera()
    boxes, disparity = _scene(camera, [25.0])
    channel = _metric_channel(disparity)
    results = channel.update(
        np.zeros((SRC_H, SRC_W, 3), np.uint8),
        0,
        boxes,
        SRC_W,
        SRC_H,
        reference=_refs(boxes, camera),
    )
    assert results[0].source is RangeSource.UNAVAILABLE
    assert channel.scale_mode == ""
    assert not channel.metric_gate_open
    channel.close()


def test_box_leave_one_out_fallback_produces_a_range():
    camera = _camera()
    ranges = [10.0, 18.0, 30.0, 50.0]
    boxes = [_box_at(camera, z) for z in ranges]
    disparity = _road_disparity_map(camera)
    for box, z in zip(boxes, ranges):
        _paint_box(disparity, box, z)
    channel = _metric_channel(disparity)
    reference = _refs(boxes, camera)
    results = channel.update(
        np.zeros((SRC_H, SRC_W, 3), np.uint8), 0, boxes, SRC_W, SRC_H, reference=reference
    )
    assert channel.scale_mode == "box_loo"
    usable = [r for r in results if r.source is RangeSource.DEPTH_MODEL]
    assert len(usable) >= 3
    for result, z in zip(results, ranges):
        if result.source is RangeSource.DEPTH_MODEL:
            assert result.distance_m == pytest.approx(z, rel=0.25)
            # The box mode is the weaker reference and must say so.
            assert result.confidence <= 0.45
    channel.close()


def test_channel_survives_an_engine_that_throws():
    def explode(_inputs):
        raise RuntimeError("CUDA context died")

    engine = FakeTrtEngine(
        {"0": (1, 3, MAP, MAP)}, explode, output_shapes={"797": (1, MAP, MAP)}
    )
    channel = DepthRangeChannel(engine=engine, cadence_frames=1)
    results = channel.update(
        np.zeros((SRC_H, SRC_W, 3), np.uint8), 0, [BoundingBox(0, 0, 10, 10, 0.9, "car")],
        SRC_W, SRC_H, camera=_camera(),
    )
    assert results[0].source is RangeSource.UNAVAILABLE
    channel.close()


def test_reference_length_must_match_the_boxes():
    camera = _camera()
    channel = DepthRangeChannel(engine=_fake_midas(_road_disparity_map(camera)), cadence_frames=1)
    with pytest.raises(PerceptionError):
        channel.update(
            np.zeros((SRC_H, SRC_W, 3), np.uint8),
            0,
            [_box_at(camera, 20.0), _box_at(camera, 40.0)],
            SRC_W,
            SRC_H,
            reference=[RangeEstimate(20.0, 0.5, RangeSource.HOMOGRAPHY)],
        )
    channel.close()


def test_estimator_validates_the_engine_contract():
    bad = FakeTrtEngine({"0": (1, 3, MAP, MAP)}, {"a": np.zeros((1, 4, 4), np.float32),
                                                  "b": np.zeros((1, 4, 4), np.float32)})
    with pytest.raises(PerceptionError):
        MiDaSDepthEstimator("", engine=bad)


def test_estimator_reads_its_binding_names_from_the_engine():
    """The v2.1 export's binding names are literally "0" and "797"."""
    estimator = MiDaSDepthEstimator("", engine=_fake_midas(np.ones((MAP, MAP), np.float32)))
    assert estimator.output_name == "797"
    assert (estimator.input_height, estimator.input_width) == (MAP, MAP)
    out = estimator.infer(np.zeros((SRC_H, SRC_W, 3), np.uint8))
    assert out.shape == (MAP, MAP)
    estimator.close()


def test_cadence_must_be_positive():
    with pytest.raises(PerceptionError):
        DepthRangeChannel(cadence_frames=0)


# --------------------------------------------------------------------------- #
# Demotion: no metres without demonstrated ordering skill
#
# These are the regression tests for the defect that took the channel down. On
# Ultra-Fast-Lane-Detection-v2/example.mp4 the shipped MiDaS v2.1 small engine
# turned a true 5.3-61.5 m spread into 3.7-13.4 m at Spearman 0.25, and the
# arbiter's min(pinhole, depth) fusion dragged every range toward ~9 m: 62
# range_channel_disagreement violations over 400 frames. Nothing here needs the
# GPU -- the guard is testable on a synthetic map, which is why it is a guard
# and not a note in the README.
# --------------------------------------------------------------------------- #


def test_metric_range_is_not_published_by_default():
    """The headline: a default channel emits no metres, however good the fit.

    The synthetic map is exactly on the affine law, so the fit succeeds and the
    scale is valid -- and it still must not publish, because nobody asked it to.
    """
    camera = _camera()
    boxes, disparity = _scene(camera, [12.0, 30.0, 60.0])
    channel = DepthRangeChannel(engine=_fake_midas(disparity), cadence_frames=1)
    assert channel.publish_metric is False

    results = channel.update(
        np.zeros((SRC_H, SRC_W, 3), np.uint8),
        0,
        boxes,
        SRC_W,
        SRC_H,
        reference=_refs(boxes, camera),
        camera=camera,
    )
    assert len(results) == 3
    for result in results:
        assert result.source is RangeSource.UNAVAILABLE
        assert result.confidence == 0.0
        assert result.distance_m == 0.0
    # The fit itself still ran and is still reported: it is the publication
    # that is gated, not the diagnostics.
    assert channel.scale_mode == "road_plane"
    assert channel.scale.valid
    assert channel.metric_gate_open is False
    channel.close()


def test_a_channel_with_no_ordering_skill_never_publishes_metres():
    """The actual MiDaS failure, reproduced on a synthetic map.

    Every object is painted at the *same* disparity regardless of its true
    range -- which is what the real engine does: on the reference clip, objects
    beyond 30 m carried the same mean disparity (399) as objects inside 20 m
    (392). A channel that cannot rank two vehicles must not be allowed to set a
    braking distance, even when explicitly asked for metres.
    """
    camera = _camera()
    ranges = [10.0, 20.0, 35.0, 55.0]
    boxes = [_box_at(camera, z) for z in ranges]
    disparity = _road_disparity_map(camera)
    for box in boxes:
        _paint_box(disparity, box, 9.0)  # every object reads ~9 m, near or far

    channel = _metric_channel(disparity)
    frame = np.zeros((SRC_H, SRC_W, 3), np.uint8)
    reference = _refs(boxes, camera)
    for frame_id in range(8):
        results = channel.update(
            frame, frame_id, boxes, SRC_W, SRC_H, reference=reference, camera=camera
        )
        assert all(r.source is RangeSource.UNAVAILABLE for r in results)
        assert all(r.confidence == 0.0 for r in results)

    stats = channel.stats()
    assert stats["publish_metric_requested"] == 1.0
    assert stats["metric_gate_open"] == 0.0
    assert stats["ordering_pairs"] >= 24.0
    # No rank spread in the disparities at all: the audit cannot even form a
    # correlation, and "no evidence" must mean "no metres".
    assert np.isnan(stats["ordering_spearman"])
    channel.close()


def test_a_channel_that_ranks_objects_backwards_never_publishes_metres():
    """Worse than useless: far objects reading nearer than near ones.

    This is frame 300 of the reference clip in miniature -- the car at pinhole
    54.3 m sampled disparity 410 while the car at 13.4 m sampled 348, so the far
    car read closer. An anti-correlated channel must be refused, not inverted.
    """
    camera = _camera()
    ranges = [10.0, 20.0, 35.0, 55.0]
    boxes = [_box_at(camera, z) for z in ranges]
    disparity = _road_disparity_map(camera)
    for box, z in zip(boxes, reversed(ranges)):
        _paint_box(disparity, box, z)  # nearest box painted at the farthest range

    channel = _metric_channel(disparity)
    frame = np.zeros((SRC_H, SRC_W, 3), np.uint8)
    reference = _refs(boxes, camera)
    for frame_id in range(8):
        results = channel.update(
            frame, frame_id, boxes, SRC_W, SRC_H, reference=reference, camera=camera
        )
        assert all(r.source is RangeSource.UNAVAILABLE for r in results)

    assert channel.ordering_spearman < 0.0
    assert channel.metric_gate_open is False
    channel.close()


def test_a_mediocre_correlation_is_still_refused():
    """0.686 was the best honest variant measured. The bar is 0.80. Refuse it.

    Pinned as its own case because "better than what we had" is the argument
    that would put this channel back on the road, and it is not the criterion.
    """
    camera = _camera()
    boxes, disparity = _scene(camera, [12.0, 30.0, 60.0])
    lenient = _metric_channel(disparity, min_ordering_spearman=0.60)
    strict = _metric_channel(disparity, min_ordering_spearman=1.01)
    frame = np.zeros((SRC_H, SRC_W, 3), np.uint8)
    reference = _refs(boxes, camera)

    ok = lenient.update(frame, 0, boxes, SRC_W, SRC_H, reference=reference, camera=camera)
    no = strict.update(frame, 0, boxes, SRC_W, SRC_H, reference=reference, camera=camera)

    assert any(r.source is RangeSource.DEPTH_MODEL for r in ok)
    assert all(r.source is RangeSource.UNAVAILABLE for r in no)
    assert MIN_ORDERING_SPEARMAN == 0.80
    lenient.close()
    strict.close()


def test_the_measured_model_correlation_is_below_the_gate_floor():
    """The recorded measurement and the bar must stay in the same file.

    If someone raises the model's measured skill, this test tells them the gate
    is the thing to re-check; if someone lowers the bar below what MiDaS small
    actually does, it fails and asks why.
    """
    assert MEASURED_OBJECT_SPEARMAN_MIDAS_SMALL_256 < MIN_ORDERING_SPEARMAN


def test_the_audit_fails_closed_without_reference_ranges():
    """No reference means no evidence means no metres, camera or not."""
    camera = _camera()
    boxes, disparity = _scene(camera, [12.0, 30.0, 60.0])
    channel = DepthRangeChannel(
        engine=_fake_midas(disparity),
        cadence_frames=1,
        publish_metric=True,
        min_ordering_pairs=3,
    )
    results = channel.update(
        np.zeros((SRC_H, SRC_W, 3), np.uint8),
        0,
        boxes,
        SRC_W,
        SRC_H,
        reference=[None, None, None],
        camera=camera,
    )
    assert all(r.source is RangeSource.UNAVAILABLE for r in results)
    assert channel.stats()["ordering_pairs"] == 0.0
    assert channel.metric_gate_open is False
    channel.close()


def test_the_audit_needs_enough_pairs_before_it_believes_itself():
    """A perfect correlation over two points is not evidence."""
    camera = _camera()
    boxes, disparity = _scene(camera, [12.0, 60.0])
    channel = DepthRangeChannel(
        engine=_fake_midas(disparity),
        cadence_frames=1,
        publish_metric=True,
        min_ordering_pairs=8,
    )
    frame = np.zeros((SRC_H, SRC_W, 3), np.uint8)
    reference = _refs(boxes, camera)

    first = channel.update(frame, 0, boxes, SRC_W, SRC_H, reference=reference, camera=camera)
    assert all(r.source is RangeSource.UNAVAILABLE for r in first)
    assert channel.stats()["ordering_pairs"] == 2.0

    for frame_id in range(1, 4):
        last = channel.update(
            frame, frame_id, boxes, SRC_W, SRC_H, reference=reference, camera=camera
        )
    assert channel.stats()["ordering_pairs"] == 8.0
    assert channel.ordering_spearman == pytest.approx(1.0)
    assert all(r.source is RangeSource.DEPTH_MODEL for r in last)
    channel.close()


def test_close_forgets_the_audit_so_a_restart_re_earns_the_gate():
    """Evidence from the previous clip must not open the gate on the next one."""
    camera = _camera()
    boxes, disparity = _scene(camera, [12.0, 30.0, 60.0])
    channel = _metric_channel(disparity)
    frame = np.zeros((SRC_H, SRC_W, 3), np.uint8)
    channel.update(frame, 0, boxes, SRC_W, SRC_H, reference=_refs(boxes, camera), camera=camera)
    assert channel.metric_gate_open is True

    channel.close()
    assert channel.metric_gate_open is False
    assert channel.stats()["ordering_pairs"] == 0.0
    assert np.isnan(channel.ordering_spearman)


# --------------------------------------------------------------------------- #
# The ordinal signal that replaced the metric one
# --------------------------------------------------------------------------- #


def test_ordinal_readings_carry_no_metric_field():
    """The replacement signal must be impossible to mistake for metres."""
    camera = _camera()
    boxes, disparity = _scene(camera, [12.0, 30.0, 60.0])
    channel = DepthRangeChannel(engine=_fake_midas(disparity), cadence_frames=1)
    channel.update(
        np.zeros((SRC_H, SRC_W, 3), np.uint8),
        0,
        boxes,
        SRC_W,
        SRC_H,
        reference=_refs(boxes, camera),
        camera=camera,
    )
    readings = channel.ordinal_readings()
    assert len(readings) == 3
    for reading in readings:
        assert isinstance(reading, OrdinalDepth)
        assert not isinstance(reading, RangeEstimate)
        assert not hasattr(reading, "distance_m")
        assert not hasattr(reading, "confidence")
        assert not hasattr(reading, "source")
    channel.close()


def test_ordinal_readings_rank_boxes_from_nearest_to_farthest():
    camera = _camera()
    ranges = [30.0, 12.0, 60.0]  # deliberately not in order
    boxes, disparity = _scene(camera, ranges)
    channel = DepthRangeChannel(engine=_fake_midas(disparity), cadence_frames=1)
    channel.update(
        np.zeros((SRC_H, SRC_W, 3), np.uint8),
        0,
        boxes,
        SRC_W,
        SRC_H,
        reference=_refs(boxes, camera),
        camera=camera,
    )
    readings = channel.ordinal_readings()
    assert [r.rank for r in readings] == [1, 0, 2]
    assert all(r.of == 3 for r in readings)
    assert [r.normalized for r in readings] == pytest.approx([0.5, 0.0, 1.0])
    assert readings[1].disparity > readings[0].disparity > readings[2].disparity
    channel.close()


def test_ordinal_readings_are_published_even_though_metres_are_not():
    """Demotion removed the metric claim, not the channel's output."""
    camera = _camera()
    boxes, disparity = _scene(camera, [12.0, 30.0, 60.0])
    channel = DepthRangeChannel(engine=_fake_midas(disparity), cadence_frames=1)
    results = channel.update(
        np.zeros((SRC_H, SRC_W, 3), np.uint8),
        0,
        boxes,
        SRC_W,
        SRC_H,
        reference=_refs(boxes, camera),
        camera=camera,
    )
    assert all(r.source is RangeSource.UNAVAILABLE for r in results)
    assert all(r is not None for r in channel.ordinal_readings())
    channel.close()


def test_ordinal_readings_are_empty_without_an_engine():
    channel = DepthRangeChannel("models/definitely-not-here.engine")
    channel.update(
        np.zeros((SRC_H, SRC_W, 3), np.uint8),
        0,
        [BoundingBox(0, 0, 10, 10, 0.9, "car")],
        SRC_W,
        SRC_H,
    )
    assert channel.ordinal_readings() == [None]
    channel.close()


def test_an_unsampleable_box_gets_no_rank_and_does_not_shift_the_others():
    camera = _camera()
    boxes, disparity = _scene(camera, [12.0, 60.0])
    channel = DepthRangeChannel(engine=_fake_midas(disparity), cadence_frames=1)
    channel.update(np.zeros((SRC_H, SRC_W, 3), np.uint8), 0, boxes, SRC_W, SRC_H, camera=camera)
    baseline = [r.rank for r in channel.ordinal_readings()]

    # A zero-width, zero-height frame makes every sample unavailable.
    channel.update(np.zeros((SRC_H, SRC_W, 3), np.uint8), 1, boxes, 0, 0, camera=camera)
    assert channel.ordinal_readings() == [None, None]
    assert baseline == [0, 1]
    channel.close()


# --------------------------------------------------------------------------- #
# The rank statistic the gate is built on
# --------------------------------------------------------------------------- #


def test_spearman_rho_matches_a_known_monotone_case():
    assert spearman_rho([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)
    assert spearman_rho([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_spearman_rho_is_invariant_under_monotone_rescaling():
    """Why the gate can judge ordering without a calibrated camera.

    The affine inversion a/(d - b) is monotone, so the gate's verdict does not
    depend on how (or whether) the scale was anchored -- which is what let the
    real-clip measurement be trusted on an uncalibrated camera.
    """
    reference = [5.0, 9.0, 14.0, 22.0, 40.0, 61.0]
    disparity = [900.0 / z + 4.0 for z in reference]
    direct = spearman_rho(reference, [-d for d in disparity])
    inverted = spearman_rho(reference, [900.0 / (d - 4.0) for d in disparity])
    assert direct == pytest.approx(1.0)
    assert inverted == pytest.approx(direct)


def test_spearman_rho_shares_ranks_between_ties():
    """A saturated depth map is all ties; it must not score as agreement."""
    assert np.isnan(spearman_rho([1.0, 2.0, 3.0, 4.0], [7.0, 7.0, 7.0, 7.0]))
    tied = spearman_rho([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 2.0, 3.0])
    assert 0.8 < tied < 1.0


def test_spearman_rho_needs_three_points_and_ignores_non_finite():
    assert np.isnan(spearman_rho([1.0, 2.0], [1.0, 2.0]))
    assert np.isnan(spearman_rho([], []))
    with_nan = spearman_rho([1.0, 2.0, 3.0, np.nan], [1.0, 2.0, 3.0, 99.0])
    assert with_nan == pytest.approx(1.0)


def test_spearman_rho_length_mismatch_raises():
    with pytest.raises(PerceptionError):
        spearman_rho([1.0, 2.0, 3.0], [1.0, 2.0])
