"""Specification of :class:`adas.tracking.MultiObjectTracker`.

These are scenario tests, not point assertions. A synthetic pinhole camera
projects vehicles with known ground truth into detector boxes, and the tests
assert what the planner and the safety arbiter actually need:

* ID stability through occlusion, a cut-in and two crossing objects;
* estimated range and closing speed against ground truth, in metres and m/s;
* that a single spurious detection is never reported at all (ADAS-DEC-09 --
  the phantom-braking path);
* that a distant track cannot steal the lead vehicle's detection
  (ADAS-DEC-07/08);
* time-to-collision, including every degenerate case.

The scenario generator quantises box coordinates to whole pixels, because pixel
quantisation *is* the noise process that made the old unfiltered range rate
useless (ADAS-DEC-06). Where extra noise is added the seed is fixed.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pytest

from adas.core.exceptions import TrackingError, ValidationError
from adas.core.models import (
    BoundingBox,
    DrivableArea,
    LaneLine,
    LaneModel,
    RangeSource,
    TrackedObject,
)
from adas.tracking import MultiObjectTracker, TrackStatus
from adas.tracking.tracker import NOMINAL_OBJECT_HEIGHT_M

SEED = 20240913
DT = 0.05
WIDTH = 1280
HEIGHT = 720
FOCAL_PX = 910.0
CAM_HEIGHT_M = 1.3
HORIZON_PX = 360.0

#: Metric width of each class, for the synthetic projector only.
CLASS_WIDTH_M = {"car": 1.8, "truck": 2.5, "bus": 2.6, "person": 0.6, "bicycle": 0.6}


# --------------------------------------------------------------------------- #
# Synthetic scenario generator
# --------------------------------------------------------------------------- #


def project(
    x_m: float,
    z_m: float,
    label: str = "car",
    confidence: float = 0.9,
    noise_px: float = 0.0,
    rng: Optional[np.random.RandomState] = None,
) -> BoundingBox:
    """Project a ground-truth vehicle onto the synthetic image plane.

    ``x_m`` is lateral (positive right of the camera axis), ``z_m`` longitudinal
    range. The camera is a zero-pitch pinhole at ``CAM_HEIGHT_M`` above the road
    with its horizon at ``HORIZON_PX``, so the box bottom is the ground-contact
    row and the box height is the class height prior projected at ``z_m``. That
    makes ``MultiObjectTracker``'s own pinhole range model exact up to the pixel
    quantisation applied here, which is the point: the residual error under test
    is the quantisation, not a modelling mismatch.
    """
    height_m = NOMINAL_OBJECT_HEIGHT_M.get(label, 1.5)
    width_m = CLASS_WIDTH_M.get(label, 1.8)
    h_px = height_m * FOCAL_PX / z_m
    w_px = width_m * FOCAL_PX / z_m
    cx = WIDTH / 2.0 + x_m * FOCAL_PX / z_m
    y2 = HORIZON_PX + CAM_HEIGHT_M * FOCAL_PX / z_m
    y1 = y2 - h_px
    x1 = cx - w_px / 2.0
    x2 = cx + w_px / 2.0
    if noise_px and rng is not None:
        x1 += rng.normal(0.0, noise_px)
        x2 += rng.normal(0.0, noise_px)
        y1 += rng.normal(0.0, noise_px)
        y2 += rng.normal(0.0, noise_px)
    x1 = max(0.0, min(float(WIDTH - 2), round(x1)))
    y1 = max(0.0, min(float(HEIGHT - 2), round(y1)))
    x2 = max(x1 + 1.0, min(float(WIDTH - 1), round(x2)))
    y2 = max(y1 + 1.0, min(float(HEIGHT - 1), round(y2)))
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence, label=label)


def straight_lane_model(confidence: float = 0.8, half_width_m: float = 1.75) -> LaneModel:
    """A metric ego lane centred on the camera axis, as the lane workstream emits it.

    ``LaneLine.coeffs`` are ground-plane quadratics ``X = a Z^2 + b Z + c`` in
    metres with ``X`` positive to the right, index 1 left boundary, 2 right.
    """
    return LaneModel(
        left_coeffs=(0.0, 0.0, 0.0),
        right_coeffs=(0.0, 0.0, 0.0),
        lane_center_px=WIDTH / 2.0,
        curvature_m=10000.0,
        confidence=confidence,
        lines=[
            LaneLine(points_px=[], coeffs=(0.0, 0.0, -half_width_m), confidence=confidence, index=1),
            LaneLine(points_px=[], coeffs=(0.0, 0.0, half_width_m), confidence=confidence, index=2),
        ],
        is_mock=False,
    )


def corridor_drivable_area(free_lo: int = 8, free_hi: int = 24) -> DrivableArea:
    """A 32x18 free-space mask whose free run is columns ``[free_lo, free_hi)``."""
    mask = [[free_lo <= col < free_hi for col in range(32)] for _ in range(18)]
    return DrivableArea(mask=mask, width=32, height=18, confidence=0.8)


def run_frames(
    tracker: MultiObjectTracker,
    frames: Sequence[Sequence[BoundingBox]],
    dt_s: float = DT,
    lane: Optional[LaneModel] = None,
    drivable: Optional[DrivableArea] = None,
) -> List[List[TrackedObject]]:
    """Feed a whole scenario through the tracker and collect every frame's output."""
    return [
        tracker.update(
            list(detections),
            dt_s=dt_s,
            frame_width=WIDTH,
            frame_height=HEIGHT,
            lane=lane,
            drivable=drivable,
        )
        for detections in frames
    ]


def id_switches(
    outputs: Sequence[Sequence[TrackedObject]],
    truth: Sequence[Dict[str, BoundingBox]],
    tolerance_px: float = 80.0,
) -> Dict[str, int]:
    """Count how many times each ground-truth object changed track id.

    Published tracks are attributed to the nearest ground-truth box centre within
    ``tolerance_px``; an object that goes unmatched in a frame is skipped rather
    than counted as a switch, so a coast is not penalised.
    """
    switches = {name: 0 for name in (truth[0] if truth else {})}
    last: Dict[str, int] = {}
    for frame_index, published in enumerate(outputs):
        gt = truth[frame_index]
        for name in gt:
            switches.setdefault(name, 0)
        for obj in published:
            cx = (obj.box.x1 + obj.box.x2) / 2.0
            cy = (obj.box.y1 + obj.box.y2) / 2.0
            best_name = None
            best_distance = tolerance_px
            for name, box in gt.items():
                gcx = (box.x1 + box.x2) / 2.0
                gcy = (box.y1 + box.y2) / 2.0
                distance = math.hypot(cx - gcx, cy - gcy)
                if distance < best_distance:
                    best_distance = distance
                    best_name = name
            if best_name is None:
                continue
            if best_name in last and last[best_name] != obj.track_id:
                switches[best_name] += 1
            last[best_name] = obj.track_id
    return switches


def _only(published: Sequence[TrackedObject]) -> TrackedObject:
    assert len(published) == 1, "expected exactly one published track, got %d" % len(published)
    return published[0]


# --------------------------------------------------------------------------- #
# Scenario: a lead vehicle with known ground truth
# --------------------------------------------------------------------------- #


def _closing_lead(frames: int, start_m: float = 45.0, closing_mps: float = 6.0):
    """Ground-truth ranges and their projected detections for a closing lead."""
    ranges = [start_m - closing_mps * DT * index for index in range(frames)]
    boxes = [project(0.0, z) for z in ranges]
    return ranges, boxes


def test_lead_vehicle_range_and_closing_speed_against_ground_truth():
    tracker = MultiObjectTracker()
    ranges, boxes = _closing_lead(80)
    outputs = run_frames(tracker, [[box] for box in boxes])

    assert outputs[0] == [] and outputs[1] == [], "a track must not publish before confirmation"
    assert len(outputs[2]) == 1, "3 hits in 3 frames must confirm the track"

    track_ids = {obj.track_id for frame in outputs[2:] for obj in frame}
    assert track_ids == {1}, "the lead must keep one id for the whole run"

    # Accuracy budget, measured on this scenario: range within 1 m and closing
    # speed within 1.2 m/s once the filter has converged (~25 frames). The
    # limit is the pixel quantisation of the box height, not the filter: at
    # 37 m one pixel of box height is 1.05 m of range.
    for index in range(25, 80):
        obj = _only(outputs[index])
        assert obj.distance_m == pytest.approx(ranges[index], abs=1.0), (
            "range error at frame %d" % index
        )
        assert obj.velocity_mps == pytest.approx(6.0, abs=1.2), (
            "closing speed error at frame %d: %.2f" % (index, obj.velocity_mps)
        )
        assert obj.ttc_s == pytest.approx(ranges[index] / 6.0, rel=0.25)
        assert obj.time_since_update == 0
        assert obj.hits == index + 1
        assert obj.age_frames == index + 1


def test_lead_vehicle_range_estimate_reports_its_provenance():
    tracker = MultiObjectTracker()
    _ranges, boxes = _closing_lead(10)
    outputs = run_frames(tracker, [[box] for box in boxes])
    obj = _only(outputs[-1])
    assert obj.range_estimate is not None
    assert obj.range_estimate.source is RangeSource.PINHOLE
    assert obj.range_estimate.truncated is False
    assert 0.0 < obj.range_estimate.confidence <= 1.0


def test_a_stationary_object_does_not_manufacture_closing_speed():
    """ADAS-DEC-06: 1 px of box jitter at 30 m used to be 13.2 m/s of 'velocity'."""
    rng = np.random.RandomState(SEED)
    tracker = MultiObjectTracker()
    frames = [[project(0.0, 30.0, noise_px=1.0, rng=rng)] for _ in range(120)]
    outputs = run_frames(tracker, frames)

    converged = [abs(_only(frame).velocity_mps) for frame in outputs[25:]]
    rms = math.sqrt(sum(value * value for value in converged) / len(converged))
    assert max(converged) < 2.5, "filtered peak reached %.2f m/s" % max(converged)
    assert rms < 1.0, "filtered RMS reached %.2f m/s" % rms

    # The same measurement sequence, differenced raw the way the old tracker did.
    naive = []
    previous = None
    for detections in frames:
        box = detections[0]
        distance = 1.5 * FOCAL_PX / (box.y2 - box.y1)
        if previous is not None:
            naive.append(abs((previous - distance) / DT))
        previous = distance
    assert max(naive) > 5.0, "the scenario is not actually noisy; the test would be vacuous"


def test_a_stationary_object_holds_its_range():
    tracker = MultiObjectTracker()
    outputs = run_frames(tracker, [[project(0.0, 30.0)] for _ in range(40)])
    for frame in outputs[3:]:
        assert _only(frame).distance_m == pytest.approx(30.0, abs=0.5)


# --------------------------------------------------------------------------- #
# Scenario: occlusion
# --------------------------------------------------------------------------- #


def test_occlusion_within_max_missed_preserves_the_track_id():
    tracker = MultiObjectTracker(max_missed=5)
    ranges, boxes = _closing_lead(60)
    frames: List[List[BoundingBox]] = []
    for index, box in enumerate(boxes):
        frames.append([] if 30 <= index <= 33 else [box])
    outputs = run_frames(tracker, frames)

    assert {obj.track_id for frame in outputs[2:] for obj in frame} == {1}
    for index in range(30, 34):
        obj = _only(outputs[index])
        assert obj.time_since_update == index - 29, "the coast must be visible to consumers"
        assert obj.range_estimate.source is RangeSource.UNAVAILABLE, (
            "there was no measurement this frame; the tracker must not pretend there was"
        )
    # Re-acquisition must not have lost the range.
    assert _only(outputs[34]).distance_m == pytest.approx(ranges[34], abs=1.5)
    assert _only(outputs[40]).velocity_mps == pytest.approx(6.0, abs=1.2)


def test_a_coasting_track_is_predicted_forward_not_frozen():
    """ADAS-DEC-16: a frozen box made a closing lead appear to stop closing."""
    tracker = MultiObjectTracker(max_missed=5)
    _ranges, boxes = _closing_lead(40)
    frames = [[box] if index < 30 else [] for index, box in enumerate(boxes)]
    outputs = run_frames(tracker, frames)

    last_measured = _only(outputs[29])
    first_coast = _only(outputs[30])
    third_coast = _only(outputs[32])
    assert first_coast.distance_m < last_measured.distance_m, "range must keep closing"
    assert third_coast.distance_m < first_coast.distance_m
    assert third_coast.box.y2 > last_measured.box.y2, "the predicted box must move"
    assert third_coast.velocity_mps == pytest.approx(6.0, abs=1.2)


def test_a_long_occlusion_deletes_the_track():
    tracker = MultiObjectTracker(max_missed=5)
    _ranges, boxes = _closing_lead(40)
    frames = [[box] if index < 20 or index >= 30 else [] for index, box in enumerate(boxes)]
    outputs = run_frames(tracker, frames)
    assert outputs[25] == [], "a track coasting past max_missed must be deleted"
    assert _only(outputs[-1]).track_id != 1, "re-acquisition after death is a new object"


def test_range_variance_grows_while_coasting():
    tracker = MultiObjectTracker(max_missed=5)
    _ranges, boxes = _closing_lead(40)
    frames = [[box] if index < 30 else [] for index, box in enumerate(boxes)]
    run_frames(tracker, frames[:30])
    fresh = tracker.diagnostics()[0]["range_sigma_m"]
    run_frames(tracker, frames[30:34])
    coasted = tracker.diagnostics()[0]["range_sigma_m"]
    assert coasted > fresh


# --------------------------------------------------------------------------- #
# Scenario: cut-in
# --------------------------------------------------------------------------- #


def test_a_close_cut_in_does_not_switch_ids():
    """The highest-risk manoeuvre for ACC, and the one the 120 px gate broke.

    The cut-in vehicle traverses 3.5 m of lateral offset at 8 m of range, which
    is 398 px of image motion -- more than the old fixed gate in total.
    """
    tracker = MultiObjectTracker()
    frames: List[List[BoundingBox]] = []
    truth: List[Dict[str, BoundingBox]] = []
    for index in range(45):
        lead_z = 25.0 - 2.0 * DT * index
        cut_x = 3.5 - 3.5 * min(1.0, max(0.0, (index - 10) / 15.0))
        lead = project(0.0, lead_z)
        cut = project(cut_x, 8.0)
        frames.append([lead, cut])
        truth.append({"lead": lead, "cut_in": cut})

    outputs = run_frames(tracker, frames)
    switches = id_switches(outputs, truth)
    assert switches == {"lead": 0, "cut_in": 0}, switches
    assert len(outputs[-1]) == 2


def test_two_crossing_objects_keep_their_ids():
    tracker = MultiObjectTracker()
    frames: List[List[BoundingBox]] = []
    truth: List[Dict[str, BoundingBox]] = []
    for index in range(40):
        # Same range, opposite lateral motion: their image paths cross at index 20.
        left = project(-4.0 + 0.2 * index, 20.0)
        right = project(4.0 - 0.2 * index, 22.0)
        frames.append([left, right])
        truth.append({"left": left, "right": right})
    outputs = run_frames(tracker, frames)
    switches = id_switches(outputs, truth, tolerance_px=40.0)
    assert switches == {"left": 0, "right": 0}, switches


# --------------------------------------------------------------------------- #
# Scenario: spurious detection (ADAS-DEC-09, phantom braking)
# --------------------------------------------------------------------------- #


def test_a_single_spurious_detection_is_never_published():
    tracker = MultiObjectTracker()
    frames: List[List[BoundingBox]] = []
    for index in range(40):
        detections = [project(0.0, 40.0)]
        if index == 20:
            # A road-surface patch detected once as a car with a huge box: the
            # old tracker turned this into a 4.5 m obstacle and commanded a
            # full-ABS brake on the very next command.
            detections.append(
                BoundingBox(x1=300.0, y1=350.0, x2=560.0, y2=650.0, confidence=0.36, label="car")
            )
        frames.append(detections)

    outputs = run_frames(tracker, frames)
    for index, published in enumerate(outputs[2:], start=2):
        assert len(published) == 1, "frame %d published %d tracks" % (index, len(published))
        assert published[0].distance_m > 30.0, (
            "frame %d published a %.1f m obstacle that never existed"
            % (index, published[0].distance_m)
        )
    # It did exist as a tentative track on the frame it was detected...
    assert tracker._next_track_id > 2
    # ...and it never reached confirmation.
    assert all(obj.hits >= 3 for frame in outputs for obj in frame)


def test_two_frames_of_a_phantom_are_still_not_enough():
    tracker = MultiObjectTracker(confirm_hits=3, confirm_window=5)
    phantom = BoundingBox(x1=300.0, y1=350.0, x2=560.0, y2=650.0, confidence=0.4, label="car")
    outputs = run_frames(tracker, [[phantom], [phantom], [], [], [], []])
    assert all(frame == [] for frame in outputs)
    assert tracker._tracks == {}


def test_the_confirmation_threshold_is_configurable_and_enforced():
    tracker = MultiObjectTracker(confirm_hits=1, confirm_window=1)
    outputs = run_frames(tracker, [[project(0.0, 30.0)]])
    assert len(outputs[0]) == 1, "confirm_hits=1 must publish immediately"

    with pytest.raises(ValidationError):
        MultiObjectTracker(confirm_hits=0)
    with pytest.raises(ValidationError):
        MultiObjectTracker(confirm_hits=4, confirm_window=3)


def test_tentative_tracks_are_available_for_overlay_but_not_for_the_planner():
    tracker = MultiObjectTracker()
    published = tracker.update([project(0.0, 30.0)], dt_s=DT, frame_width=WIDTH, frame_height=HEIGHT)
    assert published == []
    tentative = tracker.tentative_tracks()
    assert len(tentative) == 1
    assert tentative[0].hits == 1
    assert tracker.diagnostics()[0]["status"] == TrackStatus.TENTATIVE.value


# --------------------------------------------------------------------------- #
# Scenario: the ADAS-DEC-07 detection theft
# --------------------------------------------------------------------------- #


def test_a_distant_track_cannot_steal_the_near_lead_detection():
    """Track A is born first at 68 m; track B is the 6.8 m lead born later.

    Under greedy insertion-order matching A claimed B's detection and jumped
    68.3 m -> 6.8 m in one frame, producing 1230 m/s of closing speed and an
    immediate hard brake, while B died and was reborn with a new id.
    """
    tracker = MultiObjectTracker()
    far_x, near_x = -0.6, 0.6
    frames: List[List[BoundingBox]] = []
    truth: List[Dict[str, BoundingBox]] = []
    for index in range(30):
        far = project(far_x, 68.0 - 0.1 * index, confidence=0.6)
        # The near vehicle is created later, exactly as in the report.
        detections = [far]
        gt = {"far": far}
        if index >= 3:
            near = project(near_x, 6.8, confidence=0.9)
            detections.append(near)
            gt["near"] = near
        frames.append(detections)
        truth.append(gt)

    outputs = run_frames(tracker, frames)
    switches = id_switches(outputs, truth, tolerance_px=60.0)
    assert switches.get("far", 0) == 0
    assert switches.get("near", 0) == 0

    for published in outputs[8:]:
        by_id = {obj.track_id: obj for obj in published}
        assert len(by_id) == 2
        far_obj = min(published, key=lambda o: o.box.y2)
        near_obj = max(published, key=lambda o: o.box.y2)
        assert far_obj.distance_m > 50.0, "the distant track teleported to %.1f m" % far_obj.distance_m
        assert near_obj.distance_m < 10.0
        assert abs(far_obj.velocity_mps) < 5.0


def test_a_size_mismatch_alone_blocks_the_teleport():
    """Even with the near box put where the far track predicts, size vetoes it."""
    tracker = MultiObjectTracker()
    far = BoundingBox(x1=630.0, y1=290.0, x2=650.0, y2=310.0, confidence=0.9, label="car")
    for _ in range(5):
        tracker.update([far], dt_s=DT, frame_width=WIDTH, frame_height=HEIGHT)
    huge = BoundingBox(x1=600.0, y1=240.0, x2=700.0, y2=340.0, confidence=0.9, label="car")
    published = tracker.update([huge], dt_s=DT, frame_width=WIDTH, frame_height=HEIGHT)
    assert len(published) == 1
    assert published[0].track_id == 1
    assert published[0].distance_m > 40.0, "the far track absorbed a 5x larger box"
    assert published[0].time_since_update == 1, "the far track should have coasted instead"


def test_a_car_track_never_absorbs_a_bicycle_detection():
    tracker = MultiObjectTracker()
    for _ in range(5):
        tracker.update([project(0.0, 20.0, label="car")], dt_s=DT, frame_width=WIDTH, frame_height=HEIGHT)
    bicycle = project(0.0, 20.0, label="bicycle")
    published = tracker.update([bicycle], dt_s=DT, frame_width=WIDTH, frame_height=HEIGHT)
    assert len(published) == 1, "only the car track is confirmed"
    assert published[0].track_id == 1
    assert published[0].time_since_update == 1, "the car must coast, not adopt the bicycle"
    assert len(tracker._tracks) == 2, "the bicycle must start its own tentative track"


# --------------------------------------------------------------------------- #
# Range modelling
# --------------------------------------------------------------------------- #


def test_class_height_priors_fix_the_truck_range():
    """ADAS-DEC-10(a): one 1.5 m prior reported a 30 m truck at 12.9 m."""
    tracker = MultiObjectTracker()
    outputs = run_frames(tracker, [[project(0.0, 30.0, label="truck")] for _ in range(10)])
    assert _only(outputs[-1]).distance_m == pytest.approx(30.0, abs=1.5)

    single_prior = MultiObjectTracker(use_class_heights=False, object_height_m=1.5)
    legacy = run_frames(single_prior, [[project(0.0, 30.0, label="truck")] for _ in range(10)])
    assert _only(legacy[-1]).distance_m < 15.0, "documents the behaviour that was fixed"


def test_the_height_priors_agree_with_the_geometry_module():
    """The table is duplicated to keep tracking independent of perception; keep it honest."""
    geometry = pytest.importorskip("adas.perception.geometry")
    for label, height in NOMINAL_OBJECT_HEIGHT_M.items():
        assert geometry.NOMINAL_OBJECT_HEIGHT_M[label] == pytest.approx(height)


def test_a_vertically_truncated_box_is_flagged():
    """ADAS-DEC-10(b): the pinhole model over-reads on a clipped box, unsafely."""
    tracker = MultiObjectTracker()
    truncated = BoundingBox(
        x1=500.0, y1=400.0, x2=800.0, y2=float(HEIGHT - 1), confidence=0.9, label="car"
    )
    outputs = run_frames(tracker, [[truncated] for _ in range(5)])
    obj = _only(outputs[-1])
    assert obj.range_estimate.truncated is True
    assert obj.range_estimate.confidence < 0.9, "a truncated box must not report full confidence"
    assert tracker.diagnostics()[0]["range_truncated"] is True


def test_a_box_too_short_to_carry_a_range_reports_unavailable():
    """ADAS-DEC-10: a 1 px box used to yield 1365 m clipped to 200 m with no flag."""
    tracker = MultiObjectTracker()
    tiny = BoundingBox(x1=640.0, y1=358.0, x2=646.0, y2=363.0, confidence=0.5, label="car")
    outputs = run_frames(tracker, [[tiny] for _ in range(6)])
    obj = _only(outputs[-1])
    assert obj.range_estimate.source is RangeSource.UNAVAILABLE
    assert obj.range_estimate.confidence == 0.0
    assert obj.distance_m <= tracker.max_distance_m


def test_an_injected_range_channel_is_used_and_bounded():
    from adas.core.models import RangeEstimate

    calls = []

    def fake_range(box, frame_width, frame_height):
        calls.append((frame_width, frame_height))
        return RangeEstimate(12.0, 0.9, RangeSource.HOMOGRAPHY, False)

    tracker = MultiObjectTracker(range_fn=fake_range)
    outputs = run_frames(tracker, [[project(0.0, 40.0)] for _ in range(8)])
    obj = _only(outputs[-1])
    assert calls and calls[0] == (WIDTH, HEIGHT)
    assert obj.range_estimate.source is RangeSource.HOMOGRAPHY
    assert obj.distance_m == pytest.approx(12.0, abs=0.5)


def test_a_failing_range_channel_falls_back_instead_of_killing_the_frame():
    def broken(box, frame_width, frame_height):
        raise RuntimeError("depth engine died")

    tracker = MultiObjectTracker(range_fn=broken)
    outputs = run_frames(tracker, [[project(0.0, 30.0)] for _ in range(6)])
    obj = _only(outputs[-1])
    assert obj.range_estimate.source is RangeSource.PINHOLE
    assert obj.distance_m == pytest.approx(30.0, abs=1.0)


def test_a_range_channel_returning_rubbish_is_rejected():
    tracker = MultiObjectTracker(range_fn=lambda box, w, h: "twelve metres")
    outputs = run_frames(tracker, [[project(0.0, 30.0)] for _ in range(6)])
    assert _only(outputs[-1]).range_estimate.source is RangeSource.PINHOLE


def test_a_range_discontinuity_reinitialises_the_filter_rather_than_averaging_it():
    """A measurement far outside the gate means the filter is wrong, not the box.

    ``range_reinit_maha2`` is lowered here so the mechanism is exercised by a
    jump a monocular channel can plausibly produce. A jump large enough to also
    change the box size by more than ``max_size_ratio`` never reaches the range
    filter at all -- association rejects it first, which is
    ``test_a_size_mismatch_alone_blocks_the_teleport``.
    """
    tracker = MultiObjectTracker(range_reinit_maha2=9.0)
    run_frames(tracker, [[project(0.0, 60.0)] for _ in range(12)])
    before = tracker.diagnostics()[0]["distance_m"]
    outputs = run_frames(tracker, [[project(0.0, 40.0)] for _ in range(6)])
    assert before == pytest.approx(60.0, abs=2.0)
    assert tracker.diagnostics()[0]["range_reinits"] >= 1
    assert _only(outputs[-1]).distance_m == pytest.approx(40.0, abs=2.0)
    assert _only(outputs[-1]).track_id == 1, "the track itself must survive"


# --------------------------------------------------------------------------- #
# Time to collision
# --------------------------------------------------------------------------- #


def test_ttc_degenerate_cases():
    tracker = MultiObjectTracker()
    assert math.isinf(tracker.time_to_collision_s(50.0, 0.0)), "a static gap never collides"
    assert math.isinf(tracker.time_to_collision_s(50.0, -3.0)), "an opening range never collides"
    assert math.isinf(
        tracker.time_to_collision_s(50.0, 0.05)
    ), "a closing rate below the floor is not a collision course"
    assert tracker.time_to_collision_s(0.0, 5.0) == 0.0, "zero range is contact"
    assert tracker.time_to_collision_s(-1.0, 5.0) == 0.0, "a negative range is contact, not -0.2 s"
    assert math.isinf(tracker.time_to_collision_s(float("nan"), 5.0))
    assert math.isinf(tracker.time_to_collision_s(50.0, float("nan")))
    assert tracker.time_to_collision_s(30.0, 10.0) == pytest.approx(3.0)


def test_ttc_on_a_real_scenario_matches_the_ground_truth():
    tracker = MultiObjectTracker()
    ranges, boxes = _closing_lead(60, start_m=40.0, closing_mps=10.0)
    outputs = run_frames(tracker, [[box] for box in boxes])
    for index in range(25, 60):
        expected = ranges[index] / 10.0
        assert _only(outputs[index]).ttc_s == pytest.approx(expected, rel=0.2)


def test_ttc_is_infinite_for_a_receding_object():
    tracker = MultiObjectTracker()
    frames = [[project(0.0, 20.0 + 0.5 * index)] for index in range(40)]
    outputs = run_frames(tracker, frames)
    assert math.isinf(_only(outputs[-1]).ttc_s)
    assert _only(outputs[-1]).velocity_mps < 0.0, "a receding object has a negative closing speed"


# --------------------------------------------------------------------------- #
# Ego-lane decision
# --------------------------------------------------------------------------- #


def test_in_ego_lane_uses_the_metric_lane_geometry_when_it_is_available():
    lane = straight_lane_model()
    inside = MultiObjectTracker()
    outputs = run_frames(inside, [[project(0.0, 30.0)] for _ in range(5)], lane=lane)
    assert _only(outputs[-1]).in_ego_lane is True
    assert inside.diagnostics()[0]["ego_lane_source"] == "lane_geometry"

    outside = MultiObjectTracker()
    outputs = run_frames(outside, [[project(5.0, 30.0)] for _ in range(5)], lane=lane)
    assert _only(outputs[-1]).in_ego_lane is False
    assert outside.diagnostics()[0]["ego_lane_source"] == "lane_geometry"


def test_a_partial_cut_in_counts_as_in_lane():
    """Half a vehicle over the line is in the ego path; the test is occupancy, not centres."""
    lane = straight_lane_model()
    tracker = MultiObjectTracker()
    outputs = run_frames(tracker, [[project(2.0, 20.0)] for _ in range(5)], lane=lane)
    assert _only(outputs[-1]).in_ego_lane is True


def test_a_mock_or_low_confidence_lane_is_not_used_to_gate():
    mock = straight_lane_model()
    mock.is_mock = True
    tracker = MultiObjectTracker()
    run_frames(tracker, [[project(5.0, 30.0)] for _ in range(5)], lane=mock)
    assert tracker.diagnostics()[0]["ego_lane_source"] != "lane_geometry"

    weak = straight_lane_model(confidence=0.05)
    tracker = MultiObjectTracker()
    run_frames(tracker, [[project(5.0, 30.0)] for _ in range(5)], lane=weak)
    assert tracker.diagnostics()[0]["ego_lane_source"] != "lane_geometry"


def test_an_implausible_lane_width_is_not_used_to_gate():
    lane = straight_lane_model(half_width_m=6.0)  # 12 m wide: a mis-association
    tracker = MultiObjectTracker()
    run_frames(tracker, [[project(0.0, 30.0)] for _ in range(5)], lane=lane)
    assert tracker.diagnostics()[0]["ego_lane_source"] != "lane_geometry"


def test_in_ego_lane_falls_back_to_the_drivable_corridor():
    drivable = corridor_drivable_area()
    tracker = MultiObjectTracker()
    run_frames(tracker, [[project(0.0, 30.0)] for _ in range(5)], drivable=drivable)
    assert tracker.diagnostics()[0]["ego_lane_source"] == "drivable_area"
    assert tracker.diagnostics()[0]["in_ego_lane"] is True

    far_left = MultiObjectTracker()
    run_frames(far_left, [[project(-12.0, 25.0)] for _ in range(5)], drivable=drivable)
    diagnostics = far_left.diagnostics()[0]
    assert diagnostics["ego_lane_source"] == "drivable_area"
    assert diagnostics["in_ego_lane"] is False


def test_an_empty_drivable_mask_is_not_treated_as_a_measurement():
    empty = DrivableArea(mask=None, width=0, height=0, confidence=0.0)
    tracker = MultiObjectTracker()
    run_frames(tracker, [[project(0.0, 30.0)] for _ in range(5)], drivable=empty)
    assert tracker.diagnostics()[0]["ego_lane_source"] != "drivable_area"


def test_the_last_resort_heuristic_is_inclusive_and_labelled():
    """With no lane and no mask the fallback must not hide the lead vehicle."""
    tracker = MultiObjectTracker()
    run_frames(tracker, [[project(0.6, 30.0)] for _ in range(5)])
    diagnostics = tracker.diagnostics()[0]
    assert diagnostics["in_ego_lane"] is True
    assert diagnostics["ego_lane_source"] == "ego_corridor_heuristic"

    wide = MultiObjectTracker()
    run_frames(wide, [[project(7.0, 30.0)] for _ in range(5)])
    assert wide.diagnostics()[0]["in_ego_lane"] is False


def test_without_a_frame_size_the_tracker_declines_to_guess():
    tracker = MultiObjectTracker()
    for _ in range(5):
        published = tracker.update([project(0.0, 30.0)], dt_s=DT)
    diagnostics = tracker.diagnostics()[0]
    assert diagnostics["lateral_offset_known"] is False
    assert diagnostics["ego_lane_source"] == "unavailable"
    assert published[0].in_ego_lane is False
    assert published[0].lateral_offset_m == 0.0


def test_lateral_offset_is_metric_and_signed():
    tracker = MultiObjectTracker()
    run_frames(tracker, [[project(2.5, 30.0)] for _ in range(5)])
    assert tracker.diagnostics()[0]["lateral_offset_m"] == pytest.approx(2.5, abs=0.15)

    left = MultiObjectTracker()
    run_frames(left, [[project(-2.5, 30.0)] for _ in range(5)])
    assert left.diagnostics()[0]["lateral_offset_m"] == pytest.approx(-2.5, abs=0.15)


def test_an_injected_camera_supplies_the_lateral_offset():
    class FakeCamera:
        def image_to_ground(self, u_px, v_px):
            return (-3.25, 42.0)

    tracker = MultiObjectTracker(camera=FakeCamera())
    run_frames(tracker, [[project(0.0, 30.0)] for _ in range(5)])
    assert tracker.diagnostics()[0]["lateral_offset_m"] == pytest.approx(-3.25)


def test_a_failing_camera_falls_back_rather_than_raising():
    class BrokenCamera:
        def image_to_ground(self, u_px, v_px):
            raise RuntimeError("extrinsics not loaded")

    tracker = MultiObjectTracker(camera=BrokenCamera())
    run_frames(tracker, [[project(1.0, 30.0)] for _ in range(5)])
    assert tracker.diagnostics()[0]["lateral_offset_m"] == pytest.approx(1.0, abs=0.15)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #


def test_variable_dt_does_not_corrupt_the_closing_speed():
    """ADAS-DEC-17: the pipeline passed the scheduled period, not the elapsed one."""
    rng = np.random.RandomState(SEED)
    tracker = MultiObjectTracker()
    distance = 45.0
    errors = []
    for index in range(90):
        dt = float(rng.uniform(0.03, 0.12))
        distance -= 6.0 * dt
        published = tracker.update(
            [project(0.0, distance)],
            dt_s=dt,
            frame_width=WIDTH,
            frame_height=HEIGHT,
        )
        if index > 30:
            obj = _only(published)
            errors.append(abs(obj.velocity_mps - 6.0))
            assert obj.distance_m == pytest.approx(distance, abs=1.2)
    assert max(errors) < 1.5, "worst closing-speed error %.2f m/s" % max(errors)


def test_an_absurd_dt_is_clamped_not_propagated():
    tracker = MultiObjectTracker()
    run_frames(tracker, [[project(0.0, 30.0)] for _ in range(5)])
    published = tracker.update(
        [project(0.0, 30.0)], dt_s=1.0e6, frame_width=WIDTH, frame_height=HEIGHT
    )
    obj = _only(published)
    assert math.isfinite(obj.distance_m) and math.isfinite(obj.velocity_mps)
    assert abs(obj.velocity_mps) <= 90.0


def test_a_non_positive_or_non_numeric_dt_uses_the_nominal_period():
    tracker = MultiObjectTracker()
    run_frames(tracker, [[project(0.0, 30.0)] for _ in range(4)])
    for bad in (0.0, -1.0, float("nan"), "fast"):
        published = tracker.update(
            [project(0.0, 30.0)], dt_s=bad, frame_width=WIDTH, frame_height=HEIGHT
        )
        assert math.isfinite(_only(published).distance_m)


# --------------------------------------------------------------------------- #
# Robustness and API
# --------------------------------------------------------------------------- #


def test_the_legacy_call_signature_still_works():
    """``ADASPipeline.step`` calls ``update(detections, dt_s=dt)``; do not break it."""
    tracker = MultiObjectTracker()
    for _ in range(4):
        published = tracker.update([project(0.0, 25.0)], dt_s=0.05)
    assert len(published) == 1
    assert published[0].distance_m == pytest.approx(25.0, abs=1.0)


def test_the_config_wired_constructor_arguments_are_unchanged():
    tracker = MultiObjectTracker(
        max_missed=3,
        association_threshold_px=90.0,
        focal_length_px=800.0,
        object_height_m=1.6,
        min_box_height_px=2.0,
        max_distance_m=150.0,
    )
    assert tracker.max_missed == 3
    assert tracker.association.max_centre_distance_px == 90.0
    assert tracker.max_distance_m == 150.0


def test_a_zero_min_box_height_is_rejected_at_construction():
    """The old code divided by it; TrackerConfig never validated it."""
    with pytest.raises(ValidationError):
        MultiObjectTracker(min_box_height_px=0.0)


def test_an_invalid_detection_is_skipped_without_failing_the_frame():
    tracker = MultiObjectTracker()
    good = project(0.0, 30.0)
    bad = BoundingBox(x1=100.0, y1=100.0, x2=50.0, y2=200.0, confidence=0.9, label="car")
    negative = BoundingBox(x1=-5.0, y1=10.0, x2=50.0, y2=200.0, confidence=0.9, label="car")
    for _ in range(4):
        published = tracker.update(
            [good, bad, negative], dt_s=DT, frame_width=WIDTH, frame_height=HEIGHT
        )
    assert len(published) == 1


def test_none_detections_is_a_tracking_error():
    tracker = MultiObjectTracker()
    with pytest.raises(TrackingError):
        tracker.update(None, dt_s=DT)


def test_an_empty_frame_is_fine():
    tracker = MultiObjectTracker()
    assert tracker.update([], dt_s=DT) == []


def test_reset_clears_everything():
    tracker = MultiObjectTracker()
    run_frames(tracker, [[project(0.0, 30.0)] for _ in range(6)])
    assert tracker._tracks
    tracker.reset()
    assert tracker._tracks == {}
    assert tracker._next_track_id == 1
    assert tracker.diagnostics() == []


def test_output_is_ordered_by_track_id():
    tracker = MultiObjectTracker()
    frames = [[project(-3.0, 30.0), project(3.0, 20.0)] for _ in range(5)]
    published = run_frames(tracker, frames)[-1]
    assert [obj.track_id for obj in published] == sorted(obj.track_id for obj in published)


def test_diagnostics_expose_what_trackedobject_has_no_field_for():
    tracker = MultiObjectTracker()
    run_frames(tracker, [[project(0.0, 30.0)] for _ in range(6)])
    diagnostics = tracker.diagnostics()
    assert len(diagnostics) == 1
    entry = diagnostics[0]
    for key in (
        "range_sigma_m",
        "range_rate_sigma_mps",
        "gate_radius_px",
        "ego_lane_source",
        "range_reinits",
        "filter_saturations",
        "recent_match_ratio",
    ):
        assert key in entry, "missing diagnostic %r" % key
    assert entry["range_sigma_m"] > 0.0
    assert entry["gate_radius_px"] > 0.0
    assert entry["recent_match_ratio"] == pytest.approx(1.0)


def test_the_detection_budget_keeps_the_weakest_boxes_out():
    """A detector fault must not turn one frame into a multi-second solve."""
    tracker = MultiObjectTracker(max_detections=4, confirm_hits=1, confirm_window=1)
    detections = [
        project(-9.0 + 1.5 * index, 25.0, confidence=0.10 + 0.05 * index) for index in range(12)
    ]
    published = tracker.update(
        detections, dt_s=DT, frame_width=WIDTH, frame_height=HEIGHT
    )
    assert len(published) == 4, "only max_detections boxes may become tracks"
    kept = sorted(obj.box.confidence for obj in published)
    assert kept == pytest.approx([0.50, 0.55, 0.60, 0.65]), (
        "the surviving detections must be the most confident ones, got %r" % kept
    )


def test_the_track_budget_sacrifices_tentative_tracks_first():
    tracker = MultiObjectTracker(max_tracks=4)
    # Four steady objects earn confirmation.
    steady = [project(-6.0 + 3.0 * index, 25.0) for index in range(4)]
    for _ in range(5):
        published = tracker.update(steady, dt_s=DT, frame_width=WIDTH, frame_height=HEIGHT)
    assert len(published) == 4
    confirmed_ids = {obj.track_id for obj in published}

    # Now flood the frame with new objects; the confirmed four must survive.
    flood = steady + [project(-9.0 + 0.7 * index, 60.0) for index in range(10)]
    published = tracker.update(flood, dt_s=DT, frame_width=WIDTH, frame_height=HEIGHT)
    assert len(tracker._tracks) <= 4
    assert {obj.track_id for obj in published} == confirmed_ids


def test_the_constructor_rejects_an_out_of_range_budget():
    with pytest.raises(ValidationError):
        MultiObjectTracker(max_detections=0)
    with pytest.raises(ValidationError):
        MultiObjectTracker(max_tracks=0)
    with pytest.raises(ValidationError):
        MultiObjectTracker(max_detections=10 ** 6)


def test_many_objects_do_not_break_the_assignment():
    tracker = MultiObjectTracker()
    frames = []
    for index in range(8):
        frames.append(
            [project(-9.0 + 1.5 * lane_index, 25.0 + index * 0.1) for lane_index in range(12)]
        )
    published = run_frames(tracker, frames)[-1]
    assert len(published) == 12
    assert len({obj.track_id for obj in published}) == 12
