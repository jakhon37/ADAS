"""Tests for :mod:`adas.runtime.pipeline`.

These run without a GPU: the detector and lane estimator are fakes, so what is
under test is the ORCHESTRATION -- who is authoritative, what a failure means,
and what reaches the actuators -- not any model.
"""

from __future__ import annotations

import time

import pytest

from adas.control import PIDLikeLongitudinalController, SafetyLimits, SafetyMonitor
from adas.core.exceptions import ADASException, PerceptionError
from adas.core.models import (
    BoundingBox,
    ControlCommand,
    EgoState,
    LaneLine,
    LaneModel,
    MotionPlan,
    PerceptionFrame,
    SafetyState,
    TrackedObject,
)
from adas.planning import BehaviorPlanner
from adas.runtime import ADASPipeline, synthetic_frame
from adas.tracking import MultiObjectTracker

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeDetector:
    """Returns a scripted detection list, or raises on scripted frames."""

    is_mock = False

    def __init__(self, boxes=None, fail_on=()):
        self.boxes = list(boxes or [])
        self.fail_on = set(fail_on)
        self.calls = 0

    def infer(self, frame, width, height):
        self.calls += 1
        if (self.calls - 1) in self.fail_on:
            raise PerceptionError("scripted detector failure")
        return list(self.boxes)


class FakeLane:
    """Returns a scripted lane model, counting how often it is asked."""

    name = "fake"
    is_mock = False

    def __init__(self, model=None, fail_on=()):
        self.model = model
        self.fail_on = set(fail_on)
        self.calls = 0

    def estimate(self, frame, width, height):
        index = self.calls
        self.calls += 1
        if index in self.fail_on:
            raise PerceptionError("scripted lane failure")
        return self.model

    def drivable_area(self):
        return None


def lane_model(center_px=640.0, confidence=0.9):
    return LaneModel(
        left_coeffs=(0.0, 0.0, 540.0),
        right_coeffs=(0.0, 0.0, 740.0),
        lane_center_px=center_px,
        curvature_m=10000.0,
        confidence=confidence,
        lines=[LaneLine(index=1), LaneLine(index=2)],
        is_mock=False,
    )


def build(detector=None, lane=None, **kwargs) -> ADASPipeline:
    return ADASPipeline(
        detector=detector or FakeDetector(),
        lane_estimator=lane or FakeLane(),
        tracker=MultiObjectTracker(frame_width_px=1280, frame_height_px=720),
        planner=BehaviorPlanner(),
        controller=PIDLikeLongitudinalController(),
        safety_monitor=SafetyMonitor(limits=SafetyLimits()),
        **kwargs
    )


def frame(frame_id=0, width=1280, height=720):
    return PerceptionFrame(
        frame_id=frame_id,
        timestamp_s=time.monotonic(),
        rgb=synthetic_frame(width, height),
        width=width,
        height=height,
    )


def car(x1=560.0, y1=300.0, x2=720.0, y2=420.0, conf=0.9, label="car"):
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, label=label)


# --------------------------------------------------------------------------- #
# Smoke
# --------------------------------------------------------------------------- #


def test_pipeline_synthetic_smoke():
    pipeline = build(lane=FakeLane(lane_model()))
    plan, command = pipeline.step(frame(), current_speed_mps=10.0, dt_s=0.05)

    assert -22.0 <= plan.steering_angle_deg <= 22.0
    assert 0.0 <= command.throttle <= 1.0
    assert 0.0 <= command.brake <= 1.0
    assert -1.0 <= command.steering <= 1.0
    assert pipeline.last_arbitration is not None


def test_empty_road_lets_the_planner_accelerate():
    """No detections from a WORKING detector means the road is clear."""
    pipeline = build(lane=FakeLane(lane_model()))
    ego = EgoState(speed_mps=10.0, valid=True, timestamp_s=time.monotonic())
    plan, _cmd = pipeline.step(frame(), ego=ego, dt_s=0.05)

    assert plan.target_speed_mps > 0.0
    assert pipeline.perception.ok


# --------------------------------------------------------------------------- #
# The arbiter is authoritative (ADAS-DEC-01)
# --------------------------------------------------------------------------- #


class RunawayController(PIDLikeLongitudinalController):
    """A controller that always commands full throttle, whatever the plan says."""

    def to_command(self, plan, current_speed_mps, dt_s=0.05, emergency=False):
        return ControlCommand(throttle=1.0, brake=0.0, steering=0.0)


def test_arbiter_command_is_what_the_pipeline_returns():
    pipeline = build(lane=FakeLane(lane_model()))
    pipeline.controller = RunawayController()
    # A lead 3 m ahead at 20 m/s is a hazard the arbiter must not permit throttle for.
    pipeline.detector = FakeDetector([car(y1=200.0, y2=700.0)])
    ego = EgoState(speed_mps=20.0, valid=True, timestamp_s=time.monotonic())

    for i in range(6):  # let the tracker confirm the track
        _plan, command = pipeline.step(frame(i), ego=ego, dt_s=0.05)

    assert pipeline.last_arbitration is not None
    assert command is pipeline.last_arbitration.command
    assert command.throttle == 0.0, "the arbiter must veto the runaway throttle"
    assert pipeline.last_arbitration.state is not SafetyState.NOMINAL


def test_invalid_ego_state_forces_a_degraded_state():
    """No ego speed is a fault, not a reason to guess."""
    pipeline = build(lane=FakeLane(lane_model()))
    _plan, command = pipeline.step(frame(), dt_s=0.05)  # no ego, no speed

    assert pipeline.last_ego is not None and not pipeline.last_ego.valid
    assert pipeline.last_arbitration.state is not SafetyState.NOMINAL
    assert command.throttle == 0.0


def test_explicit_speed_argument_is_treated_as_asserted_and_valid():
    pipeline = build(lane=FakeLane(lane_model()))
    pipeline.step(frame(), current_speed_mps=12.0, dt_s=0.05)
    assert pipeline.last_ego.valid
    assert pipeline.last_ego.speed_mps == pytest.approx(12.0)


# --------------------------------------------------------------------------- #
# A perception failure is a fault, not an empty road (ADAS-PERC-08)
# --------------------------------------------------------------------------- #


def test_detector_failure_sets_perception_status_and_withholds_tracks():
    detector = FakeDetector([car()], fail_on={0})
    pipeline = build(detector=detector, lane=FakeLane(lane_model()))
    ego = EgoState(speed_mps=15.0, valid=True, timestamp_s=time.monotonic())

    perception_frame = frame()
    plan, _cmd = pipeline.step(perception_frame, ego=ego, dt_s=0.05)

    assert perception_frame.status is not None
    assert perception_frame.status.ok is False
    assert perception_frame.status.detector_ok is False
    assert perception_frame.status.consecutive_failures == 1
    assert "detector" in perception_frame.status.reason
    assert pipeline.last_tracks == []
    assert "degraded" in plan.reason or "dropout" in plan.reason


def test_a_perception_failure_deletes_no_track():
    detector = FakeDetector([car()], fail_on={3})
    pipeline = build(detector=detector, lane=FakeLane(lane_model()))
    ego = EgoState(speed_mps=15.0, valid=True, timestamp_s=time.monotonic())

    for i in range(3):
        pipeline.step(frame(i), ego=ego, dt_s=0.05)
    before = len(pipeline.tracker._tracks)

    pipeline.step(frame(3), ego=ego, dt_s=0.05)
    assert len(pipeline.tracker._tracks) == before, "a fault must not delete tracks"


def test_a_perception_dropout_coasts_the_tracker_instead_of_freezing_it():
    """The regression: a dropout used to leave the tracker completely untouched.

    Every track then kept its pre-dropout position AND ``time_since_update == 0``
    and an unchanged covariance for the whole dropout, so the recovery frame
    associated a real measurement against an N-frame-stale prediction that still
    claimed full confidence.  Reviewer probe, pre-fix, over a 5-frame dropout::

        6  True | id=1 age= 6 tsu= 0 d=  9.45 sig= 0.11
        ...
       10  True | id=1 age= 6 tsu= 0 d=  9.45 sig= 0.11

    The tracker must be advanced by a PREDICT with no measurement update.
    """
    detector = FakeDetector([car()], fail_on={5, 6, 7})
    pipeline = build(detector=detector, lane=FakeLane(lane_model()))
    ego = EgoState(speed_mps=15.0, valid=True, timestamp_s=time.monotonic())

    for i in range(5):
        pipeline.step(frame(i), ego=ego, dt_s=0.05)
    measured = pipeline.tracker.diagnostics()[0]
    assert measured["time_since_update"] == 0

    seen = []
    for i in range(5, 8):
        pipeline.step(frame(i), ego=ego, dt_s=0.05)
        assert not pipeline.perception.ok, "precondition: this frame is a dropout"
        seen.append(pipeline.tracker.diagnostics()[0])

    ages = [d["age_frames"] for d in seen]
    missed = [d["time_since_update"] for d in seen]
    sigmas = [d["range_sigma_m"] for d in seen]

    assert ages == [measured["age_frames"] + n for n in (1, 2, 3)], "the track must age"
    assert missed == [1, 2, 3], "each dropout frame is one missed measurement"
    assert sigmas[0] > measured["range_sigma_m"], "the covariance must grow while coasting"
    assert sigmas == sorted(sigmas), "and keep growing for as long as the coast lasts"


def test_coasted_tracks_are_published_as_coasted_never_as_measured():
    """Coasting is only honest if the consumer can tell. Mark it, do not hide it."""
    from adas.core.models import RangeSource

    detector = FakeDetector([car()], fail_on={5})
    pipeline = build(detector=detector, lane=FakeLane(lane_model()))
    ego = EgoState(speed_mps=15.0, valid=True, timestamp_s=time.monotonic())

    for i in range(5):
        pipeline.step(frame(i), ego=ego, dt_s=0.05)
    assert pipeline.last_tracks, "precondition: a confirmed track exists"
    assert all(obj.time_since_update == 0 for obj in pipeline.last_tracks)

    plan, _cmd = pipeline.step(frame(5), ego=ego, dt_s=0.05)

    assert pipeline.last_tracks, "the dropout frame must still report the coasting track"
    for obj in pipeline.last_tracks:
        assert obj.time_since_update >= 1, "a coasted track must not look measured"
        assert obj.range_estimate.source is RangeSource.UNAVAILABLE
    # And the planner is still told the truth about perception, independently.
    assert "degraded" in plan.reason or "dropout" in plan.reason


def test_a_tracker_failure_on_a_dropout_frame_does_not_escalate():
    """Coasting is best-effort on a frame that has already failed."""
    detector = FakeDetector([car()], fail_on={0})
    pipeline = build(detector=detector, lane=FakeLane(lane_model()))

    def explode(*args, **kwargs):
        raise RuntimeError("scripted tracker failure")

    pipeline.tracker.update = explode
    ego = EgoState(speed_mps=15.0, valid=True, timestamp_s=time.monotonic())

    plan, command = pipeline.step(frame(0), ego=ego, dt_s=0.05)
    assert pipeline.last_tracks == []
    assert command.throttle == 0.0


def test_consecutive_failures_accumulate_and_reset():
    detector = FakeDetector(fail_on={0, 1, 2})
    pipeline = build(detector=detector, lane=FakeLane(lane_model()))
    ego = EgoState(speed_mps=10.0, valid=True, timestamp_s=time.monotonic())

    for i in range(3):
        pipeline.step(frame(i), ego=ego, dt_s=0.05)
    assert pipeline.perception.consecutive_failures == 3

    pipeline.step(frame(3), ego=ego, dt_s=0.05)
    assert pipeline.perception.ok
    assert pipeline.perception.consecutive_failures == 0


def test_lane_failure_is_isolated_from_the_detector():
    pipeline = build(detector=FakeDetector([car()]), lane=FakeLane(lane_model(), fail_on={0}))
    ego = EgoState(speed_mps=10.0, valid=True, timestamp_s=time.monotonic())

    perception_frame = frame()
    pipeline.step(perception_frame, ego=ego, dt_s=0.05)

    assert perception_frame.status.detector_ok is True
    assert perception_frame.status.lane_ok is False
    assert perception_frame.lane is None
    assert perception_frame.detections, "the detector's output must survive a lane fault"


# --------------------------------------------------------------------------- #
# Lane scheduling
# --------------------------------------------------------------------------- #


def test_lane_backend_runs_on_its_schedule():
    lane = FakeLane(lane_model())
    pipeline = build(lane=lane, lane_every_n_frames=4, lane_max_age_frames=3)
    ego = EgoState(speed_mps=10.0, valid=True, timestamp_s=time.monotonic())

    for i in range(8):
        pipeline.step(frame(i), ego=ego, dt_s=0.05)

    assert lane.calls == 2, "8 frames at every_n=4 is 2 inferences"


def test_a_reused_lane_model_loses_confidence_with_age():
    lane = FakeLane(lane_model(confidence=0.8))
    pipeline = build(lane=lane, lane_every_n_frames=4, lane_max_age_frames=3)
    ego = EgoState(speed_mps=10.0, valid=True, timestamp_s=time.monotonic())

    fresh = frame(0)
    pipeline.step(fresh, ego=ego, dt_s=0.05)
    assert fresh.lane.confidence == pytest.approx(0.8)

    stale = frame(1)
    pipeline.step(stale, ego=ego, dt_s=0.05)
    assert stale.lane is not None
    assert stale.lane.confidence < 0.8, "a reused model must not claim to be fresh"


def test_a_lane_model_older_than_max_age_is_dropped():
    lane = FakeLane(lane_model())
    pipeline = build(lane=lane, lane_every_n_frames=10, lane_max_age_frames=2)
    ego = EgoState(speed_mps=10.0, valid=True, timestamp_s=time.monotonic())

    pipeline.step(frame(0), ego=ego, dt_s=0.05)
    last = None
    for i in range(1, 6):
        last = frame(i)
        pipeline.step(last, ego=ego, dt_s=0.05)
    assert last.lane is None, "stale evidence must expire, not be re-served forever"


# --------------------------------------------------------------------------- #
# Fail-safe
# --------------------------------------------------------------------------- #


def test_failsafe_command_brakes_and_never_throttles():
    pipeline = build(lane=FakeLane(lane_model()))
    pipeline.step(frame(), current_speed_mps=15.0, dt_s=0.05)

    command = pipeline.failsafe_command(dt_s=0.05)
    assert command.throttle == 0.0
    assert command.brake > 0.0
    assert pipeline.last_arbitration.state in (
        SafetyState.LIMITED, SafetyState.MIN_RISK_MANEUVER, SafetyState.DISENGAGE
    )


def test_failsafe_command_advances_the_arbiters_timing_baseline():
    """The regression: the fail-safe context carried no ``timestamp_s``.

    The arbiter advances ``_last_timestamp_s`` only from that field, so a run of
    fail-safe frames left it stuck at the last good frame and the recovery frame
    presented a gap of N*dt -- reported as ``timing_stale_input`` and forced to
    ``max_dt_s``, corrupting every rate estimate at the worst possible moment.
    """
    pipeline = build(lane=FakeLane(lane_model()))
    arbiter = pipeline.safety_monitor.arbiter
    period_s = 0.05

    pipeline.step(frame(), current_speed_mps=15.0, dt_s=period_s)
    after_good_frame = arbiter._last_timestamp_s
    assert after_good_frame is not None

    # Paced for real: the defect is about wall-clock time passing unrecorded, so
    # the burst has to actually take time.
    burst = 5
    for _ in range(burst):
        time.sleep(period_s)
        pipeline.failsafe_command(dt_s=period_s)

    advanced_by = arbiter._last_timestamp_s - after_good_frame
    assert advanced_by >= burst * period_s * 0.8, (
        "a burst of fail-safe frames must keep the arbiter's timing baseline "
        "current; it advanced by only %.3f s over %.3f s of wall clock"
        % (advanced_by, burst * period_s)
    )

    # The recovery frame is then a normal-length step, not an N*dt stale input.
    time.sleep(period_s)
    pipeline.step(frame(1), current_speed_mps=15.0, dt_s=period_s)
    stale = [v for v in pipeline.last_arbitration.violations if v.startswith("timing_stale")]
    assert stale == [], "recovery must not be flagged stale: %s" % stale


def test_a_tracker_failure_raises_adas_exception():
    class ExplodingTracker(MultiObjectTracker):
        def update(self, *args, **kwargs):
            raise RuntimeError("boom")

    pipeline = build(lane=FakeLane(lane_model()))
    pipeline.tracker = ExplodingTracker()
    with pytest.raises(ADASException):
        pipeline.step(frame(), current_speed_mps=10.0, dt_s=0.05)


def test_reset_clears_cross_frame_state():
    pipeline = build(detector=FakeDetector([car()]), lane=FakeLane(lane_model()))
    ego = EgoState(speed_mps=10.0, valid=True, timestamp_s=time.monotonic())
    for i in range(4):
        pipeline.step(frame(i), ego=ego, dt_s=0.05)

    pipeline.reset()
    assert pipeline.last_arbitration is None
    assert pipeline.last_tracks == []
    assert pipeline.perception.ok
    assert pipeline.metrics.total_frames == 0
    assert pipeline.safety_monitor.state is SafetyState.NOMINAL


# --------------------------------------------------------------------------- #
# Planner behaviour through the pipeline
# --------------------------------------------------------------------------- #


def test_planner_slows_for_close_vehicle():
    pipeline = build()
    close = TrackedObject(
        track_id=1,
        box=car(),
        velocity_mps=0.0,
        distance_m=5.0,
        in_ego_lane=True,
    )
    plan = pipeline.planner.plan(
        frame_width_px=1280,
        lane_center_px=640.0,
        objects=[close],
        ego=EgoState(speed_mps=15.0, valid=True),
    )
    assert plan.target_speed_mps < pipeline.planner.cruise_speed_mps
    assert "follow" in plan.reason or "aeb" in plan.reason


def test_metrics_record_every_stage():
    pipeline = build(lane=FakeLane(lane_model()))
    pipeline.step(frame(), current_speed_mps=10.0, dt_s=0.05)
    for stage in ("detect", "lane", "track", "plan", "control", "arbitrate"):
        assert stage in pipeline.metrics.stages, "stage %r was not timed" % stage
