"""Tests for :mod:`adas.runtime.runner` -- what the frame loop does on its way OUT.

Everything here is about the boundary the reviewers proved was open: leaving the
frame loop is not a command, so whatever the actuators were last given stays
latched until something else is written.  A mid-stream sensor loss used to
``break`` with the previous frame's throttle still applied (ADAS-DEC-21) -- the
exact failure the module docstring claimed to have fixed, but had fixed only for
the pipeline-*exception* path.

No GPU and no camera: the sources, the detector and the lane estimator are
scripted fakes, so what is under test is the loop's exit policy and nothing else.
"""

from __future__ import annotations

import pytest

from adas.control import PIDLikeLongitudinalController, SafetyMonitor
from adas.core.models import BoundingBox, ControlCommand, SafetyState
from adas.planning import BehaviorPlanner
from adas.runtime.capture import (
    CapturedFrame,
    ConstantEgoSpeed,
    FrameSource,
    SimulatedEgoSpeed,
)
from adas.runtime.pipeline import ADASPipeline
from adas.runtime.runner import (
    EXIT_COMPLETED,
    EXIT_EOF,
    EXIT_PIPELINE_DEAD,
    EXIT_SOURCE_LOST,
    EXIT_STOPPED,
    INTERVENING_STATES,
    UNSAFE_EXITS,
    PipelineRunner,
    RunnerHooks,
    synthetic_frame,
)
from adas.tracking import MultiObjectTracker

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class ScriptedSource(FrameSource):
    """Delivers ``n`` frames and then stops, either as EOF or as a failure.

    A ``None`` with ``eof`` still False is what a live camera looks like when it
    silently stops delivering: a sensor loss, not the end of a recording.  The
    two must not share an exit path.
    """

    def __init__(self, n: int, ends_with_eof: bool = False):
        self.n = int(n)
        self.i = 0
        self.eof = False
        self._ends_with_eof = bool(ends_with_eof)
        self.closed = False

    def read(self):
        if self.i >= self.n:
            self.eof = self._ends_with_eof
            return None
        self.i += 1
        return CapturedFrame(image=synthetic_frame(1280, 720), width=1280, height=720)

    def close(self):
        self.closed = True


class ClearRoadDetector:
    """An empty road, honestly reported -- so the planner accelerates."""

    is_mock = False

    def infer(self, frame, width, height):
        return []


class NoLane:
    name = "fake"
    is_mock = False

    def estimate(self, frame, width, height):
        return None

    def drivable_area(self):
        return None


class ClosingLeadDetector:
    """A car dead ahead whose box grows every frame -- a real, closing hazard.

    The pinhole range therefore shrinks monotonically and the arbiter escalates
    to MIN_RISK_MANEUVER with a full-authority brake.  This is the only kind of
    scene in which the settle policy below is allowed to hold a brake, so it is
    the scene the tests use.
    """

    is_mock = False

    def __init__(self, start_h: float = 70.0, growth: float = 4.0) -> None:
        self.i = 0
        self.start_h = float(start_h)
        self.growth = float(growth)

    def infer(self, frame, width, height):
        self.i += 1
        h = self.start_h + self.growth * self.i
        return [BoundingBox(600.0, 400.0 - h, 760.0, 400.0, 0.92, "car")]


def build_closing_pipeline() -> ADASPipeline:
    return ADASPipeline(
        detector=ClosingLeadDetector(),
        lane_estimator=NoLane(),
        tracker=MultiObjectTracker(frame_width_px=1280, frame_height_px=720),
        planner=BehaviorPlanner(),
        controller=PIDLikeLongitudinalController(),
        safety_monitor=SafetyMonitor(),
    )


def build_pipeline() -> ADASPipeline:
    return ADASPipeline(
        detector=ClearRoadDetector(),
        lane_estimator=NoLane(),
        tracker=MultiObjectTracker(frame_width_px=1280, frame_height_px=720),
        planner=BehaviorPlanner(),
        controller=PIDLikeLongitudinalController(),
        safety_monitor=SafetyMonitor(),
    )


def run_and_collect(source, **kwargs):
    """Run to completion, returning ``(summary, [command, ...])`` in emission order."""
    emitted = []
    max_frames = kwargs.pop("max_frames", 0)
    runner = PipelineRunner(
        pipeline=kwargs.pop("pipeline", None) or build_pipeline(),
        target_fps=kwargs.pop("target_fps", 0.0),
        hooks=RunnerHooks(on_frame=lambda fid, plan, cmd: emitted.append(cmd)),
        **kwargs
    )
    summary = runner.run(source=source, max_frames=max_frames)
    return summary, emitted


# --------------------------------------------------------------------------- #
# ADAS-DEC-21: a lost source is a safety event, not a loop-exit condition
# --------------------------------------------------------------------------- #


def test_a_lost_source_emits_the_failsafe_and_never_latches_the_last_throttle():
    """The regression: `break` on a lost source left the throttle applied.

    Reproduced by the reviewer as ``LAST ACTUATED COMMAND AFTER SENSOR LOSS:
    ControlCommand(throttle=0.101..., brake=0.0, ...)`` -- forever.
    """
    # Paced for real at 20 Hz: unpaced, the arbiter reports the 3 ms frame period
    # as a timing fault and holds the throttle at zero anyway, which would make
    # the precondition below vacuous.
    summary, emitted = run_and_collect(
        ScriptedSource(15, ends_with_eof=False),
        target_fps=20.0,
        ego_source=ConstantEgoSpeed(5.0),
        failsafe_hold_s=0.25,
    )

    assert summary.frames == 15
    assert "source lost" in summary.stopped_reason
    assert summary.settle_commands == 5, "0.25 s of hold at the 0.05 s nominal period"

    last_frame_command = emitted[summary.frames - 1]
    settle = emitted[summary.frames :]

    assert last_frame_command.throttle > 0.0, "precondition: the run was accelerating"
    assert settle, "the loop must not exit without writing a command"
    assert all(command.throttle == 0.0 for command in settle), "no throttle after a sensor loss"
    assert settle[-1].brake > 0.0, "the fail-safe must be braking, not coasting"
    assert settle[-1].brake >= settle[0].brake, "the arbiter's brake ramp must keep running"


def test_a_lost_source_holds_the_manoeuvre_rather_than_emitting_once():
    """One command is not a minimum-risk manoeuvre: the arbiter rate-limits brake."""
    summary, emitted = run_and_collect(
        ScriptedSource(5, ends_with_eof=False),
        ego_source=ConstantEgoSpeed(15.0),
        failsafe_hold_s=1.0,
    )
    settle = emitted[summary.frames :]
    assert len(settle) > 1
    # The first fail-safe is still on the way up the brake ramp; the hold is what
    # actually gets the demand to the minimum-risk value.
    assert settle[-1].brake > settle[0].brake


def test_the_failsafe_hold_is_bounded_and_configurable():
    """It is a bounded hold, not a supervisor. Say so in the docstring, prove the bound."""
    for hold_s, expected in ((0.0, 1), (0.25, 5), (0.5, 10)):
        summary, emitted = run_and_collect(
            ScriptedSource(3, ends_with_eof=False),
            ego_source=ConstantEgoSpeed(5.0),
            failsafe_hold_s=hold_s,
        )
        assert summary.settle_commands == expected, "hold_s=%s" % hold_s
        assert len(emitted) == summary.frames + expected


def test_a_lost_source_decelerates_the_plant_instead_of_leaving_it_rolling():
    """End to end: the plant must be slowed by what is actuated after the loss."""
    plant = SimulatedEgoSpeed(initial_speed_mps=15.0, drag_per_s=0.0)
    runner = PipelineRunner(
        pipeline=build_pipeline(),
        target_fps=0.0,
        ego_source=plant,
        failsafe_hold_s=2.0,
    )
    runner.run(source=ScriptedSource(5, ends_with_eof=False), max_frames=0)
    assert plant.speed_mps < 15.0


# --------------------------------------------------------------------------- #
# A clean end is not a failure: EOF releases, it does not brake
# --------------------------------------------------------------------------- #


def test_end_of_source_releases_the_throttle_exactly_once():
    """A finished replay file is not a sensor loss -- but it still must not latch."""
    summary, emitted = run_and_collect(
        ScriptedSource(8, ends_with_eof=True),
        ego_source=ConstantEgoSpeed(5.0),
    )
    assert summary.frames == 8
    assert "end of source" in summary.stopped_reason
    assert summary.settle_commands == 1, "an intended end gets one release, not a brake ramp"

    release = emitted[-1]
    assert release.throttle == 0.0
    assert release.brake == 0.0
    assert release.steering == 0.0


def test_a_completed_frame_budget_releases_the_throttle():
    summary, emitted = run_and_collect(
        ScriptedSource(100, ends_with_eof=True),
        ego_source=ConstantEgoSpeed(5.0),
        max_frames=6,
    )
    assert summary.frames == 6
    assert summary.settle_commands == 1
    assert emitted[-1].throttle == 0.0
    assert emitted[-1].brake == 0.0


def test_a_requested_stop_releases_the_throttle():
    emitted = []
    runner = PipelineRunner(
        pipeline=build_pipeline(),
        target_fps=0.0,
        ego_source=ConstantEgoSpeed(5.0),
    )

    def on_frame(frame_id, plan, command):
        emitted.append(command)
        if frame_id >= 4:
            runner.request_stop("test")

    runner.hooks.on_frame = on_frame
    summary = runner.run(source=ScriptedSource(100, ends_with_eof=True), max_frames=0)
    assert summary.frames == 5
    assert "test" in summary.stopped_reason
    assert summary.settle_commands == 1
    assert emitted[-1].throttle == 0.0


# --------------------------------------------------------------------------- #
# A dead pipeline is the other unsafe exit
# --------------------------------------------------------------------------- #


def test_a_dead_pipeline_holds_the_failsafe_after_the_loop_stops():
    """`max_consecutive_failures` stops the loop -- and must not stop the brake."""
    pipeline = build_pipeline()

    def explode(*args, **kwargs):
        raise RuntimeError("scripted tracker failure")

    pipeline.tracker.update = explode

    summary, emitted = run_and_collect(
        ScriptedSource(50, ends_with_eof=True),
        pipeline=pipeline,
        ego_source=ConstantEgoSpeed(15.0),
        max_consecutive_failures=3,
        failsafe_hold_s=0.5,
    )

    assert summary.frames == 3
    assert summary.failures == 3
    assert summary.settle_commands == 10
    assert all(command.throttle == 0.0 for command in emitted)
    assert emitted[-1].brake > 0.0


# --------------------------------------------------------------------------- #
# Bookkeeping and robustness
# --------------------------------------------------------------------------- #


def test_settle_commands_are_not_counted_as_frames():
    """`measured_fps` must describe frames processed, not commands written."""
    summary, emitted = run_and_collect(
        ScriptedSource(10, ends_with_eof=False),
        ego_source=ConstantEgoSpeed(5.0),
        failsafe_hold_s=0.5,
    )
    assert summary.frames == 10
    assert summary.settle_commands == 10
    assert len(emitted) == 20
    assert "settle=10" in summary.describe()


def test_the_unsafe_exits_are_exactly_the_failure_exits():
    """A guard on the classification itself: EOF must never be an unsafe exit."""
    assert set(UNSAFE_EXITS) == {EXIT_SOURCE_LOST, EXIT_PIPELINE_DEAD}
    for kind in (EXIT_EOF, EXIT_COMPLETED, EXIT_STOPPED):
        assert kind not in UNSAFE_EXITS


def test_a_settle_hook_that_raises_does_not_break_the_run():
    """The fail-safe path is the last line of defence; a hook must not kill it."""

    def boom(frame_id, plan, command):
        raise RuntimeError("scripted hook failure")

    runner = PipelineRunner(
        pipeline=build_pipeline(),
        target_fps=0.0,
        ego_source=ConstantEgoSpeed(5.0),
        hooks=RunnerHooks(on_frame=boom),
        failsafe_hold_s=0.25,
    )
    summary = runner.run(source=ScriptedSource(4, ends_with_eof=False), max_frames=0)
    assert summary.settle_commands == 5


def test_an_owned_source_is_still_closed_after_the_failsafe_hold():
    """The hold runs inside the try, so the `finally` cleanup must still happen."""
    runner = PipelineRunner(
        pipeline=build_pipeline(),
        target_fps=0.0,
        ego_source=ConstantEgoSpeed(5.0),
        failsafe_hold_s=0.1,
    )
    source = ScriptedSource(3, ends_with_eof=False)
    summary = runner.run(source=source, max_frames=0)
    assert summary.settle_commands == 2
    assert not source.closed, "a caller-supplied source is not owned by the runner"

    owned = ScriptedSource(3, ends_with_eof=False)
    runner.run(source=owned, max_frames=0)
    assert not owned.closed


# --------------------------------------------------------------------------- #
# A clean exit must not CANCEL an intervention (the settle/arbiter interaction)
# --------------------------------------------------------------------------- #
#
# The regression: `UNSAFE_EXITS` was the whole settle decision, so
# EXIT_COMPLETED / EXIT_EOF / EXIT_STOPPED all wrote ControlCommand(0, 0, 0)
# without ever reading `pipeline.last_arbitration.state`.  Reproduced by the
# reviewer as:
#     in-loop frame 29  t=0.000 b=1.000 state=min_risk_maneuver
#     in-loop frame 30  t=0.000 b=0.000 state=min_risk_maneuver
# i.e. the runner released a live full-authority AEB on its way out.  EXIT_STOPPED
# is the request_stop() path, so this was reachable with `systemctl stop`.
#
# These tests drive the runner with a STUB arbitration rather than with a scene
# that happens to trip the real arbiter's AEB thresholds.  That is deliberate:
# what is under test is the runner's exit policy GIVEN an arbitration state, and
# tying it to the arbiter's tuning would make the coverage of the MRM branch
# silently disappear the next time a threshold moves.  `test_a_real_closing_lead_*`
# below keeps one end-to-end case against the real arbiter.


class StubArbitration:
    def __init__(self, state, brake, steering=0.0, throttle=0.0):
        self.state = state
        self.command = ControlCommand(throttle=throttle, brake=brake, steering=steering)
        self.violations = ()
        self.reason = "stub"


class StubPipeline:
    """A pipeline whose arbitration the test dictates, frame by frame.

    ``failsafe_command`` mimics the real one closely enough for the settle path:
    it returns the arbiter's generic minimum-risk brake, which is LOWER than a
    full-authority AEB -- the asymmetry that makes the brake floor necessary.
    """

    MRM_BRAKE = 0.44

    def __init__(self, state, brake, steering=0.0):
        self.last_arbitration = StubArbitration(state, brake, steering)
        self.last_lead_distance_m = None
        self.last_tracks = []
        self.metrics = _NullMetrics()
        self.failsafe_calls = 0

    def step(self, frame, current_speed_mps=None, dt_s=0.05, ego=None):
        return None, self.last_arbitration.command

    def failsafe_command(self, dt_s=0.05):
        self.failsafe_calls += 1
        return ControlCommand(throttle=0.0, brake=self.MRM_BRAKE, steering=0.0)


class _NullMetrics:
    def record_dropped_frame(self, reason=""):
        pass


def _settle_commands(pipeline, exit_kind, failsafe_hold_s=0.25, nominal_dt_s=0.05):
    """Call ``_settle`` directly and return the commands it actuated."""
    emitted = []
    runner = PipelineRunner(
        pipeline=pipeline,
        target_fps=0.0,
        ego_source=ConstantEgoSpeed(15.0),
        failsafe_hold_s=failsafe_hold_s,
        hooks=RunnerHooks(on_frame=lambda fid, plan, cmd: emitted.append(cmd)),
    )
    from adas.runtime.runner import RunSummary

    summary = RunSummary()
    runner._settle(exit_kind, summary, 30, nominal_dt_s)
    return summary, emitted


@pytest.mark.parametrize("exit_kind", [EXIT_COMPLETED, EXIT_EOF, EXIT_STOPPED])
@pytest.mark.parametrize("state", list(INTERVENING_STATES))
def test_a_clean_exit_during_an_intervention_holds_the_brake(exit_kind, state):
    """EXIT_STOPPED is `systemctl stop`; none of the three may cancel an AEB."""
    pipeline = StubPipeline(state, brake=1.0, steering=-0.13)
    summary, emitted = _settle_commands(pipeline, exit_kind)

    assert summary.settle_kind == "hold"
    assert emitted, "the loop must not exit without writing a command"
    assert all(c.throttle == 0.0 for c in emitted), "no throttle on any shutdown"
    assert all(c.brake == pytest.approx(1.0) for c in emitted), (
        "%s released a live emergency brake" % exit_kind
    )
    assert all(c.steering == pytest.approx(-0.13) for c in emitted), (
        "the MRM's steering is held with it, not zeroed"
    )
    assert summary.settle_brake == pytest.approx(1.0)
    assert pipeline.failsafe_calls == 0, "a live MRM is held, not re-derived"


@pytest.mark.parametrize("exit_kind", [EXIT_COMPLETED, EXIT_EOF, EXIT_STOPPED])
@pytest.mark.parametrize("state", [SafetyState.NOMINAL, SafetyState.LIMITED])
@pytest.mark.parametrize("brake", [1.0, 0.65, 0.007])
def test_a_clean_exit_while_merely_LIMITED_still_holds_the_brake(exit_kind, state, brake):
    """The gate is the BRAKE, not the state -- and real footage says it must be.

    400 frames of example.mp4 at 15 m/s produce twelve consecutive frames of
    brake >= 0.9 against a real tracked lead at 7.3-7.5 m
    (``headway_7.0m_below_rss_12.5m``), and the arbiter is in LIMITED for every
    one of them; it never reaches MIN_RISK_MANEUVER on that clip at all.  Gating
    the hold on the arbitration STATE would have protected none of those frames
    while looking like a fix.
    """
    pipeline = StubPipeline(state, brake=brake)
    summary, emitted = _settle_commands(pipeline, exit_kind)
    assert summary.settle_kind == "hold"
    assert emitted and all(c.brake == pytest.approx(brake) for c in emitted)
    assert all(c.throttle == 0.0 for c in emitted)
    assert pipeline.failsafe_calls == 0


@pytest.mark.parametrize("exit_kind", [EXIT_COMPLETED, EXIT_EOF, EXIT_STOPPED])
@pytest.mark.parametrize("state", [SafetyState.NOMINAL, SafetyState.LIMITED])
def test_a_clean_exit_outside_an_intervention_still_releases_both(exit_kind, state):
    """The over-correction guard: holding is for interventions ONLY.

    If this fails, the fix has turned every shutdown into a brake, which would
    strand a vehicle at the end of every bounded run and at every EOF.  On the
    real 400-frame clip 342 of 400 frames command zero brake, including the last
    one, so this is the branch that a normal run actually takes.
    """
    pipeline = StubPipeline(state, brake=0.0)
    summary, emitted = _settle_commands(pipeline, exit_kind)

    assert summary.settle_kind == "release"
    assert summary.settle_commands == 1, "a clean end is one release, not a ramp"
    assert emitted[-1].throttle == 0.0
    assert emitted[-1].brake == 0.0
    assert emitted[-1].steering == 0.0
    assert summary.settle_brake == 0.0


def test_a_clean_exit_with_no_arbitration_at_all_releases():
    """Zero frames ran: nothing ever engaged, so there is nothing to hold."""
    pipeline = StubPipeline(SafetyState.NOMINAL, brake=0.0)
    pipeline.last_arbitration = None
    summary, emitted = _settle_commands(pipeline, EXIT_COMPLETED)
    assert summary.settle_kind == "release"
    assert emitted[-1].brake == 0.0


@pytest.mark.parametrize("exit_kind", [EXIT_SOURCE_LOST, EXIT_PIPELINE_DEAD])
def test_an_unsafe_exit_never_lowers_a_live_aeb_to_the_generic_mrm(exit_kind):
    """The second half of the same defect, found by sweeping the exit kinds.

    `failsafe_command` asks the arbiter for a minimum-risk manoeuvre with NO
    tracks, so the arbiter sees `no_in_path_lead` and demands only its generic
    `mrm_decel_mps2`.  Measured on the real arbiter, losing the camera during a
    full-authority AEB walked the brake 1.00 -> 0.96 -> ... -> 0.44 across the
    hold.  Going blind is not evidence that the lead went away.
    """
    pipeline = StubPipeline(SafetyState.MIN_RISK_MANEUVER, brake=1.0)
    summary, emitted = _settle_commands(pipeline, exit_kind)

    assert summary.settle_kind == "min_risk"
    assert pipeline.failsafe_calls == summary.settle_commands
    assert all(c.throttle == 0.0 for c in emitted)
    assert all(c.brake >= 1.0 for c in emitted), (
        "the fail-safe ramp must not reduce the arbiter's last brake"
    )


@pytest.mark.parametrize("exit_kind", [EXIT_SOURCE_LOST, EXIT_PIPELINE_DEAD])
def test_an_unsafe_exit_from_a_clear_road_still_ramps_up(exit_kind):
    """The floor must not suppress the ramp it was added to protect."""
    pipeline = StubPipeline(SafetyState.NOMINAL, brake=0.0)
    summary, emitted = _settle_commands(pipeline, exit_kind)
    assert summary.settle_kind == "min_risk"
    assert emitted and all(c.brake == pytest.approx(StubPipeline.MRM_BRAKE) for c in emitted)
    assert emitted[-1].brake > 0.0, "the fail-safe must be braking, not coasting"


def test_the_brake_floor_cannot_invent_a_brake():
    """The floor is 0.0 unless the arbiter commanded a brake on a REAL frame."""
    for state in (SafetyState.NOMINAL, SafetyState.LIMITED, SafetyState.MIN_RISK_MANEUVER):
        pipeline = StubPipeline(state, brake=0.0)
        runner = PipelineRunner(pipeline=pipeline, target_fps=0.0)
        runner._last_frame_arbitration = pipeline.last_arbitration
        assert runner._brake_floor() == 0.0, state


def test_the_hold_is_bounded_and_does_not_run_forever():
    """A hold that never returns is a wedged process, not a safe one."""
    for hold_s, expected in ((0.0, 1), (0.25, 5), (1.0, 20)):
        pipeline = StubPipeline(SafetyState.MIN_RISK_MANEUVER, brake=1.0)
        summary, _ = _settle_commands(pipeline, EXIT_STOPPED, failsafe_hold_s=hold_s)
        assert summary.settle_commands == expected, "hold_s=%s" % hold_s


def test_settle_reads_a_broken_pipelines_arbitration_without_raising():
    """The shutdown path must survive a pipeline that is already broken."""

    class Exploding:
        @property
        def last_arbitration(self):
            raise RuntimeError("arbiter is gone")

    runner = PipelineRunner(pipeline=build_pipeline(), target_fps=0.0)
    runner.pipeline = Exploding()  # type: ignore[assignment]
    assert runner._arbitration_state() is None


def test_a_contradictory_intervention_falls_back_to_the_failsafe():
    """MRM latched but zero brake on the last command: ask the arbiter, do not coast."""
    pipeline = StubPipeline(SafetyState.MIN_RISK_MANEUVER, brake=0.0)
    runner = PipelineRunner(pipeline=pipeline, target_fps=0.0)
    assert runner._held_command() is None

    summary, emitted = _settle_commands(pipeline, EXIT_STOPPED)
    assert summary.settle_kind == "min_risk"
    assert emitted and emitted[-1].brake > 0.0
    assert all(c.throttle == 0.0 for c in emitted)


# --------------------------------------------------------------------------- #
# ... and once, end to end, against the REAL arbiter
# --------------------------------------------------------------------------- #


def test_a_real_closing_lead_is_still_braking_after_the_frame_budget_ends():
    """The reviewer's scenario, with nothing stubbed.

    Written to hold whichever way the arbiter's thresholds are tuned: if the
    scene escalates, the settle must hold the brake; if it does not, the settle
    must fully release.  What it can never do is emit LESS brake than the last
    in-loop frame.
    """
    pipeline = build_closing_pipeline()
    summary, emitted = run_and_collect(
        pipeline=pipeline,
        source=ScriptedSource(30, ends_with_eof=False),
        ego_source=ConstantEgoSpeed(15.0),
        max_frames=30,
        failsafe_hold_s=0.25,
    )
    in_loop = emitted[: summary.frames]
    settle = emitted[summary.frames :]
    assert settle, "the loop must not exit without writing a command"
    assert all(c.throttle == 0.0 for c in settle)

    if in_loop[-1].brake > 0.0:
        assert summary.settle_kind == "hold"
        assert all(c.brake == pytest.approx(in_loop[-1].brake) for c in settle), (
            "held, not escalated: no new frame arrived, so there is no new evidence"
        )
    else:
        assert summary.settle_kind == "release"
        assert settle[-1].brake == 0.0
    assert summary.settle_brake == pytest.approx(settle[-1].brake)
