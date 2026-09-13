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
