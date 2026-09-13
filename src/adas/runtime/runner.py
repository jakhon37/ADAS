"""The frame loop: capture, step, actuate, pace, report.

Everything the loop does that is not "call the pipeline" is here because it is
all timing and failure policy, and both are safety-relevant:

* **Continuous mode.**  ``max_frames <= 0`` runs until stopped (ADAS-OPS-02).
  ``deploy/adas.service`` and ``deploy/Dockerfile.jetson`` both launch with
  ``--frames 0``; before this the loop was ``for frame_id in range(max_frames)``
  and the service would have exited immediately and restarted forever.

* **Measured dt.**  Every *filter* -- the range filter, the jerk limit, the
  planner's rate limiter -- is given the interval actually observed between
  frames, clamped to ``[0.5x, 3x]`` of the nominal period so one stall cannot
  multiply through every rate estimate (ADAS-DEC-17).  The clamp is logged, and
  the metrics record the unclamped value.  The simulated plant integrates the
  UNCLAMPED interval, because it is a physical integration against the wall
  clock: clamping it would make the plant's speed disagree with the timestamps
  the arbiter differences, and the arbiter would report that disagreement as
  jerk.  In a paced run the two are the same number.

* **A failed frame emits a fail-safe, not the previous command.**  When
  ``pipeline.step`` raises, the loop actuates
  :meth:`~adas.runtime.pipeline.ADASPipeline.failsafe_command` and counts the
  failure (ADAS-DEC-21).  ``continue`` -- the old behaviour -- left whatever
  command the actuators were last given latched in place.

* **Leaving the loop is not a command either.**  ``break`` latches the last
  command just as surely as ``continue`` did, so :meth:`PipelineRunner._settle`
  runs on EVERY exit and writes something before the runner returns.  A lost
  source or a dead pipeline (``max_consecutive_failures`` in a row) is a safety
  event, not merely a loop-exit condition: the runner then re-emits the
  arbiter's minimum-risk manoeuvre once per nominal period for
  ``failsafe_hold_s`` seconds, so the brake ramp actually runs instead of
  stopping at whatever the last frame produced.  The hold is BOUNDED -- the
  runner hands control back to its caller afterwards, and ``deploy/adas.service``
  restarts the process.  A clean end (a finished replay file, the ``--frames``
  budget, an operator stop) is distinguished by ``source.eof`` and gets a single
  zero-throttle, zero-brake release instead of a brake ramp.

* **Stopping is cooperative.**  :meth:`PipelineRunner.request_stop` sets a flag
  the loop checks once per iteration; the CLI's SIGTERM/SIGINT handler calls it,
  so ``systemctl stop`` and ``docker stop`` unwind through the normal shutdown
  path (close the source, log the summary, flush the event log) instead of being
  SIGKILLed mid-frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from adas.core.exceptions import SensorError
from adas.core.logger import setup_logger
from adas.core.models import ControlCommand, PerceptionFrame
from adas.runtime.capture import (
    EgoSpeedSource,
    FrameSource,
    NoEgoSpeed,
    open_source,
)
from adas.runtime.pipeline import ADASPipeline

logger = setup_logger(__name__)

#: dt is clamped into this multiple of the nominal frame period before it reaches
#: any filter. Outside it, the measurement is a scheduling artefact, not physics.
DT_CLAMP_LOW = 0.5
DT_CLAMP_HIGH = 3.0

#: How the frame loop ended.  The distinction is safety-relevant, not cosmetic:
#: the first three are intended ends and release the throttle; the last two are
#: failures and run the minimum-risk hold.  See :meth:`PipelineRunner._settle`.
EXIT_COMPLETED = "completed"
EXIT_STOPPED = "stopped"
EXIT_EOF = "eof"
EXIT_SOURCE_LOST = "source_lost"
EXIT_PIPELINE_DEAD = "pipeline_dead"

#: Exits that mean a sensor or the decision path failed mid-stream.
UNSAFE_EXITS = (EXIT_SOURCE_LOST, EXIT_PIPELINE_DEAD)


@dataclass
class RunSummary:
    """What one :meth:`PipelineRunner.run` actually did.

    ``measured_fps`` is derived from ``time.monotonic`` over the whole run, so it
    reflects real throughput including the pacing sleep.  ``busy_fps`` excludes
    the sleep and is the rate the loop *could* sustain.
    """

    frames: int = 0
    failures: int = 0
    dropped: int = 0
    reconnects: int = 0
    #: Commands emitted by :meth:`PipelineRunner._settle` AFTER the frame loop
    #: ended -- the minimum-risk hold, or the single release on a clean exit.
    #: Deliberately not counted in ``frames``: no frame was processed.
    settle_commands: int = 0
    elapsed_s: float = 0.0
    busy_s: float = 0.0
    stopped_reason: str = ""

    @property
    def measured_fps(self) -> float:
        return self.frames / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def busy_fps(self) -> float:
        return self.frames / self.busy_s if self.busy_s > 0 else 0.0

    @property
    def mean_busy_ms(self) -> float:
        return (self.busy_s / self.frames * 1000.0) if self.frames else 0.0

    def describe(self) -> str:
        return (
            "frames=%d failures=%d dropped=%d reconnects=%d settle=%d elapsed=%.2fs "
            "measured=%.2f FPS busy=%.2f FPS (%.1f ms/frame) reason=%s"
            % (
                self.frames,
                self.failures,
                self.dropped,
                self.reconnects,
                self.settle_commands,
                self.elapsed_s,
                self.measured_fps,
                self.busy_fps,
                self.mean_busy_ms,
                self.stopped_reason or "completed",
            )
        )


@dataclass
class RunnerHooks:
    """Optional operations callbacks.

    All are duck-typed and optional, so the runner unit-tests with none of them
    and the ops layer (``adas.io``) stays out of the import graph of the frame
    loop.

    Attributes:
        health: an :class:`adas.io.health.HealthState`.
        events: an :class:`adas.io.events.EventLog`.
        watchdog: an :class:`adas.io.sd_notify.WatchdogPinger`.  It is ticked with
            the id of the frame just processed and refuses to ping when that id
            has not advanced, which is what lets systemd kill a wedged process.
        on_ready: called once, after the FIRST frame completes.  The CLI uses it
            for ``sd_notify(READY=1)``: readiness means "a frame went all the way
            through", not "the process started".
        on_frame: called after every successful frame with
            ``(frame_id, plan, command)``.
    """

    health: Any = None
    events: Any = None
    watchdog: Any = None
    on_ready: Optional[Callable[[], None]] = None
    on_frame: Optional[Callable[[int, Any, ControlCommand], None]] = None


@dataclass
class PipelineRunner:
    """Runs an :class:`ADASPipeline` against a frame source.

    Args:
        pipeline: the pipeline to drive.
        target_fps: pacing target.  ``<= 0`` runs flat out (replay throughput
            measurement).
        ego_source: where ego speed comes from.  Defaults to
            :class:`~adas.runtime.capture.NoEgoSpeed`, i.e. no ego speed, which
            keeps the planner degraded -- honest, and loud in the logs.
        hooks: optional operations callbacks.
        max_consecutive_failures: consecutive failed frames after which the loop
            stops.  Each failed frame still actuates the arbiter's fail-safe.
        failsafe_hold_s: how long :meth:`_settle` keeps re-emitting the arbiter's
            minimum-risk manoeuvre after an UNSAFE exit (a lost source or a dead
            pipeline), seconds.  One command per nominal period.  ``0`` emits
            exactly one.  This is a bounded hold, not a substitute for a
            supervisor: the runner returns afterwards.
    """

    pipeline: ADASPipeline
    target_fps: float = 20.0
    ego_source: EgoSpeedSource = field(default_factory=NoEgoSpeed)
    hooks: RunnerHooks = field(default_factory=RunnerHooks)
    max_consecutive_failures: int = 10
    failsafe_hold_s: float = 1.0

    _stop: bool = field(default=False, init=False)
    _stop_reason: str = field(default="", init=False)
    _last_safety_state: str = field(default="nominal", init=False)
    last_summary: Optional[RunSummary] = field(default=None, init=False)

    # ------------------------------------------------------------------ control

    def request_stop(self, reason: str = "requested") -> None:
        """Ask the loop to finish the current frame and return.  Signal-safe."""
        self._stop = True
        self._stop_reason = reason

    @property
    def stopping(self) -> bool:
        return self._stop

    # --------------------------------------------------------------------- run

    def run_synthetic(self, max_frames: int = 60) -> RunSummary:
        """Run against blank frames.  Wiring check only -- nothing is detected."""
        return self.run(source_type="synthetic", max_frames=max_frames)

    def run(
        self,
        source_type: str = "synthetic",
        uri: str = "",
        max_frames: int = 60,
        width: int = 1280,
        height: int = 720,
        as_image: bool = False,
        source: Optional[FrameSource] = None,
        loop: bool = False,
        reconnect_attempts: int = 0,
        reconnect_delay_s: float = 1.0,
    ) -> RunSummary:
        """Run the frame loop.

        Args:
            source_type: ``synthetic`` | ``video`` | ``camera``.
            uri: file path or device specifier.
            max_frames: number of frames, or ``<= 0`` to run until stopped.
            width, height: requested capture size.
            as_image: ask a synthetic source for a real numpy array (needed by any
                backend that runs a network).
            source: a pre-opened source; the runner then does not own it.
            loop: restart a video file at EOF.
            reconnect_attempts, reconnect_delay_s: mid-stream failure policy.

        Returns:
            A :class:`RunSummary` with the MEASURED throughput.

        Raises:
            SensorError: the source could not be opened at all.
        """
        self._stop = False
        self._stop_reason = ""
        continuous = max_frames is None or max_frames <= 0
        nominal_dt_s = 1.0 / self.target_fps if self.target_fps > 0 else 0.05

        logger.info(
            "Starting run: %s at %.1f FPS from %s%s, ego=%s",
            "continuous" if continuous else "%d frames" % max_frames,
            self.target_fps,
            source_type,
            (" %s" % uri) if uri else "",
            self.ego_source.describe(),
        )

        own_source = source is None
        if source is None:
            source = open_source(
                source_type,
                uri=uri,
                width=width,
                height=height,
                as_image=as_image,
                loop=loop,
                reconnect_attempts=reconnect_attempts,
                reconnect_delay_s=reconnect_delay_s,
            )

        summary = RunSummary()
        frame_id = 0
        consecutive_failures = 0
        previous_mono: Optional[float] = None
        run_started = time.monotonic()
        exit_kind = EXIT_COMPLETED
        broke_out = False

        try:
            while not self._stop and (continuous or frame_id < max_frames):
                loop_started = time.monotonic()

                captured = self._read(source, frame_id, summary)
                if captured is None:
                    # `eof` is a finite recording that finished. Anything else is
                    # a SENSOR FAILURE mid-stream, and the two must not share an
                    # exit path: one releases, the other brakes.
                    exit_kind = (
                        EXIT_EOF if getattr(source, "eof", False) else EXIT_SOURCE_LOST
                    )
                    broke_out = True
                    break

                now = time.monotonic()
                raw_dt_s = nominal_dt_s if previous_mono is None else (now - previous_mono)
                previous_mono = now
                dt_s = self._clamp_dt(raw_dt_s, nominal_dt_s, frame_id)

                ego = self.ego_source.state(now)
                perception = PerceptionFrame(
                    frame_id=frame_id,
                    timestamp_s=now,
                    rgb=captured.image,
                    width=captured.width,
                    height=captured.height,
                )

                try:
                    plan, command = self.pipeline.step(perception, dt_s=dt_s, ego=ego)
                    consecutive_failures = 0
                except Exception as exc:  # noqa: BLE001 - every failure is fail-safed
                    consecutive_failures += 1
                    summary.failures += 1
                    plan = None
                    command = self.pipeline.failsafe_command(dt_s=dt_s)
                    logger.error(
                        "Pipeline failed on frame %s (%d consecutive): %s; actuating fail-safe "
                        "t%.2f/b%.2f/s%+.2f",
                        frame_id,
                        consecutive_failures,
                        exc,
                        command.throttle,
                        command.brake,
                        command.steering,
                        exc_info=consecutive_failures == 1,
                    )
                    self.pipeline.metrics.record_dropped_frame(reason="pipeline")
                    self._note_engine_failure(exc)

                # The plant, if there is one, is driven by the ACTUATED command --
                # the arbiter's, never the controller's -- and integrates over the
                # UNCLAMPED elapsed time. The clamp exists to stop one stall
                # multiplying through a filter; applying it to a physical
                # integration would make the plant's speed disagree with the wall
                # clock, and the arbiter (which differences measured speed against
                # real timestamps) would read that disagreement as jerk.
                self.ego_source.apply_command(command, raw_dt_s)

                summary.frames += 1
                summary.busy_s += time.monotonic() - loop_started
                self._report(frame_id, perception, plan, command, ego)

                if summary.frames == 1 and self.hooks.on_ready is not None:
                    self.hooks.on_ready()

                if consecutive_failures >= self.max_consecutive_failures:
                    self._stop_reason = "too many consecutive pipeline failures (%d)" % (
                        consecutive_failures,
                    )
                    logger.critical("Stopping: %s", self._stop_reason)
                    exit_kind = EXIT_PIPELINE_DEAD
                    broke_out = True
                    break

                frame_id += 1
                self._pace(loop_started, nominal_dt_s)

            if not broke_out:
                exit_kind = EXIT_STOPPED if self._stop else EXIT_COMPLETED

            # Whatever ended the loop, the actuators must not be left holding the
            # last frame's command.
            self._settle(exit_kind, summary, frame_id, nominal_dt_s)
        finally:
            summary.elapsed_s = time.monotonic() - run_started
            summary.reconnects = int(getattr(source, "reconnects", 0))
            summary.stopped_reason = self._stop_reason or (
                "stop requested" if self._stop else "end of source"
                if getattr(source, "eof", False)
                else "completed"
            )
            if own_source:
                source.close()

        self.last_summary = summary
        logger.info("Run completed: %s", summary.describe())
        return summary

    # ----------------------------------------------------------------- helpers

    def _read(self, source: FrameSource, frame_id: int, summary: RunSummary):
        try:
            captured = source.read()
        except SensorError as exc:
            logger.error("Source read raised on frame %s: %s", frame_id, exc)
            self._stop_reason = "source error: %s" % exc
            self._note_source("error", str(exc), frame_id)
            return None
        if captured is None:
            if getattr(source, "eof", False):
                logger.info("End of source at frame %s", frame_id)
                self._stop_reason = "end of source"
                self._note_source("eof", "", frame_id)
            else:
                logger.error("Source stopped delivering frames at frame %s", frame_id)
                self._stop_reason = "source lost"
                summary.dropped += 1
                self.pipeline.metrics.record_dropped_frame(reason="source")
                self._note_source("lost", "", frame_id)
            return None
        return captured

    def _settle(
        self,
        exit_kind: str,
        summary: RunSummary,
        frame_id: int,
        nominal_dt_s: float,
    ) -> None:
        """Write a defined command to the actuators on the way out of the loop.

        Leaving the loop is not a command.  Whatever was last written stays
        latched until something else is written, so a mid-stream sensor loss that
        merely ``break``\ s leaves the previous frame's throttle applied -- the
        ADAS-DEC-21 failure this module claims to have fixed for the *exception*
        path.  Both exits are handled here:

        * :data:`EXIT_SOURCE_LOST` and :data:`EXIT_PIPELINE_DEAD` are failures of
          the sensing or decision path.  The runner re-emits
          :meth:`~adas.runtime.pipeline.ADASPipeline.failsafe_command` -- the
          arbiter's minimum-risk manoeuvre -- once per nominal period for
          ``failsafe_hold_s`` seconds.  Re-emitting matters: the arbiter's brake
          demand is rate limited, so a single command stops the ramp partway.
          The hold is bounded (see ``failsafe_hold_s``); it brings the demand up
          and hands back to the caller, it does not supervise the vehicle
          forever.
        * :data:`EXIT_EOF`, :data:`EXIT_COMPLETED` and :data:`EXIT_STOPPED` are
          intended ends.  One zero-throttle, zero-brake, zero-steering command is
          emitted so a finished replay or a bounded ``--frames`` run does not
          leave the last frame's throttle applied.
        """
        if exit_kind in UNSAFE_EXITS:
            steps = 1
            if self.failsafe_hold_s > 0.0 and nominal_dt_s > 0.0:
                steps = max(1, int(round(self.failsafe_hold_s / nominal_dt_s)))
            logger.critical(
                "Frame loop exited on '%s' at frame %s: this is a safety event, not a "
                "clean stop. Holding the minimum-risk manoeuvre for %d command(s) "
                "(%.2f s at the %.1f ms nominal period).",
                exit_kind,
                frame_id,
                steps,
                steps * nominal_dt_s,
                nominal_dt_s * 1000.0,
            )
            last: Optional[ControlCommand] = None
            for _ in range(steps):
                started = time.monotonic()
                last = self.pipeline.failsafe_command(dt_s=nominal_dt_s)
                self._actuate_settled(last, nominal_dt_s, frame_id, summary)
                self._pace(started, nominal_dt_s)
            if last is not None:
                logger.critical(
                    "Minimum-risk hold finished after %d command(s); last actuated "
                    "t%.2f/b%.2f/s%+.2f",
                    summary.settle_commands,
                    last.throttle,
                    last.brake,
                    last.steering,
                )
            return

        logger.info(
            "Frame loop finished ('%s') at frame %s; releasing the throttle.",
            exit_kind,
            frame_id,
        )
        self._actuate_settled(
            ControlCommand(throttle=0.0, brake=0.0, steering=0.0),
            nominal_dt_s,
            frame_id,
            summary,
        )

    def _actuate_settled(
        self,
        command: ControlCommand,
        dt_s: float,
        frame_id: int,
        summary: RunSummary,
    ) -> None:
        """Push one post-loop command to the plant, the hooks and the summary.

        Deliberately NOT :meth:`_report`: there is no frame behind these
        commands, so there is nothing truthful to say about perception, the lane
        or the frame id, and the watchdog is NOT ticked -- the frame id has not
        advanced, and a runner that has lost its source must stay visible to
        systemd rather than be kept alive by its own fail-safe.
        """
        summary.settle_commands += 1
        try:
            self.ego_source.apply_command(command, dt_s)
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            logger.error("Plant refused the post-loop command: %s", exc)

        health = self.hooks.health
        if health is not None:
            try:
                arbitration = self.pipeline.last_arbitration
                with health.lock:
                    if arbitration is not None:
                        health.safety_state = arbitration.state.value
                health.mark_update()
            except Exception as exc:  # noqa: BLE001
                logger.debug("health refresh failed: %s", exc)

        self._note_safety_state(frame_id)

        if self.hooks.on_frame is not None:
            try:
                self.hooks.on_frame(frame_id, None, command)
            except Exception as exc:  # noqa: BLE001
                logger.debug("on_frame hook failed: %s", exc)

    def _clamp_dt(self, raw_dt_s: float, nominal_dt_s: float, frame_id: int) -> float:
        low = DT_CLAMP_LOW * nominal_dt_s
        high = DT_CLAMP_HIGH * nominal_dt_s
        if raw_dt_s < low:
            return low
        if raw_dt_s > high:
            logger.warning(
                "Frame %s took %.1f ms, more than %.0fx the %.1f ms nominal period; "
                "clamping dt for the filters (the metrics keep the real value)",
                frame_id,
                raw_dt_s * 1000.0,
                DT_CLAMP_HIGH,
                nominal_dt_s * 1000.0,
            )
            return high
        return raw_dt_s

    def _pace(self, loop_started: float, nominal_dt_s: float) -> None:
        if self.target_fps <= 0:
            return
        sleep_s = nominal_dt_s - (time.monotonic() - loop_started)
        if sleep_s > 0:
            time.sleep(sleep_s)

    def _report(self, frame_id: int, frame, plan, command: ControlCommand, ego) -> None:
        """Push this frame's state into the health snapshot and the watchdog."""
        health = self.hooks.health
        if health is not None:
            try:
                with health.lock:
                    health.frame_id = frame_id
                    health.source = "ok"
                    health.plan_reason = getattr(plan, "reason", "pipeline_failure")
                    health.ego_speed_mps = ego.speed_mps if ego.valid else None
                    health.ego_speed_valid = bool(ego.valid and self.ego_source.measured)
                    health.lane_is_mock = bool(getattr(frame.lane, "is_mock", True))
                    health.lead_distance_m = self.pipeline.last_lead_distance_m
                    status = frame.status
                    if status is not None:
                        health.perception_ok = status.ok
                        health.perception_consecutive_failures = status.consecutive_failures
                        health.perception_reason = status.reason
                    arbitration = self.pipeline.last_arbitration
                    if arbitration is not None:
                        health.safety_state = arbitration.state.value
                health.update_from_metrics(self.pipeline.metrics)
                health.mark_update()
            except Exception as exc:  # noqa: BLE001 - reporting must not stop the loop
                logger.debug("health refresh failed: %s", exc)

        watchdog = self.hooks.watchdog
        if watchdog is not None:
            try:
                watchdog.tick(frame_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("watchdog tick failed: %s", exc)

        self._note_safety_state(frame_id)

        if self.hooks.on_frame is not None:
            try:
                self.hooks.on_frame(frame_id, plan, command)
            except Exception as exc:  # noqa: BLE001
                logger.debug("on_frame hook failed: %s", exc)

    def _note_safety_state(self, frame_id: int) -> None:
        events = self.hooks.events
        arbitration = self.pipeline.last_arbitration
        if events is None or arbitration is None:
            return
        current = arbitration.state.value
        previous = getattr(self, "_last_safety_state", "nominal")
        if current == previous:
            return
        self._last_safety_state = current
        try:
            events.safety_state(
                previous,
                current,
                frame_id=frame_id,
                violations=list(arbitration.violations),
                reason=arbitration.reason,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("event log write failed: %s", exc)

    def _note_source(self, kind: str, detail: str, frame_id: int) -> None:
        events = self.hooks.events
        if events is None:
            return
        try:
            events.source_event(kind, uri=detail, frame_id=frame_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("event log write failed: %s", exc)
        health = self.hooks.health
        if health is not None:
            try:
                with health.lock:
                    health.source = kind
            except Exception as exc:  # noqa: BLE001
                logger.debug("health refresh failed: %s", exc)

    def _note_engine_failure(self, exc: BaseException) -> None:
        events = self.hooks.events
        if events is None:
            return
        try:
            events.engine_failure("pipeline", "execution", str(exc))
        except Exception as inner:  # noqa: BLE001
            logger.debug("event log write failed: %s", inner)


def synthetic_frame(width: int = 1280, height: int = 720) -> dict:
    """A metadata-only synthetic frame payload, as the mock backends expect."""
    return {"width": int(width), "height": int(height)}


__all__ = [
    "DT_CLAMP_HIGH",
    "DT_CLAMP_LOW",
    "EXIT_COMPLETED",
    "EXIT_EOF",
    "EXIT_PIPELINE_DEAD",
    "EXIT_SOURCE_LOST",
    "EXIT_STOPPED",
    "UNSAFE_EXITS",
    "PipelineRunner",
    "RunSummary",
    "RunnerHooks",
    "synthetic_frame",
]
