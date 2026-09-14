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

* **A clean exit still may not cancel an intervention.**  Releasing the THROTTLE
  is right on every intended end.  Releasing the BRAKE is right only when the
  arbiter was not braking.  If ``pipeline.last_arbitration.state`` is
  ``MIN_RISK_MANEUVER`` or ``DISENGAGE`` when the loop ends, the settle holds the
  arbiter's last brake and steering instead -- an operator who stops the service
  during an AEB gets a stopped service, not a released brake.  Every exit path
  now consults the arbiter; see :data:`INTERVENING_STATES` and
  :meth:`PipelineRunner._settle`.

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
from adas.core.models import ControlCommand, PerceptionFrame, SafetyState
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
#: the first three are intended ends and RELEASE THE THROTTLE; the last two are
#: failures and run the minimum-risk hold.  See :meth:`PipelineRunner._settle`.
#:
#: The exit reason is only half of the settle decision.  The other half is the
#: arbiter's state -- see :data:`INTERVENING_STATES`.
EXIT_COMPLETED = "completed"
EXIT_STOPPED = "stopped"
EXIT_EOF = "eof"
EXIT_SOURCE_LOST = "source_lost"
EXIT_PIPELINE_DEAD = "pipeline_dead"

#: Exits that mean a sensor or the decision path failed mid-stream.
UNSAFE_EXITS = (EXIT_SOURCE_LOST, EXIT_PIPELINE_DEAD)

#: Arbitration states in which the vehicle is being actively intervened on.
#:
#: An exit reason describes why the SOFTWARE stopped.  It says nothing about
#: whether the road ahead is clear.  ``_settle`` used to conflate the two: any
#: exit outside :data:`UNSAFE_EXITS` wrote ``ControlCommand(0, 0, 0)``, which on
#: a frame-budget end, an end-of-file or -- worst -- an operator ``SIGTERM``
#: actively CANCELLED an automatic emergency brake that the arbiter had
#: commanded on the previous frame.  A shutdown is not evidence that the hazard
#: went away.  While the arbiter is in one of these states the settle releases
#: the throttle and HOLDS the brake; see :meth:`PipelineRunner._settle`.
INTERVENING_STATES = (SafetyState.MIN_RISK_MANEUVER, SafetyState.DISENGAGE)


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
    #: What :meth:`PipelineRunner._settle` did: ``"release"`` (throttle and brake
    #: to zero), ``"hold"`` (throttle released, the arbiter's brake HELD because
    #: it was mid-intervention) or ``"min_risk"`` (the fail-safe ramp after an
    #: unsafe exit).  ``""`` means settle has not run.
    settle_kind: str = ""
    #: The brake value of the LAST command settle actuated.  A caller that wants
    #: to know whether the vehicle was left braking reads this, not the log.
    settle_brake: float = 0.0
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
        failsafe_hold_s: how long :meth:`_settle` keeps re-emitting a post-loop
            command -- the arbiter's minimum-risk manoeuvre after an UNSAFE exit
            (a lost source or a dead pipeline), or the arbiter's held brake when
            a CLEAN exit interrupted an intervention -- in seconds.  One command per nominal period.  ``0`` emits
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
    #: The arbitration from the last REAL frame, snapshotted by :meth:`_settle`
    #: before the hold begins.  ``failsafe_command`` overwrites
    #: ``pipeline.last_arbitration`` on every hold step.
    _last_frame_arbitration: Any = field(default=None, init=False)
    _floor_logged: bool = field(default=False, init=False)
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
        merely ``break``\\ s leaves the previous frame's throttle applied -- the
        ADAS-DEC-21 failure this module claims to have fixed for the *exception*
        path.

        The decision has TWO inputs, and the bug this replaced used only one.
        The exit reason says why the software stopped; the arbiter's state says
        what the vehicle was being asked to do at the time.  What "shutting down
        safely" means, per exit reason:

        * :data:`EXIT_SOURCE_LOST` -- the camera stopped delivering mid-stream.
          The vehicle is moving and blind.  Re-emit
          :meth:`~adas.runtime.pipeline.ADASPipeline.failsafe_command` -- the
          arbiter's minimum-risk manoeuvre -- once per nominal period for
          ``failsafe_hold_s`` seconds.  Re-emitting matters: the arbiter's brake
          demand is rate limited, so a single command stops the ramp partway.
        * :data:`EXIT_PIPELINE_DEAD` -- ``max_consecutive_failures`` frames in a
          row failed.  The decision path is gone; identical treatment.
        * :data:`EXIT_EOF` -- a finite recording ended.  There will be no more
          evidence about the road, so an intervention in progress cannot be
          shown to be over.
        * :data:`EXIT_COMPLETED` -- the ``--frames`` budget ran out.  Same: the
          budget expiring is a fact about the operator's request, not the road.
        * :data:`EXIT_STOPPED` -- ``request_stop()``, i.e. how the CLI's
          SIGINT/SIGTERM handler unwinds a ``systemctl stop``.  THIS is the one
          that matters most: an operator stopping the service mid-emergency must
          not thereby release the brake.

        So the three intended ends release the THROTTLE unconditionally -- no
        shutdown justifies drive torque -- and release the BRAKE only when the
        arbiter was NOT BRAKING.  When it was, the arbiter's own last brake and
        steering are held instead, for the same bounded ``failsafe_hold_s``.

        The gate is the last commanded brake, not the arbitration STATE, and that
        is a deliberate widening of the original fix.  400 frames of
        ``example.mp4`` at 15 m/s produce twelve consecutive full-authority
        (>= 0.9) brake frames -- frames 170-181, against a real tracked lead at
        7.3-7.5 m with the headway violation ``headway_7.0m_below_rss_12.5m`` --
        and the arbiter is in ``LIMITED`` for every one of them.  It never enters
        ``MIN_RISK_MANEUVER`` on that clip at all.  A state-only gate would
        therefore have protected exactly none of the real full-brake frames while
        appearing to fix the defect.  The invariant that survives contact with
        real footage is the simpler one: **settle never reduces the brake.**

        The cost of the wider gate is that a shutdown during light comfort
        braking latches that small brake instead of clearing it.  A 216-cell
        sweep (ego 5-30 m/s x initial range 6-60 m x a stopped / half-speed /
        speed-matched lead x EOF and frame-budget exits) holds in 190 cells and
        releases in 26; the speed-matched cells hold a brake of only ~0.07,
        which is drag rather than an intervention.  On the real clip the picture
        is the other way round -- 342 of 400 frames command exactly zero brake,
        including the last one, so a normal run still ends in a full release.
        The trade is deliberate: latching 0.07 costs a slow coast-down, and
        releasing 0.9 costs a collision.  It is reported in
        ``RunSummary.settle_brake`` and logged rather than hidden.

        :data:`INTERVENING_STATES` still matters: an arbiter that claims
        ``MIN_RISK_MANEUVER`` while commanding no brake is a contradiction, and
        that case falls back to the arbiter's fail-safe rather than coasting.

        The hold is BOUNDED in every branch: the runner hands control back to its
        caller afterwards, it does not supervise the vehicle forever.
        """
        steps = self._settle_steps(nominal_dt_s)
        # Snapshot BEFORE the hold: ``failsafe_command`` overwrites
        # ``pipeline.last_arbitration`` on every step, so reading the floor
        # inside the loop would read the decaying fail-safe back to itself.
        self._last_frame_arbitration = self._safe_last_arbitration()
        self._floor_logged = False

        if exit_kind in UNSAFE_EXITS:
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
            summary.settle_kind = "min_risk"
            last: Optional[ControlCommand] = None
            for _ in range(steps):
                started = time.monotonic()
                last = self._floored(
                    self.pipeline.failsafe_command(dt_s=nominal_dt_s), frame_id
                )
                self._actuate_settled(last, nominal_dt_s, frame_id, summary)
                self._pace(started, nominal_dt_s)
            if last is not None:
                summary.settle_brake = last.brake
                logger.critical(
                    "Minimum-risk hold finished after %d command(s); last actuated "
                    "t%.2f/b%.2f/s%+.2f",
                    summary.settle_commands,
                    last.throttle,
                    last.brake,
                    last.steering,
                )
            return

        state = self._arbitration_state()
        held = self._held_command()
        if held is None and state in INTERVENING_STATES:
            # The arbiter says it is intervening but left no usable brake
            # behind.  That is a contradiction, not a licence to coast: ask
            # the arbiter directly, exactly as an unsafe exit does.
            logger.critical(
                "Frame loop finished ('%s') at frame %s with the arbiter in '%s' but "
                "no usable brake on its last command; falling back to the arbiter's "
                "fail-safe rather than releasing.",
                exit_kind,
                frame_id,
                state.value,
            )
            summary.settle_kind = "min_risk"
            last_fs: Optional[ControlCommand] = None
            for _ in range(steps):
                started = time.monotonic()
                last_fs = self._floored(
                    self.pipeline.failsafe_command(dt_s=nominal_dt_s), frame_id
                )
                self._actuate_settled(last_fs, nominal_dt_s, frame_id, summary)
                self._pace(started, nominal_dt_s)
            if last_fs is not None:
                summary.settle_brake = last_fs.brake
            return

        if held is not None:
            # The ACTUATION rule has no threshold in it -- the brake is held
            # whatever its size.  Only the LOG LEVEL is graded, by the arbiter's
            # own severity judgement, so that a 0.07 comfort brake held at the
            # end of a normal run does not read like a cancelled AEB and drown
            # the one that matters.  No number is invented for this.
            emit = logger.critical if state in INTERVENING_STATES else logger.warning
            emit(
                "Frame loop finished ('%s') at frame %s WHILE THE ARBITER WAS BRAKING "
                "(state '%s'). Releasing the throttle and HOLDING the brake at %.2f "
                "(steering %+.2f) for %d command(s) (%.2f s). Ending the run is not "
                "evidence that the hazard cleared, and stopping the service does not "
                "cancel an intervention in progress.",
                exit_kind,
                frame_id,
                "unknown" if state is None else state.value,
                held.brake,
                held.steering,
                steps,
                steps * nominal_dt_s,
            )
            summary.settle_kind = "hold"
            summary.settle_brake = held.brake
            for _ in range(steps):
                started = time.monotonic()
                self._actuate_settled(held, nominal_dt_s, frame_id, summary)
                self._pace(started, nominal_dt_s)
            return

        logger.info(
            "Frame loop finished ('%s') at frame %s with the arbiter in '%s'; "
            "releasing the throttle and the brake.",
            exit_kind,
            frame_id,
            "never ran" if state is None else state.value,
        )
        summary.settle_kind = "release"
        summary.settle_brake = 0.0
        self._actuate_settled(
            ControlCommand(throttle=0.0, brake=0.0, steering=0.0),
            nominal_dt_s,
            frame_id,
            summary,
        )

    def _floored(self, command: ControlCommand, frame_id: int) -> ControlCommand:
        """Never actuate LESS brake on the way out than the arbiter last commanded.

        The minimum-risk hold asks the arbiter for a fail-safe with NO tracks and
        a failed perception status, so the arbiter sees ``no_in_path_lead`` and
        demands only its generic ``mrm_decel_mps2``.  When the loop exited during
        a full-authority AEB that is a REDUCTION: a lost camera walked a
        measured, closing-lead brake of 1.00 down to 0.44 over the hold via the
        brake release-rate limiter.  Going blind is not evidence that the lead
        went away, so the last arbitrated brake becomes a floor for the whole
        hold.  It can only ever raise the demand, and it cannot invent one: the
        floor is 0.0 unless the arbiter had already commanded a brake on a real
        frame.
        """
        floor = self._brake_floor()
        if command.brake >= floor:
            return command
        if not self._floor_logged:
            self._floor_logged = True
            logger.critical(
                "The fail-safe would have lowered the brake from %.2f to %.2f at frame "
                "%s; holding it at %.2f instead. Losing the source is not evidence that "
                "the hazard cleared.",
                floor,
                command.brake,
                frame_id,
                floor,
            )
        return ControlCommand(throttle=0.0, brake=floor, steering=command.steering)

    def _brake_floor(self) -> float:
        """The brake the arbiter last commanded on a REAL frame, or 0.0."""
        arbitration = self._last_frame_arbitration
        command = getattr(arbitration, "command", None) if arbitration is not None else None
        try:
            brake = float(getattr(command, "brake", 0.0))
        except (TypeError, ValueError):
            return 0.0
        return brake if 0.0 < brake <= 1.0 else 0.0

    def _settle_steps(self, nominal_dt_s: float) -> int:
        """How many post-loop commands one bounded hold is worth."""
        if self.failsafe_hold_s > 0.0 and nominal_dt_s > 0.0:
            return max(1, int(round(self.failsafe_hold_s / nominal_dt_s)))
        return 1

    def _arbitration_state(self) -> Optional[SafetyState]:
        """The arbiter's state as of the last frame, or ``None`` if it never ran.

        Read defensively: this runs on the shutdown path, where a pipeline that
        is already broken must not be able to turn a settle into a traceback and
        so leave the actuators holding the last frame's command.
        """
        arbitration = self._safe_last_arbitration()
        if arbitration is None:
            return None
        state = getattr(arbitration, "state", None)
        return state if isinstance(state, SafetyState) else None

    def _safe_last_arbitration(self):
        try:
            return self.pipeline.last_arbitration
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            logger.error("Could not read the arbitration state at exit: %s", exc)
            return None

    def _held_command(self) -> Optional[ControlCommand]:
        """The command to keep emitting when the loop ends mid-intervention.

        The throttle is released and the arbiter's own last brake and steering
        are HELD verbatim.  They are deliberately NOT re-derived and NOT
        escalated: no new frame arrived, so there is no new evidence, and the
        arbiter already floored the brake at its minimum-risk deceleration when
        it entered the state (``_synthesise_command``).  Holding is the strongest
        claim the runner can make honestly -- and it is strictly more brake than
        the zero-brake release this replaced.

        Returns ``None`` when there is no usable brake to hold, so the caller can
        fall back to the arbiter's fail-safe instead of inventing a number.
        """
        arbitration = self._safe_last_arbitration()
        command = getattr(arbitration, "command", None) if arbitration is not None else None
        if command is None:
            return None
        try:
            brake = float(command.brake)
            steering = float(command.steering)
        except (TypeError, ValueError, AttributeError):
            return None
        if not (brake > 0.0) or brake != brake:  # non-positive, or NaN
            return None
        return ControlCommand(
            throttle=0.0,
            brake=min(1.0, brake),
            steering=max(-1.0, min(1.0, steering)),
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
                        self._publish_arbitration(health, arbitration)
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
                        self._publish_arbitration(health, arbitration)
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

    def _publish_arbitration(self, health, arbitration) -> None:
        """Copy the arbiter's own decision into the health snapshot.

        The caller already holds ``health.lock``.  Reporting only: nothing here
        is read back by the decision path, and every field is a plain snapshot of
        the last :class:`~adas.core.models.ArbitrationResult`, so a stale or
        missing value degrades the endpoint and never the vehicle.

        ``state`` alone answers "is it degraded?" and nothing else.  An operator
        looking at a vehicle that is braking needs the rest: how hard the arbiter
        is demanding, whether it actually CHANGED the command it was handed (an
        arbiter that agrees with the planner is invisible in the state label),
        which lead it selected, and whether that lead's closing rate is a
        measurement or is still unknown -- because a backstop that has not
        measured a rate is a backstop that will not brake.
        """
        arbiter = getattr(self.pipeline.safety_monitor, "arbiter", None)
        lead = getattr(arbiter, "last_lead", None) if arbiter is not None else None
        health.safety_reason = arbitration.reason or ""
        health.safety_demand_mps2 = float(getattr(arbiter, "demand_mps2", 0.0) or 0.0)
        health.safety_overrode_command = bool(arbitration.violations)
        health.safety_last_violations = list(arbitration.violations)[:8]
        health.safety_lead_track_id = getattr(lead, "track_id", None)
        health.safety_rate_is_measured = bool(getattr(lead, "rate_is_measured", False))

    def _note_safety_state(self, frame_id: int) -> None:
        """Record a change of safety state: one event, one health counter.

        Deliberately edge triggered.  A safety state is a steady condition and a
        run can hold one for thousands of frames; what an operator and a fleet
        rule act on is the ENTRY and the EXIT, so exactly one event is written
        per transition and none per frame.

        The health counter is incremented independently of the event log, so that
        a deployment with no event log still reports how many times the state has
        moved -- which is the number that distinguishes a system that degraded
        once from one that is oscillating.
        """
        arbitration = self.pipeline.last_arbitration
        if arbitration is None:
            return
        current = arbitration.state.value
        previous = getattr(self, "_last_safety_state", "nominal")
        if current == previous:
            return
        self._last_safety_state = current

        health = self.hooks.health
        if health is not None:
            try:
                with health.lock:
                    health.safety_transitions += 1
                    health.safety_last_transition_frame = frame_id
            except Exception as exc:  # noqa: BLE001
                logger.debug("health transition counter failed: %s", exc)

        events = self.hooks.events
        if events is None:
            return
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
    "INTERVENING_STATES",
    "UNSAFE_EXITS",
    "PipelineRunner",
    "RunSummary",
    "RunnerHooks",
    "synthetic_frame",
]
