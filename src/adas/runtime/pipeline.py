"""End-to-end ADAS inference pipeline.

Order of one step::

    detect -> lane -> track -> depth cross-check -> plan -> control -> ARBITRATE

Three properties of this module are safety-relevant and are each covered by a
test in ``tests/test_pipeline.py``:

**The arbiter is authoritative.**  ``step`` returns the command that
:meth:`adas.control.SafetyMonitor.arbitrate` produced, never the controller's.
The previous implementation caught the safety violation, logged it at WARNING and
then actuated the rejected command anyway (ADAS-DEC-01).  ``last_arbitration``
carries the full :class:`~adas.core.models.ArbitrationResult` -- state,
violations, reason -- for the driver interface, the health endpoint and the
event log.

**A perception failure is a fault, never an empty road.**  An exception out of
the detector or the lane estimator produces a
:class:`~adas.core.models.PerceptionStatus` with ``ok=False`` and a rising
``consecutive_failures``, and the planner is told ``perception_valid=False`` so
it holds and ramps down (ADAS-PERC-08 / ADAS-OPS-01).  No detection is
fabricated and no track is corrected against one.  The tracker is still
*advanced*, by a predict-only step (:meth:`ADASPipeline._coast_tracks`): live
tracks extrapolate, their covariances grow, ``time_since_update`` rises, their
range estimate becomes ``UNAVAILABLE`` and they are deleted once they outlive
``max_missed``.  Freezing them instead -- the previous behaviour -- left every
track holding its pre-dropout position with ``time_since_update == 0``, so the
frame perception recovered on had to associate against an N-frame-stale
prediction that still claimed full confidence.  An empty ``detections`` list
from a *working* detector still means what it says: the road is clear.

**Ego speed is never invented.**  ``step`` takes an
:class:`~adas.core.models.EgoState` whose ``valid`` flag says whether the number
is usable.  There is no vehicle bus on this board, so the state is built by
:class:`adas.runtime.capture.EgoSpeedSource` from a configured constant, a
recorded channel or a simulated plant, each of which reports honestly whether it
is a measurement.  With ``valid=False`` the constant-time-gap law is undefined
and the planner degrades -- it does not substitute a plausible speed.

Timing: every duration is measured with ``time.monotonic``.  ``dt_s`` is the
*measured* interval supplied by the runner, not the scheduled period
(ADAS-DEC-17); the tracker, the range filters, the jerk limits and the arbiter's
timing checks all consume it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from adas.control import PIDLikeLongitudinalController, SafetyMonitor
from adas.control.arbiter import SafetyContext
from adas.core.exceptions import ADASException
from adas.core.logger import Throttle, setup_logger
from adas.core.metrics import PerformanceMetrics
from adas.core.models import (
    ArbitrationResult,
    ControlCommand,
    DrivableArea,
    EgoState,
    LaneModel,
    MotionPlan,
    PerceptionFrame,
    PerceptionStatus,
    RangeEstimate,
    RangeSource,
    SafetyState,
    TrackedObject,
)
from adas.planning import BehaviorPlanner
from adas.tracking import MultiObjectTracker

logger = setup_logger(__name__)

#: Per-frame INFO is unreadable at 20 Hz and hides everything that matters, so
#: the routine line is DEBUG and this throttle emits one INFO summary a second.
_SUMMARY_THROTTLE_S = 1.0


@dataclass
class ADASPipeline:
    """Perception, planning, control and arbitration for one frame at a time.

    Not thread-safe: each backend owns a TensorRT execution context and a set of
    host buffers, and the metrics object is single-writer by design.  One
    pipeline instance belongs to one thread.

    Attributes:
        detector: object exposing ``infer(frame, width, height) -> [BoundingBox]``.
        lane_estimator: object exposing ``estimate(frame, width, height) -> LaneModel | None``
            and optionally ``drivable_area() -> DrivableArea | None``.
        tracker: the multi-object tracker.
        planner: the behaviour planner.
        controller: the longitudinal/lateral controller.
        safety_monitor: holds the authoritative arbiter.
        depth_channel: optional independent range channel (MiDaS).  ``None``
            disables the range cross-check, and the arbiter then leaves every
            lead's range source at ``PINHOLE`` so a consumer can tell that no
            second opinion existed.
        camera: optional :class:`adas.perception.geometry.CameraConfig`, needed by
            the depth channel to anchor its scale on the road plane.
        ego_lane_half_width_frac: fraction of the frame width either side of the
            lane centre the arbiter treats as in-path when it has no metric lane.
        lane_every_n_frames: run the lane backend every N frames.  YOLOP costs
            ~59 ms end to end and cannot run at 20 Hz.
        lane_max_age_frames: how long a lane model may be reused after a skipped
            frame.  A reused model is republished with its confidence scaled down
            by its age -- it is old evidence, and it is labelled as old evidence.
    """

    detector: Any
    lane_estimator: Any
    tracker: MultiObjectTracker
    planner: BehaviorPlanner
    controller: PIDLikeLongitudinalController
    safety_monitor: SafetyMonitor = field(default_factory=SafetyMonitor)
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    depth_channel: Any = None
    camera: Any = None
    ego_lane_half_width_frac: float = 0.20
    lane_every_n_frames: int = 1
    lane_max_age_frames: int = 3

    # --- state ---------------------------------------------------------------
    last_arbitration: Optional[ArbitrationResult] = field(default=None, init=False)
    last_plan: Optional[MotionPlan] = field(default=None, init=False)
    last_ego: Optional[EgoState] = field(default=None, init=False)
    last_tracks: List[TrackedObject] = field(default_factory=list, init=False)
    last_lead_distance_m: Optional[float] = field(default=None, init=False)
    perception: PerceptionStatus = field(default_factory=PerceptionStatus, init=False)

    _frame_count: int = field(default=0, init=False)
    _current_speed_mps: float = field(default=0.0, init=False)
    _lane_cache: Optional[LaneModel] = field(default=None, init=False)
    _lane_cache_frame: int = field(default=-10**9, init=False)
    _drivable: Optional[DrivableArea] = field(default=None, init=False)
    _summary: Throttle = field(default_factory=lambda: Throttle(_SUMMARY_THROTTLE_S), init=False)
    _last_alert: Tuple[str, Tuple[str, ...]] = field(default=("", ()), init=False)

    # ------------------------------------------------------------------ step

    def step(
        self,
        frame: PerceptionFrame,
        current_speed_mps: Optional[float] = None,
        dt_s: float = 0.05,
        ego: Optional[EgoState] = None,
    ) -> Tuple[MotionPlan, ControlCommand]:
        """Execute one pipeline step.

        Args:
            frame: the captured frame.  ``frame.detections``, ``frame.lane``,
                ``frame.status``, ``frame.drivable`` and ``frame.ego`` are filled
                in by this call.
            current_speed_mps: convenience for callers that only have a speed.
                Passing a number asserts that it is usable, so the resulting
                :class:`EgoState` is ``valid=True``.  ``None`` (the default)
                means *no ego speed is available* and produces ``valid=False``.
                Ignored when *ego* is supplied.
            dt_s: MEASURED seconds since the previous step.
            ego: the ego state for this frame, from
                :class:`adas.runtime.capture.EgoSpeedSource`.  Preferred over
                *current_speed_mps*: it carries validity and a timestamp.

        Returns:
            ``(plan, command)`` where ``command`` is the ARBITRATED command --
            the only thing that may reach an actuator.

        Raises:
            ADASException: an unrecoverable failure in tracking, planning,
                control or arbitration.  Perception failures are handled inside
                and never raise.  A caller that catches this MUST actuate
                :meth:`failsafe_command`, not the previous command.
        """
        try:
            self._frame_count += 1
            timestamp_s = frame.timestamp_s if frame.timestamp_s else time.monotonic()

            ego = self._resolve_ego(ego, current_speed_mps, timestamp_s)
            frame.ego = ego
            self.last_ego = ego
            self._current_speed_mps = ego.speed_mps if ego.valid else 0.0

            # --- perception ---------------------------------------------------
            detector_ok, lane_ok = self._run_perception(frame)
            status = self._update_perception_status(detector_ok, lane_ok, timestamp_s, frame)
            frame.status = status

            # --- tracking -----------------------------------------------------
            # A perception fault must not be folded into the tracker as a
            # measurement -- "no detections" reads as "every object
            # disappeared", which is a measurement nobody made. It must not
            # FREEZE the tracker either: a track left untouched for the length
            # of the dropout keeps its pre-dropout position and
            # time_since_update == 0, so the recovery frame associates against a
            # stale prediction that still claims full confidence. The dropout
            # frame therefore runs a predict-only step instead; the planner is
            # separately told perception_valid=False and degrades regardless of
            # what the coasting tracks say.
            with self.metrics.stage("track"):
                if detector_ok:
                    tracked = self.tracker.update(
                        frame.detections,
                        dt_s=dt_s,
                        frame_width=frame.width,
                        frame_height=frame.height,
                        lane=frame.lane,
                        drivable=frame.drivable,
                    )
                else:
                    tracked = self._coast_tracks(frame, dt_s)
            self.last_tracks = tracked

            # --- independent range cross-check --------------------------------
            independent = self._run_depth(frame, tracked)

            # --- planning -----------------------------------------------------
            lane_center = frame.lane.lane_center_px if frame.lane is not None else None
            with self.metrics.stage("plan"):
                plan = self.planner.plan(
                    frame_width_px=frame.width,
                    lane_center_px=lane_center,
                    objects=tracked,
                    ego=ego,
                    perception_valid=detector_ok,
                    dt_s=dt_s,
                    lane=frame.lane,
                    frame_height_px=frame.height,
                )
            self.last_plan = plan

            # --- control ------------------------------------------------------
            with self.metrics.stage("control"):
                command = self.controller.to_command(
                    plan,
                    self._current_speed_mps,
                    dt_s=dt_s,
                    emergency=self._is_emergency(plan),
                )

            # --- arbitration: this is what actuates ----------------------------
            with self.metrics.stage("arbitrate"):
                context = SafetyContext(
                    ego=ego,
                    tracks=tracked,
                    perception=status,
                    dt_s=dt_s,
                    timestamp_s=timestamp_s,
                    # `timestamp_s` here is `frame.timestamp_s`, i.e. the time
                    # the image was captured, which is exactly what the range
                    # fits must be placed on.  Passed explicitly so that the
                    # arbiter does not have to assume it.
                    measurement_t_s=timestamp_s,
                    frame_width_px=frame.width,
                    frame_height_px=frame.height,
                    lane=frame.lane,
                    ego_lane_half_width_frac=self.ego_lane_half_width_frac,
                    independent_ranges=independent or None,
                    lateral_offset_m=self._lateral_offset_m(frame.lane),
                )
                result = self.safety_monitor.arbitrate(plan, command, context)
            self.last_arbitration = result
            command = result.command

            self._record(frame, tracked, plan, command, result, dt_s)
            return plan, command

        except ADASException:
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed failure
            logger.error("Pipeline step failed: %s", exc, exc_info=True)
            raise ADASException("Pipeline execution failed: %s" % exc) from exc

    # ---------------------------------------------------------------- helpers

    def _resolve_ego(
        self,
        ego: Optional[EgoState],
        current_speed_mps: Optional[float],
        timestamp_s: float,
    ) -> EgoState:
        """Normalise the three ways a caller can supply ego speed."""
        if ego is not None:
            return ego
        if current_speed_mps is None:
            return EgoState(speed_mps=0.0, valid=False, timestamp_s=timestamp_s)
        return EgoState(
            speed_mps=max(0.0, float(current_speed_mps)),
            valid=True,
            timestamp_s=timestamp_s,
        )

    def _coast_tracks(self, frame: PerceptionFrame, dt_s: float) -> List[TrackedObject]:
        """Advance the tracker by ``dt_s`` with NO measurement (a dropout frame).

        An empty detection list handed to
        :meth:`~adas.tracking.MultiObjectTracker.update` IS the predict-only
        step: with nothing to associate and nothing to spawn, every live track
        runs its Kalman predict and then ``_apply_miss``, which sets the status
        to ``COASTING``, publishes the PREDICTED box, sets the range estimate to
        ``RangeSource.UNAVAILABLE`` and increments ``time_since_update``.
        ``_prune`` then deletes anything that has coasted past ``max_missed``.

        The returned tracks are therefore explicitly marked as unmeasured --
        every one carries ``time_since_update >= 1`` and an ``UNAVAILABLE``
        range estimate -- so no consumer can mistake a coasted track for a
        measured one.

        A tracker failure on a frame that has *already* failed is logged and
        swallowed rather than escalated: the frame then behaves exactly as it
        did before coasting existed (no tracks), which is degraded but not
        worse than the previous behaviour.
        """
        try:
            return self.tracker.update(
                [],
                dt_s=dt_s,
                frame_width=frame.width,
                frame_height=frame.height,
                lane=frame.lane,
                drivable=frame.drivable,
            )
        except Exception as exc:  # noqa: BLE001 - this frame has already failed
            logger.error(
                "Tracker coast failed on dropout frame %s: %s; continuing with no tracks",
                frame.frame_id,
                exc,
            )
            return []

    def _run_perception(self, frame: PerceptionFrame) -> Tuple[bool, bool]:
        """Run the detector and the lane estimator, isolating their failures.

        Returns:
            ``(detector_ok, lane_ok)``.  A backend returning *no* result is not a
            failure -- an empty detection list is an empty road and a ``None``
            lane is "no lane visible".  Only an exception is a failure.
        """
        detector_ok = True
        lane_ok = True

        try:
            with self.metrics.stage("detect"):
                frame.detections = list(self.detector.infer(frame.rgb, frame.width, frame.height))
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            detector_ok = False
            frame.detections = []
            self.metrics.record_perception_failure(stage="detect")
            logger.error("Detector failed on frame %s: %s", frame.frame_id, exc, exc_info=True)

        try:
            with self.metrics.stage("lane"):
                frame.lane = self._run_lane(frame)
        except Exception as exc:  # noqa: BLE001
            lane_ok = False
            frame.lane = None
            self._lane_cache = None
            self.metrics.record_perception_failure(stage="lane")
            logger.error("Lane estimator failed on frame %s: %s", frame.frame_id, exc, exc_info=True)

        try:
            self._drivable = self.lane_estimator.drivable_area()
        except Exception as exc:  # noqa: BLE001
            self._drivable = None
            logger.warning("Drivable-area read failed on frame %s: %s", frame.frame_id, exc)
        frame.drivable = self._drivable

        return detector_ok, lane_ok

    def _run_lane(self, frame: PerceptionFrame) -> Optional[LaneModel]:
        """Run the lane backend on its schedule, ageing a reused model honestly.

        A backend that costs more than the frame period cannot run every frame.
        On a skipped frame the previous model is republished with its confidence
        scaled by ``1 - age/max_age`` so that a consumer gating on confidence
        naturally stops trusting it, and it is dropped entirely once it is older
        than ``lane_max_age_frames``.  Re-serving stale evidence at full
        confidence would be indistinguishable from a fresh measurement.
        """
        every = max(1, int(self.lane_every_n_frames))
        due = every == 1 or (self._frame_count - 1) % every == 0
        if due:
            model = self.lane_estimator.estimate(frame.rgb, frame.width, frame.height)
            self._lane_cache = model
            self._lane_cache_frame = self._frame_count
            return model

        cached = self._lane_cache
        if cached is None:
            return None
        age = self._frame_count - self._lane_cache_frame
        max_age = max(0, int(self.lane_max_age_frames))
        if age > max_age:
            self._lane_cache = None
            return None
        if max_age == 0 or cached.confidence <= 0.0:
            return cached
        decay = max(0.0, 1.0 - float(age) / float(max_age + 1))
        return replace(cached, confidence=cached.confidence * decay)

    def _update_perception_status(
        self,
        detector_ok: bool,
        lane_ok: bool,
        timestamp_s: float,
        frame: PerceptionFrame,
    ) -> PerceptionStatus:
        ok = detector_ok and lane_ok
        if ok:
            if not self.perception.ok:
                logger.info(
                    "Perception recovered on frame %s after %d consecutive failures",
                    frame.frame_id,
                    self.perception.consecutive_failures,
                )
            self.perception = PerceptionStatus(
                ok=True,
                consecutive_failures=0,
                last_good_timestamp_s=timestamp_s,
                detector_ok=True,
                lane_ok=True,
                reason="",
            )
        else:
            reason = ",".join(
                name for name, good in (("detector", detector_ok), ("lane", lane_ok)) if not good
            )
            self.perception = PerceptionStatus(
                ok=False,
                consecutive_failures=self.perception.consecutive_failures + 1,
                last_good_timestamp_s=self.perception.last_good_timestamp_s,
                detector_ok=detector_ok,
                lane_ok=lane_ok,
                reason=reason,
            )
        return self.perception

    def _run_depth(
        self,
        frame: PerceptionFrame,
        tracked: List[TrackedObject],
    ) -> Dict[int, RangeEstimate]:
        """Sample the depth channel for each track and return the usable results.

        The channel is fed the TRACK boxes rather than the raw detections so the
        result is index-aligned with ``tracked`` and needs no box-to-track
        matching heuristic.  Estimates that come back ``UNAVAILABLE`` are dropped
        rather than published as a zero-confidence range: the arbiter's contract
        is that a missing key means "no second opinion", which is a different
        claim from "the second opinion is 0 m".
        """
        if self.depth_channel is None or not tracked:
            return {}
        boxes = [obj.box for obj in tracked]
        reference = [obj.range_estimate for obj in tracked]
        try:
            with self.metrics.stage("depth"):
                estimates = self.depth_channel.update(
                    frame.rgb,
                    frame.frame_id,
                    boxes,
                    frame.width,
                    frame.height,
                    reference=reference,
                    camera=self.camera,
                )
        except Exception as exc:  # noqa: BLE001 - a monitor must not kill the frame
            logger.error("Depth range channel failed on frame %s: %s", frame.frame_id, exc)
            return {}

        out: Dict[int, RangeEstimate] = {}
        for obj, estimate in zip(tracked, estimates):
            if estimate is None or estimate.source == RangeSource.UNAVAILABLE:
                continue
            if estimate.confidence <= 0.0:
                continue
            out[obj.track_id] = estimate
        return out

    def _lateral_offset_m(self, lane: Optional[LaneModel]) -> Optional[float]:
        """Metric ego offset from the lane centre, or ``None`` when unavailable.

        Positive is "vehicle right of centre".  Returns ``None`` for a mock lane,
        a lane fitted without a camera, or an implausible boundary pair -- the
        arbiter then reports ``lane_offset_unavailable`` rather than silently
        passing its lane-departure check.
        """
        if lane is None or lane.is_mock:
            return None
        try:
            from adas.perception.lane import lane_geometry_from_model

            geometry = lane_geometry_from_model(lane)
        except Exception as exc:  # noqa: BLE001 - reporting helper, never fatal
            logger.debug("lane_geometry_from_model failed: %s", exc)
            return None
        if geometry is None or not geometry.plausible:
            return None
        return geometry.lateral_offset_m

    def _is_emergency(self, plan: MotionPlan) -> bool:
        """Whether the controller may use its emergency jerk and brake rates."""
        reason = plan.reason or ""
        return "aeb" in reason or "emergency" in reason

    def _record(
        self,
        frame: PerceptionFrame,
        tracked: List[TrackedObject],
        plan: MotionPlan,
        command: ControlCommand,
        result: ArbitrationResult,
        dt_s: float,
    ) -> None:
        """Metrics, lead bookkeeping and the (throttled) operator-visible line."""
        if result.violations:
            for kind in result.violations:
                self.metrics.record_safety_event(is_violation=True, kind=_metric_kind(kind))
        elif result.state is not SafetyState.NOMINAL:
            self.metrics.record_safety_event(is_violation=False, kind=result.state.value)

        self.metrics.update_frame(
            frame_time=dt_s,
            num_detections=len(frame.detections),
            num_tracks=len(tracked),
            has_lane=frame.lane is not None,
        )

        lead = min((obj for obj in tracked if obj.in_ego_lane), key=lambda o: o.distance_m, default=None)
        if lead is None:
            lead = min(tracked, key=lambda o: o.distance_m, default=None)
        self.last_lead_distance_m = lead.distance_m if lead is not None else None

        line = (
            "frame=%s det=%d trk=%d lane=%s%s plan=%s safety=%s cmd=t%.2f/b%.2f/s%+.2f"
            % (
                frame.frame_id,
                len(frame.detections),
                len(tracked),
                "yes" if frame.lane is not None else "no",
                "" if lead is None else " lead=%.1fm" % lead.distance_m,
                plan.reason,
                result.state.value,
                command.throttle,
                command.brake,
                command.steering,
            )
        )
        logger.debug(line)
        # A degradation is worth a WARNING the first time it appears and every
        # time it CHANGES; repeating it at 20 Hz buries everything else and
        # trains an operator to ignore the channel that matters most.
        alert = (result.state.value, tuple(result.violations))
        if result.state is not SafetyState.NOMINAL and alert != self._last_alert:
            logger.warning(
                "%s violations=%s", line, ",".join(result.violations) or "(none)"
            )
            self._last_alert = alert
        elif result.state is SafetyState.NOMINAL and self._last_alert != ("", ()):
            logger.info("Safety state returned to nominal: %s", line)
            self._last_alert = ("", ())
        elif self._summary.ready():
            suppressed = self._summary.last_suppressed
            logger.info("%s%s", line, "" if not suppressed else " (+%d frames)" % suppressed)

    # ----------------------------------------------------------- fail-safe API

    def failsafe_command(self, dt_s: float = 0.05) -> ControlCommand:
        """The command to actuate when :meth:`step` raised.

        Runs the arbiter with no plan, a neutral command and a failed perception
        status, which puts it into a minimum-risk manoeuvre and returns its own
        rate-shaped braking command.  Callers MUST emit this rather than latching
        the previous command on the actuators (ADAS-DEC-21): a latched throttle
        after a pipeline crash is the worst available behaviour.

        The context carries a ``timestamp_s`` on purpose.  The arbiter advances
        its timing baseline only from that field, so a run of fail-safe frames
        built without one leaves ``_last_timestamp_s`` stuck at the last good
        frame; the frame perception recovers on then presents a gap of N*dt, is
        flagged ``timing_stale_input`` and has its dt forced to ``max_dt_s`` --
        corrupting the achieved-acceleration and jerk difference quotients and
        the range-rate filter at exactly the wrong moment.  ``time.monotonic``
        is the same clock the runner stamps frames with.
        """
        try:
            result = self.safety_monitor.arbitrate(
                None,
                ControlCommand(0.0, 0.0, 0.0),
                SafetyContext(
                    ego=EgoState(
                        speed_mps=self._current_speed_mps,
                        valid=self.last_ego.valid if self.last_ego is not None else False,
                        timestamp_s=time.monotonic(),
                    ),
                    tracks=[],
                    perception=PerceptionStatus(
                        ok=False,
                        consecutive_failures=self.perception.consecutive_failures + 1,
                        reason="pipeline_exception",
                    ),
                    dt_s=dt_s,
                    timestamp_s=time.monotonic(),
                ),
            )
            self.last_arbitration = result
            return result.command
        except Exception as exc:  # noqa: BLE001 - the last line of defence
            logger.critical(
                "Safety arbiter failed while producing a fail-safe command (%s); "
                "emitting a hard brake directly.",
                exc,
            )
            return ControlCommand(throttle=0.0, brake=1.0, steering=0.0)

    @property
    def safety_state(self) -> SafetyState:
        """The arbiter's latched state, for the health endpoint and diagnostics."""
        return self.safety_monitor.state

    def reset(self) -> None:
        """Clear every piece of cross-frame state, including a DISENGAGE latch."""
        logger.info("Pipeline reset")
        self.tracker.reset()
        self.planner.reset()
        self.controller.reset()
        self.safety_monitor.reset()
        self.metrics = PerformanceMetrics()
        self.perception = PerceptionStatus()
        self.last_arbitration = None
        self.last_plan = None
        self.last_ego = None
        self.last_tracks = []
        self.last_lead_distance_m = None
        self._frame_count = 0
        self._current_speed_mps = 0.0
        self._lane_cache = None
        self._lane_cache_frame = -10**9
        self._drivable = None
        self._last_alert = ("", ())

    def close(self) -> None:
        """Release every engine this pipeline owns.  Idempotent."""
        for name in ("detector", "lane_estimator", "depth_channel"):
            component = getattr(self, name, None)
            closer = getattr(component, "close", None)
            if closer is None:
                continue
            try:
                closer()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                logger.warning("Closing %s raised: %s", name, exc)

    def __enter__(self) -> "ADASPipeline":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


#: Arbiter violation strings are formatted (they carry numbers), but a Prometheus
#: label must have low cardinality. Map to a fixed vocabulary by prefix.
_KIND_PREFIXES = (
    "plan_speed",
    "plan_steering",
    "plan_accel",
    "command",
    "throttle",
    "brake",
    "steering_rate",
    "lateral_accel",
    "lane_departure",
    "lane_offset",
    "accel",
    "decel",
    "jerk",
    "timing",
    "dt",
    "stale",
    "ego",
    "perception",
    "range",
    "ttc",
    "headway",
    "rss",
)


def _metric_kind(violation: str) -> str:
    """Reduce a formatted violation string to a low-cardinality metric label."""
    text = str(violation).strip().lower()
    for prefix in _KIND_PREFIXES:
        if text.startswith(prefix):
            return prefix
    head = text.split()[0] if text.split() else "unspecified"
    return head.split("=")[0][:32] or "unspecified"


__all__ = ["ADASPipeline"]
