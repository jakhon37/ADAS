"""Longitudinal (speed) planning: constant-time-gap ACC plus an AEB stage.

Units
-----
``distance_m``        metres, range from the ego front bumper to the lead object.
``range_rate_mps``    metres per second.  SIGN CONVENTION is ``v_rel = v_lead - v_ego``,
                      so a NEGATIVE value means the gap is closing.  This is the
                      opposite of ``TrackedObject.velocity_mps``, which the tracker
                      publishes as positive-when-closing; :class:`LeadVehicle` is
                      built by the caller and the caller must do the sign flip.
speeds                metres per second (m/s).
accelerations         metres per second squared (m/s^2), always POSITIVE magnitudes
                      for the deceleration limits.
times                 seconds (s).

Control law
-----------
One continuous law covers the whole domain -- there is no branch, no threshold and
therefore no discontinuity to chatter across::

    d_desired = d0 + T * v_ego                              (constant time gap)
    v_raw     = v_ego + k_d * (d - d_desired) + k_v * v_rel
    v_target  = clamp(v_raw, 0, v_cruise)

``d0`` is :attr:`LongitudinalLimits.min_follow_distance_m` and ``T`` is
:attr:`LongitudinalLimits.time_gap_s`.  The law is provably monotone: ``v_target``
is non-decreasing in ``d`` (``dv/dd = k_d > 0``) and non-increasing in the closing
rate (``dv/dv_rel = k_v > 0`` and closing rate is ``-v_rel``).  ``tests/
test_longitudinal.py`` sweeps the whole domain and asserts exactly that.

On top of the comfort law sits a separate autonomous-emergency-braking (AEB) stage
that keys on time-to-collision and on the deceleration the geometry actually
requires.  Both AEB predicates are monotone in the same directions as the comfort
law, so adding them cannot break monotonicity.  The AEB stage is the only path
allowed to demand :attr:`LongitudinalLimits.emergency_decel_mps2`.

The planner is STATEFUL: the COMFORT target speed is rate-limited against the
previous target so that ``max_accel_mps2`` and ``max_decel_mps2`` are honoured by
construction.  Call :meth:`LongitudinalPlanner.reset` whenever the vehicle or the
replay is restarted, or the first frame of the new run will be rate-limited
against the last frame of the old one.

The AEB stage is DELIBERATELY EXEMPT from the downward rate limit.  An emergency
stop target must lead the vehicle, not trail it: while the AEB target was also
rate-limited at ``emergency_decel_mps2 * dt`` (0.4 m/s per 50 ms frame) the
published target tracked the measured speed down, the residual speed error stayed
near 0.4 m/s, and a proportional speed controller downstream therefore commanded
essentially no brake -- the whole AEB stage was decorative in the actuation path.
When AEB fires the target is 0 m/s on that same frame; shaping the deceleration is
the job of the controller's jerk limit and of the actuator, not of the planner.
Recovery out of AEB is still rate-limited upward at ``max_accel_mps2``.

Range-rate cross-check
----------------------
The AEB predicates are functions of the CLOSING RATE, which the planner does not
measure: it arrives on :attr:`LeadVehicle.range_rate_mps` from the tracker's
range filter.  If that one channel reports zero while the range is visibly
collapsing, every AEB predicate reads "not closing" and the planner plans a
comfort follow all the way into the back of the obstacle -- observed with a
stationary lead 25 m ahead at 15 m/s, where the planner said ``follow_gap`` on
all 40 frames and the controller commanded brake 0.000 on all 40.

So the planner keeps its own, independent estimate of the range rate: a
LEAST-SQUARES FIT of range against time over the last
``range_rate_cross_check_window`` frames of one continuously measured track.
This is a measurement of the same signal the gap maths already trusts, not an
assumption about the world.

Three independent gates stand between that estimate and the law, because this is
precisely the shape of bug that has to be got right -- an internally computed
quantity that can command a brake:

1. STATISTICAL SIGNIFICANCE. The fit reports its own standard error from the
   residual scatter of the range about the fitted line, and the estimate is only
   believed when it beats the reported rate by ``range_rate_significance_sigma``
   standard errors. This self-calibrates to however noisy the range channel
   happens to be, which the planner does not control and cannot know in advance.
   The first version of this cross-check used the median of the per-frame range
   DIFFERENCES instead; differencing multiplies the range noise by ``1/dt`` (20x
   at 20 Hz) and a median only divides it by 3, so with an HONEST rate channel
   and 0.25 m of gaussian range noise it fired on 180 frames in 3000, and with
   5 m of noise it manufactured AEB frames out of a lead that was not closing at
   all. The regression plus its own error bar fires on none of those.
2. AN ABSOLUTE FLOOR. The disagreement must also exceed
   ``range_rate_disagreement_mps`` outright, so ordinary filter lag on a
   low-noise channel cannot perturb the law however tight the error bar gets.
3. A PHYSICAL CLAMP. The correction is clamped so it can never imply anything
   worse than driving at a STATIONARY object (``rate >= -v_ego``).

Fabricating a closing rate out of the ego speed alone is exactly how a phantom
full-authority brake gets built; this path fabricates nothing, has to clear its
own measured error bar, and is bounded by the ego speed even when it does.

Failure behaviour
-----------------
* ``perception_valid=False`` -- the planner NEVER interprets a missing perception
  result as an empty scene.  It holds the last valid target and ramps it down at
  ``max_decel_mps2``; after ``mrm_after_dropouts`` consecutive dropouts it ramps at
  ``mrm_decel_mps2`` instead, i.e. a controlled stop.  ``SpeedDecision.degraded``
  is set on every such frame.
* Ego speed unknown / non-finite / negative -- the constant-time-gap law is
  undefined without it, so the planner degrades exactly as for a perception
  dropout (hold and ramp down) and reports ``ego_speed_unavailable``.  It never
  guesses a speed.
* Non-finite lead range -- the lead is discarded and the planner reports
  ``invalid_range``, holding and ramping down rather than cruising.
* Coasting lead (the tracker is extrapolating, not measuring) -- the target speed
  is not allowed to increase, so the eventual deletion of the track cannot produce
  an acceleration step.

This module owns no I/O and no logging of driver data; it is pure decision logic
and is safe to unit-test at any rate.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from adas.core.exceptions import ValidationError
from adas.core.logger import setup_logger
from adas.core.models import RangeSource

logger = setup_logger(__name__)

NOMINAL_DT_S = 0.05
"""Fallback timestep (20 Hz) used only when the caller supplies an unusable dt."""


class LogGate:
    """Rate limiter for a log line that describes a LATCHED condition.

    A degraded condition that is a permanent steady state -- ``ego.source='none'``
    on a vehicle with no CAN bus is the motivating example -- produced one WARNING
    per frame, i.e. 72,000 identical lines an hour at 20 Hz, which fills the disk
    and buries every real event.  This gate emits the entry into the condition at
    full severity, at most one repeat every ``period_s`` (carrying the number of
    frames suppressed since the last line), and the exit once.

    The clock is injectable so that tests are deterministic; production callers
    pass nothing and get :func:`time.monotonic`.
    """

    __slots__ = ("period_s", "_active", "_suppressed", "_last_emit_s")

    def __init__(self, period_s: float = 60.0) -> None:
        self.period_s = float(period_s)
        self._active = False
        self._suppressed = 0
        self._last_emit_s = 0.0

    @property
    def active(self) -> bool:
        """True while the condition is latched (entered and not yet cleared)."""
        return self._active

    @property
    def suppressed(self) -> int:
        """Frames dropped since the last emitted line."""
        return self._suppressed

    def _now(self, now_s: float | None) -> float:
        return time.monotonic() if now_s is None else float(now_s)

    def mark(self, now_s: float | None = None) -> tuple[bool, int]:
        """Record that the condition holds this frame.

        Returns:
            ``(emit, suppressed)`` -- ``emit`` is True on the entry frame and on
            each periodic repeat; ``suppressed`` is the number of frames dropped
            since the previous emitted line (0 on the entry frame).
        """
        now = self._now(now_s)
        if not self._active:
            self._active = True
            self._suppressed = 0
            self._last_emit_s = now
            return True, 0
        if self.period_s > 0.0 and (now - self._last_emit_s) >= self.period_s:
            dropped = self._suppressed
            self._suppressed = 0
            self._last_emit_s = now
            return True, dropped
        self._suppressed += 1
        return False, 0

    def clear(self) -> tuple[bool, int]:
        """Record that the condition no longer holds.

        Returns:
            ``(emit, suppressed)`` -- ``emit`` is True exactly once, on the frame
            the condition is left; ``suppressed`` is the number of frames dropped
            since the last emitted line.
        """
        if not self._active:
            return False, 0
        dropped = self._suppressed
        self._active = False
        self._suppressed = 0
        return True, dropped

    def reset(self) -> None:
        """Forget the latch entirely (used by planner ``reset()``)."""
        self._active = False
        self._suppressed = 0
        self._last_emit_s = 0.0


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``. ``low`` wins if the interval is empty."""
    if value < low:
        return low
    if value > high:
        return high
    return value


@dataclass
class LeadVehicle:
    """The single object the longitudinal law reacts to.

    Attributes:
        distance_m: Range to the lead, metres. Must be finite and non-negative.
        range_rate_mps: ``v_lead - v_ego`` in m/s. Negative means closing.
        track_id: Originating track id, for logging only.
        confidence: [0, 1] quality of the range estimate. Not used to soften the
            law -- a low-confidence lead is still braked for -- but it is carried
            through into the decision so the arbiter can cross-check it.
        frames_since_measurement: 0 when this frame carried a real measurement,
            >0 when the tracker is coasting the object.
        source: Which range channel produced ``distance_m``.
    """

    distance_m: float
    range_rate_mps: float = 0.0
    track_id: int = -1
    confidence: float = 1.0
    frames_since_measurement: int = 0
    source: RangeSource = RangeSource.PINHOLE

    @property
    def is_coasting(self) -> bool:
        """True when this range was extrapolated rather than measured."""
        return self.frames_since_measurement > 0


@dataclass
class LongitudinalLimits:
    """Tuning and hard limits for :class:`LongitudinalPlanner`.

    All decelerations are positive magnitudes in m/s^2.
    """

    cruise_speed_mps: float = 15.0
    min_follow_distance_m: float = 12.0
    """``d0`` in the spacing policy: the gap held at standstill by the comfort law."""
    time_gap_s: float = 2.0
    """``T`` in the spacing policy: seconds of headway added per m/s of ego speed."""
    standstill_gap_m: float = 4.0
    """Bumper-to-bumper gap the AEB stage protects. Must be <= min_follow_distance_m."""
    k_distance: float = 0.4
    """Spacing-error gain, units 1/s. Larger = firmer gap regulation."""
    k_speed: float = 0.6
    """Relative-speed gain, dimensionless."""
    max_accel_mps2: float = 2.0
    max_decel_mps2: float = 3.0
    """Comfort deceleration. Bounds how fast the target speed may fall."""
    emergency_decel_mps2: float = 8.0
    """Only the AEB stage may demand this."""
    mrm_decel_mps2: float = 3.5
    """Controlled-stop rate used once a perception dropout becomes persistent."""
    aeb_ttc_s: float = 0.9
    warn_ttc_s: float = 1.6
    aeb_required_decel_mps2: float = 5.0
    """If closing geometry needs more than this, the comfort law is abandoned."""
    limited_after_dropouts: int = 1
    mrm_after_dropouts: int = 3
    range_rate_cross_check_window: int = 19
    """Frames of range history the independent rate estimate is fitted over.

    Must be >= 5 (the standard error needs residual degrees of freedom).  19
    frames is 0.95 s at 20 Hz.  The slope's standard error falls as
    ``sqrt(12 / (n (n^2 - 1)))``, so a longer window buys accuracy directly; what
    it costs is detection latency.  The pair (19 frames, 5 sigma) was picked off
    a measured grid -- window 9..19 x sigma 3..5, scored on corrections
    MANUFACTURED from an honest rate channel with 0.25 / 0.5 / 1 / 3 / 8 m of
    gaussian range noise over 4,800 frames each, and on how long a stuck channel
    gets to run before it is caught.  Every shorter or looser cell leaked between
    1 and 55 fabricated corrections; (19, 5) leaked none in any noise or
    range-jump case measured and still catches a rate channel stuck at 0 within
    0.95 s.  The latency is the price of not being a phantom-brake generator, and
    it is the number to revisit first if this check ever has to be faster.
    """
    range_rate_disagreement_mps: float = 3.0
    """Absolute floor on the disagreement before the measured rate is believed.

    Below this the reported rate is used unchanged, so ordinary filter lag never
    perturbs the law however small the fit's error bar happens to be.  3 m/s at a
    30 m range is 0.15 m/s^2 of required deceleration -- far below
    ``aeb_required_decel_mps2`` -- so a disagreement this size only ever matters
    close in, which is where it should.
    """
    range_rate_significance_sigma: float = 5.0
    """Standard errors of the fitted slope the disagreement must also clear.

    This is what makes the cross-check safe on a noisy range channel: the
    threshold rises automatically with the scatter of the range about the fitted
    line, so the check neither goes deaf on a clean channel nor fires on a dirty
    one.  The absolute floor above has to be cleared as well.
    """

    def __post_init__(self) -> None:
        if self.cruise_speed_mps <= 0:
            raise ValidationError(f"cruise_speed_mps must be positive, got {self.cruise_speed_mps}")
        if self.min_follow_distance_m < 0:
            raise ValidationError(
                f"min_follow_distance_m must be non-negative, got {self.min_follow_distance_m}"
            )
        if self.time_gap_s <= 0:
            raise ValidationError(f"time_gap_s must be positive, got {self.time_gap_s}")
        if self.standstill_gap_m < 0:
            raise ValidationError(f"standstill_gap_m must be non-negative, got {self.standstill_gap_m}")
        if self.standstill_gap_m > self.min_follow_distance_m:
            raise ValidationError(
                "standstill_gap_m (%.2f) must not exceed min_follow_distance_m (%.2f); the AEB "
                "stage would then fire inside the comfort law's own equilibrium gap"
                % (self.standstill_gap_m, self.min_follow_distance_m)
            )
        if self.k_distance <= 0 or self.k_speed <= 0:
            raise ValidationError("k_distance and k_speed must both be positive for monotonicity")
        for name in ("max_accel_mps2", "max_decel_mps2", "emergency_decel_mps2", "mrm_decel_mps2"):
            value = getattr(self, name)
            if value <= 0:
                raise ValidationError(f"{name} must be positive, got {value}")
        if self.emergency_decel_mps2 < self.max_decel_mps2:
            raise ValidationError(
                "emergency_decel_mps2 must be >= max_decel_mps2 or AEB would brake more gently "
                "than the comfort law"
            )
        if self.aeb_ttc_s <= 0 or self.warn_ttc_s < self.aeb_ttc_s:
            raise ValidationError("require 0 < aeb_ttc_s <= warn_ttc_s")
        if self.mrm_after_dropouts < self.limited_after_dropouts:
            raise ValidationError("mrm_after_dropouts must be >= limited_after_dropouts")
        if self.range_rate_cross_check_window < 5:
            raise ValidationError(
                "range_rate_cross_check_window must be an integer >= 5, got %r"
                % (self.range_rate_cross_check_window,)
            )
        if not (
            math.isfinite(self.range_rate_significance_sigma)
            and self.range_rate_significance_sigma > 0.0
        ):
            raise ValidationError(
                "range_rate_significance_sigma must be positive, got %r"
                % (self.range_rate_significance_sigma,)
            )
        if not (
            math.isfinite(self.range_rate_disagreement_mps)
            and self.range_rate_disagreement_mps > 0.0
        ):
            raise ValidationError(
                "range_rate_disagreement_mps must be positive, got %r"
                % (self.range_rate_disagreement_mps,)
            )


@dataclass
class SpeedDecision:
    """Result of one longitudinal planning step.

    ``target_speed_mps`` is the only field the controller needs; everything else is
    diagnostic and is consumed by the safety arbiter and by the logs.
    """

    target_speed_mps: float
    reason: str
    desired_gap_m: float = float("inf")
    gap_m: float = float("inf")
    ttc_s: float = float("inf")
    required_decel_mps2: float = 0.0
    aeb_active: bool = False
    degraded: bool = False
    dropout_frames: int = 0
    lead_track_id: int = -1
    range_rate_mps: float = 0.0
    """The closing rate the law was actually evaluated with (``v_lead - v_ego``)."""
    range_rate_corrected: bool = False
    """True when the reported rate was overridden by the planner's own median
    range derivative; see the module note on the range-rate cross-check."""


class LongitudinalPlanner:
    """Constant-time-gap ACC with an independent AEB stage and target rate limiting."""

    def __init__(
        self,
        limits: LongitudinalLimits | None = None,
        ego_speed_log_period_s: float = 60.0,
    ) -> None:
        self.limits = limits or LongitudinalLimits()
        self._prev_target_mps: float | None = None
        self._dropout_frames = 0
        self._ego_speed_gate = LogGate(ego_speed_log_period_s)
        self._rate_gate = LogGate(ego_speed_log_period_s)
        self._range_track_id = -1
        self._range_clock_s = 0.0
        self._range_hist: list = []

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Forget the previous target, the dropout counter and the log latches."""
        self._prev_target_mps = None
        self._dropout_frames = 0
        self._ego_speed_gate.reset()
        self._rate_gate.reset()
        self._forget_range_history()

    @property
    def ego_speed_available(self) -> bool:
        """False while the planner is latched in ``ego_speed_unavailable``.

        This is the machine-readable form of the condition that used to be a
        per-frame WARNING.  It is also visible on every
        :class:`SpeedDecision` as ``degraded`` plus ``reason`` and is what a
        health endpoint should report.
        """
        return not self._ego_speed_gate.active

    @property
    def previous_target_mps(self) -> float | None:
        """Last emitted target speed, or None before the first call."""
        return self._prev_target_mps

    @property
    def dropout_frames(self) -> int:
        """Consecutive frames the planner has been told perception is invalid."""
        return self._dropout_frames

    # ------------------------------------------------------------- pure maths

    def desired_gap_m(self, ego_speed_mps: float) -> float:
        """``d0 + T * v_ego`` in metres. Pure."""
        return self.limits.min_follow_distance_m + self.limits.time_gap_s * max(0.0, ego_speed_mps)

    def equilibrium_speed_mps(
        self, distance_m: float, range_rate_mps: float, ego_speed_mps: float
    ) -> float:
        """The comfort law, before AEB and before rate limiting. Pure.

        Non-decreasing in ``distance_m``; non-increasing in the closing rate
        (``-range_rate_mps``).
        """
        lim = self.limits
        gap_error = distance_m - self.desired_gap_m(ego_speed_mps)
        raw = ego_speed_mps + lim.k_distance * gap_error + lim.k_speed * range_rate_mps
        return _clamp(raw, 0.0, lim.cruise_speed_mps)

    def hazard(self, distance_m: float, range_rate_mps: float) -> tuple[float, float]:
        """Return ``(ttc_s, required_decel_mps2)`` for a lead at this range/rate. Pure.

        ``ttc_s`` is the time until the standstill gap is consumed, ``inf`` when
        not closing.  ``required_decel_mps2`` is the constant deceleration that
        would just avoid consuming the standstill gap, ``inf`` when the gap is
        already gone and the object is still closing.
        """
        gap = distance_m - self.limits.standstill_gap_m
        closing = -range_rate_mps
        if closing <= 1e-3:
            return float("inf"), 0.0
        if gap <= 0.0:
            return 0.0, float("inf")
        return gap / closing, (closing * closing) / (2.0 * gap)

    def raw_target_speed_mps(
        self,
        lead: LeadVehicle | None,
        ego_speed_mps: float,
        range_rate_mps: float | None = None,
    ) -> tuple[float, str, bool, float, float]:
        """Comfort law + AEB, before rate limiting. Pure.

        Args:
            lead: The lead object, or None for an empty scene.
            ego_speed_mps: Measured ego speed, m/s.
            range_rate_mps: Overrides ``lead.range_rate_mps`` when not None. Used
                by :meth:`plan` to substitute the planner's own median range
                derivative when the reported rate is contradicted by the range
                history; see the module note. Everything else about the law is
                unchanged, so the monotonicity properties still hold in this
                argument exactly as they do in ``lead.range_rate_mps``.

        Returns ``(target_mps, reason, aeb_active, ttc_s, required_decel_mps2)``.
        """
        lim = self.limits
        if lead is None:
            return lim.cruise_speed_mps, "cruise_clear", False, float("inf"), 0.0

        distance_m = lead.distance_m
        if not math.isfinite(distance_m) or distance_m < 0.0:
            # An implausible range is a fault, not an empty road.
            return 0.0, "invalid_range", True, 0.0, float("inf")

        rate = lead.range_rate_mps if range_rate_mps is None else range_rate_mps
        rate = rate if math.isfinite(rate) else 0.0
        ttc_s, required = self.hazard(distance_m, rate)

        if distance_m <= lim.standstill_gap_m:
            return 0.0, "aeb_standstill_gap_%.1fm" % distance_m, True, ttc_s, required
        if ttc_s < lim.aeb_ttc_s:
            return 0.0, "aeb_ttc_%.2fs" % ttc_s, True, ttc_s, required
        if required > lim.aeb_required_decel_mps2:
            return 0.0, "aeb_required_decel_%.1f" % required, True, ttc_s, required

        target = self.equilibrium_speed_mps(distance_m, rate, ego_speed_mps)
        if target >= lim.cruise_speed_mps - 1e-9:
            reason = "cruise_clear_%.1fm" % distance_m
        else:
            reason = "follow_gap_%.1fm" % distance_m
        return target, reason, False, ttc_s, required

    # ------------------------------------------------------------------ plan

    def plan(
        self,
        lead: LeadVehicle | None,
        ego_speed_mps: float | None,
        perception_valid: bool = True,
        dt_s: float = NOMINAL_DT_S,
    ) -> SpeedDecision:
        """Produce one rate-limited target speed.

        Args:
            lead: The selected lead object, or None for a genuinely empty scene.
                Pass None ONLY when perception ran successfully and saw nothing.
            ego_speed_mps: Measured ego speed in m/s, or None when unavailable.
            perception_valid: False when the perception stage failed this frame.
                The planner then degrades instead of assuming the road is clear.
            dt_s: Elapsed time since the previous call, seconds.

        Returns:
            A :class:`SpeedDecision`. ``target_speed_mps`` is always finite, in
            ``[0, cruise_speed_mps]``, and never RISES faster than
            ``max_accel_mps2``.  Downward it is limited to ``max_decel_mps2`` on
            comfort frames and to ``mrm_decel_mps2`` / ``max_decel_mps2`` on
            degraded frames, but it is NOT limited downward while
            ``aeb_active`` is set: an AEB decision publishes 0 m/s on the frame
            it fires.  The deceleration the vehicle actually experiences is
            bounded downstream, by the controller's jerk limit and the actuator.
        """
        lim = self.limits
        dt = dt_s if (math.isfinite(dt_s) and dt_s > 1e-4) else NOMINAL_DT_S

        if not perception_valid:
            self._dropout_frames += 1
            decel = (
                lim.mrm_decel_mps2
                if self._dropout_frames >= lim.mrm_after_dropouts
                else lim.max_decel_mps2
            )
            self._forget_range_history()
            self._rate_gate.clear()
            return self._ramp_down(
                decel,
                dt,
                "perception_dropout_%d" % self._dropout_frames,
                ego_speed_mps,
            )

        self._dropout_frames = 0

        if ego_speed_mps is None or not math.isfinite(ego_speed_mps) or ego_speed_mps < 0.0:
            # The time-gap law is undefined without ego speed. Degrade, never guess.
            #
            # ego.source='none' is a PERMANENT steady state on a vehicle with no
            # CAN bus, so this line is gated: once on entry, once every
            # LogGate.period_s while latched, once on recovery. Every frame still
            # carries the condition in SpeedDecision.degraded / .reason and in
            # LongitudinalPlanner.ego_speed_available.
            emit, dropped = self._ego_speed_gate.mark()
            if emit:
                logger.warning(
                    "Longitudinal planner has no usable ego speed (%r); holding and "
                    "ramping down. Condition is latched: further occurrences are "
                    "logged at most every %.0f s (%d frames suppressed since the "
                    "last line); see SpeedDecision.reason='ego_speed_unavailable'.",
                    ego_speed_mps,
                    self._ego_speed_gate.period_s,
                    dropped,
                )
            self._forget_range_history()
            self._rate_gate.clear()
            return self._ramp_down(lim.max_decel_mps2, dt, "ego_speed_unavailable", None)

        recovered, dropped = self._ego_speed_gate.clear()
        if recovered:
            logger.info(
                "Longitudinal planner ego speed restored (%.2f m/s) after %d "
                "suppressed warnings",
                ego_speed_mps,
                dropped,
            )

        if self._prev_target_mps is None:
            self._prev_target_mps = _clamp(ego_speed_mps, 0.0, lim.cruise_speed_mps)

        rate_override = self._cross_checked_range_rate(lead, ego_speed_mps, dt)
        raw, reason, aeb, ttc_s, required = self.raw_target_speed_mps(
            lead, ego_speed_mps, range_rate_mps=rate_override
        )
        if rate_override is not None:
            reason = reason + "_xrate%.1f" % rate_override

        if lead is not None and lead.is_coasting:
            reason = reason + "_coast%d" % lead.frames_since_measurement

        if aeb:
            # NO downward rate limit on an emergency stop. `raw` is 0.0 on every
            # AEB branch of raw_target_speed_mps and is published as-is, so the
            # target LEADS the vehicle and the controller sees the full speed
            # error. Rate-limiting here at emergency_decel_mps2 * dt left a
            # residual error of 0.4 m/s per frame and no brake was commanded.
            target = _clamp(raw, 0.0, lim.cruise_speed_mps)
        else:
            low = max(0.0, self._prev_target_mps - lim.max_decel_mps2 * dt)
            high = self._prev_target_mps + lim.max_accel_mps2 * dt
            if lead is not None and lead.is_coasting:
                # The tracker is extrapolating this object. Do not release
                # throttle on a range nobody measured this frame.
                high = min(high, self._prev_target_mps)
            target = _clamp(_clamp(raw, low, high), 0.0, lim.cruise_speed_mps)
        self._prev_target_mps = target

        return SpeedDecision(
            target_speed_mps=target,
            reason=reason,
            desired_gap_m=self.desired_gap_m(ego_speed_mps),
            gap_m=lead.distance_m if lead is not None else float("inf"),
            ttc_s=ttc_s,
            required_decel_mps2=required,
            aeb_active=aeb,
            degraded=False,
            dropout_frames=0,
            lead_track_id=lead.track_id if lead is not None else -1,
            range_rate_mps=(
                rate_override
                if rate_override is not None
                else (
                    lead.range_rate_mps
                    if lead is not None and math.isfinite(lead.range_rate_mps)
                    else 0.0
                )
            ),
            range_rate_corrected=rate_override is not None,
        )

    # --------------------------------------------------------------- helpers

    def _forget_range_history(self) -> None:
        """Drop the independent range-rate estimator's window."""
        self._range_track_id = -1
        self._range_clock_s = 0.0
        self._range_hist = []

    def _measured_range_rate(
        self, lead: LeadVehicle | None, dt_s: float
    ) -> tuple[float, float] | None:
        """Fitted range rate and its standard error, or None.

        Returns ``(v_rel, stderr)`` in the planner's convention (negative =
        closing), from a least-squares fit of the ranges the planner was actually
        handed against their timestamps. Returns None until a full window of
        consecutive, MEASURED samples of ONE track exists, so a track change, a
        coasting frame or a dropout restarts the evidence.

        The standard error is the point of this method as much as the slope is:
        it is what tells the caller how much of the fitted slope is range noise,
        without the planner having to know anything about the range channel.
        """
        if (
            lead is None
            or lead.track_id < 0
            or lead.is_coasting
            or not math.isfinite(lead.distance_m)
            or lead.distance_m < 0.0
        ):
            self._forget_range_history()
            return None
        if lead.track_id != self._range_track_id:
            self._range_track_id = lead.track_id
            self._range_clock_s = 0.0
            self._range_hist = [(0.0, lead.distance_m)]
            return None

        self._range_clock_s += dt_s
        self._range_hist.append((self._range_clock_s, lead.distance_m))
        window = self.limits.range_rate_cross_check_window
        if len(self._range_hist) > window:
            del self._range_hist[0 : len(self._range_hist) - window]
            # Rebase so neither the stored times nor the clock grow without
            # bound over a multi-hour run.
            base = self._range_hist[0][0]
            self._range_hist = [(t - base, d) for t, d in self._range_hist]
            self._range_clock_s -= base
        n = len(self._range_hist)
        if n < window:
            return None

        t_mean = sum(t for t, _ in self._range_hist) / n
        d_mean = sum(d for _, d in self._range_hist) / n
        s_tt = sum((t - t_mean) ** 2 for t, _ in self._range_hist)
        if s_tt <= 1e-12:  # pragma: no cover - defensive, dt is validated > 1e-4
            return None
        slope = sum((t - t_mean) * (d - d_mean) for t, d in self._range_hist) / s_tt
        intercept = d_mean - slope * t_mean
        residual_ss = sum((d - (intercept + slope * t)) ** 2 for t, d in self._range_hist)
        stderr = math.sqrt(max(0.0, residual_ss) / (n - 2) / s_tt)
        if not (math.isfinite(slope) and math.isfinite(stderr)):  # pragma: no cover
            return None
        return slope, stderr

    def _cross_checked_range_rate(
        self, lead: LeadVehicle | None, ego_speed_mps: float, dt_s: float
    ) -> float | None:
        """Return a corrected closing rate, or None to use the reported one.

        Fires only when the planner's own fitted range rate says the gap is
        collapsing faster than the rate channel claims by BOTH
        ``range_rate_disagreement_mps`` outright AND
        ``range_rate_significance_sigma`` standard errors of the fit, and the
        correction is then clamped at ``-v_ego`` -- the rate implied by a
        STATIONARY obstacle -- so no input can make this path invent a closing
        rate the ego's own speed does not already justify.
        """
        fit = self._measured_range_rate(lead, dt_s)
        if fit is None or lead is None:
            if fit is None:
                self._rate_gate.clear()
            return None
        measured, stderr = fit
        reported = lead.range_rate_mps if math.isfinite(lead.range_rate_mps) else 0.0
        threshold = max(
            self.limits.range_rate_disagreement_mps,
            self.limits.range_rate_significance_sigma * stderr,
        )
        if measured >= reported - threshold:
            self._rate_gate.clear()
            return None
        corrected = max(measured, -max(0.0, ego_speed_mps))
        if corrected >= reported - 1e-9:
            # The stationary-obstacle clamp removed the whole disagreement.
            self._rate_gate.clear()
            return None
        emit, dropped = self._rate_gate.mark()
        if emit:
            logger.warning(
                "Lead %d reports range rate %+.2f m/s but its range has been "
                "collapsing at %+.2f m/s (+/-%.2f) over %d frames; planning with "
                "%+.2f m/s. Condition is latched: repeats at most every %.0f s "
                "(%d frames suppressed since the last line); every frame carries "
                "it as SpeedDecision.range_rate_corrected.",
                lead.track_id,
                reported,
                measured,
                stderr,
                self.limits.range_rate_cross_check_window,
                corrected,
                self._rate_gate.period_s,
                dropped,
            )
        return corrected

    def _ramp_down(
        self,
        decel_mps2: float,
        dt_s: float,
        reason: str,
        ego_speed_mps: float | None,
    ) -> SpeedDecision:
        """Hold the last valid target and ramp it toward zero at ``decel_mps2``.

        Seeds from the measured ego speed on the very first call when it is known,
        and from zero when it is not -- the conservative direction in both cases.
        """
        if self._prev_target_mps is None:
            if ego_speed_mps is not None and math.isfinite(ego_speed_mps) and ego_speed_mps >= 0.0:
                self._prev_target_mps = _clamp(ego_speed_mps, 0.0, self.limits.cruise_speed_mps)
            else:
                self._prev_target_mps = 0.0
        target = max(0.0, self._prev_target_mps - decel_mps2 * dt_s)
        self._prev_target_mps = target
        return SpeedDecision(
            target_speed_mps=target,
            reason=reason,
            degraded=True,
            dropout_frames=self._dropout_frames,
        )
