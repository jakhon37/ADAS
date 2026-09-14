"""Longitudinal (speed) planning: an evidence-gated following and braking law.

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

What this module decides, and what it refuses to
------------------------------------------------
The planner publishes two things per frame: a **target speed**, which is a
comfort request the controller may use throttle to reach, and a
**deceleration demand**, which is the authoritative braking figure.  Splitting
them is the fix for a measured defect: while the only output was a
rate-limited target speed, "brake at 3 m/s^2" reached the controller as a target
falling 0.15 m/s per frame, a proportional law produced 0.4 m/s^2 against it,
and the safety arbiter was the only thing in the vehicle that actually braked.
A planner whose braking is decorative is a planner that has made the backstop
load-bearing.

The deceleration demand is the larger of two laws, and the *authority* each may
use is fixed by the *evidence* behind it:

**A constant-time-gap follow.**  The policy spacing is ``d0 + T * v_ego`` and
inside it the ego settles at a speed deficit proportional to the shortfall,
capped at :attr:`LongitudinalLimits.headway_dv_max_mps`.  This law rests on the
measured RANGE alone, which is the best-supported quantity the system has, and
its authority is capped at :attr:`LongitudinalLimits.headway_decel_mps2` --
1.5 m/s^2, half the comfort limit.  Headway keeping is not collision avoidance
and must never be mistaken for it: braking at 3.4 m/s^2 for a whole run behind a
lead that never moved satisfies every AEB test ever written and is still wrong.

**A collision-avoidance law derived from required deceleration.**  Textbook,
computed from the measured range, the planner's OWN least-squares closing rate
and a split-half estimate of the lead's acceleration -- see
:mod:`adas.control.evidence`.  It may use the vehicle's full authority, and it
is gated: the law does not engage until the measured closure is
:attr:`~adas.control.evidence.EvidenceLimits.closure_sigmas` standard errors
clear of zero.  **The gate is on the bound; the magnitude is the estimate.**
Once the gate is open the demand is sized on the unbiased closure, because the
requirement goes as the square of it and braking for a deliberately pessimistic
closure is its own kind of over-response.

**No rate is ever fabricated.**  A camera measures range; a closing rate is a
difference of ranges over time.  Until four distinct captures exist there is no
rate, and the only prior available -- "assume the object is stationary in the
world", i.e. ``rate = -v_ego`` -- is precisely what produced this codebase's
phantom full-authority brake.  While the rate is unmeasured the planner falls
back on the time-gap law, which is bounded at 1.5 m/s^2 and cannot hurt anyone.

The demand is shaped by a jerk limiter at the ceilings the safety specification
derives: 2.5 m/s^3 while the demand stays inside the comfort band, 20 m/s^3 once
it is heading past emergency grade.  Shaping here rather than in the controller
means the planner owns the whole deceleration profile and the controller can be
a pure actuator map; it also means the demand the arbiter is handed is already
occupant-safe, so the arbiter never has to reduce one for comfort.

Failure behaviour
-----------------
* ``perception_valid=False`` -- the planner NEVER interprets a missing perception
  result as an empty scene.  The avoidance demand is HELD for
  ``blind_hold_frames`` and the vehicle then makes a minimum-risk stop at
  ``mrm_decel_mps2``.  ``SpeedDecision.degraded`` is set on every such frame.
* A detection MISS with healthy perception -- the same hold, for
  ``miss_hold_frames``: a detector that reported nothing has not reported that
  the road is clear.
* Ego speed unknown / non-finite / negative -- the time-gap law is undefined
  without it, so the planner degrades exactly as for a dropout and reports
  ``ego_speed_unavailable``.  It never guesses a speed.
* Non-finite lead range -- the lead is discarded and the planner reports
  ``invalid_range``.
* A range discontinuity larger than
  :attr:`~adas.control.evidence.EvidenceLimits.jump_m` re-anchors the estimate
  and returns the rate to unknown, rather than being differentiated into a
  200 m/s closure.

This module owns no I/O beyond gated WARNING lines and is pure decision logic;
it is safe to unit-test at any rate.  It is STATEFUL (the evidence window, the
jerk shaper, the dropout counters); call :meth:`LongitudinalPlanner.reset` on
replay restart, and use one instance per pipeline.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from adas.control.evidence import (
    EvidenceBook,
    EvidenceLimits,
    JerkShaper,
    required_decel_mps2,
)
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
        range_rate_mps: ``v_lead - v_ego`` in m/s as REPORTED by the tracker.
            Negative means closing.  Carried for diagnostics and for the legacy
            pure laws; the evidence-gated path does not read it, because it
            differentiates the range itself.
        track_id: Originating track id.  Identity, not motion: a change of it
            discards the evidence, it never implies a closure.
        confidence: [0, 1] quality of the range estimate. Not used to soften the
            law -- a low-confidence lead is still braked for -- but it is carried
            through into the decision so the arbiter can cross-check it.
        frames_since_measurement: 0 when this frame carried a real measurement,
            >0 when the tracker is coasting the object.
        source: Which range channel produced ``distance_m``.
        capture_token: A counter that advances only when perception produced a
            NEW result for this track (``TrackedObject.hits``).  When it is
            unchanged the range is a repeat of a capture already held, and
            storing it twice would put two points at the same abscissa and
            flatten the fitted slope toward zero -- a fabricated "it stopped
            closing".  At 55 ms of sense latency on a 50 ms grid that happens on
            the first two frames of every run, which is exactly where the
            tightest scenarios are decided.
    """

    distance_m: float
    range_rate_mps: float = 0.0
    track_id: int = -1
    confidence: float = 1.0
    frames_since_measurement: int = 0
    source: RangeSource = RangeSource.PINHOLE
    capture_token: int = 0

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
    """Bumper-to-bumper gap the legacy AEB predicates protect.
    Must be <= min_follow_distance_m."""
    k_distance: float = 0.4
    """Spacing-error gain of the LEGACY pure law, units 1/s."""
    k_speed: float = 0.6
    """Relative-speed gain of the LEGACY pure law, dimensionless."""
    max_accel_mps2: float = 2.0
    max_decel_mps2: float = 3.0
    """Comfort deceleration. Bounds how fast the target SPEED may fall."""
    emergency_decel_mps2: float = 8.0
    """The vehicle's full braking authority; only the avoidance law may reach it."""
    mrm_decel_mps2: float = 3.0
    """Controlled-stop rate used once the vehicle has been blind for
    ``blind_hold_frames``.

    The comfort limit, not an emergency rate: the vehicle has to stop, nothing
    has been detected in front of it, and the traffic behind has no reason to
    expect more.
    """
    aeb_ttc_s: float = 0.9
    warn_ttc_s: float = 1.6
    aeb_required_decel_mps2: float = 5.0
    """Legacy predicate threshold, retained for :meth:`raw_target_speed_mps`."""
    limited_after_dropouts: int = 1
    mrm_after_dropouts: int = 3

    # --- the evidence-gated law ----------------------------------------------
    headway_decel_mps2: float = 1.5
    """Ceiling on the authority the TIME-GAP law may use, m/s^2.

    Half the comfort limit.  A gap only opens while the ego is slower than the
    lead, and a deficit of a few m/s opens forty metres in a comfortable dozen
    seconds; there is nothing a headway law needs full authority for.  This
    number is also what keeps the system out of the 3.0-3.5 m/s^2 band that no
    emergency test can see and that a "fixed" phantom brake retreats into.
    """
    headway_gain_per_m: float = 0.30
    """Speed deficit held per metre of gap shortfall, 1/s."""
    headway_dv_max_mps: float = 3.0
    """Largest speed deficit the time-gap law will hold, m/s.

    A gap opens at the deficit, so 3 m/s opens 40 m in about thirteen seconds.
    Larger deficits open the gap faster and cost speed the journey needs.
    """
    speed_error_gain: float = 0.8
    """Deceleration demanded per m/s of speed error by the time-gap law, 1/s."""
    target_clearance_m: float = 2.25
    """Room a completed avoidance stop aims to leave, metres.

    The 2.0 m a driver leaves at standstill, plus a quarter of a metre of
    aim-off, which is the range the planner cannot see: it is acting on a
    measurement one frame old, so a law that aims at exactly the required
    clearance stops a fraction inside it every time.
    """
    demand_margin_frac: float = 0.05
    demand_margin_mps2: float = 0.05
    """Prudence added to the computed avoidance requirement.

    Deliberately small: the requirement is recomputed every frame from a fresh
    measurement, so a standing margin buys nothing the next frame does not buy
    anyway, and an over-sized one is measured as over-braking -- which transfers
    the collision to the vehicle behind rather than removing it.
    """
    comfort_jerk_mps3: float = 2.5
    """Rate the demand may be built at inside the comfort band, m/s^3."""
    emergency_jerk_mps3: float = 20.0
    """Rate the demand may be built at once it is heading past emergency grade.

    Full 8 m/s^2 authority in the 0.4 s a human panic brake takes.  Faster buys
    no stopping distance -- the brake actuator's own 0.15 s rise filters it out
    -- and costs the occupant a head-toss they cannot brace for.
    """
    release_jerk_mps3: float = 25.0
    """Rate the demand is allowed to fall at, m/s^3.

    Only the RISE is a comfort hazard; a release is arrested by the seat back,
    and every other requirement here demands that an unwarranted deceleration be
    removed PROMPTLY.
    """
    emergency_grade_mps2: float = 3.5
    """The deceleration at or above which a demand counts as an emergency."""
    blind_hold_frames: int = 8
    """Frames of perception loss tolerated before a minimum-risk stop begins.

    0.4 s, which at 20 m/s is 8 m travelled without a picture.
    """
    miss_hold_frames: int = 24
    """Frames an avoidance demand survives a DETECTION miss, 1.2 s.

    Perception is healthy and reported no object; that is not evidence of an
    empty road, so the demand is held rather than dropped.
    """
    evidence: EvidenceLimits = field(default_factory=EvidenceLimits)
    """How much measurement the avoidance law demands before it engages."""

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
        if not 0.0 < self.headway_decel_mps2 <= self.max_decel_mps2:
            raise ValidationError(
                "headway_decel_mps2 (%.2f) must be positive and no larger than the comfort "
                "limit (%.2f): a following law that may use collision-avoidance authority is "
                "how a phantom brake hides below the emergency threshold"
                % (self.headway_decel_mps2, self.max_decel_mps2)
            )
        if self.headway_gain_per_m <= 0 or self.headway_dv_max_mps <= 0:
            raise ValidationError("headway gains must be positive")
        if self.speed_error_gain <= 0:
            raise ValidationError("speed_error_gain must be positive")
        if self.target_clearance_m < 0:
            raise ValidationError("target_clearance_m must be non-negative")
        if self.comfort_jerk_mps3 <= 0 or self.emergency_jerk_mps3 < self.comfort_jerk_mps3:
            raise ValidationError("require 0 < comfort_jerk_mps3 <= emergency_jerk_mps3")
        if self.release_jerk_mps3 <= 0:
            raise ValidationError("release_jerk_mps3 must be positive")
        if not self.max_decel_mps2 <= self.emergency_grade_mps2 <= self.emergency_decel_mps2:
            raise ValidationError(
                "emergency_grade_mps2 must lie between the comfort limit and full authority"
            )
        if self.blind_hold_frames < 0 or self.miss_hold_frames < 0:
            raise ValidationError("hold windows must be non-negative")


@dataclass
class SpeedDecision:
    """Result of one longitudinal planning step.

    Two outputs matter to the controller and they are not interchangeable:
    ``target_speed_mps`` is a comfort request the throttle may serve, and
    ``decel_demand_mps2`` is the authoritative braking figure, already jerk
    shaped.  Everything else is diagnostic and is consumed by the safety
    arbiter, the logs and the health endpoint.
    """

    target_speed_mps: float
    reason: str
    desired_gap_m: float = float("inf")
    gap_m: float = float("inf")
    ttc_s: float = float("inf")
    required_decel_mps2: float = 0.0
    """The AVOIDANCE law's raw requirement this frame, before jerk shaping."""
    decel_demand_mps2: float = 0.0
    """The deceleration the controller must produce, m/s^2, jerk shaped."""
    aeb_active: bool = False
    degraded: bool = False
    dropout_frames: int = 0
    lead_track_id: int = -1
    range_rate_mps: float = 0.0
    """The closing rate the law was evaluated with (``v_lead - v_ego``)."""
    range_rate_corrected: bool = False
    """True when the planner's own measurement replaced the reported rate."""
    rate_is_measured: bool = False
    """True once the closing rate comes from the planner's own window of RAW
    range measurements.  While False no collision-avoidance authority is used at
    all: the fallback is the time-gap law, bounded at ``headway_decel_mps2``."""
    closing_lcb_mps: float = 0.0
    """Lower confidence bound on the closure, m/s, positive when closing.  This
    is the number the avoidance law is GATED on; ``range_rate_mps`` is the number
    it is SIZED on."""
    closing_stderr_mps: float = 0.0
    lead_accel_mps2: float = 0.0
    """Confident lead acceleration, <= 0."""
    headway_decel_mps2: float = 0.0
    """The time-gap law's contribution to the demand this frame."""


class LongitudinalPlanner:
    """Constant-time-gap following plus an evidence-gated avoidance law."""

    def __init__(
        self,
        limits: LongitudinalLimits | None = None,
        ego_speed_log_period_s: float = 60.0,
    ) -> None:
        self.limits = limits or LongitudinalLimits()
        self._prev_target_mps: float | None = None
        self._dropout_frames = 0
        self._blind_frames = 0
        self._miss_frames = 0
        self._held_avoid_mps2 = 0.0
        self._ego_speed_gate = LogGate(ego_speed_log_period_s)
        self._rate_gate = LogGate(ego_speed_log_period_s)
        self._clock_s = 0.0
        self._evidence = EvidenceBook(self.limits.evidence)
        self._shaper = JerkShaper(
            comfort_jerk_mps3=self.limits.comfort_jerk_mps3,
            emergency_jerk_mps3=self.limits.emergency_jerk_mps3,
            emergency_decel_mps2=self.limits.emergency_grade_mps2,
            release_rate_mps3=self.limits.release_jerk_mps3,
        )

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Forget every window, counter and latch.  Call on replay restart."""
        self._prev_target_mps = None
        self._dropout_frames = 0
        self._blind_frames = 0
        self._miss_frames = 0
        self._held_avoid_mps2 = 0.0
        self._ego_speed_gate.reset()
        self._rate_gate.reset()
        self._clock_s = 0.0
        self._evidence.reset()
        self._shaper.reset(0.0)

    @property
    def ego_speed_available(self) -> bool:
        """False while the planner is latched in ``ego_speed_unavailable``."""
        return not self._ego_speed_gate.active

    @property
    def previous_target_mps(self) -> float | None:
        """Last emitted target speed, or None before the first call."""
        return self._prev_target_mps

    @property
    def dropout_frames(self) -> int:
        """Consecutive frames the planner has been told perception is invalid."""
        return self._dropout_frames

    @property
    def demand_mps2(self) -> float:
        """The jerk-shaped deceleration demand as it currently stands."""
        return self._shaper.decel_mps2

    # ------------------------------------------------------------- pure maths

    def desired_gap_m(self, ego_speed_mps: float) -> float:
        """``d0 + T * v_ego`` in metres. Pure."""
        return self.limits.min_follow_distance_m + self.limits.time_gap_s * max(0.0, ego_speed_mps)

    def equilibrium_speed_mps(
        self, distance_m: float, range_rate_mps: float, ego_speed_mps: float
    ) -> float:
        """The legacy comfort law, before AEB and before rate limiting. Pure.

        Non-decreasing in ``distance_m``; non-increasing in the closing rate
        (``-range_rate_mps``).  Retained because ``tests/test_longitudinal.py``
        sweeps it for monotonicity and because it is the clearest statement of
        the spacing policy; :meth:`plan` uses the evidence-gated form below.
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
        """Legacy comfort law + AEB predicates, before rate limiting. Pure.

        Kept as a pure, sweepable statement of the spacing policy and of the
        emergency predicates.  :meth:`plan` no longer routes its braking through
        it, because a target speed cannot express a deceleration: see the module
        docstring.

        Returns ``(target_mps, reason, aeb_active, ttc_s, required_decel_mps2)``.
        """
        lim = self.limits
        if lead is None:
            return lim.cruise_speed_mps, "cruise_clear", False, float("inf"), 0.0

        distance_m = lead.distance_m
        if not math.isfinite(distance_m) or distance_m < 0.0:
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
        """Produce one target speed and one deceleration demand.

        Args:
            lead: The selected lead object, or None for a genuinely empty scene.
                Pass None ONLY when perception ran successfully and saw nothing.
            ego_speed_mps: Measured ego speed in m/s, or None when unavailable.
            perception_valid: False when the perception stage failed this frame.
                The planner then degrades instead of assuming the road is clear.
            dt_s: Elapsed time since the previous call, seconds.

        Returns:
            A :class:`SpeedDecision`.  ``target_speed_mps`` is always finite and
            in ``[0, cruise_speed_mps]``.  ``decel_demand_mps2`` is in
            ``[0, emergency_decel_mps2]`` and never RISES faster than
            ``comfort_jerk_mps3`` unless it is heading past
            ``emergency_grade_mps2``, in which case it may rise at
            ``emergency_jerk_mps3``.
        """
        lim = self.limits
        dt = dt_s if (math.isfinite(dt_s) and dt_s > 1e-4) else NOMINAL_DT_S
        self._clock_s += dt

        if not perception_valid:
            self._dropout_frames += 1
            self._blind_frames += 1
            self._rate_gate.clear()
            return self._blind_step(
                dt, "perception_dropout_%d" % self._dropout_frames, True
            )

        self._dropout_frames = 0

        if ego_speed_mps is None or not math.isfinite(ego_speed_mps) or ego_speed_mps < 0.0:
            # The time-gap law is undefined without ego speed. Degrade, never guess.
            #
            # ego.source='none' is a PERMANENT steady state on a vehicle with no
            # CAN bus, so this line is gated: once on entry, once every
            # LogGate.period_s while latched, once on recovery.
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
            self._blind_frames += 1
            self._rate_gate.clear()
            return self._blind_step(dt, "ego_speed_unavailable", True)

        recovered, dropped = self._ego_speed_gate.clear()
        if recovered:
            logger.info(
                "Longitudinal planner ego speed restored (%.2f m/s) after %d "
                "suppressed warnings",
                ego_speed_mps,
                dropped,
            )
        self._blind_frames = 0
        v_ego = float(ego_speed_mps)
        a_ego = self._evidence.ego_accel_mps2(self._clock_s, v_ego)

        usable_lead = lead is not None and math.isfinite(lead.distance_m) and lead.distance_m >= 0.0
        if lead is not None and not usable_lead:
            # An implausible range is a fault, not an empty road: hold.
            self._miss_frames += 1
            decision = self._no_lead_step(dt, v_ego, "invalid_range")
            return decision
        if lead is None:
            self._miss_frames += 1
            return self._no_lead_step(dt, v_ego, "cruise")

        self._miss_frames = 0
        self._evidence.forget_all_but([lead.track_id])
        rng = max(0.01, float(lead.distance_m))
        evidence = self._evidence.track(lead.track_id)
        evidence.update(
            self._clock_s,
            rng,
            measured=not lead.is_coasting,
            capture_token=lead.capture_token or None,
        )
        closure = evidence.closure(a_ego)

        a_avoid = 0.0
        required = 0.0
        if closure.measured:
            # THE GATE IS ON THE BOUND, THE MAGNITUDE IS THE ESTIMATE.
            gated_closing = closure.closing_mps if closure.confident_closing else 0.0
            required = required_decel_mps2(
                range_m=rng,
                closing_mps=gated_closing,
                ego_speed_mps=v_ego,
                lead_accel_mps2=closure.lead_accel_mps2,
                target_clearance_m=lim.target_clearance_m,
                current_decel_mps2=self._shaper.decel_mps2,
                max_decel_mps2=lim.emergency_decel_mps2,
                jerk_mps3=lim.emergency_jerk_mps3,
            )
            if required > 0.05:
                a_avoid = min(
                    lim.emergency_decel_mps2,
                    required * (1.0 + lim.demand_margin_frac) + lim.demand_margin_mps2,
                )
        self._held_avoid_mps2 = a_avoid

        v_lead_est = max(0.0, v_ego - closure.closing_mps) if closure.measured else v_ego
        desired = self.desired_gap_m(v_ego)
        deficit = _clamp(lim.headway_gain_per_m * (desired - rng), 0.0, lim.headway_dv_max_mps)
        v_target = _clamp(v_lead_est - deficit, 0.0, lim.cruise_speed_mps)
        a_headway = _clamp(
            lim.speed_error_gain * (v_ego - v_target), 0.0, lim.headway_decel_mps2
        )

        target_decel = max(a_avoid, a_headway)
        demand = self._shaper.step(target_decel, dt)
        aeb_active = demand >= lim.emergency_grade_mps2
        if aeb_active:
            # An emergency publishes a target of zero, at once.  The comfort
            # setpoint is not the instrument of an emergency stop and must not
            # be allowed to argue with one.
            v_target = 0.0
        v_target = self._rate_limited_target_mps(v_target, dt, aeb_active)
        self._prev_target_mps = v_target

        ttc_s, _legacy_required = self.hazard(rng, -closure.closing_mps)
        if a_avoid > 0.0:
            reason = "avoid_%.1fm/s^2" % a_avoid
        elif a_headway > 0.0:
            reason = "follow_gap_%.1fm" % rng
        elif not closure.measured:
            reason = "follow_no_rate_%.1fm" % rng
        else:
            reason = "cruise_clear_%.1fm" % rng
        if lead.is_coasting:
            reason += "_coast%d" % lead.frames_since_measurement
        if closure.reanchored:
            reason += "_reanchored"

        return SpeedDecision(
            target_speed_mps=v_target,
            reason=reason,
            desired_gap_m=desired,
            gap_m=rng,
            ttc_s=ttc_s,
            required_decel_mps2=required,
            decel_demand_mps2=demand,
            aeb_active=aeb_active,
            degraded=False,
            dropout_frames=0,
            lead_track_id=lead.track_id,
            range_rate_mps=-closure.closing_mps,
            range_rate_corrected=closure.measured,
            rate_is_measured=closure.measured,
            closing_lcb_mps=closure.closing_lcb_mps,
            closing_stderr_mps=closure.stderr_mps,
            lead_accel_mps2=closure.lead_accel_mps2,
            headway_decel_mps2=a_headway,
        )

    # --------------------------------------------------------------- helpers

    def _rate_limited_target_mps(
        self, raw_target_mps: float, dt_s: float, aeb_active: bool
    ) -> float:
        """Bound the step in the PUBLISHED target speed, m/s.

        The single choke point through which every ``target_speed_mps`` this
        planner emits must pass, so that the bound is a property of the class and
        not of whichever branch happened to compute the number.

        The limit is one-sided with respect to an emergency, deliberately:

        * **Rising** is bounded by ``max_accel_mps2 * dt`` always.  There is no
          case in which a setpoint may step upward: the vehicle cannot follow it,
          so the only thing a step does is saturate the throttle and wind up the
          controller's integrator.
        * **Falling** is bounded by ``max_decel_mps2 * dt`` -- the COMFORT rate --
          on an ordinary frame, because the published target is a comfort request
          that the throttle serves and the authoritative braking figure is
          ``decel_demand_mps2``, which is shaped separately and is not touched
          here.
        * **An AEB frame is exempt** and publishes its target immediately.  A
          target that trailed the vehicle down at the comfort rate during an
          emergency stop is exactly how the primary path came to contribute
          0.06 m/s^2 while the arbiter did all the braking.

        Args:
            raw_target_mps: The target the law computed this frame, m/s.
            dt_s: Elapsed time since the previous call, seconds.
            aeb_active: True when this frame's demand is at emergency grade.

        Returns:
            The target to publish, m/s.  On the first call after
            :meth:`reset` there is no previous value and the raw target is
            published unchanged.
        """
        lim = self.limits
        previous = self._prev_target_mps
        if aeb_active or previous is None or not math.isfinite(previous):
            return raw_target_mps
        up = lim.max_accel_mps2 * dt_s
        down = lim.max_decel_mps2 * dt_s
        return _clamp(raw_target_mps, previous - down, previous + up)

    def _blind_step(self, dt_s: float, reason: str, degraded: bool) -> SpeedDecision:
        """One frame with no usable picture: hold, then make a controlled stop.

        A dropped frame is a dropped frame; half a second of nothing is a vehicle
        driving blind.  Until ``blind_hold_frames`` have passed the previous
        avoidance demand is HELD -- a detector that produced nothing has not
        produced evidence of an empty road -- and after that the vehicle makes a
        minimum-risk stop at ``mrm_decel_mps2``.  Nothing was detected in front
        of it, so nothing warrants more than the comfort rate.
        """
        lim = self.limits
        self._evidence.reset()
        if self._blind_frames > lim.blind_hold_frames:
            target_decel = max(self._held_avoid_mps2, lim.mrm_decel_mps2)
            reason = "%s_minimum_risk_stop" % reason
        else:
            target_decel = self._held_avoid_mps2
        demand = self._shaper.step(target_decel, dt_s)
        aeb_active = demand >= lim.emergency_grade_mps2
        v_target = self._rate_limited_target_mps(0.0, dt_s, aeb_active)
        self._prev_target_mps = v_target
        return SpeedDecision(
            target_speed_mps=v_target,
            reason=reason,
            decel_demand_mps2=demand,
            aeb_active=aeb_active,
            degraded=degraded,
            dropout_frames=self._dropout_frames,
        )

    def _no_lead_step(self, dt_s: float, v_ego: float, reason: str) -> SpeedDecision:
        """Perception is healthy and reported no usable lead.

        The avoidance demand is HELD for ``miss_hold_frames`` before it decays:
        a detector that produced nothing has not produced evidence of an empty
        road, and a hazard does not stop existing because one frame missed it.
        Once the hold expires the road really is treated as clear.
        """
        lim = self.limits
        if self._miss_frames > lim.miss_hold_frames:
            self._held_avoid_mps2 = 0.0
            self._evidence.reset()
        target_decel = self._held_avoid_mps2
        demand = self._shaper.step(target_decel, dt_s)
        aeb_active = demand >= lim.emergency_grade_mps2
        v_target = self._rate_limited_target_mps(lim.cruise_speed_mps, dt_s, aeb_active)
        self._prev_target_mps = v_target
        if self._held_avoid_mps2 > 0.0:
            reason = "detection_miss_hold_%d" % self._miss_frames
        return SpeedDecision(
            target_speed_mps=v_target,
            reason=reason,
            decel_demand_mps2=demand,
            aeb_active=aeb_active,
            degraded=False,
            dropout_frames=0,
        )
