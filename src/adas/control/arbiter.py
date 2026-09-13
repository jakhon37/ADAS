"""Authoritative safety arbitration for the ADAS decision layer.

``SafetyArbiter.arbitrate(plan, command, context)`` returns an
:class:`adas.core.models.ArbitrationResult`.  **The ``command`` field of that
result is what the actuators get.**  On any violation it is the arbiter's own
fail-safe command, never the planner's.  This replaces the previous advisory
design in which the monitor raised, the pipeline logged, and the rejected command
was actuated anyway.

Independence
------------
A monitor that shares its inputs and its arithmetic with the planner is a second
copy of the same belief, not a check.  What follows describes what the code in this
module actually does, not an aspiration:

* **Its own in-path geometry.** :meth:`SafetyArbiter._in_path` selects candidates
  from RAW box positions and the arbiter's own corridor.  It does NOT read
  ``TrackedObject.in_ego_lane``: that flag is computed by the tracker from the same
  lane model the planner uses, so trusting it would make the "independent" backstop
  inherit the very perception error it exists to catch.  The corridor is at least
  ``ArbiterLimits.min_in_path_half_width_frac`` of the frame width and is anchored
  on the IMAGE CENTRE; a lane model widens the corridor (a second anchor) only when
  it is real and confident, and can never narrow or move it.  An object counts as
  in path when any part of its box overlaps the corridor.
* **Its own range rate.** It runs an alpha-beta filter (:class:`_AlphaBetaRange`)
  over its own history of the range measurement.  It never reads
  ``TrackedObject.velocity_mps``, which is an unfiltered reciprocal derivative and
  is pure noise at range.  A track the filter has never seen (a new object, an
  occlusion exit, a range-jump re-init) is seeded with the SAFE PRIOR that the
  object is stationary in the world, i.e. ``range_rate = -ego_speed``, so the
  backstop is not blind on the frame a hazard appears.  **That seed is a statement
  about ignorance, and the arbiter labels it as one.**  A rate is ``measured`` only
  once the arbiter has ``aeb_min_rate_samples`` RAW range measurements spanning
  ``aeb_min_rate_span_s``, at which point the least-squares slope of those
  measurements replaces the seed outright.  Until then the rate-dependent emergency
  tests (TTC, geometric required deceleration, rate-derived RSS) may raise caution
  but may NOT authorise full-authority braking; only the rate-INDEPENDENT ones (a
  gap below the absolute minimum, a gap inside half the speed-matched RSS distance)
  can.  Acting on the seed itself braked a 20 m/s ego to a standstill behind a lead
  that was holding a constant 32.5 m.

  Two numbers make that split affordable rather than merely safe-sounding.  While an
  ACUTE prior is deferred -- the seeded TTC is already inside ``ttc_brake_s`` -- the
  graded response may reach ``deferred_aeb_decel_mps2`` (5.0) instead of the comfort
  rate, which is what keeps a genuinely stationary obstacle 15 m ahead stoppable
  across the wait.  And once the rate IS measured, an emergency must hold for
  ``aeb_rate_corroboration_frames`` before it latches a manoeuvre, because range
  noise makes the estimate wander; that gate holds the STATE, never the pedal, so
  the braking still goes out on the frame the closure is seen.
* **Its own hazard maths.** TTC, required deceleration and an RSS-style minimum
  gap are computed here from ``(range, range_rate, ego_speed)``; none of the
  planner's numbers are trusted.
* **A second range channel when one exists.** ``TrackedObject.range_estimate`` and
  ``SafetyContext.independent_ranges`` carry a range from a source other than the
  box-height pinhole heuristic (for example a monocular depth model).  An estimate
  below ``min_range_confidence`` is discarded as if the channel were absent.  When
  the two channels AGREE within ``range_disagreement_frac`` (with hysteresis) they are blended by
  confidence and the nearer of (blend, pinhole, depth) is used.  When they DISAGREE
  the nearer value is adopted only after ``range_corroboration_frames`` consecutive
  disagreeing frames; until then the pinhole range is used, so a single-frame depth
  outlier cannot reach the emergency path.  A disagreement in which the second
  channel reads FARTHER is never adopted.  When no second channel is present the
  lead's ``source`` stays ``RangeSource.PINHOLE``, so a consumer can always tell
  whether the cross-check actually ran.  Adopting the second channel is immediate;
  dropping it needs ``range_source_dwell_frames`` corroborating frames, and a change
  of provenance RE-ANCHORS the range filter rather than discarding it -- the
  estimator changing its mind about where a number came from is not a discontinuity
  in the world.
* **Its own kinematics.** Achieved acceleration and jerk are differenced from the
  measured ego speed, not inferred from the plan over a fictitious horizon.

What is NOT independent: ``SafetyContext.tracks`` is the tracker's output, so a
detection the perception stack never produced is invisible here too.  The arbiter
is a second opinion on the DECISION, not a second sensor.

Findings: three kinds, only one of which can disengage
------------------------------------------------------
Every frame produces up to three lists, all of which appear in
``ArbitrationResult.violations``:

``faults``
    HEALTH faults: the automation cannot perceive, cannot be commanded, or cannot
    tell the time.  Perception dropout, a missing/invalid ego state, a non-finite
    command, a missing or non-finite plan, a non-monotonic or stale timestamp.
    ONLY these accumulate toward the terminal ``DISENGAGE`` latch, because only
    these mean the system cannot continue driving.
``advisory``
    Findings that are a CONSEQUENCE OF THE ARBITER'S OWN INTERVENTION: the plan's
    apparent over-acceleration against a speed the arbiter's own braking produced,
    the deceleration and jerk the arbiter's own brake command caused, the planner's
    steering rate while an MRM is holding the wheel, a change of range provenance.
    They are reported in ``violations`` and in the logs and they change nothing
    else.  **The arbiter must not treat a consequence of its own intervention as
    evidence that the intervention is still needed** -- that is a latch built out of
    its own actuator, and it is what stopped the vehicle on an open road.
``mitigated``
    Conditions the arbiter DETECTED AND HANDLED this frame, or measured and can
    only report: an envelope clamp (lateral acceleration, steering rate, plan
    over-speed), a command clamped back into range, a range-channel disagreement,
    a range jump, an achieved-acceleration or jerk anomaly, an uncalibrated camera.
    They force ``LIMITED`` and are logged, but they NEVER latch ``DISENGAGE``.  A
    clamp that worked is normal operation; counting it as an unrecovered fault is
    how an arbiter disengages itself on an empty road.
``hazards``
    Traffic situations the arbiter exists to handle -- a close lead, a low TTC, a
    gap inside the RSS minimum.  They force ``LIMITED`` (or ``MIN_RISK_MANEUVER``
    when the AEB test trips) and never latch ``DISENGAGE``: a two-second approach
    to a stopped queue must not disengage the system at exactly the wrong moment.

States
------
``SafetyState`` from ``core/models.py``, in increasing severity:

``NOMINAL``
    No finding at all.  The output command is the input command, made no more
    energetic (throttle never raised, brake never lowered, |steering| never
    increased).
``LIMITED``
    The arbiter modified or overrode the command but the function continues:
    an AEB-independent hazard, a limit clamp, a single perception dropout, a
    disagreeing range channel.
``MIN_RISK_MANEUVER``
    The arbiter is executing a controlled stop and the planner's longitudinal
    command is ignored entirely: AEB, a persistent perception dropout, a corrupt
    command, or an unusable ego state.  Steering is HELD on the last actuated
    angle -- an MRM maintains the current path rather than straightening the wheel
    into a corner -- and is re-checked against the lateral-acceleration ceiling at
    the current speed each frame.
``DISENGAGE``
    Latched terminal state after ``disengage_after_frames`` consecutive frames with
    a HEALTH fault.  Zero throttle, controlled stop, steering held, driver must
    take over.  Cleared only by :meth:`SafetyArbiter.reset`.

Escalation is immediate; de-escalation requires ``recovery_frames`` consecutive
clean frames, so a fault cannot be cleared by one lucky frame.  ``DISENGAGE`` never
de-escalates.

Output shaping
--------------
The arbiter never attenuates a brake.  ``max(input, own requirement)`` is the only
rule on the braking direction, plus a release-rate floor that can only hold the
brake HIGHER for longer.  Jerk shaping on brake APPLICATION belongs to
:mod:`adas.control.controller`, which owns the emergency exemption; the arbiter
re-imposing a non-exempt apply-rate limit on top used to cut a handed-in
``brake = 1.0`` to 0.25 and take 200 ms (about 3 m at 15 m/s) to let it through.
Throttle is still rate-limited, because that direction only ever removes energy.

Units
-----
metres, m/s, m/s^2, m/s^3, seconds, radians for physical steering, and
dimensionless [0, 1] / [-1, 1] for actuator commands.  ``range_rate_mps`` uses
``v_rel = v_lead - v_ego`` (negative = closing) throughout, matching
:mod:`adas.planning.longitudinal` and NOT ``TrackedObject.velocity_mps``.

Threading
---------
Stateful and NOT thread-safe.  One instance per pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from adas.core.logger import setup_logger
from adas.core.models import (
    ArbitrationResult,
    ControlCommand,
    EgoState,
    LaneModel,
    MotionPlan,
    PerceptionStatus,
    RangeEstimate,
    RangeSource,
    SafetyState,
    TrackedObject,
)

logger = setup_logger(__name__)

NOMINAL_DT_S = 0.05

_SEVERITY = {
    SafetyState.NOMINAL: 0,
    SafetyState.LIMITED: 1,
    SafetyState.MIN_RISK_MANEUVER: 2,
    SafetyState.DISENGAGE: 3,
}


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _worse(a: SafetyState, b: SafetyState) -> SafetyState:
    """Return whichever of the two states is more severe."""
    return a if _SEVERITY[a] >= _SEVERITY[b] else b


# --------------------------------------------------------------------------- #
# Independent range-rate estimation
# --------------------------------------------------------------------------- #


@dataclass
class _RangeState:
    distance_m: float
    rate_mps: float = 0.0
    updates: int = 0
    seeded: bool = True
    """True while ``rate_mps`` still carries the fabricated safe prior instead of a
    value derived from the MEASURED range history.  Nothing that authorises
    full-authority braking may read a seeded rate; see
    :meth:`SafetyArbiter._classify_hazard`."""
    clock_s: float = 0.0
    history: List[Tuple[float, float]] = field(default_factory=list)
    """``(clock_s, measurement_m)`` for MEASURED samples only, inside the rate
    window.  A coasted (tracker-extrapolated) range is deliberately excluded: an
    extrapolation is not a measurement and must not be able to establish the
    closing rate that authorises an emergency stop."""


class _AlphaBetaRange:
    """Per-track alpha-beta filter over range, owned by the arbiter.

    State is ``[d, d_dot]`` with ``d`` in metres and ``d_dot`` in m/s using the
    ``v_lead - v_ego`` sign convention (negative = closing).

    The filter re-initialises rather than differentiating across an implausible
    jump.  A greedy tracker can hand the same track id a detection 60 m away from
    its last one (the 68 m -> 6.8 m teleport in the association code); dividing that
    by dt would produce 1200 m/s of "closing speed" and an instant full-brake
    command.  Re-initialising costs one frame of rate information, so the arbiter
    reports ``range_jump`` and degrades that frame.

    On an initialisation (a new key, or a re-init after a jump) the rate is set to
    ``seed_rate_mps`` rather than to zero.  Zero is the OPTIMISTIC prior -- it says
    "whatever this is, it is keeping pace with me" -- and it made the backstop blind
    on exactly the frame a hazard appeared: a stationary car 15 m ahead with the ego
    at 15 m/s reported ``ttc=inf`` and commanded no brake.  The caller passes
    ``-ego_speed``, the safe prior that a newly seen object is stationary in the
    world.  The seed is bounded by the ego speed, so it can never manufacture the
    1200 m/s that the raw difference quotient would.
    """

    def __init__(
        self,
        alpha: float = 0.45,
        beta: float = 0.08,
        max_rate_mps: float = 80.0,
        jump_gate_m: float = 5.0,
        jump_gate_frac: float = 0.35,
        history_window_s: float = 0.60,
    ) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        if not (0.0 < beta <= 1.0):
            raise ValueError("beta must be in (0, 1]")
        self.alpha = alpha
        self.beta = beta
        self.max_rate_mps = max_rate_mps
        self.jump_gate_m = jump_gate_m
        self.jump_gate_frac = jump_gate_frac
        self.history_window_s = history_window_s
        self._states: Dict[int, _RangeState] = {}

    def reset(self) -> None:
        self._states.clear()

    def forget(self, key: int) -> None:
        """Drop one track's state so the next update re-initialises and re-seeds."""
        self._states.pop(key, None)

    def forget_all_but(self, keys: List[int]) -> None:
        """Drop filters for tracks that no longer exist, so ids cannot be reused."""
        alive = set(keys)
        for key in [k for k in self._states if k not in alive]:
            del self._states[key]

    def updates(self, key: int) -> int:
        """Number of measurements folded into ``key`` since its last (re-)init."""
        state = self._states.get(key)
        return 0 if state is None else state.updates

    def is_seeded(self, key: int) -> bool:
        """True while this track's rate is still the fabricated prior.

        A seeded rate is a statement about what the arbiter does NOT know.  It is
        deliberately pessimistic so the backstop is not blind on the frame a hazard
        appears, and for exactly that reason it must never be mistaken for a
        measurement.
        """
        state = self._states.get(key)
        return True if state is None else state.seeded

    def anchor_rate(self, key: int, rate_mps: float) -> float:
        """Replace the seeded rate with a MEASURED one and clear the seeded flag.

        Called the first frame the raw-measurement window is long enough to support a
        slope.  Without it the alpha-beta loop needs about twelve frames
        to walk a -20 m/s seed back to the truth, and the AEB would stay inhibited
        (or, worse in the old code, stay armed) for all of them.
        """
        state = self._states.get(key)
        if state is None:
            return rate_mps
        state.rate_mps = _clamp(rate_mps, -self.max_rate_mps, self.max_rate_mps)
        state.seeded = False
        return state.rate_mps

    def measured_rate(
        self, key: int, min_samples: int, min_span_s: float
    ) -> float | None:
        """Least-squares slope of the RAW range measurements, or None.

        This is the only closing rate in the arbiter that contains no prior at all:
        it is computed from measurements the perception stack actually produced, on
        a window long enough that range noise cannot fake a closure.  ``None`` means
        "not enough measured evidence yet", which is a different claim from
        "not closing".
        """
        state = self._states.get(key)
        if state is None or len(state.history) < max(2, min_samples):
            return None
        span = state.history[-1][0] - state.history[0][0]
        if span < min_span_s or span <= 0.0:
            return None
        n = float(len(state.history))
        mean_t = sum(t for t, _ in state.history) / n
        mean_d = sum(d for _, d in state.history) / n
        den = sum((t - mean_t) ** 2 for t, _ in state.history)
        if den <= 1e-9:
            return None
        slope = sum((t - mean_t) * (d - mean_d) for t, d in state.history) / den
        if not math.isfinite(slope):
            return None
        return _clamp(slope, -self.max_rate_mps, self.max_rate_mps)

    def update(
        self,
        key: int,
        measurement_m: float,
        dt_s: float,
        seed_rate_mps: float = 0.0,
        rebase: bool = False,
        measured: bool = True,
    ) -> Tuple[float, float, bool]:
        """Fold one measurement in.

        Args:
            key: Track id.
            measurement_m: This frame's range measurement, metres.
            dt_s: Elapsed time since the previous update, seconds.
            seed_rate_mps: Rate to start from when this call initialises or
                re-initialises the track, in the ``v_lead - v_ego`` convention
                (negative = closing).  Pass ``-ego_speed`` for the safe prior that
                an unknown object is stationary; the default of 0.0 keeps the bare
                filter usable on its own but is the OPTIMISTIC prior.
            rebase: The measurement comes from a DIFFERENT range source than the
                previous one.  The level is re-anchored and the rate and the
                measured-slope window are carried across unchanged, because the
                estimator changing its mind about provenance is not a discontinuity
                in the world.
            measured: False for a coasted (extrapolated) range.  The filter still
                tracks it, but the sample is kept out of the raw window, so an
                extrapolation can never establish the measured closing rate the AEB
                path requires.

        Returns ``(filtered_distance_m, range_rate_mps, reinitialised)``.
        """
        seed = seed_rate_mps if math.isfinite(seed_rate_mps) else 0.0
        seed = _clamp(seed, -self.max_rate_mps, self.max_rate_mps)
        state = self._states.get(key)
        if state is None or dt_s <= 1e-4 or not math.isfinite(measurement_m):
            self._states[key] = self._fresh(measurement_m, seed, measured)
            return measurement_m, seed, state is not None

        state.clock_s += dt_s
        if rebase:
            # A range-SOURCE switch is a step in the SIGNAL, not motion of the
            # object, so it must change neither the rate estimate nor the measured
            # slope.  Re-anchor the level and carry both across, instead of
            # throwing the filter away: discarding it re-applied the -ego_speed
            # seed, and on real footage the provenance flip-flopped every frame, so
            # the fabricated rate never got a chance to converge.
            state.distance_m = measurement_m - state.rate_mps * dt_s
            if state.history:
                shift = measurement_m - state.history[-1][1]
                state.history = [(t, d + shift) for t, d in state.history]

        predicted = state.distance_m + state.rate_mps * dt_s
        residual = measurement_m - predicted
        gate = max(self.jump_gate_m, self.jump_gate_frac * max(measurement_m, state.distance_m))
        if abs(residual) > gate:
            self._states[key] = self._fresh(measurement_m, seed, measured)
            return measurement_m, seed, True

        distance = predicted + self.alpha * residual
        rate = state.rate_mps + (self.beta / dt_s) * residual
        rate = _clamp(rate, -self.max_rate_mps, self.max_rate_mps)
        state.distance_m = distance
        state.rate_mps = rate
        state.updates += 1
        if measured:
            state.history.append((state.clock_s, measurement_m))
            cutoff = state.clock_s - self.history_window_s
            while len(state.history) > 2 and state.history[0][0] < cutoff:
                state.history.pop(0)
        return distance, rate, False

    def _fresh(self, measurement_m: float, seed: float, measured: bool) -> _RangeState:
        state = _RangeState(distance_m=measurement_m, rate_mps=seed, updates=1, seeded=True)
        if measured and math.isfinite(measurement_m):
            state.history.append((0.0, measurement_m))
        return state


# --------------------------------------------------------------------------- #
# Configuration and inputs
# --------------------------------------------------------------------------- #


@dataclass
class ArbiterLimits:
    """Hard limits and thresholds. These must be STRICTER than the planner's."""

    # Longitudinal envelope
    max_speed_mps: float = 33.0
    max_acceleration_mps2: float = 3.0
    max_deceleration_mps2: float = 8.0
    comfort_decel_mps2: float = 3.0
    mrm_decel_mps2: float = 3.5
    max_jerk_mps3: float = 4.0
    max_jerk_emergency_mps3: float = 15.0
    accel_authority_mps2: float = 2.5
    brake_authority_mps2: float = 8.0
    """m/s^2 delivered at brake = 1.0. Used to turn a required deceleration into a
    pedal fraction; must match the vehicle the commands are sent to."""

    # Lateral envelope
    max_steering_angle_rad: float = 0.52
    max_steering_rate_rad_s: float = 0.5
    max_lateral_accel_mps2: float = 4.5
    max_lateral_offset_m: float = 1.5
    """Lane-departure limit, metres from the lane centre. Enforceable ONLY when
    ``SafetyContext.lateral_offset_m`` is supplied by a calibrated lane geometry;
    without it the arbiter reports ``lane_offset_unavailable`` rather than
    pretending the check ran."""
    wheelbase_m: float = 2.8
    max_road_wheel_rad: float = 0.436
    """Road-wheel angle corresponding to ``steering = 1.0`` (25 deg by default).
    Must match ``ControllerConfig.max_steering_angle_deg`` or the normalised
    command cannot be converted to a physical angle."""

    # Headway / collision avoidance
    absolute_min_gap_m: float = 2.0
    standstill_gap_m: float = 4.0
    reaction_time_s: float = 0.6
    ego_brake_capability_mps2: float = 6.0
    lead_brake_capability_mps2: float = 8.0
    ttc_brake_s: float = 0.9
    ttc_warn_s: float = 1.6
    aeb_required_decel_mps2: float = 5.0
    aeb_decel_margin: float = 1.15
    aeb_min_decel_mps2: float = 4.0
    aeb_headway_frac: float = 0.5
    """A gap below this fraction of the RSS minimum is an emergency in its own
    right: even ordinary braking by the lead would then cause a collision, so the
    geometric required deceleration (which is near zero while the lead keeps pace)
    is not a sufficient trigger."""

    # What it takes before a closing RATE may authorise full-authority braking
    deferred_aeb_decel_mps2: float = 5.0
    """Deceleration ceiling while an emergency test has tripped on the SAFE PRIOR
    but the closing rate is not yet measured, and the prior is ACUTE (the seeded
    TTC is already below ``ttc_brake_s``).

    This number is the whole cost/benefit of not letting a fabricated rate fire the
    AEB, so it is chosen against a specific case rather than by feel: a stationary
    obstacle 15 m ahead with the ego at 15 m/s needs at least 4.6 m/s^2 during the
    deferral for a full-authority stop afterwards to still clear it.  5.0 m/s^2 --
    the same number as ``aeb_required_decel_mps2``, the threshold at which the
    arbiter calls a situation an emergency at all -- buys that with margin while
    staying well under the 8.0 m/s^2 a MEASURED closure commands.  A prior may act;
    it may not act with full authority."""

    aeb_rate_corroboration_frames: int = 2
    """Consecutive frames a RATE-DEPENDENT emergency test must hold before it
    authorises full-authority braking.  Range noise makes the closing-rate estimate
    wander, and a single noisy frame used to be enough: with +/-0.30 m of pinhole
    noise on a lead holding a constant 12 m at 15 m/s, isolated frames of
    ``brake = 0.625`` appeared out of nothing.  A genuine closure persists; a noise
    excursion does not.  The RATE-INDEPENDENT tests bypass this -- they read the
    measured range, so there is nothing to corroborate."""

    aeb_min_rate_samples: int = 4
    """Raw range measurements required in the rate window before the arbiter will
    call its closing rate MEASURED.  Below this the rate is still (partly) the
    ``-ego_speed`` safe prior, which is a statement about ignorance, and the
    rate-dependent AEB tests are inhibited -- see
    :meth:`SafetyArbiter._classify_hazard`.  The rate-INDEPENDENT emergencies (a gap
    below the absolute minimum, a gap inside half the speed-matched RSS distance)
    keep full authority on a track's very first frame, because they are computed
    from the measured range and the measured ego speed and contain no prior."""
    aeb_min_rate_span_s: float = 0.15
    """Elapsed time the rate window must span as well.  Samples alone are not
    enough: four measurements 5 ms apart cannot separate a closure from range
    noise."""
    range_rate_window_s: float = 0.60
    """Length of the raw-measurement window the measured slope is fitted over.
    Longer rejects more range noise and lags a genuine lead deceleration by more;
    the alpha-beta filter, which the measured slope only ANCHORS, supplies the fast
    response after that."""

    # Range-source stability
    range_source_dwell_frames: int = 5
    """Consecutive frames the second channel must be unusable before the arbiter
    stops using it.  ADOPTING the channel is immediate -- it can only lower the
    range, which is the safe direction -- but DROPPING it needs corroboration, so a
    confidence that dithers across ``min_range_confidence`` cannot flip the range
    source every frame.  On real footage the provenance flipped 26 times in 400
    frames and each flip used to discard the range filter."""
    range_confidence_hysteresis: float = 0.10
    """Once the second channel is in use its confidence may fall this far below
    ``min_range_confidence`` before it counts as unusable."""
    range_disagreement_hysteresis: float = 0.25
    """Fractional hysteresis band around ``range_disagreement_frac``, so a relative
    difference sitting on the threshold does not toggle agreement every frame."""

    max_coast_frames: int = 5
    """How many frames the arbiter will keep assessing a track the tracker is
    COASTING (``time_since_update > 0``) through a perception dropout.  The arbiter
    used to discard the entire lead assessment on a dropout frame, so the braking it
    was applying for a real obstacle collapsed to the flat MRM deceleration the
    instant perception blinked.  Past this many frames the extrapolation is too
    stale to act on and the dropout escalation takes over on its own."""

    # Plan feasibility
    plan_horizon_s: float = 1.0
    """Horizon used ONLY to judge a plan's requested ACCELERATION -- the direction
    that adds energy. A plan that asks for a large deceleration is never a
    violation; excessive achieved deceleration is judged separately, against the
    measured speed and with hazard context."""

    # Degradation policy
    limited_after_dropouts: int = 1
    mrm_after_dropouts: int = 3
    disengage_after_frames: int = 40
    recovery_frames: int = 10
    limited_throttle_cap: float = 0.0
    """Maximum throttle allowed once degraded. 0.0 means: never accelerate while
    the arbiter is not in NOMINAL."""

    kinematics_min_speed_mps: float = 0.5
    """Below this speed the achieved-acceleration and jerk checks are disabled.
    Wheel-speed quantisation and the standstill transition dominate the difference
    quotient there, so the checks would report faults that are artefacts of the
    measurement rather than of the control."""

    # In-path corridor -- the arbiter's OWN geometry, never the tracker's flag
    min_in_path_half_width_frac: float = 0.30
    """Floor on the half width of the arbiter's in-path corridor, as a fraction of
    the frame width.  The corridor actually used is
    ``max(this, SafetyContext.ego_lane_half_width_frac)``, so the arbiter's hazard
    window is always at least as wide as the planner's however the planner is
    configured.  Widening only ever brakes for more objects."""
    lane_trust_confidence: float = 0.50
    """Minimum ``LaneModel.confidence`` (and the lane must not be ``is_mock``)
    before the lane centre may be used as a SECOND corridor anchor.  A lane centre
    is never allowed to move or narrow the corridor -- a lane-detection error must
    not be able to hide a real lead from the backstop."""
    mrm_straighten_speed_mps: float = 1.0
    """Below this speed a minimum-risk manoeuvre stops holding the steering angle
    and slews toward straight.  Above it the MRM HOLDS the last actuated angle:
    zeroing the wheel mid-corner is a lateral hazard in its own right."""

    # Timing
    min_dt_s: float = 0.005
    """Frames faster than this have their dt clamped up, silently. Running well is
    not a fault: the previous code reported ``timing_dt_out_of_range`` for a 2 ms
    frame and let that latch DISENGAGE on an unpaced run."""
    max_dt_s: float = 0.5
    max_frame_gap_s: float = 0.5

    # Diverse range channel
    range_disagreement_frac: float = 0.30
    """Relative disagreement between the two range channels that is reported and
    forces LIMITED. It is a mitigated condition, not a health fault."""
    min_range_confidence: float = 0.35
    """A second-channel :class:`~adas.core.models.RangeEstimate` below this
    confidence is treated as UNAVAILABLE rather than merely down-weighted, so a
    channel that is only guessing cannot pull the fused range around."""
    range_corroboration_frames: int = 3
    """Consecutive disagreeing frames required before a NEARER second-channel range
    is adopted. Until then the pinhole range is used, so a single-frame depth
    outlier cannot reach the emergency path. A second channel that reads FARTHER
    than the pinhole is never adopted at all."""

    # Camera calibration honesty
    allow_uncalibrated_range: bool = False
    """When False (the default) a ``SafetyContext.camera_calibrated=False`` frame
    reports ``camera_uncalibrated`` and the state can never be NOMINAL: the metric
    range every hazard test is built on is then an assumption, not a measurement.
    Set True only for a deployment that has accepted that risk explicitly."""

    # Output shaping (does not by itself escalate the state)
    throttle_rate_per_s: float = 5.0
    brake_release_rate_per_s: float = 8.0
    """Rate at which the brake may be RELEASED. There is deliberately no
    corresponding apply-rate limit here: the arbiter must never attenuate a brake.
    Jerk shaping on brake application lives in
    :class:`adas.control.controller.PIDLikeLongitudinalController`, which owns the
    emergency exemption."""

    def __post_init__(self) -> None:
        if self.max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be positive")
        if self.brake_authority_mps2 <= 0 or self.accel_authority_mps2 <= 0:
            raise ValueError("actuator authorities must be positive")
        if self.max_road_wheel_rad <= 0:
            raise ValueError("max_road_wheel_rad must be positive")
        if self.standstill_gap_m < self.absolute_min_gap_m:
            raise ValueError("standstill_gap_m must be >= absolute_min_gap_m")
        if self.ttc_warn_s < self.ttc_brake_s:
            raise ValueError("ttc_warn_s must be >= ttc_brake_s")
        if self.mrm_after_dropouts < self.limited_after_dropouts:
            raise ValueError("mrm_after_dropouts must be >= limited_after_dropouts")
        if self.recovery_frames < 1 or self.disengage_after_frames < 1:
            raise ValueError("recovery_frames and disengage_after_frames must be >= 1")
        if not (0.0 < self.aeb_headway_frac <= 1.0):
            raise ValueError("aeb_headway_frac must be in (0, 1]")
        if self.min_in_path_half_width_frac < 0.0:
            raise ValueError("min_in_path_half_width_frac must be >= 0")
        if self.range_corroboration_frames < 1:
            raise ValueError("range_corroboration_frames must be >= 1")
        if self.aeb_rate_corroboration_frames < 1:
            raise ValueError("aeb_rate_corroboration_frames must be >= 1")
        if self.aeb_min_rate_samples < 2:
            raise ValueError("aeb_min_rate_samples must be >= 2")
        if self.aeb_min_rate_span_s <= 0.0:
            raise ValueError("aeb_min_rate_span_s must be > 0")
        if self.range_rate_window_s < self.aeb_min_rate_span_s:
            raise ValueError("range_rate_window_s must be >= aeb_min_rate_span_s")
        if self.range_source_dwell_frames < 1:
            raise ValueError("range_source_dwell_frames must be >= 1")
        if not (0.0 <= self.range_disagreement_hysteresis < 1.0):
            raise ValueError("range_disagreement_hysteresis must be in [0, 1)")
        if self.range_confidence_hysteresis < 0.0:
            raise ValueError("range_confidence_hysteresis must be >= 0")
        if self.max_coast_frames < 0:
            raise ValueError("max_coast_frames must be >= 0")
        if not (
            self.comfort_decel_mps2
            <= self.deferred_aeb_decel_mps2
            <= self.max_deceleration_mps2
        ):
            raise ValueError(
                "deferred_aeb_decel_mps2 must be between comfort_decel_mps2 and "
                "max_deceleration_mps2"
            )
        if self.min_dt_s <= 0.0 or self.max_dt_s <= self.min_dt_s:
            raise ValueError("max_dt_s must be > min_dt_s > 0")


@dataclass
class SafetyContext:
    """Everything the arbiter needs that is not the plan or the command.

    This is the ``state`` argument of ``arbitrate(plan, cmd, state)``.

    Attributes:
        ego: Measured ego state. ``ego.valid`` must be True for ``speed_mps`` to be
            believed; an invalid or missing ego state forces a minimum-risk
            manoeuvre, because none of the hazard maths is defined without it.
        tracks: The RAW track list for this frame. The arbiter does its own lead
            selection over it. An empty list means "perception ran and saw
            nothing"; a perception failure must be signalled through
            ``perception``, never by passing an empty list.
        perception: Perception health for this frame.
        dt_s: Nominal timestep, seconds. Used only when ``timestamp_s`` is absent.
        timestamp_s: Monotonic capture time of this frame, seconds. When supplied
            on consecutive frames the arbiter uses the MEASURED interval instead of
            the nominal one, and flags a stale input if the gap is implausible.
        frame_width_px / frame_height_px: Image size, for the in-path gate.
        lane: Lane model. It may only WIDEN the arbiter's in-path corridor, by
            adding a second anchor, and only when it is real
            (``not is_mock``) and its confidence reaches
            ``ArbiterLimits.lane_trust_confidence``. It can never move or narrow
            the corridor: a lane-detection error must not be able to hide a lead.
        ego_lane_half_width_frac: Half width of the ego-lane band as a fraction of
            the frame width, usually the planner's value. The arbiter widens it to
            at least ``ArbiterLimits.min_in_path_half_width_frac``, so its hazard
            window is always at least as wide as the planner's.
        independent_ranges: Range estimates from a channel other than the box
            height pinhole, keyed by track id. Absent keys simply mean no second
            opinion for that track.
        lateral_offset_m: Metric ego offset from the lane centre, or None.
        camera_calibrated: Whether the camera whose geometry produced the metric
            ranges is calibrated. ``False`` reports ``camera_uncalibrated`` and
            floors the state at LIMITED unless
            ``ArbiterLimits.allow_uncalibrated_range`` is set. ``None`` means the
            caller did not say, and the arbiter makes no claim either way.
    """

    ego: EgoState | None = None
    tracks: List[TrackedObject] = field(default_factory=list)
    perception: PerceptionStatus | None = None
    dt_s: float = NOMINAL_DT_S
    timestamp_s: float | None = None
    frame_width_px: int = 0
    frame_height_px: int = 0
    lane: LaneModel | None = None
    ego_lane_half_width_frac: float = 0.30
    independent_ranges: Dict[int, RangeEstimate] | None = None
    lateral_offset_m: float | None = None
    """Ego offset from the lane centre in METRES, positive when the ego is right
    of centre. Requires a calibrated camera; pass None when only a pixel-space lane
    centre is available and the arbiter will report the check as unavailable."""
    camera_calibrated: bool | None = None
    """True/False when the caller knows, None when it does not report. See the
    class docstring; the pipeline is expected to pass
    ``camera.calibrated`` here."""


@dataclass
class LeadAssessment:
    """The arbiter's own view of the most dangerous in-path object."""

    track_id: int
    distance_m: float
    range_rate_mps: float
    ttc_s: float
    required_decel_mps2: float
    rss_min_gap_m: float
    source: RangeSource
    rss_matched_gap_m: float = 0.0
    """The RSS minimum gap computed with the OPTIMISTIC assumption that the lead is
    travelling at exactly the ego speed.  It depends on the measured range and the
    measured ego speed only -- no closing rate at all -- so it is the one headway
    test that may authorise full braking before the rate has been measured."""
    disagreement: bool = False
    reinitialised: bool = False
    coasting: bool = False
    coast_frames: int = 0
    rate_is_measured: bool = False
    """True once the closing rate is derived from the arbiter's own window of RAW
    range measurements rather than from the ``-ego_speed`` safe prior.  The
    rate-dependent AEB tests are inhibited while this is False: a fabricated rate
    may inform caution, it must never by itself authorise full-authority braking."""
    measured_rate_mps: float | None = None
    """The raw least-squares slope, when there is one; None while unmeasured."""
    rate_updates: int = 0
    """How many measurements the arbiter's own range filter has folded in since
    this track was last (re-)initialised. 1 means the rate is still the seeded
    worst-case prior rather than a measurement."""


# --------------------------------------------------------------------------- #
# The arbiter
# --------------------------------------------------------------------------- #


class SafetyArbiter:
    """Independent, authoritative arbitration between a plan and the actuators."""

    def __init__(self, limits: ArbiterLimits | None = None) -> None:
        self.limits = limits or ArbiterLimits()
        self._range_filter = _AlphaBetaRange(
            history_window_s=self.limits.range_rate_window_s
        )
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Clear every latch and every history. The only way out of DISENGAGE."""
        self._latched_state = SafetyState.NOMINAL
        self._health_fault_streak = 0
        self._clean_streak = 0
        self._dropout_streak = 0
        self._prev_speed_mps: float | None = None
        self._prev_accel_mps2: float | None = None
        self._prev_steering_rad = 0.0
        self._prev_request_rad = 0.0
        self._last_commanded_steering = 0.0
        self._prev_throttle = 0.0
        self._prev_brake = 0.0
        self._last_timestamp_s: float | None = None
        self.lane_offset_available = False
        self._range_filter.reset()
        self._disagreement_streak: Dict[int, int] = {}
        self._range_source: Dict[int, RangeSource] = {}
        self._stable_source: Dict[int, RangeSource] = {}
        self._second_channel: Dict[int, bool] = {}
        self._second_channel_pending: Dict[int, int] = {}
        self._second_channel_gap: Dict[int, int] = {}
        self._agreeing: Dict[int, bool] = {}
        # Deliberately NOT per track: keyed on the track id, an AEB condition that
        # alternated between two candidate leads would never corroborate and the
        # emergency path would never arm at all. Over-corroborating across a
        # changing lead errs toward braking; under-corroborating errs toward not.
        self._aeb_rate_streak = 0
        # How many frames ago the arbiter last overrode a channel itself. Findings
        # that are a CONSEQUENCE of the arbiter's own intervention are reported but
        # may not force a state, or the intervention becomes its own justification.
        self._long_override_age = 999
        self._lat_override_active = False
        self.total_violations = 0
        self.total_overrides = 0
        self.last_lead: LeadAssessment | None = None
        self.last_shaping: List[str] = []

    @property
    def state(self) -> SafetyState:
        """Current latched safety state."""
        return self._latched_state

    @property
    def dropout_frames(self) -> int:
        """Consecutive frames on which perception was reported unhealthy."""
        return self._dropout_streak

    # ------------------------------------------------------------------- main

    def arbitrate(
        self,
        plan: MotionPlan | None,
        command: ControlCommand,
        state: SafetyContext,
    ) -> ArbitrationResult:
        """Decide what the actuators actually receive.

        Args:
            plan: The planner's motion plan, or None when planning itself failed.
                None is treated as a planning dropout, not as "no constraint".
            command: The controller's proposed command.
            state: The :class:`SafetyContext` for this frame.

        Returns:
            An :class:`ArbitrationResult`. ``result.command`` is authoritative:
            the caller must send exactly that to the actuators and must not fall
            back to ``command`` under any circumstance.
            ``result.violations`` is ``faults + mitigated + hazards`` in that
            order; see the module docstring for what each list means and for which
            one -- only ``faults`` -- can latch ``DISENGAGE``.
        """
        ctx = state
        # Three disjoint finding lists; see the module docstring.
        #   faults    -- HEALTH faults. ONLY these accumulate toward DISENGAGE.
        #   mitigated -- detected and handled this frame (a clamp that worked), or
        #                measured and merely reported. LIMITED, never DISENGAGE.
        #   hazards   -- traffic the arbiter exists to handle. LIMITED/MRM only.
        faults: List[str] = []
        mitigated: List[str] = []
        hazards: List[str] = []
        # advisory -- reported in ``violations`` and in the logs, but NOT allowed to
        # force a state or to reset the recovery streak, because each one is a
        # CONSEQUENCE OF THE ARBITER'S OWN INTERVENTION.  See ``_self_induced``.
        advisory: List[str] = []
        self.last_shaping = []

        dt_s = self._resolve_dt(ctx, faults)
        ego_speed_mps, ego_ok = self._resolve_ego(ctx, faults, mitigated)
        cmd_in, cmd_ok = self._sanitize(command, faults, mitigated)

        if ctx.camera_calibrated is False and not self.limits.allow_uncalibrated_range:
            # Every metric range below is derived from this camera's geometry. An
            # uncalibrated camera makes them assumptions, so the arbiter says so
            # and refuses to report NOMINAL while acting on them.
            mitigated.append("camera_uncalibrated")

        perception = ctx.perception if ctx.perception is not None else PerceptionStatus()
        if perception.ok:
            self._dropout_streak = 0
        else:
            self._dropout_streak = max(
                self._dropout_streak + 1, int(perception.consecutive_failures)
            )
            faults.append(
                "perception_dropout_%d%s"
                % (self._dropout_streak, (":" + perception.reason) if perception.reason else "")
            )

        # The lead is assessed on EVERY frame, dropout included.  The tracker coasts
        # its tracks through a perception blink, and discarding the whole assessment
        # on those frames made the arbiter forget the hazard it was braking for: the
        # command collapsed from the AEB demand to the flat MRM deceleration the
        # instant perception blinked, while still closing on the obstacle.  A coasted
        # track is degraded, not absent -- ``_assess_lead`` drops it once
        # ``time_since_update`` passes ``max_coast_frames`` and never lets an
        # extrapolated range establish the measured closing rate the AEB needs.
        lead = self._assess_lead(ctx, ego_speed_mps, dt_s, mitigated, advisory)
        self.last_lead = lead

        aeb, hazard, deferred_acute, rate_pending = self._classify_hazard(lead, hazards)
        if plan is None:
            faults.append("plan_missing")
        else:
            self._check_plan(plan, ego_speed_mps, ego_ok, faults, mitigated, advisory)
        self._check_kinematics(
            ego_speed_mps, ego_ok, dt_s, aeb or hazard, mitigated, advisory
        )

        self._check_lane_departure(ctx, mitigated)
        steering_out = self._limit_steering(
            cmd_in.steering, ego_speed_mps, dt_s, mitigated, advisory
        )

        violations = faults + mitigated + advisory + hazards
        requested = self._decide_state(faults, mitigated, hazards, cmd_ok, ego_ok, aeb)
        new_state = self._latch(requested)

        out = self._synthesise_command(
            cmd_in,
            steering_out,
            new_state,
            aeb,
            hazard,
            lead,
            dt_s,
            cmd_ok,
            ego_speed_mps,
            deferred_acute,
            rate_pending,
        )

        if violations:
            self.total_violations += 1
        if new_state is not SafetyState.NOMINAL:
            self.total_overrides += 1

        self._prev_steering_rad = out.steering * self.limits.max_road_wheel_rad
        # Updated on EVERY frame, whatever the state. The previous code only wrote
        # this in NOMINAL, so an MRM entered in a bend held 0.0 -- it straightened
        # the wheel into the corner and then raised a steering_rate fault against
        # its own hold.
        self._last_commanded_steering = out.steering
        self._prev_throttle = out.throttle
        self._prev_brake = out.brake
        # Record what the arbiter itself did to the actuators this frame, so that
        # next frame's checks can tell the vehicle's response to the ARBITER apart
        # from the planner's or the world's behaviour.
        long_override = (
            new_state is not SafetyState.NOMINAL
            or out.brake > cmd_in.brake + 1e-3
            or out.throttle < cmd_in.throttle - 1e-3
        )
        self._long_override_age = 0 if long_override else min(self._long_override_age + 1, 999)
        self._lat_override_active = (
            _SEVERITY[new_state] >= _SEVERITY[SafetyState.MIN_RISK_MANEUVER]
        )
        if ego_ok:
            self._prev_speed_mps = ego_speed_mps
        if ctx.timestamp_s is not None and math.isfinite(ctx.timestamp_s):
            self._last_timestamp_s = ctx.timestamp_s

        reason = self._describe(new_state, violations, lead, aeb)
        if new_state is not SafetyState.NOMINAL:
            logger.warning(
                "Arbiter %s: cmd t=%.2f b=%.2f s=%.2f | %s",
                new_state.value,
                out.throttle,
                out.brake,
                out.steering,
                reason,
            )
        return ArbitrationResult(
            command=out, state=new_state, violations=list(violations), reason=reason
        )

    # ------------------------------------------------------------- input prep

    def _resolve_dt(self, ctx: SafetyContext, faults: List[str]) -> float:
        """Prefer the measured frame interval; clamp and flag implausible ones.

        A frame FASTER than ``min_dt_s`` is not a finding of any kind. Running well
        is not a fault: the previous version reported ``timing_dt_out_of_range`` for
        an unpaced 2 ms frame and fed it to the DISENGAGE counter, which latched the
        terminal state after 40 frames of an empty road.
        """
        lim = self.limits
        dt = ctx.dt_s if (isinstance(ctx.dt_s, (int, float)) and math.isfinite(ctx.dt_s)) else 0.0

        if ctx.timestamp_s is not None and math.isfinite(ctx.timestamp_s):
            if self._last_timestamp_s is not None:
                measured = ctx.timestamp_s - self._last_timestamp_s
                if measured <= 0.0:
                    faults.append("timing_non_monotonic_timestamp")
                elif measured > lim.max_frame_gap_s:
                    faults.append("timing_stale_input_%.3fs" % measured)
                    dt = lim.max_dt_s
                else:
                    dt = measured

        if dt <= 0.0 or not math.isfinite(dt):
            faults.append("timing_invalid_dt")
            return NOMINAL_DT_S
        if dt > lim.max_dt_s:
            # The loop is running slower than the arbiter's maths is valid for.
            # That is a health fault: the hazard estimates below are stale.
            faults.append("timing_dt_above_max_%.3fs" % dt)
            return lim.max_dt_s
        if dt < lim.min_dt_s:
            # Silently clamp. Nothing is wrong with a fast frame; the clamp only
            # keeps the per-frame rate windows from collapsing to zero.
            return lim.min_dt_s
        return dt

    def _resolve_ego(
        self, ctx: SafetyContext, faults: List[str], mitigated: List[str]
    ) -> Tuple[float, bool]:
        """Return ``(speed_mps, usable)``. An unusable ego state forces an MRM."""
        ego = ctx.ego
        if ego is None:
            faults.append("ego_state_missing")
            return 0.0, False
        if not ego.valid:
            faults.append("ego_state_invalid")
            return 0.0, False
        if not math.isfinite(ego.speed_mps) or ego.speed_mps < 0.0:
            faults.append("ego_speed_implausible")
            return 0.0, False
        if ego.speed_mps > self.limits.max_speed_mps * 1.5:
            # Reported and it degrades the state, but the ego state is still
            # USABLE, so this is not a health fault: the arbiter can and does keep
            # arbitrating on it.
            mitigated.append("ego_speed_above_envelope_%.1f" % ego.speed_mps)
            return ego.speed_mps, True
        return float(ego.speed_mps), True

    def _sanitize(
        self, cmd: ControlCommand, faults: List[str], mitigated: List[str]
    ) -> Tuple[ControlCommand, bool]:
        """Range- and NaN-check the proposed command.

        A non-finite field makes the whole command unusable; the arbiter reports
        that and hands back zeros, then escalates to a minimum-risk manoeuvre.  It
        does NOT clamp NaN, because ``max(0, min(1, nan))`` evaluates to 1.0 in
        Python and would turn a corrupt command into full braking.
        """
        values = (cmd.throttle, cmd.brake, cmd.steering)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            faults.append("command_not_finite")
            logger.error("Non-finite control command received: %r", cmd)
            return ControlCommand(throttle=0.0, brake=0.0, steering=0.0), False

        throttle, brake, steering = (float(v) for v in values)
        if not (0.0 <= throttle <= 1.0):
            mitigated.append("command_throttle_out_of_range_%.3f" % throttle)
            throttle = _clamp(throttle, 0.0, 1.0)
        if not (0.0 <= brake <= 1.0):
            mitigated.append("command_brake_out_of_range_%.3f" % brake)
            brake = _clamp(brake, 0.0, 1.0)
        if not (-1.0 <= steering <= 1.0):
            mitigated.append("command_steering_out_of_range_%.3f" % steering)
            steering = _clamp(steering, -1.0, 1.0)
        if throttle > 0.02 and brake > 0.02:
            mitigated.append("command_throttle_brake_conflict")
            throttle = 0.0
        return ControlCommand(throttle=throttle, brake=brake, steering=steering), True

    # ------------------------------------------------------------ lead + risk

    def _corridor_anchors(self, ctx: SafetyContext) -> Tuple[List[float], float]:
        """Return ``(anchor_columns_px, half_width_px)`` for the in-path corridor.

        The image centre is ALWAYS an anchor: it is where the ego is going, and it
        is the one reference column that no perception module can move.  A lane
        model adds a SECOND anchor -- widening the corridor -- but only when it is
        real and confident.  A lane centre can therefore never narrow the corridor
        or shift it off the ego's own heading.
        """
        lim = self.limits
        frac = max(float(ctx.ego_lane_half_width_frac or 0.0), lim.min_in_path_half_width_frac)
        half_px = ctx.frame_width_px * frac
        anchors = [ctx.frame_width_px / 2.0]
        lane = ctx.lane
        if (
            lane is not None
            and not lane.is_mock
            and math.isfinite(lane.lane_center_px)
            and 0.0 <= lane.lane_center_px <= ctx.frame_width_px
            and lane.confidence >= lim.lane_trust_confidence
        ):
            anchors.append(float(lane.lane_center_px))
        return anchors, half_px

    def _in_path(self, ctx: SafetyContext, obj: TrackedObject) -> bool:
        """The arbiter's OWN in-path test. Deliberately wider than the planner's.

        It does not read ``TrackedObject.in_ego_lane``.  That flag is set by the
        tracker from the same lane model the planner reads, so a lane-detection
        error would reach the planner and the "independent" backstop identically --
        the definition of a common-mode failure.  With the old short-circuit, a lead
        dead ahead at x=640 of 1280 disappeared from the arbiter's view entirely
        when the lane centre was reported at 180 px, and the arbiter passed the
        planner's throttle straight through in NOMINAL.

        An object counts as in path when ANY part of its box overlaps the corridor,
        not when its centre does: it is far better to consider an out-of-lane object
        than to miss a real lead.
        """
        if ctx.frame_width_px <= 0:
            return True
        anchors, half_px = self._corridor_anchors(ctx)
        if half_px <= 0.0:
            return True
        left = min(obj.box.x1, obj.box.x2)
        right = max(obj.box.x1, obj.box.x2)
        if not (math.isfinite(left) and math.isfinite(right)):
            return True
        for anchor in anchors:
            if right >= anchor - half_px and left <= anchor + half_px:
                return True
        return False

    def _fuse_range(
        self,
        ctx: SafetyContext,
        obj: TrackedObject,
        mitigated: List[str],
        advisory: List[str],
    ) -> Tuple[float, RangeSource, bool, bool]:
        """Combine the pinhole range with an independent channel if one exists.

        Returns ``(distance_m, source, disagreement, provisional)``.  ``provisional``
        marks a fall-back to the pinhole caused only by the second channel being
        momentarily ABSENT: the range filter is still re-anchored on the new level,
        but the arbiter does not call a gap shorter than
        ``range_source_dwell_frames`` a change of provenance.

        Behaviour, in the order the code tests it:

        1. **No usable second channel** -- absent, ``UNAVAILABLE``, non-finite
           distance, non-finite or too-low confidence.  The pinhole range is used and
           the source stays ``PINHOLE``, so a consumer can tell the cross-check did
           not run.
        2. **Agreement** within ``range_disagreement_frac`` -- confidence-weighted
           blend, then the nearer of (blend, pinhole, depth).
        3. **Disagreement, second channel NEARER** -- adopted only after
           ``range_corroboration_frames`` consecutive disagreeing frames.
        4. **Disagreement, second channel FARTHER** -- never adopted.

        Two stabilisers sit on top of that, because on real footage the provenance
        flip-flopped 26 times in 400 frames and every flip used to reset the range
        filter:

        * **Asymmetric dwell.**  ADOPTING the second channel is immediate -- it can
          only lower the range, which is the safe direction.  DROPPING it needs
          ``range_source_dwell_frames`` consecutive unusable frames, unless the
          estimate vanishes altogether, in which case there is nothing left to use.
          A confidence dithering across ``min_range_confidence`` therefore cannot
          toggle the source.
        * **Hysteresis** on the confidence gate and on the agreement threshold, so a
          signal sitting exactly on a threshold does not toggle either.
        """
        lim = self.limits
        tid = obj.track_id
        pinhole = obj.distance_m

        estimate: RangeEstimate | None = None
        if ctx.independent_ranges:
            estimate = ctx.independent_ranges.get(tid)
        if estimate is None:
            estimate = obj.range_estimate

        have = (
            estimate is not None
            and estimate.source is not RangeSource.UNAVAILABLE
            and math.isfinite(estimate.distance_m)
            and estimate.distance_m >= 0.0
        )
        # A NaN confidence used to pass every gate below (`nan < 0.35` is False),
        # NaN-weight the blend and return a NaN fused range, against which every
        # hazard comparison is False -- a real obstacle produced no finding at all.
        if have and not math.isfinite(estimate.confidence):
            mitigated.append("range_channel_confidence_not_finite_track%d" % tid)
            have = False

        using = self._second_channel.get(tid, False)
        pending = self._second_channel_pending.get(tid, 0)
        gap = 0 if have else self._second_channel_gap.get(tid, 0) + 1
        self._second_channel_gap[tid] = gap
        provisional = False
        floor = lim.min_range_confidence - (lim.range_confidence_hysteresis if using else 0.0)
        desired = bool(have and estimate.confidence >= floor)

        if desired and not using:
            using, pending = True, 0
        elif using and not desired:
            if not have:
                # The estimate simply is not there this frame. The pinhole level has
                # to be used -- there is nothing to fuse -- but a gap shorter than
                # the dwell is a gap, not a decision to stop believing the channel.
                if gap < lim.range_source_dwell_frames:
                    provisional = True
                else:
                    using, pending = False, 0
            else:
                pending += 1
                if pending >= lim.range_source_dwell_frames:
                    using, pending = False, 0
        else:
            pending = 0
        self._second_channel[tid] = using
        self._second_channel_pending[tid] = pending

        if have and estimate.confidence < lim.min_range_confidence:
            text = "range_channel_low_confidence_%.2f_track%d" % (estimate.confidence, tid)
            # Reported either way; it only DEGRADES the state when the channel was
            # actually discarded.  Holding a dipping channel under hysteresis is the
            # arbiter's own choice and must not be its own justification.
            (mitigated if not using else advisory).append(text)

        if not using or not have:
            self._disagreement_streak.pop(tid, None)
            self._agreeing.pop(tid, None)
            return pinhole, RangeSource.PINHOLE, False, provisional

        reference = max(1.0, min(pinhole, estimate.distance_m))
        relative = abs(pinhole - estimate.distance_m) / reference
        was_agreeing = self._agreeing.get(tid)
        threshold = lim.range_disagreement_frac
        if was_agreeing is True:
            threshold *= 1.0 + lim.range_disagreement_hysteresis
        elif was_agreeing is False:
            threshold *= 1.0 - lim.range_disagreement_hysteresis
        agreeing = relative <= threshold
        self._agreeing[tid] = agreeing

        if not agreeing:
            streak = self._disagreement_streak.get(tid, 0) + 1
            self._disagreement_streak[tid] = streak
            if estimate.distance_m >= pinhole:
                mitigated.append(
                    "range_channel_disagreement_%.0f%%_farther_track%d"
                    % (relative * 100.0, tid)
                )
                return pinhole, RangeSource.PINHOLE, True, False
            if streak < lim.range_corroboration_frames:
                mitigated.append(
                    "range_channel_disagreement_%.0f%%_uncorroborated_%d/%d_track%d"
                    % (relative * 100.0, streak, lim.range_corroboration_frames, tid)
                )
                return pinhole, RangeSource.PINHOLE, True, False
            mitigated.append(
                "range_channel_disagreement_%.0f%%_track%d" % (relative * 100.0, tid)
            )
            return estimate.distance_m, RangeSource.FUSED, True, False

        self._disagreement_streak.pop(tid, None)
        weight = _clamp(estimate.confidence, 0.0, 1.0)
        fused = (1.0 - weight) * pinhole + weight * estimate.distance_m
        return min(fused, pinhole, estimate.distance_m), RangeSource.FUSED, False, False

    def _rss_gap_m(self, ego_speed_mps: float, lead_speed_mps: float) -> float:
        """RSS-style minimum safe gap for a given pair of speeds, metres."""
        lim = self.limits
        rss = (
            ego_speed_mps * lim.reaction_time_s
            + (ego_speed_mps * ego_speed_mps) / (2.0 * lim.ego_brake_capability_mps2)
            - (lead_speed_mps * lead_speed_mps) / (2.0 * lim.lead_brake_capability_mps2)
        )
        return max(lim.absolute_min_gap_m, rss)

    def _assess_lead(
        self,
        ctx: SafetyContext,
        ego_speed_mps: float,
        dt_s: float,
        mitigated: List[str],
        advisory: List[str],
    ) -> LeadAssessment | None:
        """Select and characterise the most dangerous in-path object.

        Selection is by smallest time-to-collision, falling back to smallest range
        when nothing is closing -- not by ``min(distance)``, which is the rule the
        planner and the old pipeline both used and which misses a distant object
        closing fast.

        Two rate numbers, and the difference between them is the whole safety
        argument of this method:

        ``range_rate_mps``
            The alpha-beta rate.  A track the filter has not seen before is SEEDED at
            ``-ego_speed``, the safe prior that an unknown object is stationary in
            the world, so the backstop is not blind on the frame a hazard appears.
            That number is a statement about ignorance and it is FABRICATED.
        ``measured_rate_mps`` / ``rate_is_measured``
            The least-squares slope of the arbiter's own window of RAW range
            measurements, available only once there are ``aeb_min_rate_samples`` of
            them spanning ``aeb_min_rate_span_s``.  The frame it first becomes
            available it also ANCHORS the alpha-beta rate, replacing the seed with a
            measurement in one step instead of letting the loop walk it back over a
            dozen frames.

        Seeding at zero, as a still older version did, made the backstop report
        ``ttc=inf`` and command no brake on the first frame a stationary car appeared
        15 m ahead.  Seeding at ``-ego_speed`` and then letting the AEB act on it
        did the opposite: it fabricated a closing rate for a lead at a CONSTANT
        32.5 m and braked a 20 m/s ego to a standstill behind it.  Both failures are
        the same mistake -- treating a prior as a measurement -- and the fix is to
        keep the prior and label it, not to change its value.
        """
        lim = self.limits
        alive = [obj.track_id for obj in ctx.tracks]
        self._range_filter.forget_all_but(alive)
        alive_set = set(alive)
        for book in (
            self._disagreement_streak,
            self._range_source,
            self._stable_source,
            self._second_channel,
            self._second_channel_pending,
            self._second_channel_gap,
            self._agreeing,
        ):
            for key in [k for k in book if k not in alive_set]:
                del book[key]

        # Safe prior for a track with no rate history: the object is stationary in
        # the world, so it closes at the ego speed. Derived from the EGO state, not
        # from TrackedObject.velocity_mps, and therefore bounded by the ego speed.
        seed_rate_mps = -max(0.0, ego_speed_mps)

        best: LeadAssessment | None = None
        for obj in ctx.tracks:
            if not math.isfinite(obj.distance_m) or obj.distance_m < 0.0:
                mitigated.append("track%d_range_implausible" % obj.track_id)
                continue
            coast_frames = int(getattr(obj, "time_since_update", 0) or 0)
            if coast_frames > lim.max_coast_frames:
                mitigated.append(
                    "track%d_coast_%dframes_too_stale" % (obj.track_id, coast_frames)
                )
                continue
            if not self._in_path(ctx, obj):
                continue

            distance_m, source, disagreement, provisional = self._fuse_range(
                ctx, obj, mitigated, advisory
            )

            # Switching range channel is a discontinuity in the SIGNAL, not motion
            # of the object.  The filter is RE-ANCHORED, never discarded: discarding
            # it re-applied the -ego_speed seed, and the provenance flip-flopped
            # every frame on real footage, so the fabricated rate was kept alive
            # indefinitely and held a full-authority brake on a flat 7 m range.
            previous_source = self._range_source.get(obj.track_id)
            rebase = previous_source is not None and previous_source is not source
            self._range_source[obj.track_id] = source
            if coast_frames == 0 and not provisional:
                # Only a change between two MEASURED, non-provisional frames is a
                # change of provenance worth reporting.  The second channel declines to
                # estimate a range for a box the tracker is extrapolating, and on
                # real footage every single one of the 26 reported "source switches"
                # was that -- a coast frame with no depth estimate, followed by the
                # depth estimate coming straight back.  The filter is still
                # re-anchored on those frames (the level really did move); it is the
                # FINDING that was noise.
                stable = self._stable_source.get(obj.track_id)
                if stable is not None and stable is not source:
                    # Advisory: the arbiter's own estimator changing its mind about
                    # provenance is not evidence that the automation is degraded.
                    advisory.append(
                        "range_source_switch_%s_to_%s_track%d"
                        % (stable.value, source.value, obj.track_id)
                    )
                self._stable_source[obj.track_id] = source

            filtered_m, rate_mps, reinitialised = self._range_filter.update(
                obj.track_id,
                distance_m,
                dt_s,
                seed_rate_mps=seed_rate_mps,
                rebase=rebase,
                measured=(coast_frames == 0),
            )
            if reinitialised:
                mitigated.append("range_jump_track%d" % obj.track_id)

            measured_rate = self._range_filter.measured_rate(
                obj.track_id, lim.aeb_min_rate_samples, lim.aeb_min_rate_span_s
            )
            if measured_rate is not None and self._range_filter.is_seeded(obj.track_id):
                rate_mps = self._range_filter.anchor_rate(obj.track_id, measured_rate)
            rate_is_measured = not self._range_filter.is_seeded(obj.track_id)

            gap = filtered_m - lim.standstill_gap_m
            closing = -rate_mps
            if closing <= 1e-3:
                ttc_s = float("inf")
                required = 0.0
            elif gap <= 0.0:
                ttc_s = 0.0
                required = float("inf")
            else:
                ttc_s = gap / closing
                required = (closing * closing) / (2.0 * gap)

            rss = self._rss_gap_m(ego_speed_mps, max(0.0, ego_speed_mps + rate_mps))
            rss_matched = self._rss_gap_m(ego_speed_mps, ego_speed_mps)

            candidate = LeadAssessment(
                track_id=obj.track_id,
                distance_m=filtered_m,
                range_rate_mps=rate_mps,
                ttc_s=ttc_s,
                required_decel_mps2=required,
                rss_min_gap_m=rss,
                source=source,
                rss_matched_gap_m=rss_matched,
                disagreement=disagreement,
                reinitialised=reinitialised,
                coasting=coast_frames > 0,
                coast_frames=coast_frames,
                rate_is_measured=rate_is_measured,
                measured_rate_mps=measured_rate,
                rate_updates=self._range_filter.updates(obj.track_id),
            )
            if best is None or self._more_dangerous(candidate, best):
                best = candidate

        return best

    @staticmethod
    def _more_dangerous(a: LeadAssessment, b: LeadAssessment) -> bool:
        """Order two candidates: smaller TTC first, then smaller range."""
        if a.ttc_s != b.ttc_s:
            return a.ttc_s < b.ttc_s
        return a.distance_m < b.distance_m

    def _classify_hazard(
        self, lead: LeadAssessment | None, hazards: List[str]
    ) -> Tuple[bool, bool, bool, bool]:
        """Return ``(aeb, hazard)`` and append the HAZARD findings that justify them.

        These describe traffic, not system health, so none of them may latch
        DISENGAGE. ``aeb`` forces MIN_RISK_MANEUVER, ``hazard`` forces LIMITED.  The
        third element says an ACUTE emergency test tripped on the safe prior and was
        held back for want of a measurement; :meth:`_required_decel` then raises the
        graded ceiling from ``comfort_decel_mps2`` to ``deferred_aeb_decel_mps2``,
        which is what keeps a genuinely stationary obstacle stoppable across the
        deferral.

        The tests are split by what they depend on, and that split is the fix for
        the phantom AEB:

        **Rate-independent** -- computed from the MEASURED range and the MEASURED ego
        speed and nothing else.  A gap below the absolute minimum, and a gap inside
        ``aeb_headway_frac`` of the SPEED-MATCHED RSS distance (the distance you need
        even from a lead travelling at exactly your speed, purely to have time to
        react to it braking).  No prior can manufacture either, so both keep full
        authority on a track's very first frame.

        **Rate-dependent** -- TTC, the geometric required deceleration, and the RSS
        gap computed from the lead speed the closing rate implies.  These may
        authorise an emergency stop ONLY once ``lead.rate_is_measured``.  While the
        rate is still the ``-ego_speed`` safe prior they are reported and they force
        LIMITED, so the arbiter is cautious, but they cannot reach the actuators at
        full authority.  That is the whole difference between a stationary obstacle
        15 m ahead (which fails the rate-independent tests too within a few frames of
        real measurement, and is braked for at the graded rate meanwhile) and a lead
        holding a constant 32.5 m (which never fails them at all).
        """
        if lead is None:
            return False, False, False, False
        lim = self.limits
        aeb = False
        hazard = False
        deferred_acute = False
        rate_pending = False
        measured = lead.rate_is_measured

        # -- rate-independent emergencies ---------------------------------------
        if lead.distance_m < lim.absolute_min_gap_m:
            hazards.append("gap_%.2fm_below_absolute_min" % lead.distance_m)
            aeb = True
        if lead.distance_m < lim.aeb_headway_frac * lead.rss_matched_gap_m:
            hazards.append(
                "headway_%.1fm_below_%.0f%%_of_matched_rss_%.1fm"
                % (lead.distance_m, lim.aeb_headway_frac * 100.0, lead.rss_matched_gap_m)
            )
            aeb = True

        # -- rate-dependent emergencies: MEASURED closure only ------------------
        rate_findings: List[str] = []
        if measured:
            if lead.ttc_s < lim.ttc_brake_s:
                rate_findings.append("ttc_%.2fs_below_brake_threshold" % lead.ttc_s)
            if lead.required_decel_mps2 > lim.aeb_required_decel_mps2:
                rate_findings.append(
                    "required_decel_%.1f_above_%.1f"
                    % (lead.required_decel_mps2, lim.aeb_required_decel_mps2)
                )
            if lead.distance_m < lim.aeb_headway_frac * lead.rss_min_gap_m:
                rate_findings.append(
                    "headway_%.1fm_below_%.0f%%_of_rss_%.1fm"
                    % (lead.distance_m, lim.aeb_headway_frac * 100.0, lead.rss_min_gap_m)
                )
        streak = (self._aeb_rate_streak + 1) if rate_findings else 0
        self._aeb_rate_streak = streak
        if rate_findings:
            hazards.extend(rate_findings)
            if streak >= lim.aeb_rate_corroboration_frames:
                aeb = True
            else:
                # Corroboration gates the STATE, not the pedal. The braking a
                # measured closure demands goes out on this very frame -- braking is
                # the safe direction and a frame is 0.4 m/s -- but a single noisy
                # frame must not latch a minimum-risk manoeuvre, with its throttle
                # lock, its steering hold and its ten-frame recovery.
                hazards.append(
                    "aeb_rate_uncorroborated_%d/%d_track%d"
                    % (streak, lim.aeb_rate_corroboration_frames, lead.track_id)
                )
                hazard = True
                rate_pending = True

        if not aeb:
            if lead.ttc_s < lim.ttc_warn_s:
                hazards.append("ttc_%.2fs_below_warn_threshold" % lead.ttc_s)
                hazard = True
            if lead.distance_m < lead.rss_min_gap_m:
                hazards.append(
                    "headway_%.1fm_below_rss_%.1fm" % (lead.distance_m, lead.rss_min_gap_m)
                )
                hazard = True
            if not measured and (
                lead.ttc_s < lim.ttc_brake_s
                or lead.required_decel_mps2 > lim.aeb_required_decel_mps2
            ):
                # Say out loud that an emergency test tripped on a prior and was
                # deliberately held back for want of a measurement.
                hazards.append(
                    "aeb_deferred_unmeasured_rate_%.1fm/s_track%d"
                    % (lead.range_rate_mps, lead.track_id)
                )
                hazard = True
                # ACUTE only: the prior says contact is less than ttc_brake_s away.
                # The other trigger -- a large geometric required deceleration --
                # fires as far out as 65 m at 25 m/s on nothing but the prior, and
                # braking hard THERE is the phantom, not caution.
                deferred_acute = lead.ttc_s < lim.ttc_brake_s
        return aeb, hazard, deferred_acute, rate_pending

    # ------------------------------------------------------------- envelope

    def _self_induced(self, frames: int = 2) -> bool:
        """Was the ARBITER itself driving the longitudinal channel recently?

        The general principle this implements, which several individual defects were
        instances of: **the arbiter must not treat a consequence of its own
        intervention as evidence that the intervention is still needed.**  An
        arbiter that brakes, measures the resulting deceleration or the resulting
        gap between the planner's target speed and the ego speed, calls that a
        violation, and keeps braking because of it, has built a latch out of its own
        actuator.  The last two defects of this shape -- ``steering_rate`` raised
        against the arbiter's own MRM hold, and ``plan_accel`` raised against the
        speed the arbiter's own MRM braking produced -- differed only in the channel.

        ``frames`` is how long the arbiter's own action can still EXPLAIN the
        measurement in question, and it differs by finding.  An instantaneous
        derivative -- achieved deceleration, jerk -- is explained by the last frame
        or two.  A persistent state variable -- the gap between the planner's target
        speed and the measured ego speed -- is opened by the intervention and does
        not close again until the vehicle has been allowed to accelerate for a
        while, so that one uses ``recovery_frames``.  Both are the same rule with
        the right time constant, not a special case for one finding's name.

        Findings that fail this test are still reported in ``violations`` and in the
        logs.  They just may not force a state or reset the recovery streak.
        """
        return self._long_override_age <= frames


    def _check_plan(
        self,
        plan: MotionPlan,
        ego_speed_mps: float,
        ego_ok: bool,
        faults: List[str],
        mitigated: List[str],
        advisory: List[str],
    ) -> None:
        """Envelope checks on the plan itself.

        A NON-FINITE plan field is a health fault: the planner is producing
        garbage. A plan that is merely outside the envelope is mitigated -- the
        arbiter clamps the resulting command and continues.
        """
        lim = self.limits
        if not math.isfinite(plan.target_speed_mps) or plan.target_speed_mps < 0.0:
            faults.append("plan_target_speed_invalid")
        elif plan.target_speed_mps > lim.max_speed_mps:
            mitigated.append(
                "plan_speed_%.1f_above_%.1f" % (plan.target_speed_mps, lim.max_speed_mps)
            )
        if not math.isfinite(plan.steering_angle_deg):
            faults.append("plan_steering_invalid")
        elif abs(math.radians(plan.steering_angle_deg)) > lim.max_steering_angle_rad:
            mitigated.append("plan_steering_%.1fdeg_above_limit" % plan.steering_angle_deg)

        if ego_ok and math.isfinite(plan.target_speed_mps):
            requested = (plan.target_speed_mps - ego_speed_mps) / max(lim.plan_horizon_s, 1e-3)
            # Only the energy-adding direction is a plan violation. A plan that
            # asks to slow down quickly is never vetoed here; that was the old
            # monitor's most dangerous inversion.
            if requested > lim.max_acceleration_mps2:
                text = "plan_accel_%.2f_above_%.2f" % (requested, lim.max_acceleration_mps2)
                # recovery_frames, not 2: the arbiter opened this speed gap by
                # capping the throttle, and the gap survives for as long as the
                # vehicle has not been allowed to close it. With a 2-frame window
                # the state oscillated NOMINAL/LIMITED every eleventh frame and the
                # ego crawled back up to the plan target at a tenth of its
                # acceleration authority.
                if self._self_induced(lim.recovery_frames):
                    # The arbiter is already holding the longitudinal channel: the
                    # plan's requested acceleration is not being actuated, and the
                    # gap between the plan target and the measured speed is the
                    # arbiter's OWN braking. Counting it forced LIMITED, which reset
                    # the recovery streak, which is how a minimum-risk manoeuvre
                    # became self-sustaining and stopped the vehicle on an open road.
                    advisory.append(text + "_arbiter_induced")
                else:
                    mitigated.append(text)

    def _check_kinematics(
        self,
        ego_speed_mps: float,
        ego_ok: bool,
        dt_s: float,
        hazard_active: bool,
        mitigated: List[str],
        advisory: List[str],
    ) -> None:
        """Judge the ACHIEVED longitudinal acceleration and jerk.

        These are MEASUREMENTS of the vehicle's response, reported and used to
        degrade the state, never health faults: the arbiter cannot un-apply an
        acceleration that already happened, and a wheel-speed trace that clips the
        jerk limit twenty times in a 300-frame run is not grounds for a terminal
        disengagement.
        """
        if not ego_ok:
            return
        lim = self.limits
        if self._prev_speed_mps is None:
            self._prev_accel_mps2 = None
            return
        if max(ego_speed_mps, self._prev_speed_mps) < lim.kinematics_min_speed_mps:
            # See ArbiterLimits.kinematics_min_speed_mps.
            self._prev_accel_mps2 = None
            return
        # The arbiter can only ever REMOVE energy, so an excess ACCELERATION is
        # never its own doing and stays a real finding. An excess DECELERATION or a
        # jerk spike very often is: they are the vehicle answering the arbiter's own
        # brake command.
        induced = self._self_induced()
        accel = (ego_speed_mps - self._prev_speed_mps) / dt_s
        if accel > lim.max_acceleration_mps2 + 1e-6:
            mitigated.append("measured_accel_%.2f_above_%.2f" % (accel, lim.max_acceleration_mps2))
        if -accel > lim.max_deceleration_mps2 + 1e-6 and not hazard_active:
            text = "measured_decel_%.2f_above_%.2f" % (-accel, lim.max_deceleration_mps2)
            if induced:
                advisory.append(text + "_arbiter_commanded")
            else:
                mitigated.append(text + "_unjustified")
        if self._prev_accel_mps2 is not None:
            jerk = abs(accel - self._prev_accel_mps2) / dt_s
            limit = (
                lim.max_jerk_emergency_mps3
                if (hazard_active or induced)
                else lim.max_jerk_mps3
            )
            if jerk > limit + 1e-6:
                text = "jerk_%.1f_above_%.1f" % (jerk, limit)
                (advisory if induced else mitigated).append(text)
        self._prev_accel_mps2 = accel

    def _check_lane_departure(self, ctx: SafetyContext, mitigated: List[str]) -> None:
        """Enforce ``max_lateral_offset_m`` when a metric lane offset is available.

        Sets :attr:`lane_offset_available` so the caller and the logs can tell the
        difference between "inside the lane" and "we could not tell". This is the
        only honest way to expose a limit whose input requires camera calibration
        the vehicle may not have.
        """
        offset = ctx.lateral_offset_m
        if offset is None or not math.isfinite(offset):
            self.lane_offset_available = False
            return
        self.lane_offset_available = True
        if abs(offset) > self.limits.max_lateral_offset_m:
            mitigated.append(
                "lane_departure_%.2fm_above_%.2fm" % (abs(offset), self.limits.max_lateral_offset_m)
            )

    def _steering_ceiling_rad(self, ego_speed_mps: float) -> float:
        """Largest road-wheel angle whose steady-state lateral accel is in budget."""
        lim = self.limits
        ceiling = lim.max_steering_angle_rad
        if ego_speed_mps > 1.0:
            ceiling = min(
                ceiling,
                math.atan(
                    lim.max_lateral_accel_mps2 * lim.wheelbase_m / (ego_speed_mps * ego_speed_mps)
                ),
            )
        return ceiling

    def _limit_steering(
        self,
        steering: float,
        ego_speed_mps: float,
        dt_s: float,
        mitigated: List[str],
        advisory: List[str],
    ) -> float:
        """Apply the lateral-acceleration cap and the road-wheel slew-rate limit.

        Both are enforced on the PHYSICAL angle, ``steering * max_road_wheel_rad``,
        not on the normalised number. This is the check that the previous monitor
        computed into a local variable and threw away.

        The ``steering_rate`` finding is raised against the PREVIOUS REQUEST, not
        against the arbiter's own previous output.  Comparing to the output made the
        arbiter blame the planner for the arbiter's own hold: in a minimum-risk
        manoeuvre it replaced the steering with a held angle and then reported
        ``steering_rate_1.12_above_0.50`` every frame, a self-sustaining fault that
        fed the DISENGAGE counter.  The output is still slew-limited against the
        arbiter's own last command -- that is what the actuator sees -- but the
        limit is recorded as shaping, not as a finding.
        """
        lim = self.limits
        requested_rad = _clamp(steering, -1.0, 1.0) * lim.max_road_wheel_rad
        # While the arbiter is HOLDING the steering itself (MRM/DISENGAGE) the
        # planner's request is not actuated at all, so a finding about it is a
        # report, not a reason to stay in the manoeuvre. Same principle as
        # ``_self_induced``, applied to the lateral channel.
        sink = advisory if self._lat_override_active else mitigated

        max_step = lim.max_steering_rate_rad_s * dt_s
        request_delta = requested_rad - self._prev_request_rad
        if abs(request_delta) > max_step + 1e-9:
            sink.append(
                "steering_rate_%.2f_above_%.2f"
                % (abs(request_delta) / dt_s, lim.max_steering_rate_rad_s)
            )
        self._prev_request_rad = requested_rad

        angle_rad = requested_rad
        ceiling = self._steering_ceiling_rad(ego_speed_mps)
        if abs(angle_rad) > ceiling + 1e-9:
            lateral = (ego_speed_mps ** 2) * math.tan(abs(angle_rad)) / lim.wheelbase_m
            sink.append(
                "lateral_accel_%.1f_above_%.1f" % (lateral, lim.max_lateral_accel_mps2)
            )
            angle_rad = _clamp(angle_rad, -ceiling, ceiling)

        angle_rad = self._slew(angle_rad, dt_s)
        return _clamp(angle_rad / lim.max_road_wheel_rad, -1.0, 1.0)

    def _slew(self, target_rad: float, dt_s: float) -> float:
        """Rate-limit a road-wheel angle against the arbiter's own last output."""
        max_step = self.limits.max_steering_rate_rad_s * dt_s
        delta = target_rad - self._prev_steering_rad
        if abs(delta) > max_step + 1e-9:
            self.last_shaping.append("steering_slew_limited")
            return self._prev_steering_rad + math.copysign(max_step, delta)
        return target_rad

    # -------------------------------------------------------------- decision

    def _decide_state(
        self,
        faults: List[str],
        mitigated: List[str],
        hazards: List[str],
        cmd_ok: bool,
        ego_ok: bool,
        aeb: bool,
    ) -> SafetyState:
        """Map this frame's findings onto a requested state, before latching.

        Only HEALTH FAULTS accumulate toward DISENGAGE.  A hazard is a traffic
        situation the arbiter is designed to handle, and a MITIGATED finding is a
        clamp that worked -- neither is evidence that the automation can no longer
        drive.  Counting mitigated clamps was how the arbiter disengaged itself on
        an empty road at frame 47 of a 120-frame run: an unpaced loop produced a
        2 ms frame, that was reported as ``timing_dt_out_of_range``, and 40 of them
        in a row latched the terminal state.
        """
        lim = self.limits
        requested = SafetyState.NOMINAL
        if faults or mitigated or hazards:
            requested = SafetyState.LIMITED
        if self._dropout_streak >= lim.limited_after_dropouts:
            requested = _worse(requested, SafetyState.LIMITED)
        if self._dropout_streak >= lim.mrm_after_dropouts:
            requested = _worse(requested, SafetyState.MIN_RISK_MANEUVER)
        if aeb:
            requested = _worse(requested, SafetyState.MIN_RISK_MANEUVER)
        if not cmd_ok:
            requested = _worse(requested, SafetyState.MIN_RISK_MANEUVER)
        if not ego_ok:
            requested = _worse(requested, SafetyState.MIN_RISK_MANEUVER)

        if requested is SafetyState.NOMINAL:
            self._clean_streak += 1
        else:
            self._clean_streak = 0

        if faults:
            self._health_fault_streak += 1
            if self._health_fault_streak >= lim.disengage_after_frames:
                requested = SafetyState.DISENGAGE
        else:
            self._health_fault_streak = 0
        return requested

    @property
    def fault_streak(self) -> int:
        """Consecutive frames carrying an unrecovered HEALTH fault.

        This, and only this, is what ``disengage_after_frames`` counts.
        """
        return self._health_fault_streak

    def _latch(self, requested: SafetyState) -> SafetyState:
        """Escalate immediately; de-escalate only after a clean run."""
        if _SEVERITY[requested] >= _SEVERITY[self._latched_state]:
            self._latched_state = requested
        elif self._latched_state is not SafetyState.DISENGAGE:
            if self._clean_streak >= self.limits.recovery_frames:
                self._latched_state = requested
        return self._latched_state

    # --------------------------------------------------------------- output

    def _required_decel(
        self,
        state: SafetyState,
        aeb: bool,
        hazard: bool,
        lead: LeadAssessment | None,
        deferred_acute: bool = False,
        rate_pending: bool = False,
    ) -> float:
        """Deceleration the arbiter itself demands this frame, m/s^2 (>= 0)."""
        lim = self.limits
        required = 0.0
        if aeb or rate_pending:
            geometric = lead.required_decel_mps2 if lead is not None else float("inf")
            if not math.isfinite(geometric):
                geometric = lim.max_deceleration_mps2
            required = max(lim.aeb_min_decel_mps2, geometric * lim.aeb_decel_margin)
            required = min(required, lim.max_deceleration_mps2)
        if state is SafetyState.MIN_RISK_MANEUVER or state is SafetyState.DISENGAGE:
            required = max(required, lim.mrm_decel_mps2)
        elif hazard:
            geometric = lead.required_decel_mps2 if lead is not None else 0.0
            if not math.isfinite(geometric):
                geometric = lim.comfort_decel_mps2
            # The geometric requirement is near zero while the lead keeps pace, so
            # scale the response by how far inside the RSS minimum we are as well.
            deficit = 0.0
            if lead is not None and lead.rss_min_gap_m > 0.0:
                deficit = _clamp(
                    (lead.rss_min_gap_m - lead.distance_m) / lead.rss_min_gap_m, 0.0, 1.0
                )
            demanded = max(geometric, lim.comfort_decel_mps2 * deficit)
            ceiling = (
                lim.deferred_aeb_decel_mps2 if deferred_acute else lim.comfort_decel_mps2
            )
            required = max(required, min(demanded, ceiling))
        elif state is SafetyState.LIMITED and self._dropout_streak >= lim.limited_after_dropouts:
            required = max(required, lim.comfort_decel_mps2)
        return required

    def _synthesise_command(
        self,
        cmd_in: ControlCommand,
        steering: float,
        state: SafetyState,
        aeb: bool,
        hazard: bool,
        lead: LeadAssessment | None,
        dt_s: float,
        cmd_ok: bool,
        ego_speed_mps: float,
        deferred_acute: bool = False,
        rate_pending: bool = False,
    ) -> ControlCommand:
        """Build the command that actually reaches the actuators.

        Invariant, and this one is enforced rather than asserted: the output is
        never MORE energetic than the input.  ``throttle_out <= throttle_in``
        always, and ``brake_out >= brake_in`` always -- there is no path in this
        method that lowers a brake below what was handed in.  The arbiter used to
        re-apply its own 5.0/s brake apply-rate limit here whenever the AEB test had
        not tripped, which cut a handed-in ``brake = 1.0`` to 0.250, 0.500, 0.750,
        1.000 over four frames while the docstring claimed the opposite.  Brake
        jerk shaping, with its emergency exemption, belongs to the controller.

        Steering in MRM/DISENGAGE is HELD, not zeroed: see
        :meth:`_mrm_steering`.
        """
        lim = self.limits
        throttle = cmd_in.throttle
        brake = cmd_in.brake

        if state is not SafetyState.NOMINAL:
            throttle = min(throttle, lim.limited_throttle_cap)

        required = self._required_decel(state, aeb, hazard, lead, deferred_acute, rate_pending)
        if required > 0.0:
            throttle = 0.0
            brake = max(brake, _clamp(required / lim.brake_authority_mps2, 0.0, 1.0))

        if not cmd_ok:
            # A corrupt command tells us nothing about intent; fall back entirely
            # to the arbiter's own controlled stop.
            throttle = 0.0
            brake = max(brake, _clamp(lim.mrm_decel_mps2 / lim.brake_authority_mps2, 0.0, 1.0))

        if _SEVERITY[state] >= _SEVERITY[SafetyState.MIN_RISK_MANEUVER]:
            steering = self._mrm_steering(ego_speed_mps, dt_s)

        # Output shaping. Recorded but not escalating. Only two shapers survive, and
        # NEITHER can reduce the brake below the input:
        #   * the throttle apply-rate limit -- that direction removes energy;
        #   * the brake RELEASE-rate floor -- that direction only holds the brake on
        #     for longer.
        # There is deliberately no brake apply-rate limit here.
        throttle_cap = self._prev_throttle + lim.throttle_rate_per_s * dt_s
        if throttle > throttle_cap:
            self.last_shaping.append("throttle_rate_limited")
            throttle = throttle_cap
        brake_floor = self._prev_brake - lim.brake_release_rate_per_s * dt_s
        if brake < brake_floor:
            self.last_shaping.append("brake_release_limited")
            brake = brake_floor

        throttle = _clamp(throttle, 0.0, 1.0)
        brake = _clamp(max(brake, _clamp(cmd_in.brake, 0.0, 1.0)), 0.0, 1.0)
        if brake > 0.0:
            throttle = 0.0
        return ControlCommand(
            throttle=throttle, brake=brake, steering=_clamp(steering, -1.0, 1.0)
        )

    def _mrm_steering(self, ego_speed_mps: float, dt_s: float) -> float:
        """Steering for a minimum-risk manoeuvre: HOLD the path, do not straighten.

        An MRM brakes to a stop on the CURRENT path.  Replacing the steering with a
        latch that was only ever written in NOMINAL meant that any MRM entered in a
        bend -- and a bend is where the lateral clamps are active, so NOMINAL had
        not been reached for many frames -- commanded ``steering = 0.0`` while
        braking at 3.5 m/s^2.  It steered straight into the corner and then reported
        ``lateral_accel_8.4_above_4.5`` and ``steering_rate_1.12_above_0.50``
        against its own output.

        What it does instead: hold the last ACTUATED angle (recorded every frame,
        in every state), re-check it against the lateral-acceleration ceiling at the
        CURRENT speed -- the ceiling relaxes as the vehicle slows, so the hold is
        never clipped harder than the frame it was commanded on -- and slew toward
        straight only once the vehicle is essentially stopped, below
        ``mrm_straighten_speed_mps``, where holding lock serves no purpose.
        """
        lim = self.limits
        held_rad = self._last_commanded_steering * lim.max_road_wheel_rad
        if ego_speed_mps < lim.mrm_straighten_speed_mps:
            held_rad = 0.0
        else:
            ceiling = self._steering_ceiling_rad(ego_speed_mps)
            held_rad = _clamp(held_rad, -ceiling, ceiling)
        return _clamp(self._slew(held_rad, dt_s) / lim.max_road_wheel_rad, -1.0, 1.0)

    def _describe(
        self,
        state: SafetyState,
        violations: List[str],
        lead: LeadAssessment | None,
        aeb: bool,
    ) -> str:
        """One-line human-readable summary; also what goes into the logs."""
        parts = [state.value]
        if aeb:
            parts.append("AEB")
        if lead is not None:
            parts.append(
                "lead#%d d=%.1fm rate=%+.1fm/s%s ttc=%s a_req=%.1f src=%s"
                % (
                    lead.track_id,
                    lead.distance_m,
                    lead.range_rate_mps,
                    "" if lead.rate_is_measured else "(seeded)",
                    "inf" if math.isinf(lead.ttc_s) else "%.2fs" % lead.ttc_s,
                    lead.required_decel_mps2 if math.isfinite(lead.required_decel_mps2) else -1.0,
                    lead.source.value,
                )
            )
            if lead.coasting:
                parts.append("coasted=%d" % lead.coast_frames)
        else:
            parts.append("no_in_path_lead")
        if violations:
            parts.append("violations=" + ",".join(violations[:6]))
        if not self.lane_offset_available:
            parts.append("lane_offset_unavailable")
        if self.last_shaping:
            parts.append("shaped=" + ",".join(sorted(set(self.last_shaping))))
        return " | ".join(parts)
