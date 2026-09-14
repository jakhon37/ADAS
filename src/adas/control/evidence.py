"""Measurement evidence for the longitudinal decision path.

This module is the numerical foundation of the *evidence-gated authority*
design: **every quantity a longitudinal decision rests on carries an explicit
uncertainty, and the authority the system may exercise is a function of the
evidence supporting it.**

Three primitives, and nothing policy-shaped:

:class:`RangeEvidence`
    A per-track window of RAW range measurements and everything that can
    honestly be derived from it -- a closing rate, the standard error of that
    rate, a lower confidence bound on it, and a lead acceleration with its own,
    separately-estimated standard error.
:class:`EvidenceBook`
    A collection of those, keyed by track id, plus the ONE thing that is a
    property of the sensor rather than of a track: the measurement-noise
    estimate.
:func:`required_decel_mps2` and :class:`JerkShaper`
    The kinematics of stopping, and the ceiling on how fast a demand may be
    built.

What is deliberately NOT here is any decision about when to brake.  Two
consumers -- :class:`adas.planning.longitudinal.LongitudinalPlanner` and
:class:`adas.control.arbiter.SafetyArbiter` -- each own an instance, feed it
from their OWN lead selection, and apply their OWN authority policy to the
result.  They share the arithmetic of *measuring*; they do not share a belief
about the world, and they do not share state.

Four estimator decisions here are load-bearing, and each of them is the fix for
a measured defect in this codebase's history:

1. **A rate is never seeded, only measured.**  A camera measures range; a
   closing rate is a difference of ranges over time.  Until
   ``min_samples`` distinct captures exist there is no rate, and the only prior
   available -- "assume the object is stationary in the world", i.e.
   ``rate = -v_ego`` -- is exactly what produced the constant-range phantom
   brake that took a 20 m/s ego to a standstill behind a car holding 32.5 m.
   :attr:`ClosureEstimate.measured` is False until the evidence exists, and
   the consumers hold themselves to a headway-grade response while it is.

2. **Measurement noise is estimated from SECOND DIFFERENCES, not from fit
   residuals.**  The residuals of a straight line fitted to a range that is
   genuinely curving -- which is what a braking lead produces -- are dominated
   by the curvature, not by the noise.  An estimator that took its noise from
   them concludes that a lead braking at 6 m/s^2 is a noisy lead holding
   station, refuses it the lead-deceleration credit, and drives into it.  The
   second difference ``r[i] - 2 r[i-1] + r[i-2]`` annihilates any straight line
   exactly and leaves a constant acceleration with only ``a dt^2`` (0.015 m at
   6 m/s^2 on a 50 ms grid), so it measures the sensor and not the manoeuvre.

3. **Lead acceleration comes from split, NON-OVERLAPPING half windows of the
   raw range.**  Differentiating an already-smoothed speed produces errors that
   are strongly correlated between frames, because consecutive estimates share
   most of their samples; a residual-based standard error then reports almost no
   uncertainty for a trend that is entirely noise, and the consumer confidently
   concludes that a lead holding a steady speed is braking at 4 m/s^2.  Two
   half-windows with no samples in common have independent slope errors, so
   ``sqrt(se_a^2 + se_b^2) / span`` is an honest standard error.

4. **A range discontinuity is a re-anchor, not motion.**  A reported range that
   moves further in one frame than any pair of road vehicles could move drops
   the history and returns the rate to unknown.  Differentiating such a step is
   how a 10 m range correction becomes 200 m/s of closure.

Units: metres, seconds, m/s, m/s^2, m/s^3.  **Closing rates in this module are
POSITIVE WHEN CLOSING**, matching ``TrackedObject.velocity_mps`` and the
harness's own convention, and unlike the planner's internal
``v_rel = v_lead - v_ego``.  Every function says which it means.

Nothing here does I/O, holds a lock, or touches numpy; it is pure Python and
safe to unit-test at any rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    "ClosureEstimate",
    "EvidenceBook",
    "EvidenceLimits",
    "JerkShaper",
    "RangeEvidence",
    "least_squares_fit",
    "quadratic_curvature",
    "required_decel_mps2",
    "stopping_distance_m",
    "sub_emergency_guard",
]


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``; ``low`` wins if the interval is empty."""
    if value < low:
        return low
    if value > high:
        return high
    return value


# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #


@dataclass
class EvidenceLimits:
    """How much evidence a decision needs, and how it is gathered.

    These are the knobs a deployment may retune.  Both consumers construct their
    own instance, so the planner and the arbiter can demand different amounts of
    evidence for the same measurement without either one's setting reaching the
    other.
    """

    window_samples: int = 9
    """Raw range measurements the slope is fitted over.  0.45 s at 20 Hz.

    The standard error of a slope over ``n`` samples spaced ``dt`` is
    ``sigma / sqrt(dt^2 n (n^2-1) / 12)``, so widening from 5 to 9 samples cuts
    the noise on the rate by 2.4x at the cost of about 0.2 s of lag against a
    lead whose deceleration is changing.  Every scenario in the acceptance
    corpus can afford that lag; none of them can afford a 6 m/s^2 brake for a
    range estimate that wandered.
    """

    min_samples: int = 3
    """Fewest DISTINCT captures that produce a rate at all.

    Three, which is the smallest number from which a curvature -- and therefore
    a lead acceleration -- can be computed at all, and one more than the two a
    slope needs.  It is not a statistical choice and it is not where the safety
    comes from: the confidence bound is, and the bound is computed from a noise
    estimate that is POOLED OVER THE WHOLE RUN rather than from three residuals.
    Demanding four samples instead bought no extra rejection (the pooled sigma is
    the same either way) and cost a frame of latency on the tightest cells in the
    envelope, where a lead braking at 6 m/s^2 leaves three frames between the
    first measurable curvature and the last moment full braking still works.

    The physical floor underneath it is unchanged and is not negotiable: at 55 ms
    of sense latency on a 50 ms grid, decision frames 0 and 1 read the SAME
    capture, so a third distinct capture cannot exist before decision frame 3.
    """

    min_span_s: float = 0.10
    """Elapsed capture time the window must cover before a rate is believed."""

    closure_sigmas: float = 4.0
    """Standard errors of separation from zero before a closure may authorise
    collision-avoidance braking.

    THE GATE IS ON THE BOUND; THE MAGNITUDE IS THE ESTIMATE.  Four rather than
    three because the question is not "is this frame a false alarm?" but "does
    this system ever brake for nothing across a whole operating envelope?".
    Fifty scenarios of three hundred frames is fifteen thousand opportunities,
    and a one-sided three-sigma gate (1.3e-3) would be expected to open about
    twenty times; four sigma (3.2e-5) about once in two corpora.  The cost is
    nothing on a clean sensor -- the residuals are zero and the bound IS the
    estimate -- and about one frame of latency on a noisy one, because a genuine
    20 m/s closure is more than seven standard errors even on the shortest
    window this estimator will fit.
    """

    accel_sigmas: float = 5.0
    """Standard errors demanded of the LEAD ACCELERATION credit.

    Higher than :attr:`closure_sigmas`, and not for statistical reasons -- the
    false-alarm rate a given sigma count buys is the same for both -- but
    because the CONSEQUENCE is not.  A false closure enters the braking law
    squared and small: at a 20 m gap a 3-sigma spurious closure produces about
    0.15 m/s^2.  A false lead deceleration switches the law to its
    lead-is-stopping branch, where the demand scales with the EGO's speed
    squared and not with the error, and manufactures a full emergency out of
    range noise.  At +/-0.30 m of range noise the split-half estimate has a
    standard error near 15 m/s^2, so five of them is 75 m/s^2 and no draw
    survives it.
    """

    jump_m: float = 3.0
    """Range discontinuity treated as a re-anchor rather than as motion.

    At 20 Hz, 3.0 m implies 60 m/s of closure between two road vehicles, which
    no pair produces, and it is ten standard deviations of the worst range noise
    this system is specified against.
    """

    early_noise_inflation: float = 2.0
    """How much a YOUNG noise estimate is inflated by, in units of 1/m.

    The pooled noise estimate is used as ``sigma * (1 + this / m)`` where ``m`` is
    the number of differences folded in, so one difference is trusted at a third
    of its face value and thirty at 94% of it.  A noise-free channel reports
    exactly zero however young the estimate is, so this costs nothing where there
    is nothing to protect against; what it protects against is the one-in-four
    chance that the first draw from a genuinely noisy channel happens to be small
    and opens the acceleration gate on a bound that has not earned its confidence.
    """

    sigma_ewma_alpha: float = 0.1
    """Forgetting factor of the running measurement-noise estimate.

    The noise on a range measurement is a property of the SENSOR, not of the
    last four samples, so it is estimated once and remembered.  This is not a
    refinement: with a four-sample fit the residuals carry two degrees of
    freedom, a chi-squared with two degrees of freedom has plenty of mass near
    zero, and the local estimate therefore collapses to almost nothing several
    times in a three-hundred-frame run.  Every standard error computed from it
    collapses with it, the lower confidence bound stops bounding anything, and
    the system brakes hard for a lead holding a steady speed -- the original
    defect, reconstructed inside the fix for it.  Alpha 0.1 settles in about
    thirty frames.
    """

    max_lead_decel_mps2: float = 8.0
    """Largest lead deceleration the credit will believe, m/s^2."""

    def __post_init__(self) -> None:
        if self.min_samples < 3:
            raise ValueError("min_samples must be >= 3 to have a residual at all")
        if self.window_samples < 2 * self.min_samples:
            raise ValueError(
                "window_samples (%d) must be at least 2 x min_samples (%d) so the "
                "lead-acceleration estimate can use two half windows with no "
                "samples in common" % (self.window_samples, self.min_samples)
            )
        if self.closure_sigmas <= 0.0 or self.accel_sigmas <= 0.0:
            raise ValueError("confidence bounds must be positive")
        if not 0.0 < self.sigma_ewma_alpha <= 1.0:
            raise ValueError("sigma_ewma_alpha must be in (0, 1]")
        if self.jump_m <= 0.0:
            raise ValueError("jump_m must be positive")


# --------------------------------------------------------------------------- #
# The estimate
# --------------------------------------------------------------------------- #


@dataclass
class ClosureEstimate:
    """What the range history supports, and how strongly.

    Attributes:
        closing_mps: Unbiased least-squares closing rate, POSITIVE WHEN CLOSING.
            0.0 when there is no rate yet.
        closing_lcb_mps: ``closing_mps`` less
            :attr:`EvidenceLimits.closure_sigmas` standard errors.  This is what
            a decision to brake is gated on, because braking is irreversible and
            a closure indistinguishable from noise is not a reason to decelerate.
        stderr_mps: Standard error of the fitted slope.
        lead_accel_mps2: Confident lead acceleration, <= 0, already reduced by
            :attr:`EvidenceLimits.accel_sigmas` standard errors toward zero.
            0.0 means "no credible deceleration", never "the lead is not
            braking".
        samples: Distinct captures in the window.
        span_s: Elapsed capture time the window covers.
        measured: True once a rate exists at all.  **The single most important
            field in this module**: False is a statement about the system's
            ignorance, not about the world, and no full-authority action may
            rest on a frame where it is False.
        reanchored: A range discontinuity dropped the history this frame.
    """

    closing_mps: float = 0.0
    closing_lcb_mps: float = 0.0
    stderr_mps: float = 0.0
    lead_accel_mps2: float = 0.0
    samples: int = 0
    span_s: float = 0.0
    measured: bool = False
    reanchored: bool = False

    @property
    def confident_closing(self) -> bool:
        """True when the closure is separated from zero by the required bound."""
        return self.measured and self.closing_lcb_mps > 0.05


def least_squares_fit(
    samples: List[Tuple[float, float]]
) -> Optional[Tuple[float, float, Optional[float]]]:
    """``(slope, Sxx, residual sigma)`` of ``y`` against ``t``, or None.

    ``Sxx`` is the spread of the abscissae, so any noise estimate becomes a
    standard error with ``sigma / sqrt(Sxx)``.  The residual sigma is this fit's
    own opinion of the measurement noise and is None when there are too few
    degrees of freedom for one.

    Args:
        samples: ``(t_s, y)`` pairs, at least two, with distinct abscissae.

    Returns:
        The triple, or None when the fit is not defined.
    """
    n = len(samples)
    if n < 2:
        return None
    mean_t = sum(s[0] for s in samples) / n
    mean_y = sum(s[1] for s in samples) / n
    sxx = sum((s[0] - mean_t) ** 2 for s in samples)
    if sxx <= 1e-12:
        return None
    slope = sum((s[0] - mean_t) * (s[1] - mean_y) for s in samples) / sxx
    if not math.isfinite(slope):
        return None
    if n <= 2:
        return slope, sxx, None
    intercept = mean_y - slope * mean_t
    resid = sum((s[1] - (intercept + slope * s[0])) ** 2 for s in samples)
    return slope, sxx, math.sqrt(max(0.0, resid / (n - 2)))



def quadratic_curvature(
    samples: List[Tuple[float, float]]
) -> Optional[Tuple[float, float, Optional[float]]]:
    """``(c, var_factor, residual sigma)`` for ``y = a + b t + c t^2``.

    The curvature coefficient ``c`` is half the second derivative, so a range
    history fitted this way yields the RELATIVE acceleration as ``2c`` directly.

    ``var_factor`` is the ``(3,3)`` element of ``(X'X)^-1``, so a caller with any
    noise estimate ``sigma`` gets the standard error of ``c`` as
    ``sigma * sqrt(var_factor)`` -- which is the whole point of returning it
    separately.  The estimator MUST NOT take its noise from its own residuals: a
    range that is genuinely curving is exactly the case this is used on, and
    residuals taken from a three-point exact fit are identically zero.  The
    residual sigma is returned only so that a caller can use it when it is LARGER
    than its own pooled estimate, i.e. when this window has seen something the
    pooled estimate has not.  It is None below four samples, where the fit is
    exact and has no residual.

    At exactly three evenly spaced samples this reduces to the second difference
    ``(y0 - 2 y1 + y2) / dt^2`` with ``sqrt(var_factor) = sqrt(1.5) / dt^2``,
    giving the textbook ``sqrt(6) sigma / dt^2`` on the acceleration.

    Args:
        samples: ``(t_s, y)`` pairs; at least three, with at least three distinct
            abscissae.

    Returns:
        The triple, or None when the fit is not defined.
    """
    n = len(samples)
    if n < 3:
        return None
    # Centre the abscissae: it costs nothing and keeps the normal equations well
    # conditioned over a long run, where t is a clock that has been accumulating
    # for hours.
    mean_t = sum(s[0] for s in samples) / n
    ts = [s[0] - mean_t for s in samples]
    ys = [s[1] for s in samples]
    s1 = sum(ts)
    s2 = sum(t * t for t in ts)
    s3 = sum(t ** 3 for t in ts)
    s4 = sum(t ** 4 for t in ts)
    if s2 <= 1e-18 or s4 <= 1e-24:
        return None
    # M = [[n, s1, s2], [s1, s2, s3], [s2, s3, s4]]
    det = (
        n * (s2 * s4 - s3 * s3)
        - s1 * (s1 * s4 - s3 * s2)
        + s2 * (s1 * s3 - s2 * s2)
    )
    if abs(det) <= 1e-30:
        return None
    b0 = sum(ys)
    b1 = sum(t * y for t, y in zip(ts, ys))
    b2 = sum(t * t * y for t, y in zip(ts, ys))
    m = [[float(n), s1, s2], [s1, s2, s3], [s2, s3, s4]]
    rhs = [b0, b1, b2]
    sol = _solve3(m, rhs)
    if sol is None:
        return None
    a, b, c = sol
    var_factor = (n * s2 - s1 * s1) / det
    if not (math.isfinite(c) and math.isfinite(var_factor)) or var_factor < 0.0:
        return None
    sigma_local: Optional[float] = None
    if n > 3:
        resid = sum((y - (a + b * t + c * t * t)) ** 2 for t, y in zip(ts, ys))
        sigma_local = math.sqrt(max(0.0, resid / (n - 3)))
    return c, var_factor, sigma_local


def _solve3(
    m: List[List[float]], rhs: List[float]
) -> Optional[Tuple[float, float, float]]:
    """Gaussian elimination with partial pivoting on a 3x3 system."""
    a = [row[:] + [rhs[i]] for i, row in enumerate(m)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) <= 1e-30:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        inv = 1.0 / a[col][col]
        for row in range(3):
            if row == col:
                continue
            factor = a[row][col] * inv
            if factor == 0.0:
                continue
            for k in range(col, 4):
                a[row][k] -= factor * a[col][k]
    out = []
    for i in range(3):
        if abs(a[i][i]) <= 1e-30:
            return None
        out.append(a[i][3] / a[i][i])
    return out[0], out[1], out[2]

class RangeEvidence:
    """One track's raw-range window and the estimates it supports.

    Stateful and single-threaded.  The owner calls :meth:`update` once per frame
    with a RAW range measurement and the CAPTURE time it belongs to, then reads
    :meth:`closure`.

    Only measurements are stored.  A tracker-extrapolated ("coasted") range is
    not a measurement and is deliberately excluded: an extrapolation must never
    be able to establish the closing rate that authorises an emergency stop.
    """

    __slots__ = ("_limits", "_book", "_hist", "_last_token", "_reanchored")

    def __init__(self, limits: EvidenceLimits, book: "EvidenceBook") -> None:
        self._limits = limits
        self._book = book
        self._hist: List[Tuple[float, float]] = []
        self._last_token: Optional[Tuple[float, float, int]] = None
        self._reanchored = False

    # ------------------------------------------------------------------ input

    def update(
        self,
        t_s: float,
        range_m: float,
        measured: bool = True,
        capture_token: Optional[int] = None,
    ) -> None:
        """Fold one range report in.

        Args:
            t_s: The CAPTURE time the measurement describes, seconds.  Not the
                decision time: a decision loop running faster than the camera
                reads one image twice, and storing it twice puts two points at
                the same abscissa and flattens the fitted slope toward zero --
                a fabricated "it stopped closing", which is the exact signature
                this system exists to avoid.
            range_m: The raw reported range, metres.
            measured: False for a coasted range.  Ignored entirely.
            capture_token: A monotone counter that changes only when the
                perception stack produced a NEW result for this track
                (``TrackedObject.hits`` is one).  When it is unchanged the
                report is a repeat of a capture already held and is dropped,
                which is the robust form of the ``t_s`` test above for callers
                whose clock is the decision clock rather than the capture clock.
        """
        self._reanchored = False
        if not measured:
            return
        if not (math.isfinite(t_s) and math.isfinite(range_m)):
            return
        token = (float(t_s), float(range_m), int(capture_token or -1))
        if self._last_token is not None and capture_token is not None:
            if token[2] == self._last_token[2]:
                return
        if self._hist and t_s <= self._hist[-1][0] + 1e-9:
            return
        self._last_token = token

        if self._hist:
            prev_t, prev_r = self._hist[-1]
            fit = least_squares_fit(self._hist)
            slope = 0.0 if fit is None else fit[0]
            predicted = prev_r + slope * (t_s - prev_t)
            if abs(range_m - predicted) > self._limits.jump_m:
                # A re-anchor, not motion.  The rate goes back to unknown until
                # fresh captures rebuild it.
                self._hist = []
                self._reanchored = True

        self._hist.append((float(t_s), float(range_m)))
        if len(self._hist) > self._limits.window_samples:
            del self._hist[0 : len(self._hist) - self._limits.window_samples]
        self._book.note_noise(self._hist)

    def forget(self) -> None:
        """Drop the window, so the next report starts a fresh measurement."""
        self._hist = []
        self._last_token = None

    # ----------------------------------------------------------------- output

    @property
    def samples(self) -> int:
        """Distinct captures currently in the window."""
        return len(self._hist)

    @property
    def span_s(self) -> float:
        """Elapsed capture time the window covers, seconds."""
        if len(self._hist) < 2:
            return 0.0
        return self._hist[-1][0] - self._hist[0][0]

    def closure(self, ego_accel_mps2: float = 0.0) -> ClosureEstimate:
        """The closing rate, its uncertainty, and the lead's acceleration.

        Args:
            ego_accel_mps2: The ego's own acceleration from the VEHICLE BUS, not
                from the camera.  ``d(range)/dt = v_lead - v_ego`` so
                ``d2(range)/dt2 = a_lead - a_ego``, and the ego term has to be
                added back to recover the lead's acceleration over the ground.

        Returns:
            A :class:`ClosureEstimate`.  ``measured`` is False, and every other
            field zero, until the window carries enough distinct captures.
        """
        lim = self._limits
        n = len(self._hist)
        if n < lim.min_samples or self.span_s < lim.min_span_s:
            return ClosureEstimate(measured=False, samples=n, span_s=self.span_s,
                                   reanchored=self._reanchored)
        fit = least_squares_fit(self._hist)
        if fit is None:
            return ClosureEstimate(measured=False, samples=n, span_s=self.span_s,
                                   reanchored=self._reanchored)
        slope, sxx, sigma_local = fit
        stderr = self._book.slope_stderr(sxx, sigma_local)
        closing = -slope
        return ClosureEstimate(
            closing_mps=closing,
            closing_lcb_mps=closing - lim.closure_sigmas * stderr,
            stderr_mps=stderr,
            lead_accel_mps2=self._lead_accel(ego_accel_mps2),
            samples=n,
            span_s=self.span_s,
            measured=True,
            reanchored=self._reanchored,
        )

    def _lead_accel(self, ego_accel_mps2: float) -> float:
        """Confident lead acceleration from the curvature of the RAW range.

        See the module docstring, point 3.  Returns a value in
        ``[-max_lead_decel, 0]``: only the deceleration that survives
        :attr:`EvidenceLimits.accel_sigmas` standard errors is credited, and a
        lead that is accelerating is credited with nothing, because the braking
        law must not be softened by an estimate that could be noise.

        ``range'' = a_lead - a_ego``, so the ego's own acceleration -- known from
        the vehicle bus, not from the camera -- is added back.
        """
        lim = self._limits
        if not self._book.noise_estimated:
            # No measurement of the range channel's noise yet, so no honest
            # uncertainty on a curvature, so no credit.  This is the evidence gate
            # applied to the estimator's own error bar rather than to the signal.
            return 0.0
        got = quadratic_curvature(self._hist)
        if got is None:
            return 0.0
        curvature, var_factor, sigma_local = got
        a_rel = 2.0 * curvature
        sigma = max(sigma_local or 0.0, self._book.sigma_range_bound_m)
        stderr = 2.0 * sigma * math.sqrt(max(0.0, var_factor))
        a_lead = a_rel + ego_accel_mps2
        return _clamp(a_lead + lim.accel_sigmas * stderr, -lim.max_lead_decel_mps2, 0.0)


class EvidenceBook:
    """Per-track :class:`RangeEvidence`, plus the sensor's own noise estimate.

    The noise estimate lives here rather than on a track because it is a
    property of the RANGE CHANNEL: every track measured through the same camera
    shares it, and pooling the evidence is what gives it enough degrees of
    freedom to be stable.
    """

    def __init__(self, limits: Optional[EvidenceLimits] = None) -> None:
        self.limits = limits or EvidenceLimits()
        self._tracks: Dict[int, RangeEvidence] = {}
        self._sigma_var: Optional[float] = None
        self._sigma_samples = 0
        self._ego_hist: List[Tuple[float, float]] = []

    # ------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        """Forget every track and the noise estimate."""
        self._tracks.clear()
        self._sigma_var = None
        self._sigma_samples = 0
        self._ego_hist = []

    def track(self, track_id: int) -> RangeEvidence:
        """The evidence for one track, created on first use."""
        got = self._tracks.get(track_id)
        if got is None:
            got = RangeEvidence(self.limits, self)
            self._tracks[track_id] = got
        return got

    def forget_all_but(self, alive: List[int]) -> None:
        """Drop tracks that no longer exist, so an id cannot be reused stale."""
        keep = set(alive)
        for key in [k for k in self._tracks if k not in keep]:
            del self._tracks[key]

    # ------------------------------------------------------------------ noise

    @property
    def noise_estimated(self) -> bool:
        """True once the range channel's noise has been measured at all.

        A quantity with no uncertainty estimate is not evidence.  The
        acceleration credit, whose standard error carries a factor of
        ``1 / dt^2`` and is therefore ruined by any error in the noise figure,
        is withheld entirely while this is False.
        """
        return self._sigma_var is not None

    @property
    def sigma_range_m(self) -> float:
        """Running estimate of the range channel's measurement noise, metres.

        The POINT estimate, used for the closing-rate standard error.  Its
        consumer's mistakes are cheap: a closure enters the braking law squared
        and divided by the gap, so a spurious one at a 20 m gap is worth
        0.15 m/s^2, and the four-sigma bound already carries the risk.  Inflating
        it here would cost real detection latency on exactly the cases where a
        stopped obstacle has to be braked for through a noisy range -- measured:
        one extra frame, and 1.1 m of clearance instead of 2.9 m.
        """
        return 0.0 if self._sigma_var is None else math.sqrt(max(0.0, self._sigma_var))

    @property
    def sigma_range_bound_m(self) -> float:
        """Upper working bound on the range noise, metres.

        The same estimate inflated while it is YOUNG; see
        :attr:`EvidenceLimits.early_noise_inflation`.  Used only for the
        lead-ACCELERATION credit, whose consumer's mistakes are not cheap: a
        false deceleration switches the braking law to its lead-is-stopping
        branch, where the demand scales with the EGO's speed squared and not with
        the error, and manufactures a full emergency out of range noise.  The two
        bounds differ for the same reason the two confidence levels do -- the
        false-alarm rate a given bound buys is the same, and the CONSEQUENCE is
        not.

        A noise-free channel reports exactly zero here too, so this costs nothing
        where there is nothing to protect against.
        """
        if self._sigma_var is None:
            return 0.0
        sigma = math.sqrt(max(0.0, self._sigma_var))
        m = max(1, self._sigma_samples)
        return sigma * (1.0 + self.limits.early_noise_inflation / m)

    def note_noise(self, hist: List[Tuple[float, float]]) -> None:
        """Update the noise estimate from the last four EVENLY SPACED samples.

        THIRD differences.  ``r3 - 3 r2 + 3 r1 - r0`` annihilates any quadratic
        exactly, so a lead holding a constant deceleration contributes NOTHING to
        this estimate and only a change of deceleration leaks in.  That matters
        because the estimate is what bounds the lead-ACCELERATION credit, and an
        acceleration standard error carries a factor of ``1 / dt^2``: with the
        second difference this probe used to take, a lead braking at 6 m/s^2
        contributed 0.015 m, which at a 50 ms period is 6 m/s^2 of imagined
        uncertainty -- the size of the signal itself, so the manoeuvre suppressed
        its own detection and the brake arrived five frames late.

        The variance of a third difference is twenty times the measurement
        variance for independent samples, which is the ``/ 20``.  Unequally spaced
        captures are skipped: across a frame overrun the difference is not a clean
        noise probe.

        The estimate is an EWMA over the whole run because the noise is a property
        of the sensor and one quadruple of samples is one degree of freedom.
        """
        if len(hist) < 4:
            return
        (t0, r0), (t1, r1), (t2, r2), (t3, r3) = hist[-4], hist[-3], hist[-2], hist[-1]
        step = t1 - t0
        if abs((t2 - t1) - step) > 1e-6 or abs((t3 - t2) - step) > 1e-6:
            return
        var = (r3 - 3.0 * r2 + 3.0 * r1 - r0) ** 2 / 20.0
        self._sigma_samples += 1
        if self._sigma_var is None:
            self._sigma_var = var
        else:
            self._sigma_var += self.limits.sigma_ewma_alpha * (var - self._sigma_var)

    def slope_stderr(self, sxx: float, sigma_local: Optional[float]) -> float:
        """Standard error of a slope fitted over abscissae with spread ``sxx``.

        ``sigma_local`` is a single fit's own residual estimate and is used only
        when it is LARGER than the pooled one, so a window that has just seen
        something the pooled estimate has not is never ignored, while a window
        whose residuals happen to be small cannot shrink the bound to nothing.
        """
        sigma = max(sigma_local or 0.0, self.sigma_range_m)
        if sxx <= 1e-12:
            return 0.0
        return sigma / math.sqrt(sxx)

    # -------------------------------------------------------------- ego accel

    def ego_accel_mps2(self, t_s: float, speed_mps: float) -> float:
        """The ego's own acceleration, differentiated from the VEHICLE BUS.

        Not from the camera.  The speed signal is the system's own
        proprioception and is orders of magnitude cleaner than a range, so a
        plain least-squares slope over the same window is enough.
        """
        hist = self._ego_hist
        if not hist or t_s > hist[-1][0] + 1e-12:
            hist.append((float(t_s), float(speed_mps)))
            if len(hist) > self.limits.window_samples:
                del hist[0]
        fit = least_squares_fit(hist)
        return 0.0 if fit is None else fit[0]


# --------------------------------------------------------------------------- #
# The kinematics of stopping
# --------------------------------------------------------------------------- #


def stopping_distance_m(speed_mps: float, decel_mps2: float) -> float:
    """``v^2 / 2a``, metres.  Zero when the vehicle is stopped or ``a <= 0``."""
    if speed_mps <= 0.0 or decel_mps2 <= 0.0:
        return 0.0
    return (speed_mps * speed_mps) / (2.0 * decel_mps2)


def required_decel_mps2(
    range_m: float,
    closing_mps: float,
    ego_speed_mps: float,
    lead_accel_mps2: float,
    target_clearance_m: float,
    current_decel_mps2: float = 0.0,
    max_decel_mps2: float = 8.0,
    jerk_mps3: float = 20.0,
    ramp_allowance_s: float = 0.4,
) -> float:
    """The constant deceleration that keeps ``target_clearance_m``, m/s^2.

    Textbook, from measured quantities only, with two cases because they are
    physically different problems:

    * **The lead is stopping** (``lead_accel_mps2 < -0.5``).  It will still
      travel ``v_lead^2 / 2|a_lead|``, so the room the ego has to stop in is the
      gap plus that, and the ego must come to REST inside it: the demand scales
      with the EGO's speed squared.
    * **The lead is not stopping.**  The ego only has to wash out the RELATIVE
      speed inside the gap.

    The one non-textbook term is the ramp allowance.  While the demand is being
    built at ``jerk_mps3`` the average deceleration is about half the target, so
    a law that ignored the build-up would chase its own lag and settle high.
    The allowance is computed FROM THE RAMP ACTUALLY NEEDED --
    ``(target - current) / jerk`` -- rather than being a fixed fraction of a
    second, because a brake that is already applied needs no allowance at all
    and charging one anyway inflates the demand by a sixth exactly where the
    case is tightest.

    Args:
        range_m: Measured range to the lead, metres.
        closing_mps: Closing rate, POSITIVE WHEN CLOSING.  Pass 0.0 to ask what
            the geometry alone requires.
        ego_speed_mps: Measured ego speed, m/s.
        lead_accel_mps2: Confident lead acceleration, <= 0.
        target_clearance_m: Room a completed stop aims to leave, metres.
        current_decel_mps2: The deceleration already being commanded, m/s^2.
        max_decel_mps2: The vehicle's braking authority, m/s^2.
        jerk_mps3: The rate the demand will actually be built at, m/s^3.
        ramp_allowance_s: Ceiling on the closure conceded to the build-up.

    Returns:
        A deceleration in ``[0, max_decel_mps2]``.
    """
    # The clearance target is capped at the room that still exists.  Once the
    # ego is inside the standstill clearance no deceleration can restore it, and
    # demanding it anyway turns the law into a step to full authority in the last
    # metre of an otherwise correct stop -- braking hard for a gap that is no
    # longer closing.
    clearance = min(target_clearance_m, max(0.0, range_m - 0.3))
    room = max(0.05, range_m - clearance)
    closing = max(0.0, closing_mps)

    naive = (closing * closing) / (2.0 * room) if closing > 0.05 else 0.0
    if lead_accel_mps2 < -0.5:
        naive = max(naive, (ego_speed_mps * ego_speed_mps) / (2.0 * room))
    if jerk_mps3 > 0.0:
        t_ramp = min(ramp_allowance_s, max(0.0, naive - current_decel_mps2) / jerk_mps3)
    else:
        t_ramp = 0.0
    usable = max(0.05, room - closing * 0.5 * t_ramp)

    if lead_accel_mps2 < -0.5:
        lead_speed = max(0.0, ego_speed_mps - closing)
        available = max(0.05, usable + stopping_distance_m(lead_speed, -lead_accel_mps2))
        return min(max_decel_mps2, (ego_speed_mps * ego_speed_mps) / (2.0 * available))
    if closing <= 0.05:
        return 0.0
    return min(max_decel_mps2, (closing * closing) / (2.0 * usable))


class JerkShaper:
    """The ceiling on how fast a deceleration demand may be built.

    Only the RISE is limited.  A release returns the occupant toward zero g and
    is arrested by the seat back, and the brake's own hydraulic decay turns a
    step release into a ramp on the road whatever the demand does; more
    decisively, a ceiling on the release rate is a requirement to KEEP braking,
    which contradicts the requirement to remove an unwarranted deceleration
    promptly.  The two are only jointly satisfiable by never braking at all.

    The band is chosen from where the demand is HEADING: a demand bound for
    emergency grade is a collision-avoidance action and gets the emergency
    ceiling for the whole ramp, because a ramp that has to pause at the comfort
    limit on its way to 8 m/s^2 is not a collision-avoidance action at all.
    """

    def __init__(
        self,
        comfort_jerk_mps3: float = 2.5,
        emergency_jerk_mps3: float = 20.0,
        emergency_decel_mps2: float = 3.5,
        release_rate_mps3: float = 12.0,
    ) -> None:
        if comfort_jerk_mps3 <= 0.0 or emergency_jerk_mps3 < comfort_jerk_mps3:
            raise ValueError("require 0 < comfort_jerk <= emergency_jerk")
        self.comfort_jerk_mps3 = comfort_jerk_mps3
        self.emergency_jerk_mps3 = emergency_jerk_mps3
        self.emergency_decel_mps2 = emergency_decel_mps2
        self.release_rate_mps3 = release_rate_mps3
        self._decel = 0.0

    @property
    def decel_mps2(self) -> float:
        """The shaped demand as it currently stands, m/s^2."""
        return self._decel

    def reset(self, decel_mps2: float = 0.0) -> None:
        """Force the shaped demand to a value (used on a pipeline restart)."""
        self._decel = max(0.0, float(decel_mps2))

    def step(self, target_mps2: float, dt_s: float) -> float:
        """Move the shaped demand toward ``target_mps2`` and return it."""
        target = max(0.0, float(target_mps2))
        dt = dt_s if (math.isfinite(dt_s) and dt_s > 0.0) else 0.05
        if target > self._decel:
            rate = (
                self.emergency_jerk_mps3
                if target >= self.emergency_decel_mps2
                else self.comfort_jerk_mps3
            )
            self._decel = min(target, self._decel + rate * dt)
        else:
            self._decel = max(target, self._decel - self.release_rate_mps3 * dt)
        return self._decel


def sub_emergency_guard(
    target_mps2: float,
    warrant_decel_mps2: float,
    comfort_decel_mps2: float = 3.0,
    guard_margin_mps2: float = 0.05,
) -> float:
    """Keep an UNWARRANTED demand out of the sub-emergency band.

    The band between the comfort limit (3.0 m/s^2) and the emergency threshold
    (3.5 m/s^2) is the one place a longitudinal fault can hide: an emergency
    grading starts at 3.5 and cannot see 3.4, and a headway grading only asks
    whether *some* response happened and is satisfied by 3.4.  A system can
    therefore brake harder than any passenger tolerates, for a whole run,
    against a lead that never moved, and pass every other test.  That is not
    hypothetical -- it is what the previous revision of this codebase did at
    every constant-range gap from 26 m to 44 m at 20 m/s.

    The rule this function implements states the band's meaning rather than
    policing its symptom: **deceleration above the comfort limit is a
    collision-avoidance action, and a collision-avoidance action requires a
    warrant.**  So while the situation is not yet an emergency -- while the
    deceleration needed to hold the standstill clearance is still below the
    comfort limit -- the demand is held just under that limit.  Once it IS an
    emergency the demand is free, and by then the geometry is already asking for
    more than 3.5, so the band is crossed in a single frame rather than lived in.

    The guard is not a limit on how hard the vehicle may brake in an emergency.
    It is a statement that 3.0-3.5 m/s^2 buys almost nothing over 3.0 and costs
    the occupant and the vehicle behind, so it is not a place to spend time
    unless the physics has already demanded more.

    Args:
        target_mps2: The deceleration the laws want, m/s^2.
        warrant_decel_mps2: The constant deceleration that would just preserve
            the standstill clearance, computed from the measured scene with no
            margins and no ramp allowance.  This is the specification's own
            definition of when an emergency has arisen.
        comfort_decel_mps2: The comfort limit; the bottom of the band.
        guard_margin_mps2: How far below the limit an unwarranted demand is
            held.  The band is closed at the bottom -- a command of exactly the
            comfort limit is inside it -- so the guard has to leave a gap, and
            0.05 m/s^2 is the quantisation of a rate-limited brake command.

    Returns:
        The guarded target.
    """
    if warrant_decel_mps2 >= comfort_decel_mps2:
        return target_mps2
    return min(target_mps2, max(0.0, comfort_decel_mps2 - guard_margin_mps2))
