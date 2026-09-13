"""Kalman filters for the ADAS multi-object tracker.

Two filters with deliberately different motion models are provided, because the
two quantities the tracker estimates have nothing in common:

:class:`RangeFilter`
    A **constant-acceleration** filter over the ground-plane longitudinal state
    ``[d, d_dot, d_ddot]`` in metres, metres/second and metres/second^2. This is
    the channel the planner and the safety arbiter consume, so it is the one
    that must not lag. A constant-*velocity* range filter lags by ``a * tau``
    metres/second throughout a lead-vehicle braking event -- precisely the
    manoeuvre time-to-collision has to be right for -- so the third state is not
    a luxury. The price is a noisier velocity estimate, paid for by a
    range-dependent measurement variance (see :func:`range_measurement_sigma_m`)
    that automatically distrusts far objects.

:class:`BoxFilter`
    A **constant-velocity** filter over the image-plane box centre
    ``[cx, cy, vcx, vcy]`` in pixels and pixels/second, plus two independent
    scalar random walks on width and height. It exists only to give data
    association a prediction and a covariance, so a validation gate can be a
    chi-square test on the innovation instead of a fixed pixel radius. Box
    *size* is a random walk rather than a state with a velocity: a size rate
    estimated from noisy boxes diverges during a coast and inflates or collapses
    the predicted box, a well known failure of SORT-style trackers.

Units
-----
Every length is metres unless the identifier ends in ``_px``; every pixel
quantity ends in ``_px``. Time is seconds. ``d_dot`` is the derivative of
*range*, so it is **negative when the object is closing**. The tracker publishes
``TrackedObject.velocity_mps`` with the opposite (positive-when-closing) sign
that the rest of this codebase already uses; :attr:`RangeFilter.closing_mps`
performs that flip in one documented place.

Why there is no numpy in the inner loop
---------------------------------------
Measured on the target Xavier NX: a single numpy call on a 4x4 array costs
15-30 us of dispatch overhead, so a numpy ``BoxFilter.update`` measured 287 us
against 20 us for the same arithmetic in plain Python lists, and the tracker
came to 22.5 ms per frame for twenty objects -- half the 50 ms budget, for
matrices with sixteen entries. Every state and covariance here is therefore a
Python ``list``; numpy appears only at the API boundary (:attr:`KalmanFilter.x`,
:attr:`KalmanFilter.P`, and the standalone matrix builders) where callers and
tests expect arrays and the call happens once, not per track per frame.

The two structural shortcuts that buy the rest:

* :class:`KalmanFilter` observes the **leading block** of the state -- a
  measurement of length ``k`` observes ``x[:k]``, i.e. ``H = [I_k | 0]``. Both
  filters here are of that form (range observes ``d``; the box observes
  ``cx, cy``), so their state vectors are ordered to put the observed
  components first, and the four matrix products a general ``H`` would need
  never happen. ``tests/test_kalman.py`` pins the result against the textbook
  ``H``-matrix update.
* The ``dt``-dependent transition and process-noise matrices are memoised,
  because the frame period barely varies.

Failure behaviour
-----------------
* Construction validates its arguments and raises
  :class:`~adas.core.exceptions.ValidationError` rather than producing a filter
  that will misbehave later.
* A singular innovation covariance raises :class:`FilterError` from
  :meth:`KalmanFilter.update`; :meth:`KalmanFilter.mahalanobis2` returns ``inf``
  for the same condition, so a numerically degenerate track fails the
  association gate instead of matching everything.
* Nothing here clamps an estimate to a plausible-looking constant. The two
  saturations that do exist (:attr:`RangeFilter.max_speed_mps` and
  :attr:`RangeFilter.max_accel_mps2`) bound *physically impossible* states, log
  at WARNING when they engage, and are recorded in
  :attr:`RangeFilter.saturations` so a caller can tell the estimate was clipped.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from adas.core.exceptions import TrackingError, ValidationError
from adas.core.logger import setup_logger

logger = setup_logger(__name__)

__all__ = [
    "CHI2_95",
    "CHI2_99",
    "FilterError",
    "KalmanFilter",
    "RangeFilter",
    "BoxFilter",
    "range_measurement_sigma_m",
    "range_transition",
    "range_process_noise",
]


#: Chi-square inverse CDF at p = 0.95, keyed by degrees of freedom. Used for
#: diagnostics; the association gate uses the 0.99 table so a genuine manoeuvre
#: is not rejected one frame in twenty.
CHI2_95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877}

#: Chi-square inverse CDF at p = 0.99, keyed by degrees of freedom.
CHI2_99 = {1: 6.6349, 2: 9.2103, 3: 11.3449, 4: 13.2767}

#: Upper bound on each memo of ``dt``-dependent matrices. The frame period is
#: near-constant, so one or two entries are live; the bound only stops an
#: adversarial ``dt`` sequence from growing the dict without limit.
_MEMO_LIMIT = 64


class FilterError(TrackingError):
    """A Kalman update could not be computed (singular innovation covariance)."""


# --------------------------------------------------------------------------- #
# Small dense linear algebra on Python lists
# --------------------------------------------------------------------------- #


def _as_vector(values: Sequence[float]) -> List[float]:
    """Copy any sequence of numbers into a list of Python floats."""
    return [float(v) for v in values]


def _as_rows(matrix: Any, n: int) -> List[List[float]]:
    """Copy any ``(n, n)`` matrix-like into a list of lists of Python floats.

    A list of lists is returned unchanged only if it is already exactly that;
    numpy arrays, tuples and nested sequences are converted. The caller must not
    mutate the result when it was passed through.
    """
    if isinstance(matrix, list) and len(matrix) == n and matrix and isinstance(matrix[0], list):
        return matrix
    rows = [_as_vector(row) for row in matrix]
    if len(rows) != n or any(len(row) != n for row in rows):
        raise ValidationError("Expected a %dx%d matrix, got %r" % (n, n, rows))
    return rows


def _inverse_small(S: List[List[float]], k: int) -> List[List[float]]:
    """Inverse of a small symmetric matrix, as lists.

    Closed form for the 1x1 and 2x2 cases this package uses; ``numpy.linalg``
    for anything larger, which only a future model would reach.

    Raises:
        numpy.linalg.LinAlgError: if ``S`` is singular or non-finite.
    """
    if k == 1:
        s = S[0][0]
        if s == 0.0 or not math.isfinite(s):
            raise np.linalg.LinAlgError("singular 1x1 innovation covariance")
        return [[1.0 / s]]
    if k == 2:
        a = S[0][0]
        b = S[0][1]
        c = S[1][0]
        d = S[1][1]
        det = a * d - b * c
        if det == 0.0 or not math.isfinite(det):
            raise np.linalg.LinAlgError("singular 2x2 innovation covariance")
        return [[d / det, -b / det], [-c / det, a / det]]
    inverse = np.linalg.inv(np.asarray(S, dtype=np.float64))
    if not np.isfinite(inverse).all():
        raise np.linalg.LinAlgError("singular innovation covariance")
    return inverse.tolist()


class KalmanFilter:
    """Linear Kalman filter over a Python-list state, observing the leading block.

    A measurement of length ``k`` observes ``x[:k]`` directly, i.e.
    ``H = [I_k | 0]``. See the module docstring for why the general measurement
    matrix is not supported and why the arithmetic is not vectorised.

    Attributes
    ----------
    x:
        State vector as a numpy array (a copy; assign a sequence to replace it).
    P:
        State covariance as a numpy array (a copy; assign a matrix to replace
        it). Kept symmetric by construction.
    """

    __slots__ = ("_x", "_P", "_n")

    def __init__(self, x: Sequence[float], P: Any) -> None:
        state = _as_vector(x)
        n = len(state)
        if n == 0:
            raise ValidationError("Kalman state must be non-empty")
        try:
            cov = [_as_vector(row) for row in P]
        except TypeError as exc:
            raise ValidationError("Covariance must be a %dx%d matrix" % (n, n)) from exc
        if len(cov) != n or any(len(row) != n for row in cov):
            raise ValidationError(
                "Covariance shape does not match state length %d" % n
            )
        if not all(math.isfinite(v) for v in state):
            raise ValidationError("Kalman state must be finite")
        if not all(math.isfinite(v) for row in cov for v in row):
            raise ValidationError("Kalman covariance must be finite")
        self._x = state
        self._P = _symmetrise(cov, n)
        self._n = n

    # -- state access -------------------------------------------------------- #

    @property
    def dim(self) -> int:
        """Number of state variables."""
        return self._n

    @property
    def x(self) -> np.ndarray:
        """State vector as a numpy array. Reading allocates; prefer :meth:`state_at`."""
        return np.array(self._x, dtype=np.float64)

    @x.setter
    def x(self, values: Sequence[float]) -> None:
        state = _as_vector(values)
        if len(state) != self._n:
            raise ValidationError("State length must stay %d" % self._n)
        self._x = state

    @property
    def P(self) -> np.ndarray:
        """State covariance as a numpy array. Reading allocates; prefer :meth:`covariance_at`."""
        return np.array(self._P, dtype=np.float64)

    @P.setter
    def P(self, matrix: Any) -> None:
        self._P = _symmetrise([_as_vector(row) for row in matrix], self._n)

    def state_at(self, index: int) -> float:
        """One state component, without allocating an array."""
        return self._x[index]

    def covariance_at(self, row: int, column: int) -> float:
        """One covariance entry, without allocating an array."""
        return self._P[row][column]

    def set_state_at(self, index: int, value: float) -> None:
        """Overwrite one state component in place (used by physical saturation)."""
        self._x[index] = float(value)

    # -- filter -------------------------------------------------------------- #

    def predict(self, F: Any, Q: Any) -> None:
        """Propagate the state through ``F`` and add process noise ``Q``.

        ``F`` and ``Q`` may be nested lists (no conversion, the fast path used by
        the memoised models) or any matrix-like. Zero entries of ``F`` are
        skipped, which is most of them for a constant-velocity or
        constant-acceleration model.
        """
        n = self._n
        rows = _as_rows(F, n)
        noise = _as_rows(Q, n)
        x = self._x
        P = self._P

        new_x = [0.0] * n
        for i in range(n):
            Fi = rows[i]
            total = 0.0
            for j in range(n):
                f = Fi[j]
                if f:
                    total += f * x[j]
            new_x[i] = total
        self._x = new_x

        # M = F P
        M = [[0.0] * n for _ in range(n)]
        for i in range(n):
            Fi = rows[i]
            Mi = M[i]
            for k in range(n):
                f = Fi[k]
                if f:
                    Pk = P[k]
                    for j in range(n):
                        Mi[j] += f * Pk[j]

        # P' = M F^T + Q, upper triangle then mirrored so it is exactly symmetric.
        new_P = [[0.0] * n for _ in range(n)]
        for i in range(n):
            Mi = M[i]
            for j in range(i, n):
                Fj = rows[j]
                total = noise[i][j]
                for k in range(n):
                    f = Fj[k]
                    if f:
                        total += Mi[k] * f
                new_P[i][j] = total
                new_P[j][i] = total
        self._P = new_P

    def innovation(self, z: Sequence[float], R: Any) -> Tuple[List[float], List[List[float]]]:
        """Return ``(y, S)``: the residual of ``z`` against ``x[:k]`` and its covariance.

        ``k`` is ``len(z)``; ``R`` must be ``(k, k)``.
        """
        measurement = _as_vector(z)
        k = len(measurement)
        if k < 1 or k > self._n:
            raise ValidationError("Measurement length %d must be in [1, %d]" % (k, self._n))
        noise = _as_rows(R, k)
        x = self._x
        P = self._P
        y = [measurement[i] - x[i] for i in range(k)]
        S = [[P[i][j] + noise[i][j] for j in range(k)] for i in range(k)]
        return y, S

    def update(self, z: Sequence[float], R: Any) -> Tuple[List[float], List[List[float]]]:
        """Correct the state with a measurement of ``x[:len(z)]``.

        Uses the Joseph form for the covariance update, which stays positive
        semi-definite under round-off where the short form does not. Returns
        ``(y, S)`` so the caller can record the normalised innovation without
        recomputing it.

        Raises:
            ValidationError: if ``z`` or ``R`` has the wrong shape.
            FilterError: if ``S`` is singular or the resulting gain is not finite.
        """
        y, S = self.innovation(z, R)
        k = len(y)
        n = self._n
        P = self._P
        noise = _as_rows(R, k)
        try:
            S_inv = _inverse_small(S, k)
        except np.linalg.LinAlgError as exc:
            raise FilterError("Singular innovation covariance: %s" % exc) from exc

        # K = P H^T S^-1 with H = [I_k | 0], so P H^T is the first k columns of P.
        rng_k = range(k)
        K = [[0.0] * k for _ in range(n)]
        for i in range(n):
            Pi = P[i]
            Ki = K[i]
            for a in rng_k:
                total = 0.0
                for b in rng_k:
                    total += Pi[b] * S_inv[b][a]
                if not math.isfinite(total):
                    raise FilterError("Non-finite Kalman gain")
                Ki[a] = total

        x = self._x
        new_x = [0.0] * n
        for i in range(n):
            Ki = K[i]
            total = x[i]
            for a in rng_k:
                total += Ki[a] * y[a]
            new_x[i] = total
        self._x = new_x

        # A = (I - K H) P, exploiting that (I - K H) differs from the identity
        # only in its leading k columns: A[i][j] = P[i][j] - sum_a K[i][a] P[a][j].
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            Pi = P[i]
            Ki = K[i]
            Ai = A[i]
            for j in range(n):
                total = Pi[j]
                for a in rng_k:
                    total -= Ki[a] * P[a][j]
                Ai[j] = total

        # R K^T, reused across every (i, j) pair below.
        RKt = [[0.0] * n for _ in rng_k]
        for a in rng_k:
            Ra = noise[a]
            RKt_a = RKt[a]
            for j in range(n):
                Kj = K[j]
                total = 0.0
                for b in rng_k:
                    total += Ra[b] * Kj[b]
                RKt_a[j] = total

        # P' = A (I - K H)^T + K R K^T, upper triangle then mirrored.
        new_P = [[0.0] * n for _ in range(n)]
        for i in range(n):
            Ai = A[i]
            Ki = K[i]
            for j in range(i, n):
                Kj = K[j]
                total = Ai[j]
                for a in rng_k:
                    total += -Kj[a] * Ai[a] + Ki[a] * RKt[a][j]
                new_P[i][j] = total
                new_P[j][i] = total
        self._P = new_P
        return y, S

    def mahalanobis2(self, z: Sequence[float], R: Any) -> float:
        """Squared Mahalanobis distance of ``z`` from the predicted ``x[:len(z)]``.

        Returns ``inf`` (never raises, never a plausible small number) when the
        innovation covariance is singular or the result is not finite, so a
        degenerate filter rejects every candidate rather than accepting them all.
        """
        y, S = self.innovation(z, R)
        k = len(y)
        try:
            S_inv = _inverse_small(S, k)
        except np.linalg.LinAlgError:
            return float("inf")
        value = 0.0
        for a in range(k):
            inner = 0.0
            for b in range(k):
                inner += S_inv[a][b] * y[b]
            value += y[a] * inner
        if not math.isfinite(value) or value < 0.0:
            return float("inf")
        return value


def _symmetrise(matrix: List[List[float]], n: int) -> List[List[float]]:
    """Return an exactly symmetric copy. Round-off asymmetry in ``P`` compounds."""
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            value = 0.5 * (matrix[i][j] + matrix[j][i])
            out[i][j] = value
            out[j][i] = value
    return out


# --------------------------------------------------------------------------- #
# Longitudinal range filter
# --------------------------------------------------------------------------- #


def _range_transition_rows(dt: float) -> List[List[float]]:
    return [
        [1.0, dt, 0.5 * dt * dt],
        [0.0, 1.0, dt],
        [0.0, 0.0, 1.0],
    ]


def _range_process_noise_rows(dt: float, jerk_psd: float) -> List[List[float]]:
    q = jerk_psd
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    dt5 = dt4 * dt
    return [
        [q * dt5 / 20.0, q * dt4 / 8.0, q * dt3 / 6.0],
        [q * dt4 / 8.0, q * dt3 / 3.0, q * dt2 / 2.0],
        [q * dt3 / 6.0, q * dt2 / 2.0, q * dt],
    ]


def range_transition(dt_s: float) -> np.ndarray:
    """Constant-acceleration transition matrix for ``[d, d_dot, d_ddot]``."""
    return np.array(_range_transition_rows(float(dt_s)), dtype=np.float64)


def range_process_noise(dt_s: float, jerk_psd: float) -> np.ndarray:
    """Discretised continuous white-noise-jerk process noise.

    Args:
        dt_s: Step length in seconds.
        jerk_psd: Power spectral density of the jerk process, (m/s^3)^2 / Hz,
            i.e. m^2/s^5. The default used by :class:`RangeFilter` is 1.0, a
            1 m/s^3 jerk standard deviation over one second, chosen empirically
            against a quantised monocular box-height measurement: at 9.0 the
            acceleration state chases pixel noise (a stationary object at 30 m
            reads up to 3.15 m/s of closing speed instead of 1.96) while gaining
            only ~0.4 m/s of response at the end of a 4 m/s^2 braking event.
            See ``tests/test_tracking.py``.
    """
    return np.array(_range_process_noise_rows(float(dt_s), float(jerk_psd)), dtype=np.float64)


_RANGE_MODEL_MEMO: Dict[Tuple[float, float], Tuple[List[List[float]], List[List[float]]]] = {}


def _range_model(dt_s: float, jerk_psd: float) -> Tuple[List[List[float]], List[List[float]]]:
    """Memoised ``(F, Q)`` rows for the range model. The result must not be mutated."""
    key = (round(float(dt_s), 9), round(float(jerk_psd), 9))
    cached = _RANGE_MODEL_MEMO.get(key)
    if cached is None:
        if len(_RANGE_MODEL_MEMO) >= _MEMO_LIMIT:
            _RANGE_MODEL_MEMO.clear()
        cached = (_range_transition_rows(key[0]), _range_process_noise_rows(key[0], key[1]))
        _RANGE_MODEL_MEMO[key] = cached
    return cached


def range_measurement_sigma_m(
    distance_m: float,
    focal_length_px: float,
    object_height_m: float,
    box_sigma_px: float = 1.5,
    confidence: float = 1.0,
    truncated: bool = False,
    min_sigma_m: float = 0.15,
    max_sigma_m: float = 60.0,
    truncation_factor: float = 5.0,
    confidence_floor: float = 0.15,
) -> float:
    """Standard deviation (metres) of a monocular box-height range measurement.

    The pinhole model is ``d = H * f / h``, so ``|dd/dh| = d^2 / (H * f)`` and a
    detector box-height noise of ``box_sigma_px`` pixels maps to

        sigma_d = d^2 / (H * f) * sigma_h

    metres. This is the whole reason the filter is worth having: with ``H`` =
    1.5 m and ``f`` = 910 px, one pixel is 0.11 m at 10 m and 2.6 m at 50 m, so
    a fixed measurement variance would either over-trust far objects or
    over-smooth near ones.

    ``confidence`` (the detector score, or a geometry channel's own confidence)
    divides into the result: a weak detection has a looser box. ``truncated``
    multiplies by ``truncation_factor`` because a vertically clipped box makes
    the height model read too FAR by an unbounded factor, exactly when the
    object is closest -- see ADAS-DEC-10.

    Returns a value clamped to ``[min_sigma_m, max_sigma_m]``; a zero or
    negative focal length or object height yields ``max_sigma_m`` rather than a
    division by zero.
    """
    denominator = float(object_height_m) * float(focal_length_px)
    if not math.isfinite(denominator) or denominator <= 0.0:
        return float(max_sigma_m)
    d = float(distance_m)
    if not math.isfinite(d) or d < 0.0:
        return float(max_sigma_m)
    sigma = (d * d / denominator) * float(box_sigma_px)
    conf = float(confidence)
    if not math.isfinite(conf) or conf <= 0.0:
        conf = confidence_floor
    sigma /= max(confidence_floor, min(1.0, conf))
    if truncated:
        sigma *= float(truncation_factor)
    if not math.isfinite(sigma):
        return float(max_sigma_m)
    return float(min(max(sigma, float(min_sigma_m)), float(max_sigma_m)))


class RangeFilter:
    """Constant-acceleration Kalman filter over longitudinal range.

    State is ``[d, d_dot, d_ddot]`` in metres, m/s and m/s^2, where ``d`` is the
    range from the camera to the object along the ground plane. ``d_dot`` is the
    range rate, **negative when closing**; :attr:`closing_mps` is the
    positive-when-closing flip that ``TrackedObject.velocity_mps`` uses.

    Failure behaviour: :meth:`update` raises :class:`FilterError` if the
    innovation covariance is singular. States outside
    ``[-max_speed_mps, max_speed_mps]`` or ``[-max_accel_mps2, max_accel_mps2]``
    are saturated (logged, counted in :attr:`saturations`) because no road
    vehicle closes at 200 m/s; a saturated estimate is still reported, never
    silently discarded. Range itself is floored at 0 -- a negative range is
    unphysical and would invert every TTC downstream.
    """

    __slots__ = ("jerk_psd", "max_speed_mps", "max_accel_mps2", "saturations", "_kf")

    def __init__(
        self,
        distance_m: float,
        measurement_sigma_m: float,
        jerk_psd: float = 1.0,
        init_rate_sigma_mps: float = 15.0,
        init_accel_sigma_mps2: float = 4.0,
        max_speed_mps: float = 90.0,
        max_accel_mps2: float = 12.0,
    ) -> None:
        d = float(distance_m)
        if not math.isfinite(d) or d < 0.0:
            raise ValidationError(
                "RangeFilter needs a finite non-negative range, got %r" % (distance_m,)
            )
        sigma = float(measurement_sigma_m)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValidationError(
                "RangeFilter needs a positive measurement sigma, got %r" % (measurement_sigma_m,)
            )
        if jerk_psd <= 0.0:
            raise ValidationError("jerk_psd must be positive")
        self.jerk_psd = float(jerk_psd)
        self.max_speed_mps = float(max_speed_mps)
        self.max_accel_mps2 = float(max_accel_mps2)
        self.saturations = 0
        rate_var = float(init_rate_sigma_mps) ** 2
        accel_var = float(init_accel_sigma_mps2) ** 2
        self._kf = KalmanFilter(
            x=[d, 0.0, 0.0],
            P=[
                [sigma * sigma, 0.0, 0.0],
                [0.0, rate_var, 0.0],
                [0.0, 0.0, accel_var],
            ],
        )

    # -- accessors ---------------------------------------------------------- #

    @property
    def distance_m(self) -> float:
        """Filtered range, metres. Never negative."""
        value = self._kf.state_at(0)
        return value if value > 0.0 else 0.0

    @property
    def rate_mps(self) -> float:
        """Range rate ``d_dot``, m/s. Negative when the object is closing."""
        return self._kf.state_at(1)

    @property
    def closing_mps(self) -> float:
        """Closing speed, m/s, **positive when the range is shrinking**.

        This is the sign convention ``TrackedObject.velocity_mps`` uses across
        the planner, the arbiter and their tests.
        """
        value = -self._kf.state_at(1)
        # Normalise negative zero: it prints as -0.0 in every log line.
        return 0.0 if value == 0.0 else value

    @property
    def accel_mps2(self) -> float:
        """Range acceleration ``d_ddot``, m/s^2."""
        return self._kf.state_at(2)

    @property
    def distance_variance_m2(self) -> float:
        """Variance of the range estimate, m^2."""
        return self._kf.covariance_at(0, 0)

    @property
    def rate_variance_m2s2(self) -> float:
        """Variance of the range-rate estimate, (m/s)^2."""
        return self._kf.covariance_at(1, 1)

    @property
    def state(self) -> np.ndarray:
        """``[d, d_dot, d_ddot]`` as a numpy array."""
        return self._kf.x

    @property
    def covariance(self) -> np.ndarray:
        """The 3x3 state covariance as a numpy array."""
        return self._kf.P

    # -- filter ------------------------------------------------------------- #

    def predict(self, dt_s: float) -> None:
        """Propagate by ``dt_s`` seconds. A non-positive or non-finite ``dt_s`` is a no-op."""
        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            return
        F, Q = _range_model(dt, self.jerk_psd)
        self._kf.predict(F, Q)
        self._saturate()

    def update(self, distance_m: float, measurement_sigma_m: float) -> float:
        """Correct with a range measurement. Returns the squared Mahalanobis distance.

        Raises:
            ValidationError: if the measurement is not finite and non-negative,
                or the sigma is not positive.
            FilterError: if the innovation covariance is singular.
        """
        z, sigma = self._check_measurement(distance_m, measurement_sigma_m)
        y, S = self._kf.update([z], [[sigma * sigma]])
        self._saturate()
        variance = S[0][0]
        if variance <= 0.0 or not math.isfinite(variance):
            return float("inf")
        return y[0] * y[0] / variance

    def gate(self, distance_m: float, measurement_sigma_m: float) -> float:
        """Squared Mahalanobis distance of a candidate measurement, without updating."""
        z, sigma = self._check_measurement(distance_m, measurement_sigma_m)
        residual = z - self._kf.state_at(0)
        variance = self._kf.covariance_at(0, 0) + sigma * sigma
        if variance <= 0.0 or not math.isfinite(variance):
            return float("inf")
        value = residual * residual / variance
        return value if math.isfinite(value) else float("inf")

    def reinitialise(self, distance_m: float, measurement_sigma_m: float) -> None:
        """Restart the filter at a new range, discarding the rate estimate.

        Used when a measurement is so far outside the gate that the previous
        state is more likely wrong than the measurement. The rate is reset to
        zero with its full prior variance rather than carried over, because a
        rate differenced across a discontinuity is meaningless.
        """
        z, sigma = self._check_measurement(distance_m, measurement_sigma_m)
        rate_var = max(self._kf.covariance_at(1, 1), 15.0 ** 2)
        accel_var = max(self._kf.covariance_at(2, 2), 4.0 ** 2)
        self._kf.x = [z, 0.0, 0.0]
        self._kf.P = [
            [sigma * sigma, 0.0, 0.0],
            [0.0, rate_var, 0.0],
            [0.0, 0.0, accel_var],
        ]

    # -- internals ---------------------------------------------------------- #

    @staticmethod
    def _check_measurement(distance_m: float, measurement_sigma_m: float) -> Tuple[float, float]:
        z = float(distance_m)
        sigma = float(measurement_sigma_m)
        if not math.isfinite(z) or z < 0.0:
            raise ValidationError(
                "Range measurement must be finite and non-negative, got %r" % (distance_m,)
            )
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValidationError(
                "Range measurement sigma must be positive, got %r" % (measurement_sigma_m,)
            )
        return z, sigma

    def _saturate(self) -> None:
        kf = self._kf
        d = kf.state_at(0)
        rate = kf.state_at(1)
        accel = kf.state_at(2)
        clipped = False
        if d < 0.0:
            kf.set_state_at(0, 0.0)
            d = 0.0
            clipped = True
        if abs(rate) > self.max_speed_mps:
            rate = math.copysign(self.max_speed_mps, rate)
            kf.set_state_at(1, rate)
            clipped = True
        if abs(accel) > self.max_accel_mps2:
            accel = math.copysign(self.max_accel_mps2, accel)
            kf.set_state_at(2, accel)
            clipped = True
        if clipped:
            self.saturations += 1
            logger.warning(
                "RangeFilter state saturated (d=%.2f m, d_dot=%.2f m/s, d_ddot=%.2f m/s^2)",
                d,
                rate,
                accel,
            )


# --------------------------------------------------------------------------- #
# Image-plane box filter
# --------------------------------------------------------------------------- #


_CENTRE_MODEL_MEMO: Dict[Tuple[float, float], Tuple[List[List[float]], List[List[float]]]] = {}


def _centre_model(dt_s: float, accel_px_s2: float) -> Tuple[List[List[float]], List[List[float]]]:
    """Memoised ``(F, Q)`` rows for the 2-axis constant-velocity centre model.

    ``Q`` is the discrete white-noise-acceleration form with an acceleration
    standard deviation of ``accel_px_s2`` pixels/second^2 per axis. The memo key
    rounds the acceleration to 0.1 px/s^2, far finer than the box-height proxy it
    is derived from. The result must not be mutated.
    """
    key = (round(float(dt_s), 9), round(float(accel_px_s2), 1))
    cached = _CENTRE_MODEL_MEMO.get(key)
    if cached is not None:
        return cached
    if len(_CENTRE_MODEL_MEMO) >= _MEMO_LIMIT:
        _CENTRE_MODEL_MEMO.clear()
    dt = key[0]
    F = [
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    q = key[1] * key[1]
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    qpp = q * dt4 / 4.0
    qpv = q * dt3 / 2.0
    qvv = q * dt2
    Q = [
        [qpp, 0.0, qpv, 0.0],
        [0.0, qpp, 0.0, qpv],
        [qpv, 0.0, qvv, 0.0],
        [0.0, qpv, 0.0, qvv],
    ]
    cached = (F, Q)
    _CENTRE_MODEL_MEMO[key] = cached
    return cached


class BoxFilter:
    """Constant-velocity filter over an image-plane bounding box.

    The coupled state is the centre only -- ``[cx, cy, vcx, vcy]`` in pixels and
    pixels/second. Width and height are two independent scalar random walks,
    because a size *rate* fitted to noisy detector boxes diverges while a track
    coasts and produces a predicted box that either collapses or swallows the
    frame.

    The point of this filter is the *gate*. ``association_threshold_px`` was a
    fixed 120 px radius, which at 100 m is wider than the 32 px image separation
    of two adjacent-lane vehicles and at 8 m is narrower than a genuine cut-in
    (ADAS-DEC-08). :meth:`gate_centre` replaces it with a chi-square test on the
    innovation, which scales with range, with ``dt`` and with how long the track
    has been coasting, all automatically.

    Process and measurement noise both scale with box height, which is the
    cheapest available proxy for range: everything in the image plane moves
    ``f/Z`` times faster for a near object than a far one.

    Failure behaviour: :meth:`update` propagates :class:`FilterError` from a
    singular innovation covariance; :meth:`gate_centre` returns ``inf`` instead.
    Predicted width and height are floored at :data:`MIN_SIZE_PX` so a diverged
    covariance cannot produce a zero-area or inverted box.
    """

    #: Smallest predicted box side, pixels. A detector box is never smaller.
    MIN_SIZE_PX = 1.0

    __slots__ = (
        "centre_accel_scale",
        "centre_accel_floor_px_s2",
        "size_rate_scale",
        "size_rate_floor_px_s",
        "centre_sigma_scale",
        "centre_sigma_floor_px",
        "size_sigma_scale",
        "size_sigma_floor_px",
        "_kf",
        "_w",
        "_h",
        "_var_w",
        "_var_h",
    )

    def __init__(
        self,
        box_xyxy: Sequence[float],
        centre_accel_scale: float = 15.0,
        centre_accel_floor_px_s2: float = 200.0,
        size_rate_scale: float = 0.5,
        size_rate_floor_px_s: float = 4.0,
        centre_sigma_scale: float = 0.05,
        centre_sigma_floor_px: float = 2.0,
        size_sigma_scale: float = 0.08,
        size_sigma_floor_px: float = 2.0,
        init_rate_scale: float = 4.0,
        init_rate_floor_px_s: float = 100.0,
    ) -> None:
        x1, y1, x2, y2 = (float(v) for v in box_xyxy)
        w = x2 - x1
        h = y2 - y1
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or w <= 0.0 or h <= 0.0:
            raise ValidationError("BoxFilter needs a finite, positive-area box, got %r" % (box_xyxy,))
        self.centre_accel_scale = float(centre_accel_scale)
        self.centre_accel_floor_px_s2 = float(centre_accel_floor_px_s2)
        self.size_rate_scale = float(size_rate_scale)
        self.size_rate_floor_px_s = float(size_rate_floor_px_s)
        self.centre_sigma_scale = float(centre_sigma_scale)
        self.centre_sigma_floor_px = float(centre_sigma_floor_px)
        self.size_sigma_scale = float(size_sigma_scale)
        self.size_sigma_floor_px = float(size_sigma_floor_px)

        init_rate_var = max(float(init_rate_floor_px_s), float(init_rate_scale) * h) ** 2
        centre_var = self._centre_variance(h)
        size_var = self._size_variance(h)
        self._kf = KalmanFilter(
            x=[(x1 + x2) / 2.0, (y1 + y2) / 2.0, 0.0, 0.0],
            P=[
                [centre_var, 0.0, 0.0, 0.0],
                [0.0, centre_var, 0.0, 0.0],
                [0.0, 0.0, init_rate_var, 0.0],
                [0.0, 0.0, 0.0, init_rate_var],
            ],
        )
        self._w = w
        self._h = h
        self._var_w = size_var
        self._var_h = size_var

    # -- accessors ---------------------------------------------------------- #

    @property
    def centre_px(self) -> Tuple[float, float]:
        """Predicted box centre ``(cx, cy)`` in pixels."""
        return self._kf.state_at(0), self._kf.state_at(1)

    @property
    def size_px(self) -> Tuple[float, float]:
        """Predicted box ``(width, height)`` in pixels, floored at :data:`MIN_SIZE_PX`."""
        floor = self.MIN_SIZE_PX
        return (self._w if self._w > floor else floor, self._h if self._h > floor else floor)

    @property
    def size_variance_px2(self) -> Tuple[float, float]:
        """Variances of the width and height random walks, px^2."""
        return self._var_w, self._var_h

    @property
    def velocity_px_s(self) -> Tuple[float, float]:
        """Estimated centre velocity ``(vcx, vcy)`` in pixels/second."""
        return self._kf.state_at(2), self._kf.state_at(3)

    @property
    def state(self) -> np.ndarray:
        """``[cx, cy, w, h, vcx, vcy]``, assembled from the centre and size filters."""
        cx, cy = self.centre_px
        w, h = self.size_px
        vx, vy = self.velocity_px_s
        return np.array([cx, cy, w, h, vx, vy], dtype=np.float64)

    @property
    def covariance(self) -> np.ndarray:
        """The 4x4 *centre* covariance as a numpy array. Size variances are separate."""
        return self._kf.P

    def box_xyxy(self) -> Tuple[float, float, float, float]:
        """Current estimate as an ``(x1, y1, x2, y2)`` tuple."""
        cx, cy = self.centre_px
        w, h = self.size_px
        half_w = w / 2.0
        half_h = h / 2.0
        return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    def centre_gate_radius_px(self, chi2: float) -> float:
        """Radius (pixels) of the isotropic circle inscribed in the chi-square gate.

        Diagnostics only -- the gate itself is the full 2-D quadratic form. The
        value is the smallest displacement along either principal axis that the
        gate would reject, so it is a *lower* bound on the true gate extent.
        """
        variance = self._centre_variance(self.size_px[1])
        smallest = min(self._kf.covariance_at(0, 0), self._kf.covariance_at(1, 1)) + variance
        if smallest <= 0.0 or not math.isfinite(smallest) or chi2 <= 0.0:
            return 0.0
        return math.sqrt(chi2 * smallest)

    # -- filter ------------------------------------------------------------- #

    def predict(self, dt_s: float) -> None:
        """Propagate the box by ``dt_s`` seconds. A non-positive ``dt_s`` is a no-op."""
        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            return
        height = self.size_px[1]
        accel = self.centre_accel_scale * height
        if accel < self.centre_accel_floor_px_s2:
            accel = self.centre_accel_floor_px_s2
        F, Q = _centre_model(dt, accel)
        self._kf.predict(F, Q)
        size_rate = self.size_rate_scale * height
        if size_rate < self.size_rate_floor_px_s:
            size_rate = self.size_rate_floor_px_s
        growth = size_rate * size_rate * dt
        self._var_w += growth
        self._var_h += growth

    def update(self, box_xyxy: Sequence[float]) -> float:
        """Correct with an observed box. Returns the squared Mahalanobis distance of the centre."""
        cx, cy, w, h = self._measurement(box_xyxy)
        variance = self._centre_variance(h)
        y, S = self._kf.update([cx, cy], [[variance, 0.0], [0.0, variance]])
        maha2 = _quadratic_form_2(y, S)
        size_variance = self._size_variance(h)
        self._w, self._var_w = _scalar_update(self._w, self._var_w, w, size_variance)
        self._h, self._var_h = _scalar_update(self._h, self._var_h, h, size_variance)
        if self._w < self.MIN_SIZE_PX:
            self._w = self.MIN_SIZE_PX
        if self._h < self.MIN_SIZE_PX:
            self._h = self.MIN_SIZE_PX
        return maha2

    def gate_centre(
        self, cx_px: float, cy_px: float, observed_height_px: Optional[float] = None
    ) -> float:
        """Squared Mahalanobis distance (2 dof) of a candidate box centre.

        ``observed_height_px`` sizes the measurement noise; when omitted the
        filter's own predicted height is used. Returns ``inf`` for a
        non-finite candidate or a singular innovation covariance, so a
        degenerate track rejects every candidate.
        """
        height = (
            float(observed_height_px)
            if observed_height_px and math.isfinite(observed_height_px)
            else self.size_px[1]
        )
        return self.gate_centre_batch([cx_px], [cy_px], [height])[0]

    def gate_centre_batch(
        self,
        cx_px: Sequence[float],
        cy_px: Sequence[float],
        observed_height_px: Sequence[float],
    ) -> List[float]:
        """Vectorised :meth:`gate_centre` over many candidate detections.

        Association evaluates ``n_tracks * n_detections`` gates every frame, so
        this is the closed-form 2x2 quadratic form, and it returns a plain list
        of floats rather than an array -- the caller feeds it straight into
        :func:`adas.tracking.association.build_cost_matrix`, which is also plain
        Python.

        Args:
            cx_px, cy_px: Candidate centres.
            observed_height_px: Candidate box heights, used to size the
                measurement noise.

        Returns:
            Squared Mahalanobis distances, one per candidate. Entries whose
            innovation covariance is singular or whose input is non-finite are
            ``inf``, so they fail every gate.

        Raises:
            ValidationError: if the three sequences differ in length.
        """
        m = len(cx_px)
        if len(cy_px) != m or len(observed_height_px) != m:
            raise ValidationError("gate_centre_batch inputs must have the same length")
        kf = self._kf
        p00 = kf.covariance_at(0, 0)
        p01 = kf.covariance_at(0, 1)
        p11 = kf.covariance_at(1, 1)
        x0 = kf.state_at(0)
        x1 = kf.state_at(1)
        scale = self.centre_sigma_scale
        floor = self.centre_sigma_floor_px
        inf = float("inf")
        out: List[float] = []
        for index in range(m):
            cx = cx_px[index]
            cy = cy_px[index]
            if cx != cx or cy != cy or cx in (inf, -inf) or cy in (inf, -inf):
                out.append(inf)
                continue
            sigma = scale * observed_height_px[index]
            if sigma < floor:
                sigma = floor
            var = sigma * sigma
            s00 = p00 + var
            s11 = p11 + var
            det = s00 * s11 - p01 * p01
            if det <= 0.0 or not math.isfinite(det):
                out.append(inf)
                continue
            dy0 = cx - x0
            dy1 = cy - x1
            value = (dy0 * dy0 * s11 - 2.0 * dy0 * dy1 * p01 + dy1 * dy1 * s00) / det
            out.append(value if math.isfinite(value) and value >= 0.0 else inf)
        return out

    # -- internals ---------------------------------------------------------- #

    def _centre_variance(self, height_px: float) -> float:
        sigma = self.centre_sigma_scale * height_px
        if sigma < self.centre_sigma_floor_px:
            sigma = self.centre_sigma_floor_px
        return sigma * sigma

    def _size_variance(self, height_px: float) -> float:
        sigma = self.size_sigma_scale * height_px
        if sigma < self.size_sigma_floor_px:
            sigma = self.size_sigma_floor_px
        return sigma * sigma

    @staticmethod
    def _measurement(box_xyxy: Sequence[float]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = (float(v) for v in box_xyxy)
        w = x2 - x1
        h = y2 - y1
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or w <= 0.0 or h <= 0.0:
            raise ValidationError("BoxFilter measurement must be a finite, positive-area box")
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, w, h


def _quadratic_form_2(y: Sequence[float], S: Sequence[Sequence[float]]) -> float:
    """``y^T S^-1 y`` for a 2x2 ``S``. Returns ``inf`` rather than raising."""
    det = S[0][0] * S[1][1] - S[0][1] * S[1][0]
    if det <= 0.0 or not math.isfinite(det):
        return float("inf")
    value = (y[0] * y[0] * S[1][1] - 2.0 * y[0] * y[1] * S[0][1] + y[1] * y[1] * S[0][0]) / det
    return value if math.isfinite(value) and value >= 0.0 else float("inf")


def _scalar_update(x: float, variance: float, z: float, measurement_variance: float) -> Tuple[float, float]:
    """One scalar Kalman correction. Returns ``(posterior mean, posterior variance)``.

    Used for the box width and height random walks. A non-positive or non-finite
    total variance leaves the state untouched rather than dividing by zero.
    """
    total = variance + measurement_variance
    if total <= 0.0 or not math.isfinite(total):
        return x, variance
    gain = variance / total
    return x + gain * (z - x), (1.0 - gain) * variance
