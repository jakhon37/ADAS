"""Tests for the tracking Kalman filters.

These are the specification of :mod:`adas.tracking.kalman`. They check the three
properties the rest of the stack depends on and that the old un-filtered range
code violated:

1. the range estimate converges and its *rate* is usable -- ADAS-DEC-06 measured
   13.2 m/s of spurious closing speed from one pixel of box jitter at 30 m;
2. the covariance stays symmetric positive-definite through hundreds of variable
   ``dt`` steps, so the association gate derived from it stays meaningful;
3. the validation gate actually rejects an outlier instead of absorbing it.

Every stochastic test uses a fixed seed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from adas.core.exceptions import ValidationError
from adas.tracking.kalman import (
    CHI2_99,
    BoxFilter,
    FilterError,
    KalmanFilter,
    RangeFilter,
    range_measurement_sigma_m,
    range_process_noise,
    range_transition,
)

SEED = 20240913
FOCAL_PX = 910.0
CAR_HEIGHT_M = 1.5


def _rng() -> np.random.RandomState:
    return np.random.RandomState(SEED)


def _is_spd(matrix: np.ndarray, tol: float = 1e-9) -> bool:
    if not np.allclose(matrix, matrix.T, atol=tol):
        return False
    return bool(np.linalg.eigvalsh(matrix).min() > -tol)


# --------------------------------------------------------------------------- #
# Generic filter
# --------------------------------------------------------------------------- #


def test_kalman_rejects_mismatched_covariance():
    with pytest.raises(ValidationError):
        KalmanFilter(x=[0.0, 0.0], P=np.eye(3))


def test_kalman_rejects_non_finite_state():
    with pytest.raises(ValidationError):
        KalmanFilter(x=[float("nan")], P=np.eye(1))


def test_kalman_update_is_the_textbook_scalar_result():
    """One scalar state, one scalar measurement: the posterior is analytic."""
    kf = KalmanFilter(x=[0.0], P=np.array([[4.0]]))
    kf.update(np.array([10.0]), np.array([[1.0]]))
    # posterior mean = P/(P+R) * z = 4/5 * 10 = 8; posterior var = P*R/(P+R) = 0.8
    assert kf.x[0] == pytest.approx(8.0)
    assert kf.P[0, 0] == pytest.approx(0.8)


def test_kalman_update_matches_the_general_h_formulation():
    """The leading-block form must equal the textbook ``H``-matrix update.

    ``KalmanFilter`` drops the general measurement matrix for speed; this pins
    that the arithmetic is unchanged for the ``H = [I_k | 0]`` case it supports.
    """
    x0 = np.array([1.0, -2.0, 0.5, 3.0])
    P0 = np.array(
        [
            [4.0, 0.5, 0.1, 0.0],
            [0.5, 3.0, 0.0, 0.2],
            [0.1, 0.0, 2.0, 0.3],
            [0.0, 0.2, 0.3, 1.0],
        ]
    )
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    R = np.diag([0.7, 1.3])
    z = np.array([2.5, -1.0])

    # Textbook reference, computed independently of the implementation.
    y_ref = z - H.dot(x0)
    S_ref = H.dot(P0).dot(H.T) + R
    K_ref = P0.dot(H.T).dot(np.linalg.inv(S_ref))
    x_ref = x0 + K_ref.dot(y_ref)
    I_KH = np.eye(4) - K_ref.dot(H)
    P_ref = I_KH.dot(P0).dot(I_KH.T) + K_ref.dot(R).dot(K_ref.T)

    kf = KalmanFilter(x=x0, P=P0)
    y, S = kf.update(z, R)
    assert np.allclose(y, y_ref)
    assert np.allclose(S, S_ref)
    assert np.allclose(kf.x, x_ref)
    assert np.allclose(kf.P, P_ref)


def test_kalman_rejects_a_measurement_it_cannot_interpret():
    kf = KalmanFilter(x=[0.0, 0.0], P=np.eye(2))
    with pytest.raises(ValidationError):
        kf.update(np.array([1.0, 2.0, 3.0]), np.eye(3))
    with pytest.raises(ValidationError):
        kf.update(np.array([1.0]), np.eye(2))


def test_kalman_covariance_stays_spd_over_many_updates():
    kf = KalmanFilter(x=[0.0, 0.0], P=np.diag([100.0, 100.0]))
    R = np.array([[0.25]])
    rng = _rng()
    for step in range(500):
        dt = 0.05
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.diag([1e-4, 1e-2])
        kf.predict(F, Q)
        kf.update(np.array([0.5 * step + rng.normal(0.0, 0.5)]), R)
        assert _is_spd(kf.P), "covariance lost positive-definiteness at step %d" % step


def test_kalman_mahalanobis_matches_the_hand_computation():
    kf = KalmanFilter(x=[0.0, 0.0], P=np.diag([1.0, 1.0]))
    R = np.diag([1.0, 1.0])
    # S = 2I, so maha2 = (3^2 + 4^2) / 2 = 12.5
    assert kf.mahalanobis2(np.array([3.0, 4.0]), R) == pytest.approx(12.5)


def test_kalman_mahalanobis_is_inf_for_a_singular_covariance():
    kf = KalmanFilter(x=[0.0], P=np.array([[0.0]]))
    assert math.isinf(kf.mahalanobis2(np.array([1.0]), np.array([[0.0]])))


def test_kalman_update_raises_on_a_singular_innovation_covariance():
    kf = KalmanFilter(x=[0.0], P=np.array([[0.0]]))
    with pytest.raises(FilterError):
        kf.update(np.array([1.0]), np.array([[0.0]]))


# --------------------------------------------------------------------------- #
# Range model helpers
# --------------------------------------------------------------------------- #


def test_range_transition_is_constant_acceleration():
    F = range_transition(0.1)
    x = np.array([10.0, -5.0, 2.0])
    y = F @ x
    assert y[0] == pytest.approx(10.0 - 0.5 + 0.5 * 2.0 * 0.01)
    assert y[1] == pytest.approx(-5.0 + 0.2)
    assert y[2] == pytest.approx(2.0)


def test_range_process_noise_is_spd_and_scales_with_dt():
    small = range_process_noise(0.02, 9.0)
    large = range_process_noise(0.20, 9.0)
    assert _is_spd(small) and _is_spd(large)
    assert large[2, 2] > small[2, 2]
    assert large[0, 0] > small[0, 0]


def test_range_measurement_sigma_grows_with_the_square_of_range():
    """This is why the filter can trust a near object and distrust a far one."""
    near = range_measurement_sigma_m(20.0, FOCAL_PX, CAR_HEIGHT_M)
    far = range_measurement_sigma_m(100.0, FOCAL_PX, CAR_HEIGHT_M)
    assert near == pytest.approx((400.0 / (CAR_HEIGHT_M * FOCAL_PX)) * 1.5, rel=1e-9)
    assert far / near == pytest.approx(25.0, rel=1e-6)
    # Below the floor the model would claim centimetre accuracy from a detector
    # box; it is clamped instead.
    assert range_measurement_sigma_m(5.0, FOCAL_PX, CAR_HEIGHT_M) == 0.15


def test_range_measurement_sigma_is_inflated_by_truncation_and_low_confidence():
    base = range_measurement_sigma_m(20.0, FOCAL_PX, CAR_HEIGHT_M, confidence=1.0)
    truncated = range_measurement_sigma_m(20.0, FOCAL_PX, CAR_HEIGHT_M, confidence=1.0, truncated=True)
    weak = range_measurement_sigma_m(20.0, FOCAL_PX, CAR_HEIGHT_M, confidence=0.4)
    assert truncated == pytest.approx(5.0 * base)
    assert weak == pytest.approx(base / 0.4)


def test_range_measurement_sigma_degrades_safely_on_bad_geometry():
    assert range_measurement_sigma_m(10.0, 0.0, CAR_HEIGHT_M) == 60.0
    assert range_measurement_sigma_m(float("nan"), FOCAL_PX, CAR_HEIGHT_M) == 60.0
    assert range_measurement_sigma_m(1e9, FOCAL_PX, CAR_HEIGHT_M) == 60.0


# --------------------------------------------------------------------------- #
# RangeFilter
# --------------------------------------------------------------------------- #


def test_range_filter_rejects_bad_construction():
    with pytest.raises(ValidationError):
        RangeFilter(-1.0, 1.0)
    with pytest.raises(ValidationError):
        RangeFilter(10.0, 0.0)
    with pytest.raises(ValidationError):
        RangeFilter(10.0, 1.0, jerk_psd=0.0)


def test_range_filter_recovers_a_constant_closing_rate_from_pixel_noise():
    """The headline requirement. Ground truth is 8 m/s of closing.

    Measurements are the pinhole range of a *quantised* box height, which is
    exactly the noise process ADAS-DEC-06 identified: at 30 m one pixel is
    0.66 m, and raw differencing at dt=0.05 turns that into 13 m/s.
    """
    dt = 0.05
    true_rate = -8.0
    d = 40.0
    height_px = round(CAR_HEIGHT_M * FOCAL_PX / d)
    z0 = CAR_HEIGHT_M * FOCAL_PX / height_px
    filt = RangeFilter(z0, range_measurement_sigma_m(z0, FOCAL_PX, CAR_HEIGHT_M))

    worst_naive = 0.0
    previous_z = z0
    for _ in range(60):
        d += true_rate * dt
        height_px = round(CAR_HEIGHT_M * FOCAL_PX / d)
        z = CAR_HEIGHT_M * FOCAL_PX / height_px
        worst_naive = max(worst_naive, abs((previous_z - z) / dt - 8.0))
        previous_z = z
        filt.predict(dt)
        filt.update(z, range_measurement_sigma_m(z, FOCAL_PX, CAR_HEIGHT_M))

    assert filt.distance_m == pytest.approx(d, abs=0.5)
    assert filt.closing_mps == pytest.approx(8.0, abs=0.6)
    assert filt.rate_mps == pytest.approx(-8.0, abs=0.6)
    # The unfiltered differencing the old tracker used is far worse on the very
    # same measurement sequence. This is the regression guard.
    assert worst_naive > 3.0 * abs(filt.closing_mps - 8.0)


def test_range_filter_tracks_a_decelerating_lead_which_is_why_it_is_constant_acceleration():
    """A CV filter lags by a*tau throughout a braking event; a CA filter does not."""
    dt = 0.05
    d = 40.0
    rate = -12.0
    accel = 4.0  # the lead brakes, so the range rate returns toward zero
    filt = RangeFilter(d, range_measurement_sigma_m(d, FOCAL_PX, CAR_HEIGHT_M))
    rng = _rng()
    for _ in range(60):
        rate = min(0.0, rate + accel * dt)
        d = max(1.0, d + rate * dt)
        z = max(0.5, d + rng.normal(0.0, 0.05))
        filt.predict(dt)
        filt.update(z, range_measurement_sigma_m(z, FOCAL_PX, CAR_HEIGHT_M))
    assert filt.distance_m == pytest.approx(d, abs=0.6)
    assert filt.rate_mps == pytest.approx(rate, abs=1.0)
    assert filt.accel_mps2 > 0.5, "the acceleration state must pick up the braking"


def test_range_filter_handles_variable_dt():
    """The pipeline rate is not constant; a filter that assumes it is drifts."""
    rng = _rng()
    d = 50.0
    filt = RangeFilter(d, 0.5)
    total = 0.0
    for _ in range(120):
        dt = float(rng.uniform(0.02, 0.20))
        total += dt
        d -= 3.0 * dt
        filt.predict(dt)
        filt.update(max(0.1, d + rng.normal(0.0, 0.2)), 0.5)
    assert d > 5.0, "scenario must stay in front of the camera"
    assert filt.distance_m == pytest.approx(d, abs=0.5)
    assert filt.closing_mps == pytest.approx(3.0, abs=0.6)
    assert total > 0.0


def test_range_filter_predict_ignores_non_positive_dt():
    filt = RangeFilter(20.0, 0.5)
    before = filt.state
    filt.predict(0.0)
    filt.predict(-1.0)
    filt.predict(float("nan"))
    assert np.allclose(filt.state, before)


def test_range_filter_gates_an_outlier():
    """The 68 m -> 6.8 m teleport of ADAS-DEC-07 must be far outside the gate."""
    filt = RangeFilter(68.3, range_measurement_sigma_m(68.3, FOCAL_PX, CAR_HEIGHT_M))
    for _ in range(10):
        filt.predict(0.05)
        filt.update(68.3, range_measurement_sigma_m(68.3, FOCAL_PX, CAR_HEIGHT_M))
    filt.predict(0.05)
    maha2 = filt.gate(6.8, range_measurement_sigma_m(6.8, FOCAL_PX, CAR_HEIGHT_M))
    assert maha2 > 100.0


def test_range_filter_reinitialise_discards_the_rate():
    filt = RangeFilter(40.0, 0.5)
    for _ in range(20):
        filt.predict(0.05)
        filt.update(40.0 - 0.4 * (_ + 1), 0.5)
    assert filt.closing_mps > 3.0
    filt.reinitialise(10.0, 0.3)
    assert filt.distance_m == pytest.approx(10.0)
    assert filt.closing_mps == pytest.approx(0.0)
    assert filt.rate_variance_m2s2 >= 15.0 ** 2


def test_range_filter_saturates_impossible_states_and_says_so():
    filt = RangeFilter(100.0, 0.5)
    for _ in range(30):
        filt.predict(0.05)
        # Alternating extremes: no real object does this, so the filter is
        # driven outside the physical envelope on purpose.
        filt.update(0.0 if _ % 2 else 200.0, 0.05)
    assert abs(filt.rate_mps) <= filt.max_speed_mps
    assert abs(filt.accel_mps2) <= filt.max_accel_mps2
    assert filt.saturations > 0
    assert filt.distance_m >= 0.0


def test_range_filter_rejects_a_bad_measurement():
    filt = RangeFilter(10.0, 0.5)
    with pytest.raises(ValidationError):
        filt.update(-1.0, 0.5)
    with pytest.raises(ValidationError):
        filt.update(10.0, 0.0)
    with pytest.raises(ValidationError):
        filt.update(float("inf"), 0.5)


# --------------------------------------------------------------------------- #
# BoxFilter
# --------------------------------------------------------------------------- #


def _box(cx: float, cy: float, w: float, h: float):
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def test_box_filter_rejects_a_degenerate_box():
    with pytest.raises(ValidationError):
        BoxFilter((10.0, 10.0, 10.0, 20.0))
    with pytest.raises(ValidationError):
        BoxFilter((10.0, 10.0, 20.0, float("nan")))


def test_box_filter_learns_a_constant_velocity():
    filt = BoxFilter(_box(100.0, 300.0, 80.0, 100.0))
    for step in range(1, 25):
        filt.predict(0.05)
        filt.update(_box(100.0 + 10.0 * step, 300.0, 80.0, 100.0))
    vx, vy = filt.velocity_px_s
    assert vx == pytest.approx(200.0, rel=0.1)
    assert abs(vy) < 20.0


def test_box_filter_prediction_leads_the_measurement():
    filt = BoxFilter(_box(100.0, 300.0, 80.0, 100.0))
    for step in range(1, 15):
        filt.predict(0.05)
        filt.update(_box(100.0 + 10.0 * step, 300.0, 80.0, 100.0))
    filt.predict(0.05)
    cx, _cy = filt.centre_px
    assert cx > 100.0 + 10.0 * 14, "a coasting prediction must move, not freeze"


def test_box_filter_gate_is_tight_for_a_small_far_box_and_loose_for_a_large_near_one():
    """The whole point of replacing the fixed 120 px radius (ADAS-DEC-08).

    Two adjacent-lane vehicles at 100 m are 32 px apart; the gate must not span
    that. A cut-in at 8 m moves hundreds of pixels; the gate must.
    """
    far = BoxFilter(_box(640.0, 360.0, 16.0, 14.0))
    near = BoxFilter(_box(640.0, 360.0, 205.0, 171.0))
    for _ in range(10):
        far.predict(0.05)
        far.update(_box(640.0, 360.0, 16.0, 14.0))
        near.predict(0.05)
        near.update(_box(640.0, 360.0, 205.0, 171.0))
    far.predict(0.05)
    near.predict(0.05)
    far_radius = far.centre_gate_radius_px(CHI2_99[2])
    near_radius = near.centre_gate_radius_px(CHI2_99[2])
    assert far_radius < 32.0, "far gate %.1f px would merge adjacent lanes" % far_radius
    assert near_radius > 3.0 * far_radius


def test_box_filter_gate_widens_while_coasting():
    filt = BoxFilter(_box(640.0, 360.0, 100.0, 100.0))
    for _ in range(10):
        filt.predict(0.05)
        filt.update(_box(640.0, 360.0, 100.0, 100.0))
    filt.predict(0.05)
    fresh = filt.centre_gate_radius_px(CHI2_99[2])
    for _ in range(4):
        filt.predict(0.05)
    coasted = filt.centre_gate_radius_px(CHI2_99[2])
    assert coasted > fresh


def test_box_filter_batch_gate_matches_the_scalar_gate():
    filt = BoxFilter(_box(640.0, 360.0, 100.0, 120.0))
    filt.predict(0.05)
    cx = np.array([640.0, 700.0, 500.0, 641.0])
    cy = np.array([360.0, 380.0, 300.0, 359.0])
    h = np.array([120.0, 118.0, 130.0, 121.0])
    batch = filt.gate_centre_batch(cx, cy, h)
    scalar = [filt.gate_centre(cx[i], cy[i], h[i]) for i in range(4)]
    assert np.allclose(batch, scalar, rtol=1e-9, atol=1e-9)


def test_box_filter_batch_gate_rejects_mismatched_lengths():
    filt = BoxFilter(_box(640.0, 360.0, 100.0, 120.0))
    with pytest.raises(ValidationError):
        filt.gate_centre_batch(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0]))


def test_box_filter_batch_gate_is_inf_for_non_finite_candidates():
    filt = BoxFilter(_box(640.0, 360.0, 100.0, 120.0))
    out = filt.gate_centre_batch(
        np.array([float("nan")]), np.array([360.0]), np.array([120.0])
    )
    assert math.isinf(out[0])


def test_box_filter_size_never_collapses_while_coasting():
    filt = BoxFilter(_box(640.0, 360.0, 40.0, 30.0))
    for _ in range(200):
        filt.predict(0.05)
    w, h = filt.size_px
    assert w >= BoxFilter.MIN_SIZE_PX and h >= BoxFilter.MIN_SIZE_PX
    x1, y1, x2, y2 = filt.box_xyxy()
    assert x2 > x1 and y2 > y1


def test_box_filter_exposes_the_centre_covariance_and_the_size_variances():
    """Size is two scalar random walks, not part of the coupled state."""
    filt = BoxFilter(_box(640.0, 360.0, 100.0, 120.0))
    assert filt.covariance.shape == (4, 4)
    var_w, var_h = filt.size_variance_px2
    assert var_w > 0.0 and var_h > 0.0
    filt.predict(0.5)
    grown_w, grown_h = filt.size_variance_px2
    assert grown_w > var_w and grown_h > var_h
    filt.update(_box(640.0, 360.0, 100.0, 120.0))
    shrunk_w, shrunk_h = filt.size_variance_px2
    assert shrunk_w < grown_w and shrunk_h < grown_h


def test_box_filter_state_keeps_the_documented_layout():
    filt = BoxFilter(_box(640.0, 360.0, 100.0, 120.0))
    state = filt.state
    assert state.shape == (6,)
    assert state[0] == pytest.approx(640.0)
    assert state[1] == pytest.approx(360.0)
    assert state[2] == pytest.approx(100.0)
    assert state[3] == pytest.approx(120.0)
    assert state[4] == pytest.approx(0.0)
    assert state[5] == pytest.approx(0.0)


def test_box_filter_covariance_stays_spd():
    rng = _rng()
    filt = BoxFilter(_box(640.0, 360.0, 100.0, 120.0))
    for _ in range(300):
        filt.predict(float(rng.uniform(0.02, 0.2)))
        filt.update(
            _box(
                640.0 + rng.normal(0.0, 3.0),
                360.0 + rng.normal(0.0, 3.0),
                100.0 + rng.normal(0.0, 2.0),
                120.0 + rng.normal(0.0, 2.0),
            )
        )
        assert _is_spd(filt.covariance)
