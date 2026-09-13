"""Tests for the ground-plane camera model (CPU only, no engine required)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from adas.core.exceptions import ConfigurationError
from adas.core.models import BoundingBox, RangeSource
from adas.perception.geometry import (
    MAX_CURVATURE_RADIUS_M,
    MAX_TRUSTED_RANGE_M,
    UNCALIBRATED_CONFIDENCE_CAP,
    BoxTruncation,
    CameraConfig,
    LaneCalibrator,
    calibrate_from_lane_observations,
    calibrate_mount_height_from_lane,
    curvature_radius_m,
    default_camera,
    estimate_range,
    estimate_vanishing_point,
    fit_ground_polyline,
    fuse_range,
    ground_plane_range,
    lane_geometry,
    pinhole_range,
    truncation_flags,
)


def make_camera(pitch_deg: float = 2.0, calibrated: bool = True) -> CameraConfig:
    return CameraConfig(
        image_width=1280,
        image_height=720,
        fx=910.0,
        fy=910.0,
        cx=640.0,
        cy=360.0,
        mount_height_m=1.30,
        pitch_deg=pitch_deg,
        calibrated=calibrated,
        label="test",
    )


# --------------------------------------------------------------------------- #
# Construction and validation
# --------------------------------------------------------------------------- #


def test_zero_mount_height_is_rejected():
    with pytest.raises(ConfigurationError):
        CameraConfig(mount_height_m=0.0)


def test_negative_focal_is_rejected():
    with pytest.raises(ConfigurationError):
        CameraConfig(fx=-1.0)


def test_extreme_pitch_is_rejected():
    with pytest.raises(ConfigurationError):
        CameraConfig(pitch_deg=90.0)


def test_non_finite_principal_point_is_rejected():
    with pytest.raises(ConfigurationError):
        CameraConfig(cx=float("nan"))


def test_from_fov_matches_the_geometry_it_claims():
    cam = CameraConfig.from_fov(1280, 720, 70.0)
    recovered = 2.0 * math.degrees(math.atan((1280 / 2.0) / cam.fx))
    assert recovered == pytest.approx(70.0, abs=1e-6)
    assert cam.calibrated is False


def test_from_fov_rejects_impossible_fov():
    with pytest.raises(ConfigurationError):
        CameraConfig.from_fov(1280, 720, 180.0)


def test_default_camera_is_flagged_uncalibrated():
    cam = default_camera()
    assert cam.calibrated is False
    assert cam.fx == pytest.approx(910.0)


# --------------------------------------------------------------------------- #
# Horizon
# --------------------------------------------------------------------------- #


def test_horizon_is_the_principal_row_when_pitch_is_zero():
    cam = make_camera(pitch_deg=0.0)
    assert cam.horizon_y_px == pytest.approx(cam.cy)


def test_nose_down_pitch_raises_the_horizon_in_the_image():
    assert make_camera(pitch_deg=5.0).horizon_y_px < make_camera(pitch_deg=0.0).horizon_y_px


def test_pixels_at_or_above_the_horizon_do_not_project():
    cam = make_camera()
    horizon = cam.horizon_y_px
    assert cam.image_to_ground(640.0, horizon) is None
    assert cam.image_to_ground(640.0, horizon - 5.0) is None
    assert cam.image_to_ground(640.0, horizon + 40.0) is not None


def test_range_grows_without_bound_as_the_horizon_is_approached():
    cam = make_camera()
    near = cam.image_to_ground(640.0, 700.0)
    far = cam.image_to_ground(640.0, cam.horizon_y_px + 10.0)
    assert near is not None and far is not None
    assert far[1] > near[1] * 10.0


def test_with_horizon_px_inverts_the_horizon_property():
    cam = make_camera()
    for target in (300.0, 360.0, 420.0, 500.0):
        assert cam.with_horizon_px(target).horizon_y_px == pytest.approx(target, abs=1e-6)


# --------------------------------------------------------------------------- #
# Homography round trips
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("pitch", [-6.0, 0.0, 2.0, 12.0])
def test_image_ground_round_trip_is_exact(pitch):
    cam = make_camera(pitch_deg=pitch)
    for u in (10.0, 320.0, 640.0, 1270.0):
        for v in (cam.horizon_y_px + 30.0, 500.0, 719.0):
            ground = cam.image_to_ground(u, v)
            if ground is None:
                continue
            back = cam.ground_to_image(*ground)
            assert back is not None
            assert back[0] == pytest.approx(u, abs=1e-6)
            assert back[1] == pytest.approx(v, abs=1e-6)


@pytest.mark.parametrize("pitch", [0.0, 3.0])
def test_ground_image_round_trip_is_exact(pitch):
    cam = make_camera(pitch_deg=pitch)
    for x_m in (-5.0, 0.0, 1.75, 8.0):
        for z_m in (3.0, 12.0, 60.0, 150.0):
            pixel = cam.ground_to_image(x_m, z_m)
            assert pixel is not None
            back = cam.image_to_ground(*pixel)
            assert back is not None
            assert back[0] == pytest.approx(x_m, abs=1e-6)
            assert back[1] == pytest.approx(z_m, abs=1e-6)


def test_homography_matrices_are_mutual_inverses():
    cam = make_camera()
    identity = cam.ground_to_image_matrix @ cam.image_to_ground_matrix
    assert np.allclose(identity / identity[2, 2], np.eye(3), atol=1e-9)


def test_principal_column_maps_to_zero_lateral_offset():
    cam = make_camera()
    ground = cam.image_to_ground(cam.cx, 700.0)
    assert ground is not None
    assert ground[0] == pytest.approx(0.0, abs=1e-9)


def test_ranges_beyond_the_trust_horizon_are_refused():
    cam = make_camera()
    assert cam.image_to_ground(640.0, cam.horizon_y_px + 0.05) is None


def test_project_points_drops_unprojectable_pixels():
    cam = make_camera()
    pts = [(640.0, 200.0), (640.0, 500.0), (640.0, 700.0)]
    ground = cam.project_points(pts)
    assert len(ground) == 2
    assert all(z <= MAX_TRUSTED_RANGE_M for _x, z in ground)


def test_scaled_to_preserves_the_projected_world_point():
    cam = make_camera()
    big = cam.scaled_to(2560, 1440)
    a = cam.image_to_ground(700.0, 600.0)
    b = big.image_to_ground(1400.0, 1200.0)
    assert a is not None and b is not None
    assert b[0] == pytest.approx(a[0], rel=1e-9)
    assert b[1] == pytest.approx(a[1], rel=1e-9)


def test_scaled_to_is_identity_for_the_same_size():
    cam = make_camera()
    assert cam.scaled_to(1280, 720) is cam


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #


def test_truncation_flags_detect_each_edge():
    box = BoundingBox(0.0, 0.0, 1279.0, 719.0, 0.9, "car")
    trunc = truncation_flags(box, 1280, 720)
    assert trunc.left and trunc.right and trunc.top and trunc.bottom
    assert trunc.any_edge and trunc.vertical


def test_default_truncation_is_all_clear():
    assert BoxTruncation().any_edge is False
    assert BoxTruncation(top=True).vertical is True
    assert BoxTruncation(left=True).vertical is False


def test_interior_box_is_not_truncated():
    box = BoundingBox(100.0, 200.0, 300.0, 400.0, 0.9, "car")
    trunc = truncation_flags(box, 1280, 720)
    assert not trunc.any_edge
    assert not trunc.vertical


# --------------------------------------------------------------------------- #
# Range models
# --------------------------------------------------------------------------- #


def test_pinhole_range_matches_the_closed_form():
    cam = make_camera()
    box = BoundingBox(600.0, 400.0, 700.0, 500.0, 0.9, "car")
    est = pinhole_range(box, cam)
    assert est.source is RangeSource.PINHOLE
    assert est.distance_m == pytest.approx(1.5 * 910.0 / 100.0, rel=1e-9)


def test_pinhole_range_uses_the_class_height_not_one_constant():
    cam = make_camera()
    geometry = (600.0, 400.0, 700.0, 500.0)
    car = pinhole_range(BoundingBox(*geometry, 0.9, "car"), cam)
    truck = pinhole_range(BoundingBox(*geometry, 0.9, "truck"), cam)
    assert truck.distance_m > car.distance_m * 2.0


def test_pinhole_range_on_a_degenerate_box_is_unavailable():
    cam = make_camera()
    est = pinhole_range(BoundingBox(10.0, 10.0, 20.0, 10.0, 0.9, "car"), cam)
    assert est.source is RangeSource.UNAVAILABLE
    assert est.confidence == 0.0


def test_ground_plane_range_recovers_a_planted_distance():
    cam = make_camera()
    # Put a 1.5 m tall car with its wheels at exactly 25 m.
    bottom = cam.ground_to_image(0.0, 25.0)
    top = cam.ground_to_image(0.0, 25.0)
    assert bottom is not None and top is not None
    box = BoundingBox(600.0, bottom[1] - 55.0, 680.0, bottom[1], 0.9, "car")
    est = ground_plane_range(box, cam, 1280, 720)
    assert est.source is RangeSource.HOMOGRAPHY
    assert est.distance_m == pytest.approx(25.0, rel=1e-6)


def test_ground_plane_range_above_the_horizon_is_unavailable():
    cam = make_camera()
    box = BoundingBox(600.0, 100.0, 700.0, cam.horizon_y_px - 10.0, 0.9, "car")
    est = ground_plane_range(box, cam, 1280, 720)
    assert est.source is RangeSource.UNAVAILABLE
    assert est.confidence == 0.0


def test_uncalibrated_camera_caps_range_confidence():
    cam = make_camera(calibrated=False)
    box = BoundingBox(600.0, 500.0, 700.0, 700.0, 0.9, "car")
    assert pinhole_range(box, cam).confidence <= UNCALIBRATED_CONFIDENCE_CAP
    assert ground_plane_range(box, cam, 1280, 720).confidence <= UNCALIBRATED_CONFIDENCE_CAP


def test_top_truncated_box_falls_back_to_the_ground_plane():
    """A box clipped at the TOP shrinks, so H*f/h reads far. The contact point is
    still visible, so the homography is unaffected and must win."""
    cam = make_camera()
    z_true = 30.0
    contact = cam.ground_to_image(0.0, z_true)
    assert contact is not None
    height_px = 1.5 * cam.fy / z_true
    intact = BoundingBox(600.0, contact[1] - height_px, 700.0, contact[1], 0.9, "car")
    # Same vehicle, top half hidden behind the A-pillar / frame edge.
    clipped = BoundingBox(600.0, 0.0, 700.0, contact[1], 0.9, "car")

    naive_intact = pinhole_range(intact, cam)
    naive_clipped = pinhole_range(clipped, cam)
    assert naive_intact.distance_m == pytest.approx(z_true, rel=0.02)
    # The clipped box is TALLER here (it reaches row 0), so pinhole reads near --
    # what matters is that the value is no longer the true range.
    assert abs(naive_clipped.distance_m - z_true) > 1.0

    est = estimate_range(clipped, cam, 1280, 720)
    assert est.truncated
    assert est.source is RangeSource.HOMOGRAPHY
    assert est.distance_m == pytest.approx(z_true, rel=1e-6)


def test_bottom_truncated_box_reports_a_bounded_upper_bound():
    """A vehicle so close its wheels leave the frame. Both monocular models
    over-read; the homography's error is at least BOUNDED by the range of the
    bottom image row, so that is what is published, flagged truncated."""
    cam = make_camera()
    z_true = 2.2
    contact = cam.ground_to_image(0.0, z_true)
    assert contact is not None and contact[1] > 719.0  # genuinely off the bottom
    height_px = 1.5 * cam.fy / z_true
    clipped = BoundingBox(500.0, contact[1] - height_px, 780.0, 719.0, 0.9, "car")

    est = estimate_range(clipped, cam, 1280, 720)
    assert est.truncated
    assert est.source is RangeSource.HOMOGRAPHY
    # Upper bound: the true vehicle is nearer than the bottom row of the image.
    bottom_row_range = cam.image_to_ground(640.0, 719.0)
    assert bottom_row_range is not None
    assert est.distance_m == pytest.approx(bottom_row_range[1], rel=1e-6)
    assert est.distance_m > z_true
    # And the confidence has been knocked down so nothing treats it as a fix.
    assert est.confidence < 0.5


def test_truncated_estimate_with_no_ground_solution_is_unavailable():
    cam = make_camera()
    # Box clipped at the top whose bottom edge is still above the horizon.
    clipped = BoundingBox(600.0, 0.0, 700.0, cam.horizon_y_px - 20.0, 0.9, "car")
    est = estimate_range(clipped, cam, 1280, 720)
    assert est.source is RangeSource.UNAVAILABLE
    assert est.confidence == 0.0
    assert est.truncated


def test_fuse_range_ignores_unavailable_inputs():
    good = pinhole_range(BoundingBox(600.0, 400.0, 700.0, 500.0, 0.9, "car"), make_camera())
    bad = pinhole_range(BoundingBox(10.0, 10.0, 20.0, 10.0, 0.9, "car"), make_camera())
    fused = fuse_range(good, bad)
    assert fused is good


def test_fuse_range_with_nothing_usable_is_unavailable():
    fused = fuse_range()
    assert fused.source is RangeSource.UNAVAILABLE
    assert fused.confidence == 0.0


def test_fuse_range_lands_between_its_inputs():
    cam = make_camera()
    contact = cam.ground_to_image(0.0, 20.0)
    assert contact is not None
    box = BoundingBox(600.0, contact[1] - 68.0, 700.0, contact[1], 0.9, "car")
    ground = ground_plane_range(box, cam, 1280, 720)
    pinhole = pinhole_range(box, cam)
    fused = fuse_range(ground, pinhole)
    assert fused.source is RangeSource.FUSED
    lo, hi = sorted((ground.distance_m, pinhole.distance_m))
    assert lo - 1e-6 <= fused.distance_m <= hi + 1e-6


def test_estimate_range_is_consistent_for_a_synthetic_car():
    """Plant a 1.5 m car at 30 m and check both models agree to a few percent."""
    cam = make_camera()
    bottom = cam.ground_to_image(0.0, 30.0)
    assert bottom is not None
    height_px = 1.5 * cam.fy / 30.0
    box = BoundingBox(600.0, bottom[1] - height_px, 700.0, bottom[1], 0.9, "car")
    est = estimate_range(box, cam, 1280, 720)
    assert est.source is RangeSource.FUSED
    assert est.distance_m == pytest.approx(30.0, rel=0.1)


# --------------------------------------------------------------------------- #
# Lane fitting on the ground plane
# --------------------------------------------------------------------------- #


def project_lane(cam: CameraConfig, x_m: float, z_range=(4.0, 40.0), n: int = 25):
    """Pixels of a straight ground line at constant lateral offset ``x_m``."""
    out = []
    for z in np.linspace(z_range[0], z_range[1], n):
        pixel = cam.ground_to_image(x_m, float(z))
        if pixel is not None:
            out.append(pixel)
    return out


def test_fit_ground_polyline_recovers_a_straight_line():
    cam = make_camera()
    fit = fit_ground_polyline(project_lane(cam, -1.8), cam)
    assert fit is not None
    assert fit.x_at(10.0) == pytest.approx(-1.8, abs=1e-3)
    assert fit.coeffs[0] == pytest.approx(0.0, abs=1e-5)
    assert fit.rms_residual_m < 1e-3


def test_fit_ground_polyline_recovers_a_curve():
    cam = make_camera()
    pts = []
    for z in np.linspace(4.0, 40.0, 30):
        x = 0.001 * z * z - 1.8
        pixel = cam.ground_to_image(float(x), float(z))
        if pixel is not None:
            pts.append(pixel)
    fit = fit_ground_polyline(pts, cam)
    assert fit is not None
    assert fit.coeffs[0] == pytest.approx(0.001, rel=0.05)


def test_fit_ground_polyline_needs_enough_range_span():
    cam = make_camera()
    pts = project_lane(cam, -1.8, z_range=(10.0, 10.4), n=10)
    assert fit_ground_polyline(pts, cam) is None


def test_fit_ground_polyline_returns_none_for_too_few_points():
    cam = make_camera()
    assert fit_ground_polyline(project_lane(cam, -1.8, n=2), cam) is None


def test_curvature_radius_of_a_straight_lane_is_capped():
    assert curvature_radius_m((0.0, 0.0, -1.8)) == MAX_CURVATURE_RADIUS_M


def test_curvature_radius_matches_the_analytic_value():
    # x = a z^2 with a = 0.002 -> R = 1 / |2a| = 250 m at z = 0.
    assert curvature_radius_m((0.002, 0.0, 0.0), 0.0) == pytest.approx(250.0, rel=1e-9)


def test_lane_geometry_recovers_width_and_offset():
    cam = make_camera()
    left = fit_ground_polyline(project_lane(cam, -1.8), cam)
    right = fit_ground_polyline(project_lane(cam, 1.8), cam)
    geom = lane_geometry(left, right, eval_range_m=10.0)
    assert geom is not None
    assert geom.lane_width_m == pytest.approx(3.6, abs=0.01)
    assert geom.lateral_offset_m == pytest.approx(0.0, abs=0.01)
    assert geom.both_boundaries and geom.plausible


def test_lane_geometry_signs_offset_positive_when_the_vehicle_is_right_of_centre():
    cam = make_camera()
    # Lane centred at X = -0.5 m: the vehicle (X = 0) sits 0.5 m to its right.
    left = fit_ground_polyline(project_lane(cam, -2.3), cam)
    right = fit_ground_polyline(project_lane(cam, 1.3), cam)
    geom = lane_geometry(left, right, eval_range_m=10.0)
    assert geom is not None
    assert geom.lateral_offset_m == pytest.approx(0.5, abs=0.01)


def test_lane_geometry_never_extrapolates_below_its_support():
    cam = make_camera()
    left = fit_ground_polyline(project_lane(cam, -1.8, z_range=(12.0, 40.0)), cam)
    right = fit_ground_polyline(project_lane(cam, 1.8, z_range=(12.0, 40.0)), cam)
    geom = lane_geometry(left, right, eval_range_m=0.0)
    assert geom is not None
    assert geom.eval_range_m >= 11.9


def test_lane_geometry_with_one_boundary_flags_the_assumption():
    cam = make_camera()
    left = fit_ground_polyline(project_lane(cam, -1.8), cam)
    geom = lane_geometry(left, None, eval_range_m=10.0)
    assert geom is not None
    assert geom.both_boundaries is False
    assert geom.lane_width_m == 0.0


def test_lane_geometry_marks_an_impossible_width_implausible():
    cam = make_camera()
    left = fit_ground_polyline(project_lane(cam, -4.0), cam)
    right = fit_ground_polyline(project_lane(cam, 4.0), cam)
    geom = lane_geometry(left, right, eval_range_m=10.0)
    assert geom is not None
    assert geom.lane_width_m == pytest.approx(8.0, abs=0.05)
    assert geom.plausible is False


def test_lane_geometry_with_no_boundaries_is_none():
    assert lane_geometry(None, None) is None


# --------------------------------------------------------------------------- #
# Self-calibration
# --------------------------------------------------------------------------- #


def test_vanishing_point_recovers_the_synthetic_horizon():
    cam = make_camera(pitch_deg=4.0)
    left = project_lane(cam, -1.825, z_range=(5.0, 80.0), n=40)
    right = project_lane(cam, 1.825, z_range=(5.0, 80.0), n=40)
    vp = estimate_vanishing_point(left, right)
    assert vp is not None
    assert vp[1] == pytest.approx(cam.horizon_y_px, abs=0.5)
    assert vp[0] == pytest.approx(cam.cx, abs=1.0)


def test_vanishing_point_rejects_parallel_boundaries():
    # Two vertical image lines never meet.
    left = [(500.0, float(y)) for y in range(400, 700, 20)]
    right = [(700.0, float(y)) for y in range(400, 700, 20)]
    assert estimate_vanishing_point(left, right) is None


def test_vanishing_point_rejects_short_polylines():
    assert estimate_vanishing_point([(1.0, 2.0)], [(3.0, 4.0)]) is None


def test_mount_height_calibration_recovers_the_true_height():
    truth = make_camera(pitch_deg=3.0)
    truth = truth.with_mount_height(1.42)
    left = project_lane(truth, -1.825, z_range=(5.0, 40.0), n=30)
    right = project_lane(truth, 1.825, z_range=(5.0, 40.0), n=30)
    wrong = truth.with_mount_height(1.0)
    solved = calibrate_mount_height_from_lane(wrong, left, right, known_lane_width_m=3.65)
    assert solved is not None
    assert solved == pytest.approx(1.42, rel=1e-6)


def test_full_calibration_round_trip():
    truth = CameraConfig(
        1280, 720, 910.0, 910.0, 640.0, 360.0, 1.42, -3.5, calibrated=True, label="truth"
    )
    left = project_lane(truth, -1.825, z_range=(5.0, 60.0), n=40)
    right = project_lane(truth, 1.825, z_range=(5.0, 60.0), n=40)
    guess = CameraConfig(
        1280, 720, 910.0, 910.0, 640.0, 360.0, 1.0, 2.0, calibrated=False, label="guess"
    )
    solved = calibrate_from_lane_observations(guess, left, right, known_lane_width_m=3.65)
    assert solved is not None
    assert solved.calibrated is True
    assert solved.pitch_deg == pytest.approx(truth.pitch_deg, abs=0.05)
    assert solved.mount_height_m == pytest.approx(truth.mount_height_m, rel=0.01)


def test_calibrator_withholds_a_camera_until_it_has_samples():
    truth = CameraConfig(1280, 720, 910.0, 910.0, 640.0, 360.0, 1.42, -3.5, True, "truth")
    guess = CameraConfig(1280, 720, 910.0, 910.0, 640.0, 360.0, 1.0, 2.0, False, "guess")
    cal = LaneCalibrator(guess, known_lane_width_m=3.65, min_samples=10)
    left = project_lane(truth, -1.825, z_range=(5.0, 60.0), n=40)
    right = project_lane(truth, 1.825, z_range=(5.0, 60.0), n=40)
    for _ in range(5):
        assert cal.observe(left, right)
    assert cal.camera() is None
    for _ in range(5):
        cal.observe(left, right)
    solved = cal.camera()
    assert solved is not None
    assert solved.mount_height_m == pytest.approx(1.42, rel=0.01)


def test_calibrator_refuses_an_implausible_height():
    truth = CameraConfig(1280, 720, 910.0, 910.0, 640.0, 360.0, 1.42, -3.5, True, "truth")
    guess = CameraConfig(1280, 720, 910.0, 910.0, 640.0, 360.0, 1.0, 2.0, False, "guess")
    # Claim the lane is 30 m wide: the solved height goes far out of band.
    cal = LaneCalibrator(guess, known_lane_width_m=30.0, min_samples=4)
    left = project_lane(truth, -1.825, z_range=(5.0, 60.0), n=40)
    right = project_lane(truth, 1.825, z_range=(5.0, 60.0), n=40)
    for _ in range(4):
        cal.observe(left, right)
    assert cal.camera() is None


def test_calibrator_counts_rejected_observations():
    guess = CameraConfig(1280, 720, 910.0, 910.0, 640.0, 360.0, 1.0, 2.0, False, "guess")
    cal = LaneCalibrator(guess, min_samples=2)
    assert not cal.observe([(1.0, 2.0)], [(3.0, 4.0)])
    assert cal.rejected == 1
    assert cal.camera() is None
