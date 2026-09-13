"""Tests for detection-to-track association.

The Hungarian implementation is checked three ways: against a brute-force
optimum on small matrices, against ``scipy.optimize.linear_sum_assignment`` when
scipy is importable (it is, 1.3.3, on the target board -- the solver is
implemented here anyway so the safety path owns no extra dependency), and on the
specific shapes that broke the old greedy matcher.

The gating tests are written directly from the verified defects: ADAS-DEC-07's
68 m -> 6.8 m detection theft, ADAS-DEC-08's range-blind 120 px radius, and the
absent class gate that let a car associate with a bicycle.

Fixed seed throughout.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from adas.core.exceptions import ValidationError
from adas.tracking.association import (
    MAX_ASSIGNMENT_DIM,
    AssociationParams,
    build_cost_matrix,
    class_compatible,
    class_group,
    hungarian,
    iou_xyxy,
    solve_assignment,
)
from adas.tracking.kalman import CHI2_99

SEED = 20240913

try:  # pragma: no cover - depends on the environment
    from scipy.optimize import linear_sum_assignment as _scipy_lsa

    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_lsa = None
    HAVE_SCIPY = False


def _brute_force_optimum(cost: np.ndarray) -> float:
    """Minimum total cost over every maximum-cardinality assignment. Small n only."""
    n, m = cost.shape
    if n > m:
        return _brute_force_optimum(cost.T)
    best = math.inf
    for columns in itertools.permutations(range(m), n):
        total = sum(cost[i, columns[i]] for i in range(n))
        best = min(best, total)
    return best


def _total(cost: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    return float(sum(cost[r, c] for r, c in zip(rows.tolist(), cols.tolist())))


# --------------------------------------------------------------------------- #
# Hungarian solver
# --------------------------------------------------------------------------- #


def test_hungarian_empty_inputs():
    for shape in ((0, 0), (0, 3), (3, 0)):
        rows, cols = hungarian(np.zeros(shape))
        assert rows.size == 0 and cols.size == 0


def test_hungarian_rejects_a_non_2d_matrix():
    with pytest.raises(ValidationError):
        hungarian(np.zeros(4))


def test_hungarian_refuses_an_absurdly_large_problem():
    """A detector fault must fail loudly, not stall the control loop."""
    oversized = MAX_ASSIGNMENT_DIM + 1
    with pytest.raises(ValidationError):
        hungarian(np.zeros((oversized, 2)))
    with pytest.raises(ValidationError):
        hungarian(np.zeros((2, oversized)))


def test_hungarian_matches_brute_force_on_small_random_matrices():
    rng = np.random.RandomState(SEED)
    for _ in range(200):
        n = int(rng.randint(1, 6))
        m = int(rng.randint(1, 6))
        cost = rng.uniform(-5.0, 5.0, size=(n, m))
        rows, cols = hungarian(cost)
        assert rows.size == min(n, m)
        assert len(set(rows.tolist())) == rows.size
        assert len(set(cols.tolist())) == cols.size
        assert _total(cost, rows, cols) == pytest.approx(_brute_force_optimum(cost))


@pytest.mark.skipif(not HAVE_SCIPY, reason="scipy is not importable here")
def test_hungarian_matches_scipy_on_larger_matrices():
    rng = np.random.RandomState(SEED + 1)
    for _ in range(40):
        n = int(rng.randint(1, 25))
        m = int(rng.randint(1, 25))
        cost = rng.uniform(0.0, 100.0, size=(n, m))
        rows, cols = hungarian(cost)
        srows, scols = _scipy_lsa(cost)
        assert _total(cost, rows, cols) == pytest.approx(
            float(cost[srows, scols].sum()), rel=1e-9, abs=1e-9
        )


def test_hungarian_never_returns_an_inadmissible_pair():
    cost = np.array([[1.0, np.inf], [np.inf, np.inf]])
    rows, cols = hungarian(cost)
    assert rows.tolist() == [0] and cols.tolist() == [0]


def test_hungarian_prefers_a_complete_admissible_matching():
    """An ``inf`` must cost more than any achievable finite total."""
    cost = np.array(
        [
            [0.0, 900.0],
            [np.inf, 1.0],
        ]
    )
    rows, cols = hungarian(cost)
    assert sorted(zip(rows.tolist(), cols.tolist())) == [(0, 0), (1, 1)]


def test_hungarian_handles_an_all_inadmissible_matrix():
    rows, cols = hungarian(np.full((3, 3), np.inf))
    assert rows.size == 0 and cols.size == 0


def test_hungarian_is_not_order_dependent():
    """The defect ADAS-DEC-07 describes: the answer must not depend on row order."""
    rng = np.random.RandomState(SEED + 2)
    cost = rng.uniform(0.0, 1.0, size=(6, 6))
    rows, cols = hungarian(cost)
    baseline = _total(cost, rows, cols)
    order = rng.permutation(6)
    permuted = cost[order, :]
    prows, pcols = hungarian(permuted)
    assert _total(permuted, prows, pcols) == pytest.approx(baseline)


def test_hungarian_beats_greedy_on_the_teleport_matrix():
    """Greedy row-order matching picks 0->0 then 1->1 and pays 1.9; the optimum is 0.2."""
    cost = np.array(
        [
            [0.45, 0.10],  # old distant track: slightly prefers the near detection
            [0.80, 0.05],  # the real lead track: strongly prefers it
        ]
    )
    greedy_rows = [0, 1]
    greedy_cols = [1, 0]  # row 0 grabs its own best first, row 1 takes what is left
    greedy_total = cost[0, 1] + cost[1, 0]
    rows, cols = hungarian(cost)
    assert _total(cost, rows, cols) < greedy_total
    assert sorted(zip(rows.tolist(), cols.tolist())) == [(0, 0), (1, 1)]
    assert greedy_rows and greedy_cols  # documents what greedy would have done


# --------------------------------------------------------------------------- #
# solve_assignment
# --------------------------------------------------------------------------- #


def test_solve_assignment_accepts_plain_lists():
    """build_cost_matrix returns nested lists; the solver must take them as-is."""
    result = solve_assignment([[0.1, 5.0], [5.0, 0.2]], max_cost=1.0)
    assert result.matches == [(0, 0), (1, 1)]


def test_solve_assignment_reports_the_leftovers():
    cost = np.array([[0.1, 5.0, 5.0], [5.0, 0.2, 5.0]])
    result = solve_assignment(cost)
    assert result.matches == [(0, 0), (1, 1)]
    assert result.unmatched_rows == []
    assert result.unmatched_cols == [2]
    assert result.costs == pytest.approx([0.1, 0.2])


def test_solve_assignment_rejects_pairs_above_max_cost():
    cost = np.array([[0.1, 5.0], [5.0, 0.95]])
    result = solve_assignment(cost, max_cost=0.9)
    assert result.matches == [(0, 0)]
    assert result.unmatched_rows == [1]
    assert result.unmatched_cols == [1]


def test_solve_assignment_with_no_admissible_pair():
    result = solve_assignment(np.full((2, 2), np.inf))
    assert result.matches == []
    assert result.unmatched_rows == [0, 1]
    assert result.unmatched_cols == [0, 1]


# --------------------------------------------------------------------------- #
# Geometry and classes
# --------------------------------------------------------------------------- #


def test_iou_of_identical_boxes_is_one():
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # 10x10 and 10x10 offset by 5 in x: intersection 5x10 = 50, union 150.
    assert iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50.0 / 150.0)


def test_iou_of_a_degenerate_box_is_zero_not_an_exception():
    assert iou_xyxy((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0
    assert iou_xyxy((0, 0, float("nan"), 10), (0, 0, 10, 10)) == 0.0


def test_class_groups_allow_car_truck_but_not_car_bicycle():
    assert class_compatible("car", "truck")
    assert class_compatible("bus", "vehicle")
    assert class_compatible("bicycle", "motorcycle")
    assert not class_compatible("car", "bicycle")
    assert not class_compatible("person", "car")


def test_unknown_labels_associate_only_with_themselves():
    assert class_group("traffic_cone") == "other:traffic_cone"
    assert class_compatible("traffic_cone", "TRAFFIC_CONE")
    assert not class_compatible("traffic_cone", "car")


# --------------------------------------------------------------------------- #
# Cost matrix construction
# --------------------------------------------------------------------------- #


def _box(cx, cy, w, h):
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def test_build_cost_matrix_shape_validation():
    with pytest.raises(ValidationError):
        build_cost_matrix(np.zeros((1, 1)), [_box(0, 0, 10, 10)], ["car"], [], [])
    with pytest.raises(ValidationError):
        build_cost_matrix(
            np.zeros((2, 2)), [_box(0, 0, 10, 10)], ["car"], [_box(0, 0, 10, 10)], ["car"]
        )
    with pytest.raises(ValidationError):
        build_cost_matrix(np.zeros((1, 1)), [_box(0, 0, 10, 10)], ["car", "car"], [_box(0, 0, 10, 10)], ["car"])


def test_build_cost_matrix_perfect_match_is_cheap():
    box = _box(640.0, 360.0, 100.0, 100.0)
    cost = build_cost_matrix(np.zeros((1, 1)), [box], ["car"], [box], ["car"])
    assert cost[0][0] == pytest.approx(0.0)


def test_build_cost_matrix_gates_on_class():
    box = _box(640.0, 360.0, 100.0, 100.0)
    cost = build_cost_matrix(np.zeros((1, 1)), [box], ["car"], [box], ["bicycle"])
    assert math.isinf(cost[0][0])
    relaxed = build_cost_matrix(
        np.zeros((1, 1)),
        [box],
        ["car"],
        [box],
        ["bicycle"],
        AssociationParams(require_class_match=False),
    )
    assert math.isfinite(relaxed[0][0])


def test_build_cost_matrix_gates_on_size_ratio():
    """This single veto blocks the 68 m -> 6.8 m teleport of ADAS-DEC-07."""
    far_track = _box(640.0, 300.0, 20.0, 20.0)
    near_detection = _box(700.0, 340.0, 200.0, 200.0)
    # Pretend the Kalman gate is wide open, so only the size veto can act.
    cost = build_cost_matrix(
        np.zeros((1, 1)), [far_track], ["car"], [near_detection], ["car"]
    )
    assert math.isinf(cost[0][0])


def test_build_cost_matrix_gates_on_absolute_centre_distance():
    track = _box(100.0, 100.0, 50.0, 50.0)
    detection = _box(400.0, 100.0, 50.0, 50.0)
    cost = build_cost_matrix(np.zeros((1, 1)), [track], ["car"], [detection], ["car"])
    assert math.isinf(cost[0][0])


def test_build_cost_matrix_gates_on_chi_square():
    track = _box(640.0, 360.0, 100.0, 100.0)
    detection = _box(700.0, 360.0, 100.0, 100.0)
    over = np.array([[CHI2_99[2] + 1.0]])
    under = np.array([[CHI2_99[2] - 1.0]])
    assert math.isinf(build_cost_matrix(over, [track], ["car"], [detection], ["car"])[0][0])
    assert math.isfinite(build_cost_matrix(under, [track], ["car"], [detection], ["car"])[0][0])


def test_the_chi_square_gate_has_a_floor_so_an_overconfident_filter_cannot_starve():
    """A detection essentially on top of the prediction is never gated out."""
    track = _box(640.0, 360.0, 100.0, 100.0)
    detection = _box(642.0, 361.0, 100.0, 100.0)  # 2.2 px away
    cost = build_cost_matrix(
        np.array([[1.0e6]]), [track], ["car"], [detection], ["car"]
    )
    assert math.isfinite(cost[0][0])


def test_build_cost_matrix_is_inf_for_an_inf_gate_value():
    box = _box(640.0, 360.0, 100.0, 100.0)
    cost = build_cost_matrix(np.array([[np.inf]]), [box], ["car"], [box], ["car"])
    assert math.isinf(cost[0][0])


def test_costs_are_normalised_into_the_unit_interval():
    rng = np.random.RandomState(SEED + 3)
    tracks = [_box(640.0, 360.0, 100.0, 100.0) for _ in range(5)]
    dets = [
        _box(640.0 + rng.uniform(-40, 40), 360.0 + rng.uniform(-40, 40), 100.0, 100.0)
        for _ in range(5)
    ]
    maha = rng.uniform(0.0, CHI2_99[2], size=(5, 5))
    cost = build_cost_matrix(maha, tracks, ["car"] * 5, dets, ["car"] * 5)
    finite = [value for row in cost for value in row if math.isfinite(value)]
    assert finite
    assert min(finite) >= 0.0 and max(finite) <= 1.0


def test_association_params_validate_their_arguments():
    with pytest.raises(ValidationError):
        AssociationParams(chi2_gate=0.0)
    with pytest.raises(ValidationError):
        AssociationParams(max_size_ratio=1.0)
    with pytest.raises(ValidationError):
        AssociationParams(max_centre_distance_px=0.0)
    with pytest.raises(ValidationError):
        AssociationParams(weight_maha=0.0, weight_iou=0.0, weight_size=0.0)


def test_end_to_end_the_near_lead_keeps_its_own_detection():
    """ADAS-DEC-07 in full: an older distant track must not steal the lead's box.

    Track order is deliberately distant-first, which is what made the old greedy
    matcher fail.
    """
    distant = _box(640.0, 300.0, 20.0, 20.0)
    lead = _box(700.0, 340.0, 200.0, 200.0)
    detections = [_box(702.0, 342.0, 202.0, 201.0), _box(641.0, 301.0, 20.0, 21.0)]
    maha = np.array([[8.0, 0.5], [0.5, 8.0]])
    cost = build_cost_matrix(
        maha, [distant, lead], ["car", "car"], detections, ["car", "car"]
    )
    result = solve_assignment(cost, max_cost=AssociationParams().max_cost)
    assert sorted(result.matches) == [(0, 1), (1, 0)]
