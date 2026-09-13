"""Detection-to-track association: gating, cost, and optimal assignment.

Everything here is a **pure function of plain sequences**. No filter objects, no
track objects, no logging of state. The tracker computes the Mahalanobis matrix
from its Kalman filters and hands it in; this module decides which pairs are
admissible, what each admissible pair costs, and which global assignment is
cheapest. That split is what makes the association layer testable without
constructing a pipeline.

Why a global optimum
--------------------
The previous tracker walked ``self._tracks`` in dict insertion order and let each
track claim its own nearest unassigned detection. Ordering, not cost, decided the
assignment, so an old distant track could steal the near lead vehicle's detection
and teleport 68 m -> 6.8 m in one frame (ADAS-DEC-07). :func:`hungarian` solves
the whole matrix at once, so the result does not depend on track creation order.

Why a chi-square gate instead of a pixel radius
-----------------------------------------------
The image-plane separation of two objects a fixed metric distance apart scales
as ``f * X / Z``. A single 120 px radius is simultaneously far too loose at 100 m
(two adjacent-lane vehicles are 32 px apart there) and too tight for a cut-in at
8 m (398 px of lateral motion). The gate here is a chi-square test on the Kalman
innovation, which is range-, dt- and coast-duration-adaptive by construction,
backed up by three hard vetoes that a covariance cannot argue with: a maximum
centre displacement, a maximum box-size ratio, and class compatibility.

Units
-----
Pixels for every image-plane quantity. ``maha2`` is a squared Mahalanobis
distance and therefore dimensionless. Costs are dimensionless and normalised to
``[0, 1]`` so the weights are directly interpretable; an inadmissible pair costs
``inf``.

Cost
----
Plain Python throughout, for the reason given in
:mod:`adas.tracking.kalman`: on the target Xavier NX a single numpy call costs
15-30 us of dispatch, so a fully vectorised :func:`build_cost_matrix` measured
1.09 ms on a 2x2 problem against ~0.05 ms for the same work in a loop. The loop
also gets to *skip*: the cheap vetoes (class, centre distance, size ratio) reject
most pairs in a handful of operations and only survivors pay for an IoU and a
logarithm, so the cost grows with the number of *plausible* pairs rather than
with ``n * m``. :func:`hungarian` is O(n^2 m) and refuses a problem larger than
:data:`MAX_ASSIGNMENT_DIM` rather than stalling a frame.

Failure behaviour
-----------------
* :func:`hungarian` raises :class:`~adas.core.exceptions.ValidationError` on a
  non-2-D cost matrix, or one larger than :data:`MAX_ASSIGNMENT_DIM`. Pairs whose
  cost is ``inf`` or ``nan`` are never returned as matches, even when the matrix
  admits no complete matching without them.
* :func:`build_cost_matrix` never invents a cost: a pair that fails any gate is
  ``inf``, and ``inf`` is preserved end to end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from adas.core.exceptions import ValidationError
from adas.tracking.kalman import CHI2_99

__all__ = [
    "AssociationParams",
    "Assignment",
    "CLASS_GROUPS",
    "MAX_ASSIGNMENT_DIM",
    "build_cost_matrix",
    "class_compatible",
    "class_group",
    "hungarian",
    "iou_xyxy",
    "solve_assignment",
]


#: Largest assignment problem this module will attempt, per side. A road scene
#: after non-maximum suppression is an order of magnitude below this; the bound
#: exists so a detector fault that emits thousands of boxes fails loudly instead
#: of stalling the control loop. The tracker caps its inputs before calling.
MAX_ASSIGNMENT_DIM = 256


#: Detector labels that may be associated with one another. A car/truck flip
#: between consecutive frames is routine for a COCO detector, but a car and a
#: bicycle are never the same object and letting them associate produces an
#: instant range discontinuity because their height priors differ by 0.2 m and
#: their box aspect by a factor of three.
CLASS_GROUPS: Dict[str, FrozenSet[str]] = {
    "vehicle": frozenset({"car", "truck", "bus", "train", "vehicle"}),
    "two_wheeler": frozenset({"bicycle", "motorcycle"}),
    "vru": frozenset({"person"}),
}

_GROUP_OF_LABEL: Dict[str, str] = {
    member: group for group, members in CLASS_GROUPS.items() for member in members
}


def class_group(label: str) -> str:
    """Association group of a detector label.

    An unrecognised label maps to ``"other:<label>"``, so unknown classes
    associate only with themselves rather than with everything.
    """
    key = str(label).strip().lower()
    group = _GROUP_OF_LABEL.get(key)
    if group is not None:
        return group
    return "other:%s" % key


def class_compatible(a: str, b: str) -> bool:
    """True when two labels may describe the same physical object."""
    return class_group(a) == class_group(b)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes.

    Returns 0.0 for degenerate or non-finite boxes rather than raising: a
    detector can emit one, and refusing to score it simply means the pair is
    matched on the other cost terms.
    """
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    aw = ax2 - ax1
    ah = ay2 - ay1
    bw = bx2 - bx1
    bh = by2 - by1
    if not (aw > 0.0 and ah > 0.0 and bw > 0.0 and bh > 0.0):
        return 0.0
    ix = (ax2 if ax2 < bx2 else bx2) - (ax1 if ax1 > bx1 else bx1)
    if ix <= 0.0:
        return 0.0
    iy = (ay2 if ay2 < by2 else by2) - (ay1 if ay1 > by1 else by1)
    if iy <= 0.0:
        return 0.0
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    if union <= 0.0:
        return 0.0
    value = inter / union
    return value if math.isfinite(value) else 0.0


# --------------------------------------------------------------------------- #
# Optimal assignment
# --------------------------------------------------------------------------- #


def _lap(cost: Sequence[Sequence[float]], n: int, m: int) -> Tuple[List[int], List[int]]:
    """Jonker-Volgenant shortest augmenting path on a finite ``(n, m)`` cost, ``n <= m``.

    ``cost`` is a sequence of ``n`` row sequences of length ``m``, all finite.
    Returns ``(rows, cols)`` of the optimal complete matching of the rows.
    """
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)  # p[j] = 1-based row matched to column j
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            row = cost[i0 - 1]
            offset = u[i0]
            delta = inf
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = row[j - 1] - offset - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 < 0 or delta == inf:
                # Every remaining column is unreachable. Cannot happen with a
                # fully finite matrix; guard rather than loop forever.
                raise ValidationError("Assignment problem is infeasible (non-finite reduced cost)")
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    rows: List[int] = []
    cols: List[int] = []
    for j in range(1, m + 1):
        if p[j] != 0:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    return rows, cols


def _as_cost_rows(cost: Any) -> Tuple[List[List[float]], int, int]:
    """Normalise a cost matrix into ``(rows, n, m)`` of Python floats."""
    if hasattr(cost, "tolist"):
        rows = cost.tolist()
    else:
        rows = [list(row) for row in cost]
    if not isinstance(rows, list) or (rows and not isinstance(rows[0], list)):
        raise ValidationError("Cost matrix must be 2-D")
    n = len(rows)
    m = len(rows[0]) if n else 0
    if any(len(row) != m for row in rows):
        raise ValidationError("Cost matrix rows must all have the same length")
    if n > MAX_ASSIGNMENT_DIM or m > MAX_ASSIGNMENT_DIM:
        raise ValidationError(
            "Assignment problem %dx%d exceeds MAX_ASSIGNMENT_DIM=%d" % (n, m, MAX_ASSIGNMENT_DIM)
        )
    return rows, n, m


def _hungarian_rows(rows: List[List[float]], n: int, m: int) -> List[Tuple[int, int]]:
    """Core of :func:`hungarian` on plain lists. Returns admissible ``(row, col)`` pairs."""
    if n == 0 or m == 0:
        return []

    low = math.inf
    high = -math.inf
    for row in rows:
        for value in row:
            if value == value and value != math.inf and value != -math.inf:
                if value < low:
                    low = value
                if value > high:
                    high = value
    if low == math.inf:
        return []  # nothing admissible

    # Shift so the cheapest admissible pair is 0. A constant shift does not
    # change which assignment is optimal (every assignment uses the same number
    # of cells), and it makes the BIG substitution easy to reason about: BIG
    # exceeds the total of every admissible assignment, so the solver minimises
    # the number of inadmissible cells first and their cost second.
    big = (high - low) * float(min(n, m) + 1) + 1.0
    admissible = [
        [value == value and value != math.inf and value != -math.inf for value in row]
        for row in rows
    ]
    work = [
        [(rows[i][j] - low) if admissible[i][j] else big for j in range(m)] for i in range(n)
    ]

    if n > m:
        work = [[work[i][j] for i in range(n)] for j in range(m)]
        solved_rows, solved_cols = _lap(work, m, n)
        pairs = list(zip(solved_cols, solved_rows))
    else:
        solved_rows, solved_cols = _lap(work, n, m)
        pairs = list(zip(solved_rows, solved_cols))

    keep = [(r, c) for r, c in pairs if admissible[r][c]]
    keep.sort()
    return keep


def hungarian(cost: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Minimum-cost assignment. Drop-in replacement for ``scipy.optimize.linear_sum_assignment``.

    Implemented here rather than taken from scipy so the tracker has no runtime
    dependency beyond numpy (the board ships scipy 1.3.3, but a safety-relevant
    path should not acquire a dependency it can own in eighty lines).
    ``tests/test_association.py`` cross-checks this against scipy whenever scipy
    is importable, and against a brute-force optimum on small matrices.

    Args:
        cost: ``(n, m)`` array or nested sequence. ``inf`` (or ``nan``) marks an
            inadmissible pair.

    Returns:
        ``(row_indices, col_indices)``, sorted by row index, containing only
        pairs with a finite cost. The returned matching is optimal over all
        matchings that use no inadmissible pair, and is maximal in size among
        those. Rows or columns left out are unassigned.

    Raises:
        ValidationError: if ``cost`` is not 2-D or either dimension exceeds
            :data:`MAX_ASSIGNMENT_DIM`.
    """
    rows, n, m = _as_cost_rows(cost)
    pairs = _hungarian_rows(rows, n, m)
    return (
        np.array([r for r, _ in pairs], dtype=np.int64),
        np.array([c for _, c in pairs], dtype=np.int64),
    )


@dataclass
class Assignment:
    """Result of :func:`solve_assignment`.

    Attributes:
        matches: ``(row, col)`` pairs, sorted by row.
        unmatched_rows: Row indices with no partner (unmatched tracks).
        unmatched_cols: Column indices with no partner (unmatched detections).
        costs: Cost of each accepted match, aligned with ``matches``.
    """

    matches: List[Tuple[int, int]] = field(default_factory=list)
    unmatched_rows: List[int] = field(default_factory=list)
    unmatched_cols: List[int] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)


def solve_assignment(cost: Any, max_cost: float = float("inf")) -> Assignment:
    """Solve the assignment and reject pairs above ``max_cost``.

    A pair that the global optimum selected but whose individual cost exceeds
    ``max_cost`` is rejected and both of its members are reported unmatched. The
    optimum is computed first and filtered afterwards -- filtering before would
    reintroduce the order dependence the Hungarian solve exists to remove.
    """
    rows, n, m = _as_cost_rows(cost)
    pairs = _hungarian_rows(rows, n, m)
    matches: List[Tuple[int, int]] = []
    costs: List[float] = []
    matched_rows = [False] * n
    matched_cols = [False] * m
    for r, c in pairs:
        value = rows[r][c]
        if value != value or value == math.inf or value > max_cost:
            continue
        matches.append((r, c))
        costs.append(value)
        matched_rows[r] = True
        matched_cols[c] = True
    return Assignment(
        matches=matches,
        unmatched_rows=[i for i in range(n) if not matched_rows[i]],
        unmatched_cols=[j for j in range(m) if not matched_cols[j]],
        costs=costs,
    )


# --------------------------------------------------------------------------- #
# Cost construction
# --------------------------------------------------------------------------- #


@dataclass
class AssociationParams:
    """Gates and weights for :func:`build_cost_matrix`.

    Gates (hard, a failure makes the pair ``inf``):
        chi2_gate: Chi-square threshold on the 2-dof centre innovation.
            Defaults to the 99th percentile, so a genuine manoeuvre is rejected
            about one frame in a hundred rather than one in twenty.
        max_centre_distance_px: Absolute cap on centre displacement, pixels. A
            diverged covariance cannot open the gate wider than this.
        min_centre_distance_px: Absolute gate floor, pixels. A pair closer than
            this is never rejected by the chi-square test, so an over-confident
            filter on a stationary object cannot reject its own detection.
        gate_floor_height_frac: Range-adaptive gate floor, as a fraction of the
            larger box height. A near object subtends a large box and can move a
            large number of pixels in one frame -- a cut-in at 8 m traverses
            398 px of lateral offset -- while a converged filter's own
            covariance is only a few pixels wide, so the chi-square gate alone
            would reject the manoeuvre it most needs to follow. Half a box
            height is 85 px at 8 m and 7 px at 100 m, which is exactly the
            scaling the fixed 120 px radius lacked. The floor only prevents
            *rejection*; the cost still prefers the better pair, and the class
            and size vetoes still apply.
        max_size_ratio: Largest permitted ratio between detection and predicted
            box height (and width) in one frame. 1.6 blocks the 68 m -> 6.8 m
            teleport of ADAS-DEC-07 on its own.
        require_class_match: Enforce :func:`class_compatible`.

    Weights (soft, they order the admissible pairs; they sum to 1 by default so
    a cost is directly readable as a fraction of the worst admissible match):
        weight_maha: Weight on ``sqrt(maha2 / chi2_gate)``.
        weight_iou: Weight on ``1 - IoU`` against the *predicted* box.
        weight_size: Weight on ``|log(h_det / h_pred)| / log(max_size_ratio)``.

    max_cost: Matches costlier than this are rejected by
        :func:`solve_assignment`. 0.9 keeps a fast cut-in (which can lose IoU
        entirely) while rejecting a pair that is poor on every term at once.
    """

    chi2_gate: float = CHI2_99[2]
    max_centre_distance_px: float = 120.0
    min_centre_distance_px: float = 6.0
    gate_floor_height_frac: float = 0.5
    max_size_ratio: float = 1.6
    require_class_match: bool = True

    weight_maha: float = 0.6
    weight_iou: float = 0.3
    weight_size: float = 0.1

    max_cost: float = 0.9

    def __post_init__(self) -> None:
        if self.chi2_gate <= 0.0:
            raise ValidationError("chi2_gate must be positive")
        if self.max_size_ratio <= 1.0:
            raise ValidationError("max_size_ratio must exceed 1.0")
        if self.max_centre_distance_px <= 0.0:
            raise ValidationError("max_centre_distance_px must be positive")
        if self.min_centre_distance_px < 0.0:
            raise ValidationError("min_centre_distance_px must be non-negative")
        if self.gate_floor_height_frac < 0.0:
            raise ValidationError("gate_floor_height_frac must be non-negative")
        total = self.weight_maha + self.weight_iou + self.weight_size
        if total <= 0.0:
            raise ValidationError("Association weights must sum to a positive value")


def build_cost_matrix(
    maha2: Any,
    track_boxes: Sequence[Sequence[float]],
    track_labels: Sequence[str],
    det_boxes: Sequence[Sequence[float]],
    det_labels: Sequence[str],
    params: Optional[AssociationParams] = None,
) -> List[List[float]]:
    """Assemble the gated association cost matrix.

    Args:
        maha2: ``(n_tracks, n_dets)`` squared Mahalanobis distances of each
            detection centre from each track's predicted centre, as nested
            sequences or an array. The tracker computes these from
            :meth:`~adas.tracking.kalman.BoxFilter.gate_centre_batch`; ``inf``
            entries are already-rejected pairs.
        track_boxes: Predicted track boxes, ``(x1, y1, x2, y2)`` in pixels.
        track_labels: Track class labels, aligned with ``track_boxes``.
        det_boxes: Detection boxes, ``(x1, y1, x2, y2)`` in pixels.
        det_labels: Detection class labels, aligned with ``det_boxes``.
        params: Gates and weights. Defaults to :class:`AssociationParams`.

    Returns:
        An ``n_tracks x n_dets`` list of lists. Admissible pairs carry a cost in
        ``[0, 1]``; every pair failing any gate is ``inf``.

    Raises:
        ValidationError: if the shapes are inconsistent.
    """
    par = params if params is not None else AssociationParams()
    n = len(track_boxes)
    m = len(det_boxes)
    if len(track_labels) != n or len(det_labels) != m:
        raise ValidationError("Label sequences must align with the box sequences")
    gate_rows = maha2.tolist() if hasattr(maha2, "tolist") else [list(row) for row in maha2]
    if len(gate_rows) != n or any(len(row) != m for row in gate_rows):
        raise ValidationError(
            "maha2 must be %d tracks x %d detections" % (n, m)
        )

    inf = float("inf")
    cost = [[inf] * m for _ in range(n)]
    if n == 0 or m == 0:
        return cost

    chi2 = par.chi2_gate
    max_distance = par.max_centre_distance_px
    max_distance_sq = max_distance * max_distance
    min_floor = par.min_centre_distance_px
    floor_frac = par.gate_floor_height_frac
    high_ratio = par.max_size_ratio
    low_ratio = 1.0 / high_ratio
    log_scale = math.log(high_ratio)
    weight_maha = par.weight_maha
    weight_iou = par.weight_iou
    weight_size = par.weight_size
    inv_weight = 1.0 / (weight_maha + weight_iou + weight_size)
    check_class = par.require_class_match

    # Per-detection scalars, computed once instead of n times.
    dets: List[Tuple[float, float, float, float, float, float, float, float, str]] = []
    for j in range(m):
        x1, y1, x2, y2 = (float(v) for v in det_boxes[j])
        dets.append(
            (
                x1,
                y1,
                x2,
                y2,
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                x2 - x1,
                y2 - y1,
                class_group(det_labels[j]) if check_class else "",
            )
        )

    for i in range(n):
        tx1, ty1, tx2, ty2 = (float(v) for v in track_boxes[i])
        t_w = tx2 - tx1
        t_h = ty2 - ty1
        if t_w <= 0.0 or t_h <= 0.0:
            continue
        t_cx = (tx1 + tx2) / 2.0
        t_cy = (ty1 + ty2) / 2.0
        t_group = class_group(track_labels[i]) if check_class else ""
        gate_row = gate_rows[i]
        out_row = cost[i]
        t_area = t_w * t_h

        for j in range(m):
            dx1, dy1, dx2, dy2, d_cx, d_cy, d_w, d_h, d_group = dets[j]
            if check_class and t_group != d_group:
                continue
            if d_w <= 0.0 or d_h <= 0.0:
                continue

            offset_x = d_cx - t_cx
            offset_y = d_cy - t_cy
            distance_sq = offset_x * offset_x + offset_y * offset_y
            if distance_sq > max_distance_sq:
                continue

            height_ratio = d_h / t_h
            if height_ratio < low_ratio or height_ratio > high_ratio:
                continue
            width_ratio = d_w / t_w
            if width_ratio < low_ratio or width_ratio > high_ratio:
                continue

            gate_value = gate_row[j]
            if gate_value != gate_value or gate_value == inf:
                continue
            if gate_value > chi2:
                floor = floor_frac * (t_h if t_h > d_h else d_h)
                if floor < min_floor:
                    floor = min_floor
                if distance_sq > floor * floor:
                    continue

            maha_term = math.sqrt(gate_value / chi2) if gate_value > 0.0 else 0.0
            if maha_term > 1.0:
                maha_term = 1.0

            ix = (tx2 if tx2 < dx2 else dx2) - (tx1 if tx1 > dx1 else dx1)
            iy = (ty2 if ty2 < dy2 else dy2) - (ty1 if ty1 > dy1 else dy1)
            if ix > 0.0 and iy > 0.0:
                inter = ix * iy
                union = t_area + d_w * d_h - inter
                iou_term = 1.0 - (inter / union) if union > 0.0 else 1.0
            else:
                iou_term = 1.0

            size_term = abs(math.log(height_ratio)) / log_scale
            if size_term > 1.0:
                size_term = 1.0

            value = (
                weight_maha * maha_term + weight_iou * iou_term + weight_size * size_term
            ) * inv_weight
            out_row[j] = value if math.isfinite(value) else inf

    return cost
