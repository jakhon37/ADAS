"""Multi-object tracker: Kalman motion models, optimal association, track lifecycle.

What this replaces
------------------
The previous implementation was greedy nearest-centre association in dict
insertion order, with no motion model, a fixed 120 px gate, no class check, and
track birth that published a brand-new object to the planner on its very first
frame. The consequences were verified, not hypothetical:

* **ADAS-DEC-06** -- range was an unfiltered ``H*f/h`` reciprocal differenced raw,
  so one pixel of box jitter at 30 m produced 13 m/s of "velocity". Now every
  track carries a constant-acceleration :class:`~adas.tracking.kalman.RangeFilter`
  with a range-dependent measurement variance.
* **ADAS-DEC-07 / 08** -- greedy ordering let a distant track steal the near lead
  vehicle's detection. Now association is a single global Hungarian solve over a
  gated cost matrix (see :mod:`adas.tracking.association`).
* **ADAS-DEC-09** -- one false-positive detection became a tracked obstacle and
  commanded full brake. Now a track must be matched on ``confirm_hits`` of its
  first ``confirm_window`` frames before :meth:`MultiObjectTracker.update`
  reports it at all.
* **ADAS-DEC-16** -- coasting tracks were published with a frozen box, so a
  closing lead appeared to stop closing. Now a coasting track is *predicted*
  forward, its covariance grows, ``time_since_update`` counts the coast, and its
  ``range_estimate`` is reported as ``UNAVAILABLE`` because there was no
  measurement that frame.
* **ADAS-DEC-10** -- one 1.5 m height prior for cars, trucks and bicycles. Now
  the prior is keyed on the (vote-smoothed) class label, and a vertically
  truncated box is flagged and has its measurement variance inflated fivefold.

Coordinates and units
---------------------
Image quantities are pixels with ``u`` right and ``v`` down from the top-left of
the source frame. Ground quantities are metres in the vehicle frame: ``X`` right,
``Z`` forward. ``distance_m`` is longitudinal range in metres.
``TrackedObject.velocity_mps`` keeps this codebase's existing
**positive-when-closing** convention (the planner flips it to
``v_rel = v_lead - v_ego`` itself). ``ttc_s`` is seconds until the range reaches
zero; consumers apply their own standstill gap.

Integration points (all optional, all duck-typed so this module never imports
the perception package and cannot be broken by its churn)
---------------------------------------------------------------------------
``range_fn``
    ``callable(box, frame_width, frame_height) -> RangeEstimate``. Wire it to
    ``adas.perception.geometry.estimate_range`` bound to a ``CameraConfig``; it
    fuses a ground-plane channel with the pinhole one and, critically, discards
    the pinhole model on a truncated box. Without it this module falls back to
    its own class-keyed pinhole estimator, which flags truncation but cannot
    correct it.
``camera``
    Any object exposing ``image_to_ground(u_px, v_px) -> (x_m, z_m) | None``
    (``CameraConfig`` satisfies this). Used for the object's metric lateral
    offset. Without it the tracker uses a documented small-angle pinhole
    approximation about the image centre.
``lane`` / ``drivable``
    ``LaneModel`` and ``DrivableArea`` from :mod:`adas.core.models`, passed per
    frame. Used, in that order of preference, to decide ``in_ego_lane``.

Cost
----
Measured on the target Xavier NX (MODE_20W_6CORE, board otherwise idle), whole
``update()`` including range filtering, association, lifecycle and the ego-lane
decision:

===============  ================
live tracks      ms per frame
===============  ================
2                1.3
8                4.7
20               12.5
===============  ================

Against a 50 ms budget at 20 Hz, of which the perception engines already claim
~21 ms. The work is bounded above by ``max_detections`` and ``max_tracks``; see
:mod:`adas.tracking.kalman` for why none of the arithmetic goes through numpy.

Failure behaviour
-----------------
* An invalid detection is dropped with a warning; the frame still processes.
* A per-track filter failure is contained to that track (it is marked missed and
  logged at WARNING), never aborting the frame -- the old code wrapped the whole
  update in one ``except Exception`` and turned a single bad box into a lost
  frame.
* :class:`~adas.core.exceptions.TrackingError` is raised only for a failure that
  affects the whole update (a malformed detection list, an unusable cost solve).
* No published quantity is ever a plausible constant. A range that could not be
  measured is reported as ``RangeSource.UNAVAILABLE`` with confidence 0; an
  ``in_ego_lane`` decision that had no lane geometry to work with is taken by a
  deliberately *inclusive* heuristic and recorded as such in
  :meth:`MultiObjectTracker.diagnostics`.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from adas.core.exceptions import TrackingError, ValidationError
from adas.core.logger import setup_logger
from adas.core.models import (
    BoundingBox,
    DrivableArea,
    LaneModel,
    RangeEstimate,
    RangeSource,
    TrackedObject,
)
from adas.core.validation import validate_bounding_box
from adas.tracking.association import (
    MAX_ASSIGNMENT_DIM,
    AssociationParams,
    build_cost_matrix,
    solve_assignment,
)
from adas.tracking.kalman import (
    BoxFilter,
    FilterError,
    RangeFilter,
    range_measurement_sigma_m,
)

logger = setup_logger(__name__)

__all__ = ["MultiObjectTracker", "TrackStatus", "NOMINAL_OBJECT_HEIGHT_M"]


#: Nominal object heights (metres) for the pinhole range model, keyed by the
#: COCO label strings the detectors emit. Deliberately identical to
#: ``adas.perception.geometry.NOMINAL_OBJECT_HEIGHT_M`` -- duplicated rather than
#: imported so the tracker does not depend on the perception package, and
#: cross-checked in ``tests/test_tracking.py``.
NOMINAL_OBJECT_HEIGHT_M = {
    "person": 1.70,
    "bicycle": 1.70,
    "motorcycle": 1.60,
    "car": 1.50,
    "bus": 3.20,
    "truck": 3.60,
    "train": 4.00,
    "vehicle": 1.50,
}

#: Ego-lane index in ``LaneModel.lines`` for the left boundary, the right
#: boundary, and the derived centreline. Mirrors ``adas.perception.lane``.
_LANE_LEFT_INDEX = 1
_LANE_RIGHT_INDEX = 2
_LANE_CENTRE_INDEX = -1

#: Plausibility band for a fitted lane width (metres). Outside this the lane fit
#: is almost certainly a mis-association and is not used to gate objects.
_MIN_LANE_WIDTH_M = 2.0
_MAX_LANE_WIDTH_M = 5.0

#: Columns sampled across a drivable-area row when tracing the free corridor.
_DRIVABLE_SAMPLES = 64


class TrackStatus(str, Enum):
    """Lifecycle state of a track.

    ``TENTATIVE`` tracks are never returned by :meth:`MultiObjectTracker.update`
    -- that single rule is what stops one spurious detection from becoming an
    obstacle the ACC planner brakes for.
    """

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"


@dataclass
class _Track:
    """Internal per-track state. Not part of the public API."""

    track_id: int
    box: BoundingBox
    box_filter: BoxFilter
    range_filter: RangeFilter
    label_votes: Dict[str, int]
    history: Deque[bool]
    status: TrackStatus = TrackStatus.TENTATIVE
    age_frames: int = 0
    hits: int = 0
    time_since_update: int = 0
    confidence: float = 0.0
    range_estimate: RangeEstimate = field(
        default_factory=lambda: RangeEstimate(0.0, 0.0, RangeSource.UNAVAILABLE, False)
    )
    lateral_offset_m: float = 0.0
    lateral_offset_known: bool = False
    in_ego_lane: bool = False
    ego_lane_source: str = "unavailable"
    range_reinits: int = 0

    @property
    def label(self) -> str:
        """Majority class label seen so far; ties break toward the first seen."""
        best_label = self.box.label
        best_count = -1
        for name, count in self.label_votes.items():
            if count > best_count:
                best_label = name
                best_count = count
        return best_label

    def vote(self, label: str) -> None:
        self.label_votes[label] = self.label_votes.get(label, 0) + 1


@dataclass
class MultiObjectTracker:
    """Kalman multi-object tracker over the ground plane.

    The first six fields are the ones ``adas.cli.build_pipeline`` wires from
    ``TrackerConfig``; their names and meanings are unchanged so the existing
    configuration keeps working. Everything after them is new and defaulted.

    Args:
        max_missed: Frames a *confirmed* track may coast without a measurement
            before deletion.
        association_threshold_px: Absolute cap on the association gate, pixels.
            No longer the gate itself -- the gate is a chi-square test on the
            Kalman innovation -- but a hard ceiling a diverged covariance cannot
            exceed.
        focal_length_px: Camera focal length, pixels. Used by the fallback
            pinhole range model and the fallback lateral-offset model.
        object_height_m: Height prior (metres) for labels absent from
            :data:`NOMINAL_OBJECT_HEIGHT_M`, and for every label when
            ``use_class_heights`` is False.
        min_box_height_px: Floor applied to a box height before dividing, purely
            to avoid a division by zero.
        max_distance_m: Ceiling on a reported range, metres.
        confirm_hits: Matches required within ``confirm_window`` frames before a
            track is reported to the planner.
        confirm_window: Length of the confirmation window, frames.
        nominal_dt_s: Expected frame period, used only to decide when a supplied
            ``dt_s`` is anomalous enough to warn about.
        min_dt_s, max_dt_s: Hard clamp on ``dt_s``. A pipeline stall must not
            multiply through every rate estimate (ADAS-DEC-17).
        box_sigma_px: Assumed detector box-height noise, pixels. Drives the
            range measurement variance.
        jerk_psd: Process-noise PSD of the range filter, m^2/s^5. See
            :class:`~adas.tracking.kalman.RangeFilter` for why 1.0 and not 9.0.
        min_range_box_height_px: Boxes shorter than this yield no usable range;
            the estimate is published as ``UNAVAILABLE`` and the range filter is
            not corrected that frame.
        use_class_heights: Key the height prior on the class label.
        ego_half_width_m: Half width of the ego vehicle, metres. Reserved for
            callers that want the corridor centred on the vehicle body.
        lane_margin_m: Lateral tolerance added to each lane boundary before
            testing occupancy, metres. Inclusive by design.
        nominal_lane_width_m: Assumed lane width when only one boundary was
            measured, metres.
        min_lane_confidence: ``LaneModel.confidence`` below which the lane is not
            trusted to gate objects.
        min_drivable_confidence: ``DrivableArea.confidence`` below which the mask
            is not trusted to gate objects.
        heuristic_corridor_scale: Widening factor on the image-band fallback
            corridor. Greater than 1 on purpose -- see :meth:`_image_band`.
        ttc_min_closing_mps: Closing speeds below this report ``ttc_s = inf``.
        range_reinit_maha2: Squared Mahalanobis distance beyond which a range
            measurement is treated as evidence that the *filter* is wrong, and
            the filter restarts at the measurement.
        max_detections: Largest number of detections accepted in one frame. A
            detector fault that emits hundreds of boxes must not turn one frame
            into a multi-second assignment solve, so the lowest-confidence
            detections above this bound are dropped with a WARNING. Dropping the
            *weakest* is the conservative choice; a real obstacle is not the
            300th most confident box in the frame.
        max_tracks: Largest number of live tracks. Above it, tentative tracks
            are discarded fewest-hits-first, then the longest-coasting confirmed
            tracks. A confirmed, currently-measured track is never discarded to
            make room.
        frame_width_px, frame_height_px: Default frame size, used when
            :meth:`update` is called without one. ``0`` means unknown, in which
            case truncation cannot be detected and ``in_ego_lane`` stays False.
        camera: Optional duck-typed camera (see the module docstring).
        range_fn: Optional duck-typed range channel (see the module docstring).
        association: Gates and weights for the cost matrix. Built from
            ``association_threshold_px`` when not supplied.
    """

    max_missed: int = 5
    association_threshold_px: float = 120.0
    focal_length_px: float = 910.0
    object_height_m: float = 1.5
    min_box_height_px: float = 1.0
    max_distance_m: float = 200.0

    confirm_hits: int = 3
    confirm_window: int = 5
    nominal_dt_s: float = 0.05
    min_dt_s: float = 0.005
    max_dt_s: float = 0.5
    box_sigma_px: float = 1.5
    jerk_psd: float = 1.0
    min_range_box_height_px: float = 8.0
    use_class_heights: bool = True
    ego_half_width_m: float = 1.1
    lane_margin_m: float = 0.5
    nominal_lane_width_m: float = 3.5
    min_lane_confidence: float = 0.3
    min_drivable_confidence: float = 0.3
    heuristic_corridor_scale: float = 1.6
    ttc_min_closing_mps: float = 0.1
    range_reinit_maha2: float = 100.0
    max_detections: int = 64
    max_tracks: int = 64

    frame_width_px: int = 0
    frame_height_px: int = 0
    camera: Any = None
    range_fn: Optional[Callable[[BoundingBox, int, int], RangeEstimate]] = None
    association: Optional[AssociationParams] = None

    _next_track_id: int = 1
    _tracks: Dict[int, _Track] = field(default_factory=dict)
    _last_dt_s: float = 0.0

    def __post_init__(self) -> None:
        if self.confirm_hits < 1:
            raise ValidationError("confirm_hits must be at least 1")
        if self.confirm_window < self.confirm_hits:
            raise ValidationError("confirm_window must be at least confirm_hits")
        if self.max_missed < 0:
            raise ValidationError("max_missed must be non-negative")
        if self.min_dt_s <= 0.0 or self.max_dt_s < self.min_dt_s:
            raise ValidationError("require 0 < min_dt_s <= max_dt_s")
        if self.focal_length_px <= 0.0:
            raise ValidationError("focal_length_px must be positive")
        if self.object_height_m <= 0.0:
            raise ValidationError("object_height_m must be positive")
        if self.min_box_height_px <= 0.0:
            # TrackerConfig permits 0.0 and the old code divided by it.
            raise ValidationError("min_box_height_px must be positive")
        if self.max_detections < 1 or self.max_detections > MAX_ASSIGNMENT_DIM:
            raise ValidationError(
                "max_detections must be in [1, %d]" % MAX_ASSIGNMENT_DIM
            )
        if self.max_tracks < 1 or self.max_tracks > MAX_ASSIGNMENT_DIM:
            raise ValidationError("max_tracks must be in [1, %d]" % MAX_ASSIGNMENT_DIM)
        if self.association is None:
            self.association = AssociationParams(
                max_centre_distance_px=float(self.association_threshold_px)
            )

    # ------------------------------------------------------------------ API #

    def update(
        self,
        detections: Sequence[BoundingBox],
        dt_s: float = 0.05,
        frame_width: int = 0,
        frame_height: int = 0,
        lane: Optional[LaneModel] = None,
        drivable: Optional[DrivableArea] = None,
    ) -> List[TrackedObject]:
        """Advance every track by ``dt_s`` and fold in this frame's detections.

        Args:
            detections: This frame's boxes. Invalid ones are dropped with a
                warning rather than failing the frame.
            dt_s: Measured time since the previous call, seconds. Clamped to
                ``[min_dt_s, max_dt_s]``; a value outside
                ``[0.5*nominal, 3*nominal]`` is logged.
            frame_width, frame_height: Source frame size in pixels. Needed to
                detect a truncated box and to place an object laterally; falls
                back to ``frame_width_px`` / ``frame_height_px``, and ``0`` means
                unknown.
            lane: This frame's lane model, for the ego-lane decision.
            drivable: This frame's drivable-area mask, for the ego-lane decision
                when no metric lane geometry is available.

        Returns:
            The CONFIRMED tracks only, in ascending track-id order. Tentative
            tracks are withheld; :meth:`tentative_tracks` exposes them for
            logging and overlay.

        Raises:
            TrackingError: only for a failure affecting the whole update.
        """
        if detections is None:
            raise TrackingError("detections must be a sequence, got None")

        width = int(frame_width) if frame_width else int(self.frame_width_px)
        height = int(frame_height) if frame_height else int(self.frame_height_px)
        dt = self._resolve_dt(dt_s)
        self._last_dt_s = dt

        valid = self._validated(detections)

        try:
            self._predict_all(dt)
            matches, unmatched_tracks, unmatched_dets = self._associate(valid)
        except TrackingError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise TrackingError("Tracker association failed: %s" % exc) from exc

        for track_id, det_index in matches:
            self._apply_match(self._tracks[track_id], valid[det_index], width, height)
        for track_id in unmatched_tracks:
            self._apply_miss(self._tracks[track_id], width, height)
        for det_index in unmatched_dets:
            self._spawn(valid[det_index], width, height)

        self._prune()

        return self._publish(lane, drivable, width, height)

    def tentative_tracks(self) -> List[TrackedObject]:
        """Unconfirmed tracks, for logging and overlay only.

        These must never reach the planner or the safety arbiter: a single
        false-positive detection produces exactly one of these.
        """
        return [
            self._to_tracked_object(track)
            for track in self._ordered_tracks()
            if track.status is TrackStatus.TENTATIVE
        ]

    def diagnostics(self) -> List[Dict[str, Any]]:
        """Per-track internals, for logs, tests and the recorder.

        Includes the quantities ``TrackedObject`` has no field for: the range
        variance, the association gate radius, how the ``in_ego_lane`` decision
        was reached, and how often the range filter had to be reinitialised.
        """
        out: List[Dict[str, Any]] = []
        for track in self._ordered_tracks():
            gate_px = track.box_filter.centre_gate_radius_px(self.association.chi2_gate)
            out.append(
                {
                    "track_id": track.track_id,
                    "status": track.status.value,
                    "label": track.label,
                    "age_frames": track.age_frames,
                    "hits": track.hits,
                    "time_since_update": track.time_since_update,
                    "recent_match_ratio": (
                        float(sum(1 for hit in track.history if hit)) / float(len(track.history))
                        if track.history
                        else 0.0
                    ),
                    "distance_m": track.range_filter.distance_m,
                    "range_sigma_m": math.sqrt(max(0.0, track.range_filter.distance_variance_m2)),
                    "closing_mps": track.range_filter.closing_mps,
                    "range_rate_sigma_mps": math.sqrt(
                        max(0.0, track.range_filter.rate_variance_m2s2)
                    ),
                    "range_accel_mps2": track.range_filter.accel_mps2,
                    "range_source": track.range_estimate.source.value,
                    "range_truncated": track.range_estimate.truncated,
                    "range_reinits": track.range_reinits,
                    "filter_saturations": track.range_filter.saturations,
                    "gate_radius_px": gate_px,
                    "lateral_offset_m": track.lateral_offset_m,
                    "lateral_offset_known": track.lateral_offset_known,
                    "in_ego_lane": track.in_ego_lane,
                    "ego_lane_source": track.ego_lane_source,
                }
            )
        return out

    def reset(self) -> None:
        """Drop every track and restart id allocation."""
        logger.info("Tracker reset (%d tracks discarded)", len(self._tracks))
        self._tracks.clear()
        self._next_track_id = 1
        self._last_dt_s = 0.0

    # -------------------------------------------------------------- timing #

    def _resolve_dt(self, dt_s: float) -> float:
        """Clamp ``dt_s`` into the usable band and warn when it is anomalous."""
        try:
            dt = float(dt_s)
        except (TypeError, ValueError):
            logger.warning("Non-numeric dt_s %r; using nominal %.3f s", dt_s, self.nominal_dt_s)
            return float(self.nominal_dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            logger.warning("Invalid dt_s %r; using nominal %.3f s", dt_s, self.nominal_dt_s)
            return float(self.nominal_dt_s)
        clamped = min(max(dt, self.min_dt_s), self.max_dt_s)
        if clamped != dt:
            logger.warning("dt_s %.4f s clamped to %.4f s", dt, clamped)
        elif dt < 0.5 * self.nominal_dt_s or dt > 3.0 * self.nominal_dt_s:
            logger.warning(
                "dt_s %.4f s is far from the nominal %.4f s; rate estimates will be noisy",
                dt,
                self.nominal_dt_s,
            )
        return clamped

    # ---------------------------------------------------------- detections #

    def _validated(self, detections: Sequence[BoundingBox]) -> List[BoundingBox]:
        valid: List[BoundingBox] = []
        for det in detections:
            try:
                validate_bounding_box(det)
            except ValidationError as exc:
                logger.warning("Invalid detection skipped: %s", exc)
                continue
            if not all(
                math.isfinite(float(v)) for v in (det.x1, det.y1, det.x2, det.y2)
            ):
                logger.warning("Non-finite detection skipped: %r", det)
                continue
            valid.append(det)
        if len(valid) > self.max_detections:
            valid.sort(key=lambda box: float(box.confidence), reverse=True)
            logger.warning(
                "%d detections exceeds max_detections=%d; dropping the %d weakest",
                len(valid),
                self.max_detections,
                len(valid) - self.max_detections,
            )
            valid = valid[: self.max_detections]
        return valid

    # ------------------------------------------------------------- predict #

    def _predict_all(self, dt: float) -> None:
        for track in list(self._tracks.values()):
            try:
                track.box_filter.predict(dt)
                track.range_filter.predict(dt)
            except (FilterError, ValidationError) as exc:
                logger.warning("Track %d prediction failed: %s", track.track_id, exc)
            track.age_frames += 1
            track.time_since_update += 1

    # --------------------------------------------------------- association #

    def _associate(
        self, detections: List[BoundingBox]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Two-stage cascade: confirmed tracks bid first, then tentative ones.

        Both stages are global Hungarian solves, so neither is order dependent.
        The cascade only expresses a priority between *classes* of track: a
        confirmed lead vehicle must not lose its detection to a two-frame-old
        tentative track that happens to sit slightly closer to it.
        """
        matches: List[Tuple[int, int]] = []
        remaining = list(range(len(detections)))

        for stage_confirmed in (True, False):
            stage_tracks = [
                track
                for track in self._ordered_tracks()
                if (track.status is not TrackStatus.TENTATIVE) == stage_confirmed
            ]
            if not stage_tracks or not remaining:
                continue
            stage_dets = [detections[i] for i in remaining]
            assignment = self._solve_stage(stage_tracks, stage_dets)
            matched_local = set()
            for track_index, det_index in assignment.matches:
                matches.append((stage_tracks[track_index].track_id, remaining[det_index]))
                matched_local.add(det_index)
            remaining = [remaining[i] for i in range(len(remaining)) if i not in matched_local]

        matched_ids = {track_id for track_id, _ in matches}
        unmatched_tracks = [
            track.track_id for track in self._ordered_tracks() if track.track_id not in matched_ids
        ]
        return matches, unmatched_tracks, remaining

    def _solve_stage(self, tracks: List[_Track], detections: List[BoundingBox]):
        m = len(detections)
        det_boxes = [
            (float(d.x1), float(d.y1), float(d.x2), float(d.y2)) for d in detections
        ]
        det_labels = [d.label for d in detections]
        det_cx = [(b[0] + b[2]) / 2.0 for b in det_boxes]
        det_cy = [(b[1] + b[3]) / 2.0 for b in det_boxes]
        det_h = [b[3] - b[1] for b in det_boxes]

        maha2: List[List[float]] = []
        track_boxes: List[Tuple[float, float, float, float]] = []
        track_labels: List[str] = []
        for track in tracks:
            track_boxes.append(track.box_filter.box_xyxy())
            track_labels.append(track.label)
            try:
                maha2.append(track.box_filter.gate_centre_batch(det_cx, det_cy, det_h))
            except (FilterError, ValidationError) as exc:  # pragma: no cover - defensive
                logger.warning("Track %d gating failed: %s", track.track_id, exc)
                maha2.append([float("inf")] * m)

        cost = build_cost_matrix(
            maha2, track_boxes, track_labels, det_boxes, det_labels, self.association
        )
        return solve_assignment(cost, max_cost=self.association.max_cost)

    # ------------------------------------------------------------ updating #

    def _apply_match(
        self, track: _Track, det: BoundingBox, width: int, height: int
    ) -> None:
        track.time_since_update = 0
        track.hits += 1
        track.vote(det.label)
        track.confidence = float(det.confidence)
        track.box = det
        track.history.append(True)

        try:
            track.box_filter.update((det.x1, det.y1, det.x2, det.y2))
        except (FilterError, ValidationError) as exc:
            logger.warning("Track %d box update failed: %s", track.track_id, exc)

        estimate = self._measure_range(det, track.label, width, height)
        track.range_estimate = estimate
        if estimate.source is not RangeSource.UNAVAILABLE and estimate.distance_m > 0.0:
            sigma = self._range_sigma(estimate, track.label)
            try:
                maha2 = track.range_filter.gate(estimate.distance_m, sigma)
                if maha2 > self.range_reinit_maha2:
                    logger.warning(
                        "Track %d range measurement %.1f m is %.0f sigma from the "
                        "filtered %.1f m; reinitialising the range filter",
                        track.track_id,
                        estimate.distance_m,
                        math.sqrt(maha2),
                        track.range_filter.distance_m,
                    )
                    track.range_filter.reinitialise(estimate.distance_m, sigma)
                    track.range_reinits += 1
                else:
                    track.range_filter.update(estimate.distance_m, sigma)
            except (FilterError, ValidationError) as exc:
                logger.warning("Track %d range update failed: %s", track.track_id, exc)

        if track.status is TrackStatus.TENTATIVE:
            if track.hits >= self.confirm_hits:
                track.status = TrackStatus.CONFIRMED
                logger.debug(
                    "Track %d confirmed (%d hits in %d frames)",
                    track.track_id,
                    track.hits,
                    track.age_frames,
                )
        else:
            track.status = TrackStatus.CONFIRMED

    def _apply_miss(self, track: _Track, width: int, height: int) -> None:
        """No detection this frame: coast on the prediction, publish no measurement.

        The published box is the Kalman *prediction*, not the frozen last
        detection. A frozen box made a closing lead appear to stop closing for
        the whole coast (ADAS-DEC-16); the prediction keeps the range rate alive
        while the covariance grows to say how much less it is now worth.
        ``time_since_update`` is the flag consumers use to detect this state.
        """
        track.history.append(False)
        track.range_estimate = RangeEstimate(0.0, 0.0, RangeSource.UNAVAILABLE, False)
        if track.status is TrackStatus.CONFIRMED:
            track.status = TrackStatus.COASTING
        track.box = self._predicted_box(track, width, height)

    def _predicted_box(self, track: _Track, width: int, height: int) -> BoundingBox:
        """Predicted box as a :class:`BoundingBox`, clipped into the frame.

        Clipping mirrors what the detector itself would have reported for an
        object leaving the frame; without a known frame size only the
        non-negativity that ``validate_bounding_box`` requires is enforced.
        """
        x1, y1, x2, y2 = track.box_filter.box_xyxy()
        max_x = float(width - 1) if width > 0 else None
        max_y = float(height - 1) if height > 0 else None
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        if max_x is not None and max_x > 1.0:
            x1 = max(0.0, min(x1, max_x - 1.0))
            x2 = min(max(x2, x1 + 1.0), max_x)
        x2 = max(x2, x1 + 1.0)
        if max_y is not None and max_y > 1.0:
            y1 = max(0.0, min(y1, max_y - 1.0))
            y2 = min(max(y2, y1 + 1.0), max_y)
        y2 = max(y2, y1 + 1.0)
        return BoundingBox(
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
            confidence=track.confidence,
            label=track.label,
        )

    def _spawn(self, det: BoundingBox, width: int, height: int) -> None:
        estimate = self._measure_range(det, det.label, width, height)
        sigma = self._range_sigma(estimate, det.label)
        initial = estimate.distance_m
        if estimate.source is RangeSource.UNAVAILABLE or initial <= 0.0:
            # No usable measurement. Seed the filter from the raw pinhole value
            # with the largest sigma we allow, so the first real measurement
            # dominates immediately, and keep publishing UNAVAILABLE until one
            # arrives.
            initial = self._pinhole_distance_m(det, det.label)
            sigma = 60.0
        try:
            track = _Track(
                track_id=self._next_track_id,
                box=det,
                box_filter=BoxFilter((det.x1, det.y1, det.x2, det.y2)),
                range_filter=RangeFilter(initial, sigma, jerk_psd=self.jerk_psd),
                label_votes={det.label: 1},
                history=deque([True], maxlen=max(1, self.confirm_window)),
                age_frames=1,
                hits=1,
                time_since_update=0,
                confidence=float(det.confidence),
                range_estimate=estimate,
            )
        except ValidationError as exc:
            logger.warning("Refusing to create a track from %r: %s", det, exc)
            return
        if self.confirm_hits <= 1:
            track.status = TrackStatus.CONFIRMED
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        logger.debug("Track %d created (%s, %.1f m)", track.track_id, det.label, initial)

    def _prune(self) -> None:
        for track_id in list(self._tracks):
            track = self._tracks[track_id]
            if track.status is TrackStatus.TENTATIVE:
                # M-of-N: a track that has not earned confirm_hits inside the
                # window is a false positive and is discarded without ever
                # having been reported.
                if track.time_since_update > 0 and track.age_frames >= self.confirm_window:
                    logger.debug(
                        "Tentative track %d discarded (%d hits in %d frames)",
                        track_id,
                        track.hits,
                        track.age_frames,
                    )
                    del self._tracks[track_id]
                elif track.time_since_update > self.max_missed:
                    del self._tracks[track_id]
            elif track.time_since_update > self.max_missed:
                logger.debug(
                    "Track %d deleted after coasting %d frames", track_id, track.time_since_update
                )
                del self._tracks[track_id]
        self._enforce_track_budget()

    def _enforce_track_budget(self) -> None:
        """Keep the live track set inside ``max_tracks``.

        Sacrifices tentative tracks with the fewest hits first, then the
        longest-coasting confirmed ones. A confirmed track that was measured
        this frame is never dropped: that is the object the planner is most
        likely to be following.
        """
        excess = len(self._tracks) - self.max_tracks
        if excess <= 0:
            return
        candidates = sorted(
            self._tracks.values(),
            key=lambda track: (
                track.status is not TrackStatus.TENTATIVE,
                track.time_since_update == 0,
                track.hits,
                -track.time_since_update,
            ),
        )
        for track in candidates[:excess]:
            logger.warning(
                "Track budget %d exceeded; dropping track %d (%s, %d hits)",
                self.max_tracks,
                track.track_id,
                track.status.value,
                track.hits,
            )
            self._tracks.pop(track.track_id, None)

    # ------------------------------------------------------------- publish #

    def _publish(
        self,
        lane: Optional[LaneModel],
        drivable: Optional[DrivableArea],
        width: int,
        height: int,
    ) -> List[TrackedObject]:
        out: List[TrackedObject] = []
        for track in self._ordered_tracks():
            try:
                self._resolve_geometry(track, lane, drivable, width, height)
            except Exception as exc:  # pragma: no cover - defensive, per track
                logger.warning("Track %d geometry failed: %s", track.track_id, exc)
                track.in_ego_lane = False
                track.ego_lane_source = "error"
            if track.status is TrackStatus.TENTATIVE:
                continue
            out.append(self._to_tracked_object(track))
        return out

    def _to_tracked_object(self, track: _Track) -> TrackedObject:
        distance_m = min(float(self.max_distance_m), track.range_filter.distance_m)
        closing = track.range_filter.closing_mps
        return TrackedObject(
            track_id=track.track_id,
            box=track.box,
            velocity_mps=closing,
            distance_m=distance_m,
            age_frames=track.age_frames,
            hits=track.hits,
            time_since_update=track.time_since_update,
            lateral_offset_m=track.lateral_offset_m,
            ttc_s=self.time_to_collision_s(distance_m, closing),
            range_estimate=track.range_estimate,
            in_ego_lane=track.in_ego_lane,
        )

    def time_to_collision_s(self, distance_m: float, closing_mps: float) -> float:
        """Seconds until the range reaches zero at the current closing speed.

        ``closing_mps`` uses the positive-when-closing convention of
        ``TrackedObject.velocity_mps``.

        Degenerate cases, all of them explicit:

        * a non-finite range or closing speed -> ``inf`` (unknown, not urgent);
        * range already at or below zero -> ``0.0`` (contact);
        * an opening range, or a closing speed below ``ttc_min_closing_mps``
          -> ``inf``. The floor exists because ``d / eps`` is arbitrarily large
          but *finite*, and a finite huge TTC reads as a real number to a
          downstream threshold while an infinite one reads as "not closing".

        Consumers that care about a standstill gap (the arbiter does) subtract
        it from the range themselves; this is a pure contact time.
        """
        d = float(distance_m)
        v = float(closing_mps)
        if not math.isfinite(d) or not math.isfinite(v):
            return float("inf")
        if d <= 0.0:
            return 0.0
        if v <= self.ttc_min_closing_mps:
            return float("inf")
        return d / v

    def _ordered_tracks(self) -> List[_Track]:
        return [self._tracks[key] for key in sorted(self._tracks)]

    # --------------------------------------------------------------- range #

    def _object_height_m(self, label: str) -> float:
        if not self.use_class_heights:
            return float(self.object_height_m)
        return float(
            NOMINAL_OBJECT_HEIGHT_M.get(str(label).strip().lower(), self.object_height_m)
        )

    def _pinhole_distance_m(self, box: BoundingBox, label: str) -> float:
        """Raw ``H*f/h`` range, metres, clamped to ``max_distance_m``. Never raises."""
        box_height = max(float(self.min_box_height_px), float(box.y2) - float(box.y1))
        distance = (self._object_height_m(label) * float(self.focal_length_px)) / box_height
        if not math.isfinite(distance) or distance < 0.0:
            return float(self.max_distance_m)
        return float(min(distance, self.max_distance_m))

    def _measure_range(
        self, box: BoundingBox, label: str, width: int, height: int
    ) -> RangeEstimate:
        """This frame's raw range measurement for one detection.

        Prefers the injected ``range_fn`` (a calibrated geometry channel that can
        fall back to the road plane when the box is truncated). Falls back to the
        class-keyed pinhole model, which flags truncation but cannot correct it:
        a bottom-clipped box reads too FAR, so the estimate is published with a
        reduced confidence and ``truncated=True`` and the caller inflates its
        variance fivefold.
        """
        if self.range_fn is not None:
            try:
                estimate = self.range_fn(box, width, height)
            except Exception as exc:
                logger.warning("Injected range_fn failed (%s); using the pinhole fallback", exc)
                estimate = None
            if estimate is not None:
                if (
                    isinstance(estimate, RangeEstimate)
                    and math.isfinite(estimate.distance_m)
                    and estimate.distance_m >= 0.0
                ):
                    return RangeEstimate(
                        distance_m=float(min(estimate.distance_m, self.max_distance_m)),
                        confidence=float(max(0.0, min(1.0, estimate.confidence))),
                        source=estimate.source,
                        truncated=bool(estimate.truncated),
                    )
                logger.warning("Injected range_fn returned %r; using the pinhole fallback", estimate)

        box_height = float(box.y2) - float(box.y1)
        truncated = False
        if height > 0:
            truncated = float(box.y2) >= float(height) - 2.0 or float(box.y1) <= 1.0
        distance = self._pinhole_distance_m(box, label)
        if box_height < self.min_range_box_height_px:
            # A box this short cannot carry a range: at f=910 and H=1.5 one pixel
            # of a 7 px box moves the answer by 28 m. Say so.
            return RangeEstimate(distance, 0.0, RangeSource.UNAVAILABLE, truncated)
        confidence = float(max(0.0, min(1.0, box.confidence)))
        if truncated:
            confidence *= 0.3
        return RangeEstimate(distance, confidence, RangeSource.PINHOLE, truncated)

    def _range_sigma(self, estimate: RangeEstimate, label: str) -> float:
        return range_measurement_sigma_m(
            estimate.distance_m,
            focal_length_px=self.focal_length_px,
            object_height_m=self._object_height_m(label),
            box_sigma_px=self.box_sigma_px,
            confidence=estimate.confidence if estimate.confidence > 0.0 else 0.15,
            truncated=estimate.truncated,
        )

    # ------------------------------------------------------------ geometry #

    def _resolve_geometry(
        self,
        track: _Track,
        lane: Optional[LaneModel],
        drivable: Optional[DrivableArea],
        width: int,
        height: int,
    ) -> None:
        distance_m = min(float(self.max_distance_m), track.range_filter.distance_m)
        offset, known = self._lateral_offset_m(track.box, distance_m, width)
        track.lateral_offset_m = offset
        track.lateral_offset_known = known

        half_width_m = self._object_half_width_m(track.box, distance_m)

        decision = self._lane_corridor(lane, distance_m, offset, half_width_m) if known else None
        if decision is not None:
            track.in_ego_lane, track.ego_lane_source = decision, "lane_geometry"
            return

        decision = self._drivable_corridor(track.box, drivable, lane, width, height)
        if decision is not None:
            track.in_ego_lane, track.ego_lane_source = decision, "drivable_area"
            return

        if known:
            decision = self._ego_corridor(offset, half_width_m)
            track.in_ego_lane, track.ego_lane_source = decision, "ego_corridor_heuristic"
            return

        decision = self._image_band(track.box, lane, distance_m, width)
        if decision is not None:
            track.in_ego_lane, track.ego_lane_source = decision, "image_band_heuristic"
            return

        track.in_ego_lane = False
        track.ego_lane_source = "unavailable"

    def _lateral_offset_m(
        self, box: BoundingBox, distance_m: float, width: int
    ) -> Tuple[float, bool]:
        """Signed lateral position of the object, metres, positive to the RIGHT.

        Measured at the box's ground-contact point (bottom-centre). With a
        calibrated camera this is a road-plane back-projection. Without one it is
        the small-angle pinhole approximation ``(u - W/2) * Z / f``, which assumes
        zero camera yaw and a principal point at the image centre -- good to a
        few tens of centimetres for a dash mount, and flagged as *unknown* when
        even the frame width is not available so no caller mistakes 0.0 for a
        measurement.
        """
        u = (float(box.x1) + float(box.x2)) / 2.0
        v = float(box.y2)
        if self.camera is not None:
            try:
                ground = self.camera.image_to_ground(u, v)
            except Exception as exc:
                logger.warning("camera.image_to_ground failed: %s", exc)
                ground = None
            if ground is not None:
                try:
                    x_m = float(ground[0])
                except (TypeError, ValueError, IndexError):
                    x_m = float("nan")
                if math.isfinite(x_m):
                    return x_m, True
        if width <= 0:
            return 0.0, False
        x_m = (u - width / 2.0) * distance_m / float(self.focal_length_px)
        if not math.isfinite(x_m):
            return 0.0, False
        return x_m, True

    def _object_half_width_m(self, box: BoundingBox, distance_m: float) -> float:
        """Half of the object's metric width, from its box width and range.

        Clamped to ``[0.3, 1.5]`` m: narrower is not a road user, wider is a
        detector box that has swallowed something else.
        """
        width_px = float(box.x2) - float(box.x1)
        half = 0.5 * width_px * distance_m / float(self.focal_length_px)
        if not math.isfinite(half):
            return 0.9
        return float(min(1.5, max(0.3, half)))

    @staticmethod
    def _poly_x_at(coeffs: Any, z_m: float) -> Optional[float]:
        try:
            a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        except (TypeError, ValueError, IndexError):
            return None
        value = a * z_m * z_m + b * z_m + c
        return value if math.isfinite(value) else None

    def _lane_corridor(
        self,
        lane: Optional[LaneModel],
        distance_m: float,
        lateral_offset_m: float,
        half_width_m: float,
    ) -> Optional[bool]:
        """Occupancy test against the metric ego-lane boundaries.

        Returns ``None`` when there is no trustworthy metric lane to test
        against -- a mock lane, a low-confidence fit, a fit with no ground-plane
        coefficients, or an implausible lane width. Returning ``None`` hands the
        decision to the next-best source rather than guessing.
        """
        if lane is None or getattr(lane, "is_mock", True):
            return None
        if float(getattr(lane, "confidence", 0.0)) < self.min_lane_confidence:
            return None
        lines = getattr(lane, "lines", None) or []
        by_index: Dict[int, Any] = {}
        for line in lines:
            coeffs = getattr(line, "coeffs", None)
            if coeffs is None:
                continue
            by_index[int(getattr(line, "index", -99))] = coeffs

        z = max(0.0, float(distance_m))
        left = self._poly_x_at(by_index[_LANE_LEFT_INDEX], z) if _LANE_LEFT_INDEX in by_index else None
        right = (
            self._poly_x_at(by_index[_LANE_RIGHT_INDEX], z) if _LANE_RIGHT_INDEX in by_index else None
        )
        centre = (
            self._poly_x_at(by_index[_LANE_CENTRE_INDEX], z)
            if _LANE_CENTRE_INDEX in by_index
            else None
        )

        half_lane = self.nominal_lane_width_m / 2.0
        if left is None and right is None:
            if centre is None:
                return None
            left, right = centre - half_lane, centre + half_lane
        elif left is None:
            left = right - self.nominal_lane_width_m
        elif right is None:
            right = left + self.nominal_lane_width_m
        if left > right:
            left, right = right, left
        lane_width = right - left
        if not (_MIN_LANE_WIDTH_M <= lane_width <= _MAX_LANE_WIDTH_M):
            logger.debug(
                "Lane width %.2f m at %.1f m is implausible; not gating on lane geometry",
                lane_width,
                z,
            )
            return None

        low = left - self.lane_margin_m
        high = right + self.lane_margin_m
        return bool(lateral_offset_m + half_width_m >= low and lateral_offset_m - half_width_m <= high)

    def _drivable_corridor(
        self,
        box: BoundingBox,
        drivable: Optional[DrivableArea],
        lane: Optional[LaneModel],
        width: int,
        height: int,
    ) -> Optional[bool]:
        """Is the object standing inside the free-space run that contains the ego path?

        Traces the contiguous free run of the drivable mask along the object's
        ground-contact row, starting from the ego reference column, and asks
        whether the object's own column band overlaps it. This is a *corridor*
        test, not a lane test: on a multi-lane road the free run spans every
        lane, so the answer is deliberately inclusive. Returns ``None`` when
        there is no usable mask.
        """
        if drivable is None or width <= 0 or height <= 0:
            return None
        if getattr(drivable, "mask", None) is None:
            return None
        if int(getattr(drivable, "width", 0)) <= 0 or int(getattr(drivable, "height", 0)) <= 0:
            return None
        if float(getattr(drivable, "confidence", 0.0)) < self.min_drivable_confidence:
            return None

        y_frac = min(1.0, max(0.0, float(box.y2) / float(height)))
        reference_px = (
            float(lane.lane_center_px)
            if lane is not None and math.isfinite(float(getattr(lane, "lane_center_px", float("nan"))))
            else width / 2.0
        )
        ref_frac = min(1.0, max(0.0, reference_px / float(width)))

        samples = _DRIVABLE_SAMPLES
        try:
            free = [
                bool(drivable.is_free((index + 0.5) / samples, y_frac)) for index in range(samples)
            ]
        except Exception as exc:
            logger.warning("DrivableArea.is_free failed: %s", exc)
            return None

        start = min(samples - 1, max(0, int(ref_frac * samples)))
        if not free[start]:
            # The ego's own column is not drivable at this row (a vehicle occupies
            # it, or the mask is wrong). No corridor to test against.
            return None
        low = start
        while low > 0 and free[low - 1]:
            low -= 1
        high = start
        while high < samples - 1 and free[high + 1]:
            high += 1

        obj_low = min(1.0, max(0.0, float(box.x1) / float(width))) * samples
        obj_high = min(1.0, max(0.0, float(box.x2) / float(width))) * samples
        return bool(obj_high >= low and obj_low <= high + 1)

    def _ego_corridor(self, lateral_offset_m: float, half_width_m: float) -> bool:
        """Metric straight-ahead corridor about the vehicle axis. Documented heuristic.

        Used when the object's lateral position is known in metres but there is
        no lane geometry and no drivable mask to shape the path. It assumes the
        ego continues straight, which is wrong in a curve, so the corridor is
        deliberately wide: half of a nominal lane scaled by
        ``heuristic_corridor_scale``, never narrower than the ego body
        (``ego_half_width_m``), plus ``lane_margin_m``.

        Inclusive on purpose, for the reason given in :meth:`_image_band`.
        """
        half = max(
            self.ego_half_width_m,
            self.nominal_lane_width_m / 2.0 * self.heuristic_corridor_scale,
        )
        return bool(abs(lateral_offset_m) - half_width_m <= half + self.lane_margin_m)

    def _image_band(
        self,
        box: BoundingBox,
        lane: Optional[LaneModel],
        distance_m: float,
        width: int,
    ) -> Optional[bool]:
        """Range-scaled image-x band around the lane centre. Documented fallback.

        The half width in pixels is the projection of half a nominal lane,
        widened by ``heuristic_corridor_scale``:
        ``f * (lane_width/2 * scale) / max(Z, 1)``. Unlike the fixed
        ``ego_lane_half_width_frac`` band this shrinks with range, so it does not
        merge three lanes together at 100 m.

        It is inclusive on purpose. ``BehaviorPlanner._objects_ahead`` discards
        every object *not* marked ``in_ego_lane`` as soon as any object is, so a
        false negative on the true lead vehicle hides it from the planner --
        the unsafe direction. A false positive only costs extra caution.

        Returns ``None`` when the frame width is unknown.
        """
        if width <= 0:
            return None
        reference_px = width / 2.0
        if lane is not None:
            candidate = float(getattr(lane, "lane_center_px", float("nan")))
            if math.isfinite(candidate) and 0.0 <= candidate <= float(width):
                reference_px = candidate
        z = max(1.0, float(distance_m))
        half_px = (
            float(self.focal_length_px)
            * (self.nominal_lane_width_m / 2.0 * self.heuristic_corridor_scale)
            / z
        )
        half_px = min(0.5 * width, max(0.02 * width, half_px))
        object_half_px = (float(box.x2) - float(box.x1)) / 2.0
        centre_px = (float(box.x1) + float(box.x2)) / 2.0
        return bool(abs(centre_px - reference_px) <= half_px + object_half_px)
