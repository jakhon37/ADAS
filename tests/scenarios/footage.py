"""Justification analysis of arbiter interventions on real recorded footage.

Why this exists
---------------
The synthetic sweep proves things about scenes we invented.  Real footage is
where the *inputs* are wrong in ways nobody invents: a track that flickers, a
range that jumps when the box is clipped, a lane that is not there.  Both
historical longitudinal blockers were caught here and nowhere else:

* the phantom AEB, when somebody listed all 42 intervening frames of a 400-frame
  run and found that every one of them was justified by a *seeded* closing rate
  while the measured range history was flat, and
* the fix report's claim of "0 frames at brake = 1.00", which was disproved by
  reading the actual per-frame brake out of the same run.

So the rule of this module is: **for every frame that intervenes, print the
evidence and let the scene decide.**  Not the arbiter's opinion of the scene --
the measurements it was handed.

What "justified" means here
---------------------------
The verdict is computed from the *raw tracker range history*, which is the
measurement channel, never from the arbiter's filtered range or its own closing
rate (those are the thing under test).  A least-squares slope over the preceding
window gives an observed closing rate -- the same estimator
:class:`tests.scenarios.plant.Sensor` now uses to report ``velocity_mps`` in
the synthetic world, deliberately, so that "the closing rate the measurements
support" means one thing in both halves of the harness.  That, plus the range
and the ego speed, goes into the same physics as :mod:`tests.scenarios.sweep`:

``JUSTIFIED``               the observed scene needs more than comfort braking.
``JUSTIFIED_HEADWAY``       a soft response on a gap below the safe following
                            distance.  Correct; opening a gap is not an AEB.
``JUSTIFIED_RANGE_ALONE``   the closing rate could not yet be measured, but the
                            range is short enough that even a *stationary*
                            obstacle there would demand more than comfort
                            braking, so the range alone justifies it.
``JUSTIFIED_DEGRADED``      perception or ego speed was unhealthy this frame;
                            the intervention is a health response, not a traffic
                            one, and this analyser does not grade it.
``UNJUSTIFIED_NO_TARGET``   nothing was in the ego path at all.
``UNJUSTIFIED_NOT_CLOSING`` there is a lead, but its measured range was flat or
                            opening.  This is the phantom signature.
``UNJUSTIFIED_INFERRED_RATE`` full-authority braking whose only support is the
                            seeded ``-ego_speed`` prior, at a range where a
                            stationary obstacle would not have demanded it.
``UNJUSTIFIED_OVERREACTION`` a real but mild closure answered with full
                            authority.
``INDETERMINATE``           too little history to say; reported, never counted
                            as either.

And the inverse, which matters just as much: every frame that did *not*
intervene is checked against the same physics, and a frame whose measured scene
demanded more than comfort braking is counted as ``MISSED_REACTION``.

A caveat this module prints in its own report
---------------------------------------------
On the bundled clip the ego speed is *simulated* (a constant, by default 15 m/s)
while the footage came from a real vehicle, so absolute time-to-collision
figures are not physical.  What is still exactly valid, and is the whole point,
is the internal consistency question: given the range history the arbiter itself
was handed, and the ego speed it itself believed, does its own action follow?

Recording and replay
--------------------
:func:`record_run` drives the real pipeline (YOLOX + UFLD) once and writes a
recording; every later analysis reads that file, so the GPU is needed once, not
every time.  The live path must be run under the board's GPU mutex::

    ssh jetson-nx 'flock /tmp/jetson-gpu.lock -c "cd ~/myspace/ADAS && \\
        python3 scripts/analyze_run.py record --out /tmp/run400.json"'
    python3 scripts/analyze_run.py analyze /tmp/run400.json
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tests.scenarios.instrument import (
    ArbiterInstrument,
    FrameRecord,
    frames_from_jsonable,
    frames_to_jsonable,
)
from tests.scenarios.oracle import (
    JUSTIFICATION_MARGIN_FACTOR,
    JUSTIFICATION_TOLERANCE_MPS2,
    required_decel_mps2,
)
from tests.scenarios.sweep import SweepSpec, safe_following_gap_m

__all__ = [
    "AnalysisParams",
    "RunRecording",
    "Intervention",
    "MissedReaction",
    "AnalysisReport",
    "record_run",
    "analyse",
    "format_report",
]

RECORDING_VERSION = 1


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AnalysisParams:
    """Knobs for the justification analysis.

    The physics constants live in :class:`~tests.scenarios.sweep.SweepSpec` and
    are shared with the sweep on purpose: one specification, two harnesses.
    """

    spec: SweepSpec = field(default_factory=SweepSpec)

    brake_threshold: float = 0.10
    """A brake above this counts as an intervention worth justifying."""
    hard_brake_threshold: float = 0.50
    """A brake above this is a full-authority intervention at the wheels.  The
    *verdicts* use ``spec.emergency_decel_mps2`` on the arbiter's own
    deceleration instead, so that a pedal pressed by the planner is never
    counted against the arbiter; this stays only for the frame-selection filter
    and the report."""
    intervening_states: Tuple[str, ...] = ("limited", "min_risk_maneuver", "disengage")

    history_frames: int = 12
    """How many preceding frames of raw range go into the report and the slope."""
    min_history_samples: int = 4
    """Fewer raw samples than this and the observed rate is called unknown."""
    min_history_span_s: float = 0.15
    """Samples closer together in time than this cannot resolve a closure."""
    not_closing_rate_mps: float = -0.5
    """A measured slope above this (i.e. flatter) is 'not closing'."""
    margin_factor: float = JUSTIFICATION_MARGIN_FACTOR
    """Multiplier on the true requirement before a brake is called excessive.
    ``oracle.JUSTIFICATION_MARGIN_FACTOR``: the requirement is a theoretical
    minimum that assumes an exact range and an instant brake, and a designer has
    to allow for neither being true."""
    margin_tolerance_mps2: float = JUSTIFICATION_TOLERANCE_MPS2
    """Additive slack on top of that, ``oracle.JUSTIFICATION_TOLERANCE_MPS2``."""
    significance_sigma: float = 2.0
    """How many standard errors from zero a fitted slope must be before it counts
    as a measured closure rather than range noise."""


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #


@dataclass
class RunRecording:
    """A captured run: metadata plus one :class:`FrameRecord` per arbitration."""

    meta: Dict[str, Any] = field(default_factory=dict)
    frames: List[FrameRecord] = field(default_factory=list)

    def save(self, path: str) -> None:
        """Write JSON.  Sorted keys and a fixed indent, so runs diff cleanly."""
        payload = {
            "version": RECORDING_VERSION,
            "meta": self.meta,
            "frames": frames_to_jsonable(self.frames),
        }
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "RunRecording":
        """Read a recording written by :meth:`save`."""
        with open(path) as handle:
            payload = json.load(handle)
        return cls(
            meta=payload.get("meta", {}),
            frames=frames_from_jsonable(payload.get("frames", [])),
        )


def record_run(
    source: str = "Ultra-Fast-Lane-Detection-v2/example.mp4",
    frames: int = 400,
    ego_speed_mps: float = 15.0,
    detector: str = "yolox",
    lane: str = "ufld",
    target_fps: float = 20.0,
    config_path: Optional[str] = None,
) -> RunRecording:
    """Drive the real pipeline once, instrumented, and return the recording.

    This is the only function in the harness that needs the GPU.  Run it under
    the board's GPU mutex (``flock /tmp/jetson-gpu.lock``); everything else
    works from the file it writes.

    Args:
        source: Video path, relative to the repo root.
        frames: How many frames to run.
        ego_speed_mps: The simulated ego speed. The clip carries no ego telemetry;
            see the caveat in the module docstring.
        detector: Detector backend, e.g. ``"yolox"``.
        lane: Lane backend, e.g. ``"ufld"``.
        target_fps: Runner pacing.
        config_path: Optional JSON config to start from.

    Returns:
        A :class:`RunRecording`.

    Raises:
        Whatever the pipeline build raises when an engine is missing.
    """
    from adas.cli import build_pipeline
    from adas.core.config import default_config, load_config
    from adas.runtime.capture import SimulatedEgoSpeed
    from adas.runtime.runner import PipelineRunner

    cfg = load_config(config_path) if config_path else default_config()
    cfg.detector.backend = detector
    cfg.lane.backend = lane
    cfg.source.type = "video"
    cfg.source.uri = source
    cfg.__post_init__()

    pipeline, cfg = build_pipeline(config=cfg)
    instrument = ArbiterInstrument()
    try:
        with instrument:
            runner = PipelineRunner(
                pipeline,
                target_fps=target_fps,
                ego_source=SimulatedEgoSpeed(ego_speed_mps),
            )
            summary = runner.run(
                source_type="video", uri=cfg.source.uri, max_frames=frames, as_image=True
            )
    finally:
        pipeline.close()

    meta = {
        "source": source,
        "requested_frames": frames,
        "ego_speed_mps": ego_speed_mps,
        "ego_speed_source": "simulated",
        "detector": detector,
        "lane": lane,
        "target_fps": target_fps,
        "arbitrations": len(instrument.frames),
        "loop_frames": summary.frames,
        "settle_commands": summary.settle_commands,
        "settle_kind": summary.settle_kind,
        "settle_brake": summary.settle_brake,
        "stopped_reason": summary.stopped_reason,
        "missing_hooks": list(instrument.missing_hooks),
    }
    return RunRecording(meta=meta, frames=list(instrument.frames))


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


@dataclass
class RangeSample:
    """One raw range measurement for the lead track."""

    frame_index: int
    timestamp_s: Optional[float]
    raw_distance_m: Optional[float]
    filtered_distance_m: Optional[float]


@dataclass
class Intervention:
    """One intervening frame, its evidence and its verdict."""

    frame_index: int
    timestamp_s: Optional[float]
    state: str
    brake_in: Optional[float]
    brake_out: Optional[float]
    brake_actuated: Optional[float]
    demanded_decel_mps2: Optional[float]
    ego_speed_mps: Optional[float]
    hard: bool

    lead_track_id: Optional[int] = None
    lead_raw_range_m: Optional[float] = None
    lead_filtered_range_m: Optional[float] = None
    lead_rate_used_mps: Optional[float] = None
    rate_is_measured: Optional[bool] = None
    rate_updates: int = 0
    range_source: str = ""
    coasting: bool = False

    ego_lane_occupied_tracker: int = 0
    ego_lane_occupied_arbiter: Optional[int] = None
    n_tracks: int = 0

    history: List[RangeSample] = field(default_factory=list)
    observed_rate_mps: Optional[float] = None
    """Least-squares slope of the RAW range over the window; negative closes."""
    observed_rate_stderr_mps: float = 0.0
    """Standard error of that slope.  A slope within
    ``significance_sigma`` standard errors of zero is not a measurement."""
    rate_significant: bool = False
    observed_span_s: float = 0.0
    observed_samples: int = 0
    required_decel_mps2: Optional[float] = None
    """What the observed scene actually demands, m/s^2."""
    safe_gap_m: Optional[float] = None

    attribution: str = "PASSTHROUGH"
    """Who caused the pedal, and who wanted it.  See :func:`_attribute`.

    ``ARBITER_BRAKE``            the arbiter raised the brake above the one it
                                 was handed: the wheels moved because of it.
    ``ARBITER_DEMAND_ABSORBED``  the arbiter demanded deceleration of its own,
                                 but the incoming command already exceeded it,
                                 so the pedal shows nothing.  This is the
                                 dangerous one to miss: the demand is a real
                                 arbiter decision that only failed to show
                                 because the planner was braking anyway.
    ``ARBITER_STATE_ONLY``       a degraded state with no braking of its own.
    ``PASSTHROUGH``              the brake came from the planner/controller and
                                 the arbiter neither added to it nor objected.
    """
    arbiter_decel_mps2: float = 0.0
    """The deceleration the ARBITER itself is responsible for this frame: its own
    demand, or the amount by which it raised the incoming brake, whichever is
    larger.  Every justification verdict is about this number, never about the
    pedal, which may be the planner's."""
    arbiter_added_brake: bool = False
    counterfactual_brake: Optional[float] = None
    """``demand / brake_authority`` -- the brake the arbiter's own demand
    corresponds to, i.e. what it would have commanded on its own."""
    aeb_flag: Optional[bool] = None
    hazard_flag: Optional[bool] = None

    verdict: str = "INDETERMINATE"
    why: str = ""
    findings: List[str] = field(default_factory=list)
    reason: str = ""
    """The arbiter's own one-line reason string, verbatim.  Useful when the
    findings list is empty, which is how a latched state shows up."""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready."""
        out = dict(self.__dict__)
        out["history"] = [h.__dict__ for h in self.history]
        return out


@dataclass
class MissedReaction:
    """A frame whose measured scene demanded braking and got none."""

    frame_index: int
    timestamp_s: Optional[float]
    state: str
    brake_actuated: Optional[float]
    ego_speed_mps: Optional[float]
    lead_raw_range_m: Optional[float]
    observed_rate_mps: Optional[float]
    required_decel_mps2: float

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready."""
        return dict(self.__dict__)


@dataclass
class AnalysisReport:
    """The outcome of :func:`analyse`."""

    meta: Dict[str, Any] = field(default_factory=dict)
    frames_total: int = 0
    frames_with_lead: int = 0
    state_histogram: Dict[str, int] = field(default_factory=dict)
    brake_histogram: Dict[str, int] = field(default_factory=dict)
    max_brake_actuated: float = 0.0
    max_demand_mps2: float = 0.0
    frames_brake_full: int = 0
    frames_arbiter_raised_brake: int = 0
    interventions: List[Intervention] = field(default_factory=list)
    verdict_counts: Dict[str, int] = field(default_factory=dict)
    attribution_counts: Dict[str, int] = field(default_factory=dict)
    missed: List[MissedReaction] = field(default_factory=list)
    non_traffic_findings: Dict[str, int] = field(default_factory=dict)
    """What the state-only, non-traffic degradations were raised by."""
    lead_track_ids: List[int] = field(default_factory=list)
    lead_switches: int = 0
    frames_lead_lost_then_regained: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def unjustified(self) -> List[Intervention]:
        """Interventions whose verdict starts with ``UNJUSTIFIED``.

        Pass-through planner braking is excluded by construction: its verdict is
        ``PASSTHROUGH_PLANNER_BRAKE``, because it is not the arbiter's action.
        """
        return [i for i in self.interventions if i.verdict.startswith("UNJUSTIFIED")]

    @property
    def arbiter_interventions(self) -> List["Intervention"]:
        """Interventions the arbiter actually caused or wanted."""
        return [i for i in self.interventions if i.attribution != "PASSTHROUGH"]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready."""
        return {
            "meta": self.meta,
            "frames_total": self.frames_total,
            "frames_with_lead": self.frames_with_lead,
            "state_histogram": self.state_histogram,
            "brake_histogram": self.brake_histogram,
            "max_brake_actuated": self.max_brake_actuated,
            "max_demand_mps2": self.max_demand_mps2,
            "frames_brake_full": self.frames_brake_full,
            "frames_arbiter_raised_brake": self.frames_arbiter_raised_brake,
            "verdict_counts": self.verdict_counts,
            "attribution_counts": self.attribution_counts,
            "non_traffic_findings": self.non_traffic_findings,
            "lead_track_ids": self.lead_track_ids,
            "lead_switches": self.lead_switches,
            "interventions": [i.to_dict() for i in self.interventions],
            "missed": [m.to_dict() for m in self.missed],
            "warnings": self.warnings,
        }


def _slope(samples: Sequence[Tuple[float, float]]) -> Tuple[Optional[float], float]:
    """Least-squares slope of ``(t, y)`` pairs and its standard error.

    The standard error matters more than it looks.  Monocular range at 8 m
    jitters by more than a metre frame to frame, so a slope fitted over half a
    second can read -2 m/s on one frame and -4 m/s on the next from the same
    scene.  A slope that is not several standard errors from zero is not a
    measured closure, and this harness refuses to treat it as one -- which is
    the same discipline the arbiter is supposed to apply to its own rate.

    Returns:
        ``(slope, stderr)``; ``(None, 0.0)`` when the fit is degenerate.
    """
    n = len(samples)
    if n < 2:
        return None, 0.0
    mean_t = sum(t for t, _ in samples) / n
    mean_y = sum(y for _, y in samples) / n
    den = sum((t - mean_t) ** 2 for t, _ in samples)
    if den <= 1e-12:
        return None, 0.0
    slope = sum((t - mean_t) * (y - mean_y) for t, y in samples) / den
    if n <= 2:
        return slope, float("inf")
    intercept = mean_y - slope * mean_t
    resid = sum((y - (intercept + slope * t)) ** 2 for t, y in samples)
    return slope, math.sqrt((resid / (n - 2)) / den)


def _raw_for(record: FrameRecord, track_id: int) -> Optional[float]:
    """The raw tracker range for ``track_id`` on this frame, if present."""
    for track in record.tracks:
        if track.track_id == track_id:
            return track.distance_m
    return None


def _history(
    frames: Sequence[FrameRecord], index: int, track_id: int, window: int
) -> List[RangeSample]:
    """Raw and filtered range for ``track_id`` over the preceding ``window``."""
    start = max(0, index - window + 1)
    out: List[RangeSample] = []
    for i in range(start, index + 1):
        rec = frames[i]
        raw = _raw_for(rec, track_id)
        filtered = (
            rec.lead.distance_m if (rec.lead is not None and rec.lead.track_id == track_id) else None
        )
        if raw is None and filtered is None:
            continue
        out.append(
            RangeSample(
                frame_index=rec.frame_index,
                timestamp_s=rec.timestamp_s,
                raw_distance_m=raw,
                filtered_distance_m=filtered,
            )
        )
    return out


def _fallback_time(sample: RangeSample, dt_s: float) -> float:
    """Timestamp, or frame index scaled by ``dt_s`` when timestamps are absent."""
    if sample.timestamp_s is not None:
        return sample.timestamp_s
    return sample.frame_index * dt_s


def _observed_rate(
    history: Sequence[RangeSample], params: AnalysisParams
) -> Tuple[Optional[float], float, float, int]:
    """``(slope, stderr, span_s, n)`` of the RAW range over the window.

    ``slope`` is ``None`` when the window is too short or too brief to resolve a
    closure -- the same honesty the arbiter is supposed to apply to its own rate.
    """
    pairs = [
        (_fallback_time(h, params.spec.dt_s), h.raw_distance_m)
        for h in history
        if h.raw_distance_m is not None
    ]
    if len(pairs) < 2:
        return None, 0.0, 0.0, len(pairs)
    span = pairs[-1][0] - pairs[0][0]
    if len(pairs) < params.min_history_samples or span < params.min_history_span_s:
        return None, 0.0, span, len(pairs)
    slope, stderr = _slope(pairs)
    return slope, stderr, span, len(pairs)


def _allowance(required_mps2: float, params: AnalysisParams) -> float:
    """The largest deceleration still proportionate to ``required_mps2``.

    ``oracle``'s convention: 50 % above the theoretical minimum plus half a
    metre per second squared, which covers a 20 % range under-estimate together
    with the brake rise time and a frame of latency.  Above it the demand is not
    explained by the kinematics.
    """
    if not math.isfinite(required_mps2):
        return float("inf")
    return required_mps2 * params.margin_factor + params.margin_tolerance_mps2


def _finding_keys(item: Intervention) -> List[str]:
    """Group findings for a tally, collapsing per-frame numbers away.

    A state with no findings at all is keyed on the shape of its reason string,
    which is how a *latched* degradation -- one with no live evidence behind it
    at all -- becomes visible as a category instead of 33 unique strings.
    """
    if item.findings:
        return [f.split("_")[0] for f in item.findings]
    parts = [p.strip() for p in (item.reason or "").split("|")]
    tail = parts[1] if len(parts) > 1 else (parts[0] if parts else "")
    if tail.startswith("lead#"):
        tail = "lead present, no finding raised"
    return ["latched:" + (tail or "no reason recorded")]


def _attribute(item: Intervention, params: AnalysisParams) -> str:
    """Decide who is responsible for this frame's longitudinal action.

    The whole point of separating DEMAND from COMMAND: a brake at the wheels is
    not evidence about the arbiter, and an arbiter with no brake at the wheels is
    not evidence of restraint.  Only the demand says what the arbiter wanted.
    """
    if item.arbiter_added_brake:
        return "ARBITER_BRAKE"
    demand = item.demanded_decel_mps2 or 0.0
    if demand > 0.0:
        return "ARBITER_DEMAND_ABSORBED"
    if item.state in params.intervening_states:
        return "ARBITER_STATE_ONLY"
    return "PASSTHROUGH"


def _judge(item: Intervention, params: AnalysisParams) -> None:
    """Fill ``item.verdict`` and ``item.why`` from the measured evidence."""
    spec = params.spec
    ego = item.ego_speed_mps if item.ego_speed_mps is not None else 0.0

    if item.attribution == "PASSTHROUGH":
        item.verdict = "PASSTHROUGH_PLANNER_BRAKE"
        item.why = (
            "brake %s came in from the planner/controller and left unchanged; the "
            "arbiter demanded %.2f m/s^2 of its own and did not object. This frame "
            "is evidence about the planner, not the arbiter."
            % (_fmt(item.brake_in), item.demanded_decel_mps2 or 0.0)
        )
        return

    if item.attribution == "ARBITER_STATE_ONLY" and not (item.aeb_flag or item.hazard_flag):
        item.verdict = "NON_TRAFFIC_STATE"
        item.why = (
            "state %s with no hazard flag, no AEB flag and no braking demand of its "
            "own: a health, kinematics or latched-state degradation, not a claim "
            "about traffic. findings=[%s] reason=%r"
            % (item.state, ", ".join(item.findings)[:80], item.reason)
        )
        return

    if item.lead_track_id is None:
        if not item.ego_lane_occupied_arbiter and not item.ego_lane_occupied_tracker:
            item.verdict = "UNJUSTIFIED_NO_TARGET"
            item.why = (
                "no in-path object at all this frame (%d tracks, %s in path by the "
                "arbiter's own geometry)"
                % (item.n_tracks, item.ego_lane_occupied_arbiter)
            )
        else:
            item.verdict = "UNJUSTIFIED_NO_LEAD"
            item.why = "the ego lane was occupied but no lead was assessed"
        return

    gap = item.lead_raw_range_m
    if gap is None:
        gap = item.lead_filtered_range_m
    if gap is None:
        item.verdict = "INDETERMINATE"
        item.why = "no range recorded for the lead"
        return
    item.safe_gap_m = safe_following_gap_m(spec, ego, ego)

    usable_rate = item.observed_rate_mps if item.rate_significant else None
    if usable_rate is None:
        # The closing rate is not measurable, or the fitted slope is
        # indistinguishable from range noise.  The only thing that can justify a
        # hard brake now is the RANGE, so ask the question the geometry can
        # answer on its own: if that object were standing still, would this much
        # braking be needed?
        worst = required_decel_mps2(gap, ego, 0.0, 0.0)
        detail = (
            "%d samples over %.2f s, slope %s +/- %s m/s"
            % (
                item.observed_samples,
                item.observed_span_s,
                _fmt(item.observed_rate_mps, "%+.2f"),
                _fmt(item.observed_rate_stderr_mps, "%.2f"),
            )
        )
        if worst >= spec.comfort_decel_mps2:
            item.verdict = "JUSTIFIED_RANGE_ALONE"
            item.why = (
                "no usable closing rate (%s), but a STATIONARY obstacle at %.1f m "
                "would need %s m/s^2 at %.1f m/s, so the range alone carries it"
                % (detail, gap, _fmt(worst, "%.1f"), ego)
            )
        elif item.hard:
            item.verdict = "UNJUSTIFIED_INFERRED_RATE"
            item.why = (
                "the arbiter demanded %.2f m/s^2 with no usable closing rate (%s; "
                "its own rate %s m/s is marked measured=%s); even a STATIONARY "
                "obstacle at %.1f m needs only %.1f m/s^2 at %.1f m/s"
                % (
                    item.arbiter_decel_mps2,
                    detail,
                    _fmt(item.lead_rate_used_mps, "%+.2f"),
                    item.rate_is_measured,
                    gap,
                    worst,
                    ego,
                )
            )
        elif gap < (item.safe_gap_m or 0.0):
            item.verdict = "JUSTIFIED_HEADWAY"
            item.why = "gap %.1f m is below the %.1f m safe following gap at %.1f m/s" % (
                gap,
                item.safe_gap_m,
                ego,
            )
        else:
            item.verdict = "INDETERMINATE"
            item.why = "no usable range history (%s) and the headway is safe" % detail
        return

    rate = usable_rate
    lead_speed = max(0.0, ego + rate)
    item.required_decel_mps2 = required_decel_mps2(gap, ego, lead_speed, 0.0)

    if rate > params.not_closing_rate_mps and item.hard:
        item.verdict = "UNJUSTIFIED_NOT_CLOSING"
        item.why = (
            "measured range slope %+.2f +/- %.2f m/s over %.2f s (%d samples) -- the "
            "lead is not closing; the arbiter demanded %.2f m/s^2 at %.1f m"
            % (
                rate,
                item.observed_rate_stderr_mps,
                item.observed_span_s,
                item.observed_samples,
                item.arbiter_decel_mps2,
                gap,
            )
        )
        return
    if item.required_decel_mps2 >= spec.comfort_decel_mps2:
        item.verdict = "JUSTIFIED"
        item.why = (
            "measured closure %+.2f +/- %.2f m/s at %.1f m, ego %.1f m/s -> %s m/s^2 "
            "needed, above the %.1f m/s^2 comfort boundary"
            % (
                rate,
                item.observed_rate_stderr_mps,
                gap,
                ego,
                _fmt(item.required_decel_mps2, "%.1f"),
                spec.comfort_decel_mps2,
            )
        )
        return
    if item.hard:
        item.verdict = "UNJUSTIFIED_OVERREACTION"
        item.why = (
            "measured closure %+.2f +/- %.2f m/s at %.1f m needs only %.1f m/s^2; the "
            "arbiter demanded %.2f m/s^2 (pedal reached %s)"
            % (
                rate,
                item.observed_rate_stderr_mps,
                gap,
                item.required_decel_mps2,
                item.arbiter_decel_mps2,
                _fmt(item.brake_actuated),
            )
        )
        return
    allowance = _allowance(item.required_decel_mps2, params)
    if item.required_decel_mps2 > 0.0 and item.arbiter_decel_mps2 <= allowance:
        item.verdict = "JUSTIFIED_GRADED"
        item.why = (
            "measured closure %+.2f m/s at %.1f m needs %.1f m/s^2 and the arbiter "
            "asked for %.2f m/s^2: a graded response no larger than the closure "
            "demands" % (rate, gap, item.required_decel_mps2, item.arbiter_decel_mps2)
        )
        return
    if gap < (item.safe_gap_m or 0.0):
        item.verdict = "JUSTIFIED_HEADWAY"
        item.why = "gap %.1f m is below the %.1f m safe following gap at %.1f m/s" % (
            gap,
            item.safe_gap_m,
            ego,
        )
        return
    if item.arbiter_decel_mps2 > allowance:
        item.verdict = "UNJUSTIFIED_OVERREACTION"
        item.why = (
            "measured closure %+.2f m/s at %.1f m needs %.1f m/s^2; the arbiter asked "
            "for %.2f m/s^2 on a safe headway (%.1f m)"
            % (rate, gap, item.required_decel_mps2, item.arbiter_decel_mps2, item.safe_gap_m or 0.0)
        )
        return
    item.verdict = "UNJUSTIFIED_NO_HAZARD"
    item.why = (
        "gap %.1f m with a measured %+.2f m/s slope needs %.1f m/s^2 and the headway "
        "is safe (%.1f m); nothing to react to"
        % (gap, rate, item.required_decel_mps2, item.safe_gap_m or 0.0)
    )


def analyse(
    recording: RunRecording, params: Optional[AnalysisParams] = None
) -> AnalysisReport:
    """Classify every intervention in a recording, and count the inverse.

    Args:
        recording: A run captured by :func:`record_run`.
        params: Thresholds and physics; defaults to :class:`AnalysisParams`.

    Returns:
        An :class:`AnalysisReport`.
    """
    params = params or AnalysisParams()
    spec = params.spec
    frames = recording.frames
    report = AnalysisReport(meta=dict(recording.meta))
    report.frames_total = len(frames)

    missing = recording.meta.get("missing_hooks") or []
    if missing:
        report.warnings.append(
            "the instrument could not attach these hooks, so the matching fields "
            "are empty: %s" % ", ".join(missing)
        )
    if recording.meta.get("ego_speed_source") == "simulated":
        report.warnings.append(
            "ego speed is SIMULATED at %.1f m/s on real footage; absolute TTC "
            "figures are not physical. The justification test is internal "
            "consistency: does the arbiter's action follow from the range history "
            "and the ego speed it itself used?" % (recording.meta.get("ego_speed_mps") or 0.0)
        )

    previous_lead_id: Optional[int] = None
    for index, rec in enumerate(frames):
        state = rec.state or ""
        report.state_histogram[state] = report.state_histogram.get(state, 0) + 1
        brake = rec.brake_actuated if rec.brake_actuated is not None else rec.brake_out
        brake = 0.0 if brake is None else brake
        report.max_brake_actuated = max(report.max_brake_actuated, brake)
        if rec.demanded_decel_mps2 is not None:
            report.max_demand_mps2 = max(report.max_demand_mps2, rec.demanded_decel_mps2)
        if brake >= 0.90:
            report.frames_brake_full += 1
        if rec.brake_from_arbiter:
            report.frames_arbiter_raised_brake += 1
        bucket = (
            "0.00" if brake <= 0.0
            else "(0,0.10)" if brake < 0.10
            else "[0.10,0.25)" if brake < 0.25
            else "[0.25,0.50)" if brake < 0.50
            else "[0.50,0.90)" if brake < 0.90
            else ">=0.90 FULL"
        )
        report.brake_histogram[bucket] = report.brake_histogram.get(bucket, 0) + 1
        if rec.lead is not None:
            report.frames_with_lead += 1
            if rec.lead.track_id not in report.lead_track_ids:
                report.lead_track_ids.append(rec.lead.track_id)
            if previous_lead_id is not None and rec.lead.track_id != previous_lead_id:
                report.lead_switches += 1
            previous_lead_id = rec.lead.track_id
        elif previous_lead_id is not None:
            report.frames_lead_lost_then_regained += 1
            previous_lead_id = None

        lead = rec.lead
        history = _history(frames, index, lead.track_id, params.history_frames) if lead else []
        obs_rate, rate_stderr, span, n_samples = _observed_rate(history, params)
        rate_significant = (
            obs_rate is not None
            and math.isfinite(rate_stderr)
            and abs(obs_rate) >= params.significance_sigma * rate_stderr
        )

        demand = rec.demanded_decel_mps2 or 0.0
        counterfactual = demand / spec.brake_authority_mps2 if demand else 0.0
        raised = 0.0
        if rec.brake_out is not None and rec.brake_in is not None:
            raised = max(0.0, rec.brake_out - rec.brake_in) * spec.brake_authority_mps2
        arbiter_decel = max(demand, raised)
        intervening = (
            state in params.intervening_states
            or brake > params.brake_threshold
            or demand > 0.0
        )
        if intervening:
            item = Intervention(
                frame_index=rec.frame_index,
                timestamp_s=rec.timestamp_s,
                state=state,
                brake_in=rec.brake_in,
                brake_out=rec.brake_out,
                brake_actuated=rec.brake_actuated,
                demanded_decel_mps2=rec.demanded_decel_mps2,
                ego_speed_mps=rec.ego_speed_mps,
                hard=(
                    arbiter_decel >= spec.emergency_decel_mps2
                    or state in ("min_risk_maneuver", "disengage")
                ),
                arbiter_decel_mps2=arbiter_decel,
                arbiter_added_brake=rec.brake_from_arbiter,
                counterfactual_brake=counterfactual,
                aeb_flag=rec.aeb,
                hazard_flag=rec.hazard,
                ego_lane_occupied_tracker=rec.n_tracker_in_lane,
                ego_lane_occupied_arbiter=rec.n_arbiter_in_path,
                n_tracks=rec.n_tracks,
                history=history,
                observed_rate_mps=obs_rate,
                observed_rate_stderr_mps=rate_stderr,
                rate_significant=rate_significant,
                observed_span_s=span,
                observed_samples=n_samples,
                findings=list(rec.hazard_findings or rec.violations),
                reason=rec.reason,
            )
            if lead is not None:
                item.lead_track_id = lead.track_id
                item.lead_raw_range_m = rec.lead_raw_distance_m
                item.lead_filtered_range_m = lead.distance_m
                item.lead_rate_used_mps = lead.range_rate_mps
                item.rate_is_measured = lead.rate_is_measured
                item.rate_updates = lead.rate_updates
                item.range_source = lead.source
                item.coasting = lead.coasting
            item.attribution = _attribute(item, params)
            if rec.perception_ok is False or rec.ego_valid is False:
                item.verdict = "JUSTIFIED_DEGRADED"
                item.why = (
                    "perception_ok=%s ego_valid=%s (%d consecutive perception "
                    "failures): a health response, not a traffic one"
                    % (rec.perception_ok, rec.ego_valid, rec.perception_failures)
                )
            else:
                _judge(item, params)
            report.interventions.append(item)
            report.verdict_counts[item.verdict] = report.verdict_counts.get(item.verdict, 0) + 1
            report.attribution_counts[item.attribution] = (
                report.attribution_counts.get(item.attribution, 0) + 1
            )
            if item.verdict == "NON_TRAFFIC_STATE":
                for key in _finding_keys(item):
                    report.non_traffic_findings[key] = (
                        report.non_traffic_findings.get(key, 0) + 1
                    )
            continue

        # The inverse: a measured hazard that got no reaction at all -- neither a
        # brake nor a demand nor a state change.
        if lead is None or not rate_significant or rec.ego_speed_mps is None:
            continue
        gap = rec.lead_raw_distance_m
        if gap is None:
            gap = lead.distance_m
        need = required_decel_mps2(
            gap, rec.ego_speed_mps, max(0.0, rec.ego_speed_mps + obs_rate), 0.0
        )
        if need >= spec.comfort_decel_mps2:
            report.missed.append(
                MissedReaction(
                    frame_index=rec.frame_index,
                    timestamp_s=rec.timestamp_s,
                    state=state,
                    brake_actuated=rec.brake_actuated,
                    ego_speed_mps=rec.ego_speed_mps,
                    lead_raw_range_m=gap,
                    observed_rate_mps=obs_rate,
                    required_decel_mps2=need if math.isfinite(need) else 1e9,
                )
            )
    return report


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _spark(history: Sequence[RangeSample], limit: int = 12) -> str:
    """Compact ``raw`` range history, newest last."""
    tail = list(history)[-limit:]
    return " ".join(
        "-" if h.raw_distance_m is None else "%.1f" % h.raw_distance_m for h in tail
    )


def format_report(
    report: AnalysisReport, max_interventions: int = 60, show_all: bool = False
) -> str:
    """Human-readable report.  The unjustified frames are printed in full."""
    lines: List[str] = []
    add = lines.append
    meta = report.meta
    add("=" * 78)
    add("REAL-FOOTAGE INTERVENTION JUSTIFICATION ANALYSIS")
    add("=" * 78)
    add("source            : %s" % meta.get("source"))
    add(
        "run               : %s arbitrated frames (%s loop frames requested %s), "
        "detector %s, lane %s"
        % (
            report.frames_total,
            meta.get("loop_frames"),
            meta.get("requested_frames"),
            meta.get("detector"),
            meta.get("lane"),
        )
    )
    add(
        "ego speed         : %s m/s (%s)"
        % (meta.get("ego_speed_mps"), meta.get("ego_speed_source"))
    )
    add("frames with a lead: %d" % report.frames_with_lead)
    for warning in report.warnings:
        add("NOTE: " + warning)
    add("")
    add("state histogram   : %s" % report.state_histogram)
    add("brake histogram   : %s" % report.brake_histogram)
    add("max brake actuated: %.3f" % report.max_brake_actuated)
    add("max arbiter demand: %.2f m/s^2" % report.max_demand_mps2)
    add("frames at brake >= 0.90       : %d" % report.frames_brake_full)
    add("frames the arbiter RAISED the incoming brake: %d" % report.frames_arbiter_raised_brake)
    add(
        "lead continuity   : %d distinct lead track ids %s, %d lead switches, "
        "%d frames where the lead was lost after having one"
        % (
            len(report.lead_track_ids),
            report.lead_track_ids[:12],
            report.lead_switches,
            report.frames_lead_lost_then_regained,
        )
    )
    add("")
    add("-" * 78)
    add("INTERVENTIONS: %d" % len(report.interventions))
    add("-" * 78)
    add("  attribution -- who caused the pedal, and who wanted it:")
    for name, count in sorted(report.attribution_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        add("    %-28s %4d" % (name, count))
    add("  arbiter's own interventions (everything but PASSTHROUGH): %d"
        % len(report.arbiter_interventions))
    add("")
    add("  justification verdicts:")
    for verdict, count in sorted(report.verdict_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        add("    %-28s %4d" % (verdict, count))
    unjustified = report.unjustified
    add("    %-28s %4d" % ("TOTAL UNJUSTIFIED", len(unjustified)))
    if report.non_traffic_findings:
        add("")
        add("  the NON_TRAFFIC_STATE degradations were raised by: %s"
            % report.non_traffic_findings)
    add("")

    chosen = report.interventions if show_all else unjustified
    label = "EVERY INTERVENTION" if show_all else "UNJUSTIFIED INTERVENTIONS, in full"
    add("-" * 78)
    add("%s (showing up to %d)" % (label, max_interventions))
    add("-" * 78)
    if not chosen:
        add("  none")
    for item in chosen[:max_interventions]:
        add(
            "  f%-5d %-18s %-24s brake in=%s out=%s act=%s | demand=%s m/s^2 "
            "(= brake %s)  ego=%s m/s  %s"
            % (
                item.frame_index,
                item.state,
                item.attribution,
                _fmt(item.brake_in),
                _fmt(item.brake_out),
                _fmt(item.brake_actuated),
                _fmt(item.demanded_decel_mps2, "%.2f"),
                _fmt(item.counterfactual_brake),
                _fmt(item.ego_speed_mps, "%.1f"),
                "HARD" if item.hard else "soft",
            )
        )
        add(
            "         lead=%s raw=%s filtered=%s source=%s rate_used=%s measured=%s updates=%d coasting=%s"
            % (
                item.lead_track_id,
                _fmt(item.lead_raw_range_m, "%.1f"),
                _fmt(item.lead_filtered_range_m, "%.1f"),
                item.range_source or "-",
                _fmt(item.lead_rate_used_mps, "%+.2f"),
                item.rate_is_measured,
                item.rate_updates,
                item.coasting,
            )
        )
        add(
            "         ego lane occupied: tracker=%d arbiter=%s   tracks=%d   "
            "aeb=%s hazard=%s"
            % (
                item.ego_lane_occupied_tracker,
                item.ego_lane_occupied_arbiter,
                item.n_tracks,
                item.aeb_flag,
                item.hazard_flag,
            )
        )
        add(
            "         raw range history (%d samples, %.2f s): %s"
            % (item.observed_samples, item.observed_span_s, _spark(item.history))
        )
        add(
            "         observed slope=%s +/- %s m/s (significant=%s)  required=%s m/s^2  "
            "safe gap=%s m  arbiter's own decel=%.2f m/s^2"
            % (
                _fmt(item.observed_rate_mps, "%+.2f"),
                _fmt(item.observed_rate_stderr_mps, "%.2f"),
                item.rate_significant,
                _fmt(item.required_decel_mps2, "%.2f"),
                _fmt(item.safe_gap_m, "%.1f"),
                item.arbiter_decel_mps2,
            )
        )
        add("         VERDICT %s -- %s" % (item.verdict, item.why))
        if item.findings:
            add("         findings: " + ", ".join(item.findings)[:150])
        if item.reason:
            add("         arbiter reason: " + item.reason[:150])
    add("")
    add("-" * 78)
    add("MISSED REACTIONS: %d frames whose MEASURED scene demanded braking and got none" % len(report.missed))
    add("-" * 78)
    if not report.missed:
        add("  none")
    for miss in report.missed[:max_interventions]:
        add(
            "  f%-5d %-18s brake=%s  range=%.1f m  slope=%+.2f m/s  ego=%.1f m/s  needs %.1f m/s^2"
            % (
                miss.frame_index,
                miss.state,
                _fmt(miss.brake_actuated),
                miss.lead_raw_range_m or 0.0,
                miss.observed_rate_mps or 0.0,
                miss.ego_speed_mps or 0.0,
                miss.required_decel_mps2,
            )
        )
    return "\n".join(lines)


def _fmt(value: Optional[float], fmt: str = "%.2f") -> str:
    """Format a float that may be ``None`` or infinite."""
    if value is None:
        return "-"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return fmt % value
