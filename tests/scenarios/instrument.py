"""Non-invasive instrumentation of :class:`adas.control.arbiter.SafetyArbiter`.

Why this exists
---------------
Three rounds of longitudinal-safety fixes oscillated because nobody could answer
one question from the outside: *when the wheels braked, who asked for it?*  The
planner and the controller also produce a brake command, so a brake seen at the
actuators proves nothing about the arbiter, and the arbiter's own
``ArbitrationResult`` does not say what it would have demanded on its own.  The
phantom AEB of fix round 1 was only attributed once somebody separated

    DEMAND    -- the deceleration the arbiter itself asked for this frame, and
    COMMAND   -- the brake that was handed *in* by the controller, and
    ACTUATED  -- the brake that came *out* and reached the wheels,

and that separation is first class here: every :class:`FrameRecord` carries all
three, plus the evidence the arbiter was reasoning from (which lead, at what
range, from which range source, and crucially whether the closing rate was
*measured* or *inferred from a safe prior*).

Design constraints
------------------
* **It must not modify the arbiter.**  Nothing here edits ``arbiter.py``; the
  capture is done by wrapping bound methods on the class object for the lifetime
  of a context manager, and unwinding them exactly on exit.
* **It must survive the redesign.**  The arbiter is about to be rewritten.  Every
  hook is optional: a method that no longer exists is recorded in
  :attr:`ArbiterInstrument.missing_hooks` and the rest of the capture still
  works.  Consumers therefore have to treat every field as possibly ``None``,
  and the analysis modules do.
* **It must be deterministic.**  No wall clock, no randomness, no I/O.  Frames
  are numbered by call order, and the simulated/recorded ``timestamp_s`` from the
  :class:`~adas.control.arbiter.SafetyContext` is what gets stored.

The hooks
---------
=========================== =============================================
``SafetyArbiter.arbitrate``  frame boundary; context summary and outcome
``SafetyArbiter._in_path``   per-track in-path verdict (was the lane occupied?)
``SafetyArbiter._assess_lead`` the selected lead and its whole assessment
``SafetyArbiter._classify_hazard`` the hazard verdict and the findings raised
``SafetyArbiter._required_decel`` the DEMAND, in m/s^2
``SafetyArbiter._synthesise_command`` COMMAND in, ACTUATED out
=========================== =============================================

Typical use::

    from tests.scenarios.instrument import ArbiterInstrument

    with ArbiterInstrument() as inst:
        run_whatever_calls_the_arbiter()
    for rec in inst.frames:
        print(rec.frame_index, rec.demanded_decel_mps2, rec.brake_out)

Everything in this module is import-safe without a GPU and without TensorRT.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "TrackObservation",
    "LeadRecord",
    "FrameRecord",
    "ArbiterInstrument",
    "frames_to_jsonable",
    "frames_from_jsonable",
]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass
class TrackObservation:
    """One tracker output as the arbiter received it, before any arbiter filter.

    This is the *measurement* channel.  ``distance_m`` here is what the tracker
    handed in; :attr:`LeadRecord.distance_m` is what the arbiter's own range
    filter believed after folding that measurement in.  The gap between the two
    is how a seeded (fabricated) closing rate is detected on real footage.
    """

    track_id: int
    distance_m: float
    velocity_mps: float
    label: str = ""
    in_ego_lane_flag: bool = False
    """``TrackedObject.in_ego_lane`` -- the *tracker's* lane opinion."""
    arbiter_in_path: Optional[bool] = None
    """The arbiter's own independent in-path verdict, when the hook attached."""
    box_cx_px: float = 0.0
    box_cy_px: float = 0.0
    box_w_px: float = 0.0
    box_h_px: float = 0.0
    confidence: float = 0.0


@dataclass
class LeadRecord:
    """Flattened :class:`~adas.control.arbiter.LeadAssessment`.

    Kept as a plain record rather than the live object so a run can be written to
    JSON and analysed later on a machine with no ``adas`` install.
    """

    track_id: int
    distance_m: float
    range_rate_mps: float
    ttc_s: float
    required_decel_mps2: float
    rss_min_gap_m: float
    rss_matched_gap_m: float
    source: str
    rate_is_measured: bool
    measured_rate_mps: Optional[float]
    rate_updates: int
    coasting: bool
    coast_frames: int
    disagreement: bool
    reinitialised: bool


@dataclass
class FrameRecord:
    """Everything one ``arbitrate()`` call saw, decided, demanded and emitted."""

    frame_index: int
    arbiter_id: int
    """``id()`` of the arbiter instance, so a sweep that builds one arbiter per
    cell can split a single capture back into cells."""

    # ---- inputs -----------------------------------------------------------
    timestamp_s: Optional[float] = None
    dt_s: Optional[float] = None
    ego_speed_mps: Optional[float] = None
    ego_valid: Optional[bool] = None
    perception_ok: Optional[bool] = None
    perception_failures: int = 0
    plan_target_speed_mps: Optional[float] = None
    plan_reason: str = ""
    tracks: List[TrackObservation] = field(default_factory=list)
    n_tracks: int = 0
    n_tracker_in_lane: int = 0
    """How many tracks the *tracker* called in-lane."""
    n_arbiter_in_path: Optional[int] = None
    """How many tracks the *arbiter* called in-path.  ``0`` with a non-empty
    track list is the signature of an intervention against an empty ego lane."""

    # ---- the arbiter's belief --------------------------------------------
    lead: Optional[LeadRecord] = None
    lead_raw_distance_m: Optional[float] = None
    """The raw tracker range for the lead track id, i.e. the measurement the
    arbiter's filter was fed.  ``None`` when the lead is not in ``tracks``."""

    # ---- the arbiter's decision ------------------------------------------
    aeb: Optional[bool] = None
    hazard: Optional[bool] = None
    deferred_acute: Optional[bool] = None
    rate_pending: Optional[bool] = None
    hazard_findings: List[str] = field(default_factory=list)

    # ---- demand vs command vs actuated -----------------------------------
    demanded_decel_mps2: Optional[float] = None
    """What the arbiter itself asked for, m/s^2.  Independent of the incoming
    command; this is the number that attributes a brake to the arbiter."""
    throttle_in: Optional[float] = None
    brake_in: Optional[float] = None
    steering_in: Optional[float] = None
    throttle_out: Optional[float] = None
    brake_out: Optional[float] = None
    steering_out: Optional[float] = None
    throttle_actuated: Optional[float] = None
    brake_actuated: Optional[float] = None
    steering_actuated: Optional[float] = None
    """``*_out`` is what ``_synthesise_command`` returned; ``*_actuated`` is what
    ``arbitrate`` finally returned.  They are normally equal -- when they are
    not, something after synthesis moved the pedal."""

    # ---- outcome ----------------------------------------------------------
    state: str = ""
    violations: List[str] = field(default_factory=list)
    reason: str = ""

    # -------------------------------------------------------------- helpers
    @property
    def brake_from_arbiter(self) -> bool:
        """True when the arbiter raised the brake above what it was handed."""
        if self.brake_out is None or self.brake_in is None:
            return False
        return self.brake_out > self.brake_in + 1e-9

    @property
    def pedal_conflict(self) -> bool:
        """True when the emitted command asks for throttle AND brake at once.

        A control fault, not a compromise.  The synthetic plant used to hide
        this by computing ``2.5*throttle - 8.0*brake``, which nets a full
        throttle against a full brake into an unremarkable -5.5 m/s^2;
        :class:`tests.scenarios.plant.Plant` now applies brake override and
        records the conflict on the frame instead of absorbing it.  This
        property asks the same question of a REAL recorded run, where there is
        no plant to ask it of.

        Uses the actuated pedals when they were captured, falling back to the
        synthesised ones, because the actuated pair is what reaches the wheels.
        """
        thr = self.throttle_actuated if self.throttle_actuated is not None else self.throttle_out
        brk = self.brake_actuated if self.brake_actuated is not None else self.brake_out
        if thr is None or brk is None:
            return False
        return thr > 0.0 and brk > 0.0

    @property
    def intervened_state(self) -> bool:
        """True when the state itself is a degradation (not ``nominal``)."""
        return bool(self.state) and self.state != "nominal"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict (dataclasses all the way down)."""
        return asdict(self)


def frames_to_jsonable(frames: List[FrameRecord]) -> List[Dict[str, Any]]:
    """Convert records for :func:`json.dump`."""
    return [f.to_dict() for f in frames]


def frames_from_jsonable(rows: List[Dict[str, Any]]) -> List[FrameRecord]:
    """Rebuild records written by :func:`frames_to_jsonable`.

    Unknown keys are dropped rather than raising, so a recording made by an older
    revision of this module still loads.
    """
    out: List[FrameRecord] = []
    frame_keys = set(FrameRecord.__dataclass_fields__)  # type: ignore[attr-defined]
    track_keys = set(TrackObservation.__dataclass_fields__)  # type: ignore[attr-defined]
    lead_keys = set(LeadRecord.__dataclass_fields__)  # type: ignore[attr-defined]
    for row in rows:
        kw = {k: v for k, v in row.items() if k in frame_keys}
        kw["tracks"] = [
            TrackObservation(**{k: v for k, v in t.items() if k in track_keys})
            for t in (row.get("tracks") or [])
        ]
        lead = row.get("lead")
        kw["lead"] = (
            LeadRecord(**{k: v for k, v in lead.items() if k in lead_keys}) if lead else None
        )
        out.append(FrameRecord(**kw))
    return out


# --------------------------------------------------------------------------- #
# The instrument
# --------------------------------------------------------------------------- #


def _enum_value(obj: Any) -> str:
    """``SafetyState.LIMITED`` -> ``'limited'``; anything else -> ``str()``."""
    return str(getattr(obj, "value", obj))


def _num(value: Any) -> Optional[float]:
    """Coerce to float, mapping non-numbers (and NaN-free infinities) safely."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ArbiterInstrument:
    """Capture the arbiter's per-frame decision inputs, demand and output.

    The wrapping is done on the *class*, so every :class:`SafetyArbiter` built
    while the instrument is attached is captured -- including one built deep
    inside :class:`adas.runtime.pipeline.ADASPipeline`, which is the only way to
    instrument a real video run without editing the pipeline.

    Args:
        module: Already-imported ``adas.control.arbiter`` module, or ``None`` to
            import it.  Passing it explicitly is useful when a caller has
            re-imported ``adas`` from a non-default ``sys.path``.
        max_tracks: Cap on tracks stored per frame, so a pathological detector
            frame cannot blow up a recording.  ``None`` for no cap.
        capture_tracks: Set ``False`` to store only the lead and the counts.

    Attributes:
        frames: The captured records, in call order.
        missing_hooks: Names of methods that could not be wrapped, because the
            arbiter no longer has them.  A non-empty list means the
            corresponding fields of every record are ``None``; report it, do not
            silently analyse a half-empty capture.
    """

    #: Methods wrapped, in the order they run inside one ``arbitrate`` call.
    HOOKS: Tuple[str, ...] = (
        "arbitrate",
        "_in_path",
        "_assess_lead",
        "_classify_hazard",
        "_required_decel",
        "_synthesise_command",
    )

    def __init__(
        self,
        module: Any = None,
        max_tracks: Optional[int] = 16,
        capture_tracks: bool = True,
    ) -> None:
        self._module = module or importlib.import_module("adas.control.arbiter")
        self._cls = getattr(self._module, "SafetyArbiter")
        self._max_tracks = max_tracks
        self._capture_tracks = capture_tracks
        self.frames: List[FrameRecord] = []
        self.missing_hooks: List[str] = []
        self._originals: Dict[str, Any] = {}
        self._attached = False
        self._cur: Optional[FrameRecord] = None
        self._in_path_calls: List[Tuple[int, bool]] = []

    # ---------------------------------------------------------------- attach
    def attach(self) -> "ArbiterInstrument":
        """Wrap the hooks.  Idempotent; safe to call once per context."""
        if self._attached:
            return self
        self.missing_hooks = []
        for name in self.HOOKS:
            original = getattr(self._cls, name, None)
            if original is None:
                self.missing_hooks.append(name)
                continue
            self._originals[name] = original
            setattr(self._cls, name, self._make_wrapper(name, original))
        self._attached = True
        return self

    def detach(self) -> None:
        """Restore every wrapped method.  Always runs, even on an exception."""
        for name, original in self._originals.items():
            setattr(self._cls, name, original)
        self._originals.clear()
        self._attached = False
        self._cur = None

    def __enter__(self) -> "ArbiterInstrument":
        return self.attach()

    def __exit__(self, *exc: Any) -> bool:
        self.detach()
        return False

    def reset(self) -> None:
        """Drop captured frames but stay attached (used between sweep cells)."""
        self.frames = []
        self._cur = None
        self._in_path_calls = []

    # -------------------------------------------------------------- wrapping
    def _make_wrapper(self, name: str, original: Any) -> Any:
        handler = getattr(self, "_hook_" + name.lstrip("_"))

        def wrapper(inner_self: Any, *args: Any, **kwargs: Any) -> Any:
            return handler(original, inner_self, args, kwargs)

        wrapper.__name__ = name
        wrapper.__doc__ = getattr(original, "__doc__", None)
        wrapper.__wrapped__ = original  # type: ignore[attr-defined]
        return wrapper

    # ------------------------------------------------------------- the hooks
    def _hook_arbitrate(self, original, arb, args, kwargs):
        """Frame boundary.  Opens a record, fills the inputs, closes on return."""
        plan = kwargs.get("plan", args[0] if len(args) > 0 else None)
        command = kwargs.get("command", args[1] if len(args) > 1 else None)
        ctx = kwargs.get("state", kwargs.get("context", args[2] if len(args) > 2 else None))

        rec = FrameRecord(frame_index=len(self.frames), arbiter_id=id(arb))
        self._fill_inputs(rec, plan, command, ctx)
        self._cur = rec
        self._in_path_calls = []
        try:
            result = original(arb, *args, **kwargs)
        finally:
            if self._in_path_calls:
                rec.n_arbiter_in_path = sum(1 for _, ok in self._in_path_calls if ok)
                verdicts = dict(self._in_path_calls)
                for tob in rec.tracks:
                    if tob.track_id in verdicts:
                        tob.arbiter_in_path = verdicts[tob.track_id]
            self.frames.append(rec)
            self._cur = None
            self._in_path_calls = []
        cmd = getattr(result, "command", None)
        if cmd is not None:
            rec.throttle_actuated = _num(getattr(cmd, "throttle", None))
            rec.brake_actuated = _num(getattr(cmd, "brake", None))
            rec.steering_actuated = _num(getattr(cmd, "steering", None))
        rec.state = _enum_value(getattr(result, "state", ""))
        rec.violations = list(getattr(result, "violations", []) or [])
        rec.reason = str(getattr(result, "reason", "") or "")
        return result

    def _fill_inputs(self, rec: FrameRecord, plan: Any, command: Any, ctx: Any) -> None:
        if plan is not None:
            rec.plan_target_speed_mps = _num(getattr(plan, "target_speed_mps", None))
            rec.plan_reason = str(getattr(plan, "reason", "") or "")
        if command is not None:
            rec.throttle_in = _num(getattr(command, "throttle", None))
            rec.brake_in = _num(getattr(command, "brake", None))
            rec.steering_in = _num(getattr(command, "steering", None))
        if ctx is None:
            return
        rec.dt_s = _num(getattr(ctx, "dt_s", None))
        rec.timestamp_s = _num(getattr(ctx, "timestamp_s", None))
        ego = getattr(ctx, "ego", None)
        if ego is not None:
            rec.ego_speed_mps = _num(getattr(ego, "speed_mps", None))
            rec.ego_valid = bool(getattr(ego, "valid", False))
        perception = getattr(ctx, "perception", None)
        if perception is not None:
            rec.perception_ok = bool(getattr(perception, "ok", True))
            rec.perception_failures = int(getattr(perception, "consecutive_failures", 0) or 0)
        tracks = list(getattr(ctx, "tracks", []) or [])
        rec.n_tracks = len(tracks)
        rec.n_tracker_in_lane = sum(1 for t in tracks if getattr(t, "in_ego_lane", False))
        if not self._capture_tracks:
            return
        kept = tracks if self._max_tracks is None else tracks[: self._max_tracks]
        for t in kept:
            box = getattr(t, "box", None)
            x1 = _num(getattr(box, "x1", 0.0)) or 0.0
            y1 = _num(getattr(box, "y1", 0.0)) or 0.0
            x2 = _num(getattr(box, "x2", 0.0)) or 0.0
            y2 = _num(getattr(box, "y2", 0.0)) or 0.0
            rec.tracks.append(
                TrackObservation(
                    track_id=int(getattr(t, "track_id", -1)),
                    distance_m=_num(getattr(t, "distance_m", 0.0)) or 0.0,
                    velocity_mps=_num(getattr(t, "velocity_mps", 0.0)) or 0.0,
                    label=str(getattr(box, "label", "") or ""),
                    in_ego_lane_flag=bool(getattr(t, "in_ego_lane", False)),
                    box_cx_px=0.5 * (x1 + x2),
                    box_cy_px=0.5 * (y1 + y2),
                    box_w_px=x2 - x1,
                    box_h_px=y2 - y1,
                    confidence=_num(getattr(box, "confidence", 0.0)) or 0.0,
                )
            )

    def _hook_in_path(self, original, arb, args, kwargs):
        """Record the arbiter's own in-path verdict for each candidate track."""
        result = original(arb, *args, **kwargs)
        if self._cur is not None:
            obj = kwargs.get("obj", args[1] if len(args) > 1 else None)
            tid = getattr(obj, "track_id", None)
            if tid is not None:
                self._in_path_calls.append((int(tid), bool(result)))
        return result

    def _hook_assess_lead(self, original, arb, args, kwargs):
        """Record the whole lead assessment, plus the raw range it came from."""
        lead = original(arb, *args, **kwargs)
        rec = self._cur
        if rec is not None and lead is not None:
            rec.lead = LeadRecord(
                track_id=int(getattr(lead, "track_id", -1)),
                distance_m=_num(getattr(lead, "distance_m", 0.0)) or 0.0,
                range_rate_mps=_num(getattr(lead, "range_rate_mps", 0.0)) or 0.0,
                ttc_s=_num(getattr(lead, "ttc_s", float("inf"))) or 0.0,
                required_decel_mps2=_num(getattr(lead, "required_decel_mps2", 0.0)) or 0.0,
                rss_min_gap_m=_num(getattr(lead, "rss_min_gap_m", 0.0)) or 0.0,
                rss_matched_gap_m=_num(getattr(lead, "rss_matched_gap_m", 0.0)) or 0.0,
                source=_enum_value(getattr(lead, "source", "")),
                rate_is_measured=bool(getattr(lead, "rate_is_measured", False)),
                measured_rate_mps=_num(getattr(lead, "measured_rate_mps", None)),
                rate_updates=int(getattr(lead, "rate_updates", 0) or 0),
                coasting=bool(getattr(lead, "coasting", False)),
                coast_frames=int(getattr(lead, "coast_frames", 0) or 0),
                disagreement=bool(getattr(lead, "disagreement", False)),
                reinitialised=bool(getattr(lead, "reinitialised", False)),
            )
            for tob in rec.tracks:
                if tob.track_id == rec.lead.track_id:
                    rec.lead_raw_distance_m = tob.distance_m
                    break
        return lead

    def _hook_classify_hazard(self, original, arb, args, kwargs):
        """Record the hazard verdict tuple and the findings it appended."""
        hazards = kwargs.get("hazards", args[1] if len(args) > 1 else None)
        before = len(hazards) if isinstance(hazards, list) else 0
        result = original(arb, *args, **kwargs)
        rec = self._cur
        if rec is not None:
            flags = list(result) if isinstance(result, tuple) else [result]
            names = ("aeb", "hazard", "deferred_acute", "rate_pending")
            for name, value in zip(names, flags):
                setattr(rec, name, bool(value))
            if isinstance(hazards, list):
                rec.hazard_findings = [str(h) for h in hazards[before:]]
        return result

    def _hook_required_decel(self, original, arb, args, kwargs):
        """Record the DEMAND: the deceleration the arbiter itself asked for."""
        value = original(arb, *args, **kwargs)
        if self._cur is not None:
            self._cur.demanded_decel_mps2 = _num(value)
        return value

    def _hook_synthesise_command(self, original, arb, args, kwargs):
        """Record COMMAND in vs ACTUATED out across the synthesis step."""
        cmd_in = kwargs.get("cmd_in", args[0] if len(args) > 0 else None)
        out = original(arb, *args, **kwargs)
        rec = self._cur
        if rec is not None:
            if cmd_in is not None:
                rec.throttle_in = _num(getattr(cmd_in, "throttle", None))
                rec.brake_in = _num(getattr(cmd_in, "brake", None))
                rec.steering_in = _num(getattr(cmd_in, "steering", None))
            if out is not None:
                rec.throttle_out = _num(getattr(out, "throttle", None))
                rec.brake_out = _num(getattr(out, "brake", None))
                rec.steering_out = _num(getattr(out, "steering", None))
        return out
