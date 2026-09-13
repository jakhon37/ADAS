"""In-process metric registry rendered as Prometheus text (exposition format 0.0.4).

Why this exists
---------------
``adas.core.metrics.PerformanceMetrics`` answers "how did that run go?" for one
process, once, at the end. A fleet needs the other question answered continuously:
*is this unit perceiving the road right now, which build is on it, and how often has
it fallen back to a minimum-risk manoeuvre?* Those are counters, and counters have to
live somewhere a scrape can read them while the pipeline is running.

This module is the registry. It is deliberately dependency-free — ``prometheus_client``
is not installed on the board and the project forbids adding pip packages — and
deliberately small: three sample types (:class:`Counter`, :class:`Gauge`,
:class:`Histogram`) plus an ``info``-style gauge that carries build identity in its
labels and the constant value ``1``.

Units
-----
Every series name carries its unit as a suffix, as Prometheus requires: ``_seconds``,
``_ms``, ``_mb``, ``_bytes``, and ``_total`` for counters. Speeds are metres per
second, distances metres, latencies milliseconds, sizes mebibytes where the name says
``_mb`` and bytes where it says ``_bytes``.

Concurrency
-----------
Every mutation and the whole of :meth:`Registry.render` take one registry-wide
:class:`threading.RLock`. The HTTP handler thread scrapes while the pipeline thread
counts, so an unlocked read could observe a half-written label map. The lock is held
for microseconds and is not on any latency path.

Failure behaviour
-----------------
Nothing here raises in steady state. Programming errors raise at *registration* time
(illegal metric name, duplicate name with a different type or label set) so they
surface on first import rather than mid-drive. Passing an unknown label to
``inc``/``set``/``observe`` raises :class:`ValueError`: that is a code bug, and a
silently dropped sample would be worse than a traceback in a unit test. Counters
refuse a negative increment for the same reason.
"""
from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("adas.metrics")

#: Prometheus metric-name grammar.
_NAME_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
#: Prometheus label-name grammar (``__``-prefixed names are reserved).
_LABEL_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_LabelKey = Tuple[str, ...]

#: Default histogram buckets in milliseconds. Chosen for a 20 Hz (50 ms) budget:
#: dense either side of 50 ms, with a long tail so an engine hang is still bucketed.
DEFAULT_MS_BUCKETS = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0,
                      250.0, 500.0, 1000.0)


def _fmt(value: float) -> str:
    """Render one sample value in the exposition format.

    NaN and the infinities have their own spellings; everything else goes out as a
    plain number so a scraper never has to parse an int-vs-float distinction.
    """
    v = float(value)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    if v == int(v) and abs(v) < 1e15:
        return "%d" % int(v)
    return repr(v)


def _escape(value: Any) -> str:
    """Escape a label *value*: backslash, double quote and newline, in that order."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class _Metric:
    """One named series family. Subclassed by Counter, Gauge and Histogram."""

    kind = "untyped"

    def __init__(self, name: str, documentation: str, labelnames: Sequence[str],
                 registry: "Registry") -> None:
        if not _NAME_RE.match(name):
            raise ValueError("illegal metric name %r" % (name,))
        for label in labelnames:
            if not _LABEL_RE.match(label) or label.startswith("__"):
                raise ValueError("illegal label name %r on %s" % (label, name))
        if len(set(labelnames)) != len(labelnames):
            raise ValueError("duplicate label name on %s" % name)
        self.name = name
        self.documentation = documentation or name
        self.labelnames = tuple(labelnames)
        self._registry = registry
        self._values: Dict[_LabelKey, float] = {}
        if not self.labelnames:
            # An unlabelled series must exist at zero from the first scrape so a
            # scraper can tell "nothing has happened yet" from "this unit does not
            # report that at all".
            self._values[()] = 0.0

    def _key(self, labels: Dict[str, Any]) -> _LabelKey:
        if set(labels) != set(self.labelnames):
            raise ValueError(
                "%s expects labels %s, got %s"
                % (self.name, list(self.labelnames), sorted(labels))
            )
        return tuple(str(labels[n]) for n in self.labelnames)

    def samples(self) -> List[Tuple[_LabelKey, float]]:
        """Snapshot of every series in this family. Caller holds the registry lock."""
        return sorted(self._values.items())

    def value(self, **labels: Any) -> float:
        """Current value of one series (0.0 if it has never been touched)."""
        key = self._key(labels)
        with self._registry.lock:
            return self._values.get(key, 0.0)

    def clear(self) -> None:
        """Drop every series in this family. Tests only."""
        with self._registry.lock:
            self._values.clear()
            if not self.labelnames:
                self._values[()] = 0.0

    def render_lines(self) -> List[str]:
        """Exposition lines for this family, header excluded."""
        out = []
        for key, value in self.samples():
            out.append(_sample_line(self.name, self.labelnames, key, value))
        return out


def _sample_line(name: str, labelnames: Sequence[str], key: _LabelKey,
                 value: float, extra: Sequence[Tuple[str, str]] = ()) -> str:
    pairs = list(zip(labelnames, key)) + list(extra)
    if pairs:
        rendered = ",".join('%s="%s"' % (ln, _escape(lv)) for ln, lv in pairs)
        return "%s{%s} %s" % (name, rendered, _fmt(value))
    return "%s %s" % (name, _fmt(value))


class Counter(_Metric):
    """Monotonically increasing total. Resets only when the process restarts."""

    kind = "counter"

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        """Add *amount* (must be >= 0) to one series."""
        if amount < 0:
            raise ValueError("counter %s cannot decrease" % self.name)
        key = self._key(labels)
        with self._registry.lock:
            self._values[key] = self._values.get(key, 0.0) + float(amount)


class Gauge(_Metric):
    """A value that goes up and down: a speed, a queue depth, a boolean 0/1."""

    kind = "gauge"

    def set(self, value: float, **labels: Any) -> None:
        """Replace one series' value."""
        key = self._key(labels)
        with self._registry.lock:
            self._values[key] = float(value)

    def set_bool(self, flag: Any, **labels: Any) -> None:
        """Publish a boolean as 1/0, the Prometheus convention for state."""
        self.set(1.0 if flag else 0.0, **labels)

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        key = self._key(labels)
        with self._registry.lock:
            self._values[key] = self._values.get(key, 0.0) + float(amount)

    def dec(self, amount: float = 1.0, **labels: Any) -> None:
        self.inc(-amount, **labels)


class Histogram(_Metric):
    """Cumulative bucket histogram — the only honest way to publish a latency.

    A gauge of "p95" computed in the application cannot be aggregated across a fleet
    and cannot be re-quantiled after the fact; a histogram can (``histogram_quantile``
    over ``rate(..._bucket[5m])``). Buckets are upper bounds, inclusive, and the
    implicit ``+Inf`` bucket equals ``_count``.

    Args:
        buckets: ascending upper bounds. Units are whatever the metric name says —
            for every histogram in this module, milliseconds.
    """

    kind = "histogram"

    def __init__(self, name: str, documentation: str, labelnames: Sequence[str],
                 registry: "Registry", buckets: Sequence[float] = DEFAULT_MS_BUCKETS) -> None:
        bounds = tuple(float(b) for b in buckets)
        if not bounds:
            raise ValueError("histogram %s needs at least one bucket" % name)
        if list(bounds) != sorted(bounds) or len(set(bounds)) != len(bounds):
            raise ValueError("histogram %s buckets must be ascending and unique" % name)
        if "le" in labelnames:
            raise ValueError("histogram %s cannot use the reserved label 'le'" % name)
        _Metric.__init__(self, name, documentation, labelnames, registry)
        self.buckets = bounds
        self._values = {}  # histograms track state in _counts/_sums instead
        self._counts: Dict[_LabelKey, List[float]] = {}
        self._sums: Dict[_LabelKey, float] = {}
        self._totals: Dict[_LabelKey, float] = {}
        if not self.labelnames:
            self._ensure(())

    def _ensure(self, key: _LabelKey) -> None:
        if key not in self._counts:
            self._counts[key] = [0.0] * len(self.buckets)
            self._sums[key] = 0.0
            self._totals[key] = 0.0

    def observe(self, value: float, **labels: Any) -> None:
        """Record one measurement. NaN is ignored (a failed timer is not a latency)."""
        v = float(value)
        if math.isnan(v):
            return
        key = self._key(labels)
        with self._registry.lock:
            self._ensure(key)
            counts = self._counts[key]
            for i, bound in enumerate(self.buckets):
                if v <= bound:
                    counts[i] += 1.0
            self._sums[key] += v
            self._totals[key] += 1.0

    def count(self, **labels: Any) -> float:
        key = self._key(labels)
        with self._registry.lock:
            return self._totals.get(key, 0.0)

    def sum(self, **labels: Any) -> float:
        key = self._key(labels)
        with self._registry.lock:
            return self._sums.get(key, 0.0)

    def value(self, **labels: Any) -> float:
        """Observation count, so ``value()`` means something for every metric type."""
        return self.count(**labels)

    def quantile(self, q: float, **labels: Any) -> Optional[float]:
        """Bucket-interpolated quantile, or None before the first observation.

        Resolution is limited by the bucket edges — this is for a log line or a
        ``/healthz`` field, not for a fleet SLO. Aggregate the raw buckets for that.
        """
        key = self._key(labels)
        with self._registry.lock:
            total = self._totals.get(key, 0.0)
            if total <= 0:
                return None
            counts = list(self._counts[key])
        target = q * total
        prev_bound = 0.0
        prev_count = 0.0
        for bound, cumulative in zip(self.buckets, counts):
            if cumulative >= target:
                span = cumulative - prev_count
                if span <= 0:
                    return bound
                frac = (target - prev_count) / span
                return prev_bound + frac * (bound - prev_bound)
            prev_bound = bound
            prev_count = cumulative
        return self.buckets[-1]

    def clear(self) -> None:
        with self._registry.lock:
            self._counts.clear()
            self._sums.clear()
            self._totals.clear()
            if not self.labelnames:
                self._ensure(())

    def samples(self) -> List[Tuple[_LabelKey, float]]:
        return sorted((k, v) for k, v in self._totals.items())

    def render_lines(self) -> List[str]:
        out = []
        for key in sorted(self._counts):
            counts = self._counts[key]
            for bound, cumulative in zip(self.buckets, counts):
                out.append(_sample_line(self.name + "_bucket", self.labelnames, key,
                                        cumulative, (("le", _fmt(bound)),)))
            out.append(_sample_line(self.name + "_bucket", self.labelnames, key,
                                    self._totals[key], (("le", "+Inf"),)))
            out.append(_sample_line(self.name + "_sum", self.labelnames, key,
                                    self._sums[key]))
            out.append(_sample_line(self.name + "_count", self.labelnames, key,
                                    self._totals[key]))
        return out


class Registry:
    """A named collection of metric families with one lock and one text rendering."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._metrics: Dict[str, _Metric] = {}

    def _make(self, cls: type, name: str, documentation: str,
              labelnames: Sequence[str], **kwargs: Any) -> Any:
        with self.lock:
            existing = self._metrics.get(name)
            if existing is not None:
                # Re-importing a module must not explode; a genuine collision must.
                if not isinstance(existing, cls) or existing.labelnames != tuple(labelnames):
                    raise ValueError(
                        "metric %s already registered as %s%s"
                        % (name, existing.kind, list(existing.labelnames))
                    )
                return existing
            metric = cls(name, documentation, labelnames, self, **kwargs)
            self._metrics[name] = metric
            return metric

    def counter(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> Counter:
        return self._make(Counter, name, documentation, labelnames)

    def gauge(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> Gauge:
        return self._make(Gauge, name, documentation, labelnames)

    def histogram(self, name: str, documentation: str, labelnames: Sequence[str] = (),
                  buckets: Sequence[float] = DEFAULT_MS_BUCKETS) -> Histogram:
        return self._make(Histogram, name, documentation, labelnames, buckets=buckets)

    def get(self, name: str) -> Optional[_Metric]:
        with self.lock:
            return self._metrics.get(name)

    def names(self) -> List[str]:
        with self.lock:
            return sorted(self._metrics)

    def reset(self) -> None:
        """Zero every series without unregistering anything. Tests only."""
        with self.lock:
            for metric in self._metrics.values():
                metric.clear()

    def render(self) -> str:
        """The whole registry as Prometheus text, families in stable name order.

        Every registered family contributes its ``# HELP``/``# TYPE`` header even when
        it has no samples yet: a scrape must let an operator tell "this unit reports no
        engine errors" from "this unit does not report engine errors". Always ends with
        a newline; an exposition body that does not is rejected by some scrapers.
        """
        out: List[str] = []
        with self.lock:
            for name in sorted(self._metrics):
                metric = self._metrics[name]
                out.append("# HELP %s %s" % (name, metric.documentation.replace("\n", " ")))
                out.append("# TYPE %s %s" % (name, metric.kind))
                out.extend(metric.render_lines())
        return "\n".join(out) + "\n"


#: The registry every ADAS component writes to. One per process, by design.
REGISTRY = Registry()

# --------------------------------------------------------------------- identity

BUILD_INFO = REGISTRY.gauge(
    "adas_build_info",
    "Build identity of the running unit; value is always 1.",
    ("version", "git_sha", "trt_version", "config", "detector", "lane"),
)
UP = REGISTRY.gauge("adas_up", "1 while the ADAS process is serving health.")
UPTIME_SECONDS = REGISTRY.gauge("adas_uptime_seconds", "Seconds since process start.")

# ----------------------------------------------------------------------- frames

FRAMES_TOTAL = REGISTRY.counter(
    "adas_frames_total", "Frames handed to the pipeline by the capture source."
)
FRAMES_PROCESSED_TOTAL = REGISTRY.counter(
    "adas_frames_processed_total", "Frames that completed a full pipeline step."
)
FRAMES_DROPPED_TOTAL = REGISTRY.counter(
    "adas_frames_dropped_total",
    "Frames discarded before or during inference, by reason.",
    ("reason",),
)
FRAME_ID = REGISTRY.gauge(
    "adas_frame_id", "Frame id of the most recently processed frame (not a rate)."
)
FPS = REGISTRY.gauge("adas_fps", "Processed frames per second over the last window.")

STAGE_LATENCY_MS = REGISTRY.gauge(
    "adas_stage_latency_ms",
    "Pipeline stage latency in milliseconds, by quantile.",
    ("stage", "quantile"),
)
STAGE_SECONDS_TOTAL = REGISTRY.counter(
    "adas_stage_seconds_total",
    "Cumulative wall time spent in each pipeline stage, seconds.",
    ("stage",),
)
STAGE_DURATION_MS = REGISTRY.histogram(
    "adas_stage_duration_ms",
    "Distribution of per-stage durations in milliseconds.",
    ("stage",),
)

# ------------------------------------------------------------------- perception

PERCEPTION_OK = REGISTRY.gauge(
    "adas_perception_ok", "1 while perception produced a usable frame."
)
PERCEPTION_FAILURES_TOTAL = REGISTRY.counter(
    "adas_perception_failures_total",
    "Perception exceptions, by the stage that raised.",
    ("stage",),
)
PERCEPTION_CONSECUTIVE_FAILURES = REGISTRY.gauge(
    "adas_perception_consecutive_failures",
    "Consecutive failed perception steps; 0 when healthy.",
)
DETECTIONS = REGISTRY.gauge("adas_detections", "Detections in the most recent frame.")
TRACKS = REGISTRY.gauge("adas_tracks", "Confirmed tracks in the most recent frame.")
LANE_OK = REGISTRY.gauge("adas_lane_ok", "1 when the last frame produced a lane model.")
LANE_IS_MOCK = REGISTRY.gauge(
    "adas_lane_is_mock", "1 when the lane model is a deterministic stub, not a measurement."
)

# ----------------------------------------------------------------------- safety

SAFETY_STATE = REGISTRY.gauge(
    "adas_safety_state",
    "1 on the currently active SafetyState, 0 on the others.",
    ("state",),
)
SAFETY_STATE_TRANSITIONS_TOTAL = REGISTRY.counter(
    "adas_safety_state_transitions_total",
    "SafetyState transitions.",
    ("from_state", "to_state"),
)
SAFETY_VIOLATIONS_TOTAL = REGISTRY.counter(
    "adas_safety_violations_total", "Hard safety-limit violations, by kind.", ("kind",)
)
SAFETY_WARNINGS_TOTAL = REGISTRY.counter(
    "adas_safety_warnings_total", "Soft safety warnings that did not veto a command.", ("kind",)
)
PLAN_TARGET_SPEED_MPS = REGISTRY.gauge(
    "adas_plan_target_speed_mps", "Target speed of the most recent motion plan, m/s."
)
EGO_SPEED_MPS = REGISTRY.gauge("adas_ego_speed_mps", "Ego speed used by the last step, m/s.")
EGO_SPEED_VALID = REGISTRY.gauge(
    "adas_ego_speed_valid", "1 when the ego speed is a measurement, 0 when simulated or stale."
)
LEAD_DISTANCE_M = REGISTRY.gauge(
    "adas_lead_distance_m", "Range to the closest in-lane object, metres (+Inf when none)."
)

# ---------------------------------------------------------------------- engines

MODEL_LOADED = REGISTRY.gauge(
    "adas_model_loaded", "1 when the engine deserialised and is usable.", ("model",)
)
MODEL_FAILURES = REGISTRY.gauge(
    "adas_model_consecutive_failures",
    "Consecutive inference failures for this engine; 0 when healthy.",
    ("model",),
)
MODEL_MEAN_MS = REGISTRY.gauge(
    "adas_model_mean_ms", "Rolling mean inference latency in milliseconds.", ("model",)
)
ENGINE_ERRORS_TOTAL = REGISTRY.counter(
    "adas_engine_errors_total",
    "Engine failures by kind (load, shape, execution, hang).",
    ("engine", "kind"),
)

# ----------------------------------------------------------------------- source

SOURCE_UP = REGISTRY.gauge("adas_source_up", "1 while the capture source is delivering frames.")
SOURCE_RECONNECTS_TOTAL = REGISTRY.counter(
    "adas_source_reconnects_total", "Capture source reopen attempts."
)

# -------------------------------------------------------------------- event log

EVENT_WRITES_TOTAL = REGISTRY.counter(
    "adas_event_writes_total", "Safety event records appended to the JSONL."
)
EVENT_WRITE_FAILURES_TOTAL = REGISTRY.counter(
    "adas_event_write_failures_total", "Failed JSONL appends, by errno name.", ("errno",)
)
EVENT_DROPPED_TOTAL = REGISTRY.counter(
    "adas_event_dropped_total", "Events discarded because the log was unwritable.", ("reason",)
)
EVENT_ROTATIONS_TOTAL = REGISTRY.counter(
    "adas_event_rotations_total", "In-process JSONL rotations, by trigger.", ("reason",)
)
EVENT_FSYNCS_TOTAL = REGISTRY.counter(
    "adas_event_fsyncs_total", "fsync() calls on the JSONL (durability barriers)."
)
EVENTS_WRITABLE = REGISTRY.gauge(
    "adas_events_writable", "1 while safety events are reaching durable storage."
)
EVENTS_BYTES = REGISTRY.gauge("adas_events_bytes", "Size of the live JSONL in bytes.")
EVENTS_BY_TYPE_TOTAL = REGISTRY.counter(
    "adas_events_by_type_total", "Safety events written, by type and severity.",
    ("type", "severity"),
)

# ---------------------------------------------------------------------- storage

DISK_FREE_MB = REGISTRY.gauge(
    "adas_disk_free_mb", "Free space on the filesystem holding this path, MiB.", ("path",)
)
DISK_FULL = REGISTRY.gauge(
    "adas_disk_full", "1 while any writer has hit ENOSPC/EDQUOT and stopped."
)

# -------------------------------------------------------------------- lifecycle

WATCHDOG_PINGS_TOTAL = REGISTRY.counter(
    "adas_watchdog_pings_total", "WATCHDOG=1 datagrams sent to systemd."
)
WATCHDOG_SKIPPED_TOTAL = REGISTRY.counter(
    "adas_watchdog_skipped_total",
    "Watchdog deadlines passed with no frame progress, so no ping was sent.",
)
HEALTH_STALE = REGISTRY.gauge(
    "adas_health_stale", "1 when the health snapshot has not been refreshed in time."
)
HEALTH_REQUESTS_TOTAL = REGISTRY.counter(
    "adas_health_requests_total", "HTTP requests to the health server.", ("route", "code")
)
DEGRADED = REGISTRY.gauge("adas_degraded", "1 while the unit is running in a degraded mode.")
RAM_MB = REGISTRY.gauge("adas_ram_mb", "Resident set size of this process, MiB.")


# ------------------------------------------------------------------- convenience


def record_stage(stage: str, duration_ms: float) -> None:
    """Record one stage duration into both the histogram and the cumulative counter.

    Args:
        stage: ``capture``, ``detect``, ``lane``, ``track``, ``plan``, ``control``,
            ``e2e`` — the vocabulary in :mod:`adas.core.metrics`.
        duration_ms: milliseconds. A negative value is ignored rather than corrupting
            the counter, because it can only come from a wall-clock step backwards.
    """
    if duration_ms < 0 or math.isnan(duration_ms):
        return
    STAGE_DURATION_MS.observe(duration_ms, stage=stage)
    STAGE_SECONDS_TOTAL.inc(duration_ms / 1000.0, stage=stage)


def set_safety_state(state: Any, previous: Any = None) -> None:
    """Publish the active SafetyState as a one-hot gauge and count the transition.

    Accepts the enum or its string value, so callers do not have to import
    :class:`adas.core.models.SafetyState` just to report.
    """
    name = str(getattr(state, "value", state))
    for known in ("nominal", "limited", "min_risk_maneuver", "disengage"):
        SAFETY_STATE.set_bool(known == name, state=known)
    if name not in ("nominal", "limited", "min_risk_maneuver", "disengage"):
        SAFETY_STATE.set_bool(True, state=name)
    if previous is not None:
        prev = str(getattr(previous, "value", previous))
        if prev != name:
            SAFETY_STATE_TRANSITIONS_TOTAL.inc(from_state=prev, to_state=name)


def ram_mb() -> Optional[float]:
    """Resident set size of this process in MiB, or None off Linux.

    Reads ``/proc/self/statm`` rather than shelling out; the second field is resident
    pages. Returns None (never a guess) when /proc is unavailable.
    """
    try:
        with open("/proc/self/statm") as f:
            resident_pages = int(f.read().split()[1])
    except (OSError, IndexError, ValueError):
        return None
    value = resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)
    RAM_MB.set(value)
    return value


def _git_sha(root: str) -> str:
    """Best-effort source revision. Returns '' rather than guessing.

    Order: ``$ADAS_GIT_SHA`` (set by the installer), ``/etc/adas/release`` (written by
    ``deploy/setup_jetson.sh``), then ``git rev-parse`` in a checkout. A production
    unit has no ``.git``, which is exactly why the installer writes the release file.
    """
    env = os.environ.get("ADAS_GIT_SHA", "").strip()
    if env:
        return env[:40]
    try:
        with open("/etc/adas/release") as f:
            for line in f:
                if line.startswith("git_sha="):
                    return line.split("=", 1)[1].strip()[:40]
    except OSError:
        pass
    if os.path.isdir(os.path.join(root, ".git")):
        try:
            out = subprocess.check_output(
                ["git", "-C", root, "rev-parse", "--short=12", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return out.decode("ascii", "replace").strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return ""


def set_build_info(version: str = "", git_sha: Optional[str] = None, trt_version: str = "",
                   config: str = "", detector: str = "", lane: str = "") -> Dict[str, str]:
    """Publish ``adas_build_info`` and return the labels that were used.

    Every label is a string; unknown values become ``"unknown"`` rather than being
    omitted, because a scrape with a missing label silently changes the series identity
    and breaks ``count by (git_sha)`` across a fleet. ``detector``/``lane`` carry the
    active perception backends so a fleet query can separate real-model units from
    units still running the mock estimators.
    """
    if not version:
        try:
            import adas

            version = getattr(adas, "__version__", "")
        except Exception:  # partially-initialised package during import
            version = ""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    labels = {
        "version": version or "unknown",
        "git_sha": (git_sha if git_sha is not None else _git_sha(root)) or "unknown",
        "trt_version": trt_version or "unknown",
        "config": config or "unknown",
        "detector": detector or "unknown",
        "lane": lane or "unknown",
    }
    BUILD_INFO.set(1.0, **labels)
    return labels


def render() -> str:
    """The default registry as Prometheus text."""
    return REGISTRY.render()


def reset() -> None:
    """Zero every series in the default registry. Tests only."""
    REGISTRY.reset()


__all__ = [
    "Counter",
    "DEFAULT_MS_BUCKETS",
    "Gauge",
    "Histogram",
    "REGISTRY",
    "Registry",
    "ram_mb",
    "record_stage",
    "render",
    "reset",
    "set_build_info",
    "set_safety_state",
]
