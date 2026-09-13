"""Per-process performance metrics for the ADAS pipeline.

This is the *application's* view of its own throughput: counts, real elapsed time, and
a latency distribution per pipeline stage. The fleet-facing view — Prometheus counters
and gauges that a scraper reads while the process runs — lives in :mod:`adas.io.metrics`
and is fed automatically from here, so instrumenting a stage once publishes it both to
the end-of-run summary and to ``/metrics``.

What changed and why
--------------------
The previous version accumulated ``total_processing_time`` from whatever ``frame_time``
the caller passed, and the caller passed the *nominal* frame period ``1/target_fps``.
``avg_fps`` was therefore mathematically forced to equal ``target_fps``, with
``min == max == avg``: a number that could never disagree with the configuration, and
so could never detect a regression. The four stage timers (``perception_time`` and
friends) were declared and never assigned by anything.

Now :meth:`PerformanceMetrics.update_frame` reads ``time.monotonic()`` itself. The
``frame_time`` argument is still accepted and still accumulated (it is the caller's
*claim*, exposed as :attr:`reported_frame_time_s`), but :attr:`measured_fps` and the
latency quantiles come from the clock, and they disagree with the nominal rate exactly
when something is wrong. ``time.monotonic`` — not ``time.time`` — because this board
has no battery-backed RTC: it boots with a wrong wall clock and steps when NTP syncs,
and a backward step through ``time.time`` arithmetic produces negative latencies and a
stalled frame loop.

Units
-----
Seconds for every ``*_s`` / ``*_time`` field, milliseconds for every ``*_ms`` field and
for :meth:`PerformanceMetrics.latency_ms`, frames per second for the rates.

Failure behaviour
-----------------
Nothing here raises. An unknown stage name is recorded under its own label rather than
rejected (a missing measurement is worse than an unexpected one), and a negative
duration — only possible from a clock that went backwards — is dropped and counted.
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from adas.core.logger import log_event, setup_logger
from adas.io import metrics as _registry

logger = setup_logger(__name__)

#: Stage names the pipeline is expected to use. Others are accepted but not documented.
STAGES = ("capture", "detect", "lane", "track", "plan", "control", "e2e")

#: How many recent samples each stage keeps for quantile estimation. 600 samples is
#: 30 s at 20 Hz — long enough for a stable p95, short enough that a fault shows up
#: within half a minute instead of being averaged away over an hour-long drive.
DEFAULT_WINDOW = 600


class LatencyWindow:
    """Fixed-size ring of recent durations in milliseconds, with quantiles.

    A ring, not a running mean: a mean cannot show a tail, and the tail is what makes a
    20 Hz pipeline miss its deadline. Quantiles are computed by sorting the window
    (<= ``capacity`` elements, so a few microseconds) rather than kept incrementally,
    because an incremental estimator that is wrong is worse than a cheap exact one.
    """

    def __init__(self, capacity: int = DEFAULT_WINDOW) -> None:
        self.capacity = max(1, int(capacity))
        self._values: List[float] = []
        self._index = 0
        self.count = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self.min_ms = float("inf")

    def observe(self, ms: float) -> None:
        """Record one duration in milliseconds. Negative and NaN values are ignored.

        Negative can only come from a clock that went backwards; NaN from an
        uninitialised timer. Either would poison every quantile from then on.
        """
        if ms < 0 or math.isnan(ms):
            return
        value = float(ms)
        if len(self._values) < self.capacity:
            self._values.append(value)
        else:
            self._values[self._index] = value
            self._index = (self._index + 1) % self.capacity
        self.count += 1
        self.total_ms += value
        self.max_ms = max(self.max_ms, value)
        self.min_ms = min(self.min_ms, value)

    @property
    def mean_ms(self) -> float:
        """Mean over the whole run (not just the window). 0.0 before any sample."""
        return self.total_ms / self.count if self.count else 0.0

    def quantile(self, q: float) -> float:
        """Nearest-rank quantile over the window. 0.0 before any sample."""
        if not self._values:
            return 0.0
        ordered = sorted(self._values)
        idx = int(round(q * (len(ordered) - 1)))
        return ordered[max(0, min(len(ordered) - 1, idx))]

    @property
    def p50(self) -> float:
        return self.quantile(0.50)

    @property
    def p95(self) -> float:
        return self.quantile(0.95)

    @property
    def p99(self) -> float:
        return self.quantile(0.99)


@dataclass
class PerformanceMetrics:
    """Counts, real timing and per-stage latency for one pipeline process.

    Instances are **not** thread-safe: one pipeline thread owns one instance. The
    health endpoint reads through :meth:`snapshot`, which copies, so a scrape cannot
    observe a half-updated stage map.
    """

    total_frames: int = 0
    total_detections: int = 0
    total_tracks: int = 0
    frames_with_lane: int = 0

    #: Sum of the ``frame_time`` values the caller reported, seconds. This is the
    #: caller's claim about the frame period, not a measurement; see
    #: :attr:`measured_elapsed_s`.
    reported_frame_time_s: float = 0.0

    # Safety counters. `record_safety_event` distinguishes the two; a violation is a
    # breached hard limit, a warning is a soft limit that did not veto the command.
    safety_warnings: int = 0
    safety_violations: int = 0
    perception_failures: int = 0
    source_reconnects: int = 0
    frames_dropped: int = 0
    clock_regressions: int = 0

    window: int = DEFAULT_WINDOW
    clock: Callable[[], float] = time.monotonic

    #: stage name -> LatencyWindow. Populated on first use of each stage.
    stages: Dict[str, LatencyWindow] = field(default_factory=dict)

    _first_frame_mono: Optional[float] = field(default=None, init=False, repr=False)
    _last_frame_mono: Optional[float] = field(default=None, init=False, repr=False)
    _started_mono: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._started_mono = self.clock()

    # ---------------------------------------------------------------- recording

    def update_frame(self, frame_time: float = 0.0, num_detections: int = 0,
                     num_tracks: int = 0, has_lane: bool = False) -> None:
        """Record one completed frame.

        Args:
            frame_time: the caller's nominal or measured frame period, seconds. It is
                accumulated into :attr:`reported_frame_time_s` for compatibility but is
                **not** used for :attr:`measured_fps`, which reads the clock.
            num_detections: detections in this frame.
            num_tracks: confirmed tracks after association.
            has_lane: whether a lane model was produced.

        Also derives the real inter-frame interval from the monotonic clock and
        records it under the ``interval`` stage, so a caller that instruments no stages
        at all still gets an honest rate and latency distribution. ``interval`` is the
        period between frames; ``e2e`` (if the caller times it) is the work inside one.
        """
        now = self.clock()
        if self._last_frame_mono is not None:
            delta_ms = (now - self._last_frame_mono) * 1000.0
            if delta_ms < 0:
                # time.monotonic must not go backwards; if it does, something replaced
                # the clock (a test stub, a fake). Count it rather than poisoning p95.
                self.clock_regressions += 1
            else:
                self._stage("interval").observe(delta_ms)
                _registry.record_stage("interval", delta_ms)
        else:
            self._first_frame_mono = now
        self._last_frame_mono = now

        self.total_frames += 1
        self.total_detections += num_detections
        self.total_tracks += num_tracks
        if has_lane:
            self.frames_with_lane += 1
        self.reported_frame_time_s += float(frame_time)

        _registry.FRAMES_PROCESSED_TOTAL.inc()
        _registry.FRAME_ID.set(self.total_frames)
        _registry.DETECTIONS.set(num_detections)
        _registry.TRACKS.set(num_tracks)
        _registry.LANE_OK.set_bool(has_lane)
        _registry.FPS.set(self.measured_fps)

    def record_stage(self, stage: str, duration_ms: float) -> None:
        """Record one stage duration in milliseconds, locally and in the registry."""
        self._stage(stage).observe(duration_ms)
        _registry.record_stage(stage, duration_ms)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block and record it under *name*, even if the block raises.

        Example:
            >>> with metrics.stage("detect"):
            ...     detections = detector.infer(frame.rgb, w, h)

        The duration is recorded in a ``finally``, so a failed stage still contributes
        its latency: an engine that takes 900 ms and then throws is exactly the thing
        you want to see in the histogram.
        """
        start = self.clock()
        try:
            yield
        finally:
            self.record_stage(name, (self.clock() - start) * 1000.0)

    def record_safety_event(self, is_violation: bool = False, kind: str = "unspecified") -> None:
        """Count one safety event.

        Args:
            is_violation: True for a breached hard limit (something was or should have
                been vetoed), False for a soft warning that let the command stand.
            kind: short machine-readable label, e.g. ``following_distance``,
                ``decel_limit``, ``steering_rate``. It becomes a Prometheus label, so
                keep the cardinality low — a fixed vocabulary, never a formatted value.
        """
        if is_violation:
            self.safety_violations += 1
            _registry.SAFETY_VIOLATIONS_TOTAL.inc(kind=kind)
        else:
            self.safety_warnings += 1
            _registry.SAFETY_WARNINGS_TOTAL.inc(kind=kind)

    def record_perception_failure(self, stage: str = "detect") -> None:
        """Count one perception exception, by the stage that raised it."""
        self.perception_failures += 1
        _registry.PERCEPTION_FAILURES_TOTAL.inc(stage=stage)

    def record_dropped_frame(self, reason: str = "capture") -> None:
        """Count one frame that never completed a pipeline step."""
        self.frames_dropped += 1
        _registry.FRAMES_DROPPED_TOTAL.inc(reason=reason)

    def record_source_reconnect(self) -> None:
        """Count one capture-source reopen attempt."""
        self.source_reconnects += 1
        _registry.SOURCE_RECONNECTS_TOTAL.inc()

    def _stage(self, name: str) -> LatencyWindow:
        win = self.stages.get(name)
        if win is None:
            win = LatencyWindow(self.window)
            self.stages[name] = win
        return win

    # ---------------------------------------------------------------- reporting

    @property
    def measured_elapsed_s(self) -> float:
        """Wall time from the first to the most recent frame, seconds (monotonic)."""
        if self._first_frame_mono is None or self._last_frame_mono is None:
            return 0.0
        return max(0.0, self._last_frame_mono - self._first_frame_mono)

    @property
    def uptime_s(self) -> float:
        """Seconds since this metrics object was constructed."""
        return max(0.0, self.clock() - self._started_mono)

    @property
    def measured_fps(self) -> float:
        """Frames per second measured against the monotonic clock.

        Uses ``total_frames - 1`` intervals over the measured span, so two frames one
        second apart is 1.0 fps rather than 2.0. Returns 0.0 before the second frame,
        which is honest: one frame does not define a rate.
        """
        span = self.measured_elapsed_s
        if span <= 0 or self.total_frames < 2:
            return 0.0
        return (self.total_frames - 1) / span

    @property
    def avg_fps(self) -> float:
        """Alias of :attr:`measured_fps`, kept for source compatibility.

        The old implementation divided by the *reported* frame times and therefore
        always returned the configured target rate.
        """
        return self.measured_fps

    @property
    def avg_frame_time(self) -> float:
        """Mean measured inter-frame interval, seconds. 0.0 before the second frame."""
        win = self.stages.get("interval")
        return win.mean_ms / 1000.0 if win and win.count else 0.0

    @property
    def min_frame_time(self) -> float:
        """Shortest measured inter-frame interval, seconds. 0.0 before any sample."""
        win = self.stages.get("interval")
        if not win or not win.count:
            return 0.0
        return win.min_ms / 1000.0

    @property
    def max_frame_time(self) -> float:
        """Longest measured inter-frame interval, seconds. 0.0 before any sample."""
        win = self.stages.get("interval")
        return win.max_ms / 1000.0 if win and win.count else 0.0

    @property
    def lane_detection_rate(self) -> float:
        """Percentage of frames that produced a lane model."""
        if self.total_frames == 0:
            return 0.0
        return (self.frames_with_lane / self.total_frames) * 100.0

    def latency_ms(self) -> Dict[str, float]:
        """``{"<stage>_p50": ms, "<stage>_p95": ms, ...}`` for the health endpoint.

        Only stages that have at least one sample appear, so a missing key means "never
        measured" rather than "measured as zero".
        """
        out: Dict[str, float] = {}
        for name, win in self.stages.items():
            if not win.count:
                continue
            out["%s_p50" % name] = round(win.p50, 3)
            out["%s_p95" % name] = round(win.p95, 3)
        return out

    def snapshot(self) -> Dict[str, Any]:
        """A plain-dict copy of everything, safe to hand to another thread or to JSON."""
        return {
            "frames": self.total_frames,
            "frames_dropped": self.frames_dropped,
            "detections": self.total_detections,
            "tracks": self.total_tracks,
            "lane_rate_pct": round(self.lane_detection_rate, 1),
            "fps": round(self.measured_fps, 2),
            "elapsed_s": round(self.measured_elapsed_s, 2),
            "uptime_s": round(self.uptime_s, 2),
            "safety_warnings": self.safety_warnings,
            "safety_violations": self.safety_violations,
            "perception_failures": self.perception_failures,
            "source_reconnects": self.source_reconnects,
            "clock_regressions": self.clock_regressions,
            "latency_ms": self.latency_ms(),
            "stages": {
                name: {
                    "count": win.count,
                    "mean_ms": round(win.mean_ms, 3),
                    "p50_ms": round(win.p50, 3),
                    "p95_ms": round(win.p95, 3),
                    "p99_ms": round(win.p99, 3),
                    "max_ms": round(win.max_ms, 3),
                }
                for name, win in sorted(self.stages.items())
                if win.count
            },
        }

    def summary(self) -> str:
        """Multi-line human summary. Never prints ``inf`` for an unmeasured value."""
        lines = [
            "Performance summary:",
            "  Frames:           %d processed, %d dropped" % (self.total_frames,
                                                              self.frames_dropped),
            "  Rate:             %.2f fps measured over %.1f s"
            % (self.measured_fps, self.measured_elapsed_s),
            "  Detections:       %d (avg %.2f/frame)"
            % (self.total_detections, self.total_detections / max(1, self.total_frames)),
            "  Tracks:           %d (avg %.2f/frame)"
            % (self.total_tracks, self.total_tracks / max(1, self.total_frames)),
            "  Lane detected:    %.1f%% of frames" % self.lane_detection_rate,
            "  Perception fails: %d" % self.perception_failures,
            "  Safety:           %d violations, %d warnings"
            % (self.safety_violations, self.safety_warnings),
            "  Source reconnects: %d" % self.source_reconnects,
        ]
        if self.clock_regressions:
            lines.append("  Clock regressions: %d (monotonic clock went backwards)"
                         % self.clock_regressions)
        stage_rows = [(n, w) for n, w in sorted(self.stages.items()) if w.count]
        if stage_rows:
            lines.append("  Stage latency (ms)      n     mean      p50      p95      max")
            for name, win in stage_rows:
                lines.append("    %-18s %6d %8.2f %8.2f %8.2f %8.2f"
                             % (name, win.count, win.mean_ms, win.p50, win.p95, win.max_ms))
        else:
            lines.append("  Stage latency:    not instrumented (no stage() calls)")
        return "\n".join(lines)

    def log_summary(self) -> None:
        """Log the summary at INFO under the ``summary`` event name.

        Call this in the runner's ``finally:`` and, for a long-running service, on an
        interval — it is the only place the measured rate ever gets stated.
        """
        log_event(logger, "summary", self.summary(), **{
            k: v for k, v in self.snapshot().items() if not isinstance(v, dict)
        })

    def reset(self) -> None:
        """Zero everything. The registry counters are *not* reset: they are monotonic
        by contract and a scraper computes rates from their deltas."""
        self.total_frames = 0
        self.total_detections = 0
        self.total_tracks = 0
        self.frames_with_lane = 0
        self.reported_frame_time_s = 0.0
        self.safety_warnings = 0
        self.safety_violations = 0
        self.perception_failures = 0
        self.source_reconnects = 0
        self.frames_dropped = 0
        self.clock_regressions = 0
        self.stages = {}
        self._first_frame_mono = None
        self._last_frame_mono = None
        self._started_mono = self.clock()


@dataclass
class SystemHealthMonitor:
    """Progress-aware heartbeat, and the liveness verdict behind it.

    This used to be dead code: it was exported but never constructed, and its
    ``check_watchdog`` measured only "did anyone call heartbeat recently", which a
    bare timer satisfies while the frame loop is wedged. It now tracks the frame id,
    so :meth:`is_alive` means *frames are still being produced* — the only liveness
    claim worth making — and :class:`adas.io.sd_notify.WatchdogPinger` enforces the
    same rule against systemd.

    Units: seconds throughout. ``time.monotonic``, never ``time.time``.
    """

    heartbeat_interval_s: float = 5.0
    stall_timeout_s: float = 10.0
    clock: Callable[[], float] = time.monotonic

    start_mono: float = field(default=0.0, init=False)
    last_heartbeat_mono: float = field(default=0.0, init=False)
    last_progress_mono: float = field(default=0.0, init=False)
    last_frame_id: int = field(default=-1, init=False)
    stalls: int = field(default=0, init=False)
    _stalled: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        now = self.clock()
        self.start_mono = now
        self.last_heartbeat_mono = now
        self.last_progress_mono = now

    @property
    def uptime_s(self) -> float:
        return max(0.0, self.clock() - self.start_mono)

    def heartbeat(self, frame_id: int = -1) -> bool:
        """Record progress and, at most every ``heartbeat_interval_s``, log it.

        Args:
            frame_id: id of the frame just processed. Pass -1 when calling outside the
                frame loop; that counts as *no* progress, deliberately.

        Returns:
            True when this call logged a heartbeat line.
        """
        now = self.clock()
        if frame_id >= 0 and frame_id != self.last_frame_id:
            self.last_frame_id = frame_id
            self.last_progress_mono = now
        if now - self.last_heartbeat_mono < self.heartbeat_interval_s:
            return False
        self.last_heartbeat_mono = now
        logger.info("heartbeat uptime=%.1fs frame_id=%s stalled_for=%.1fs",
                    self.uptime_s, self.last_frame_id, self.stalled_for_s())
        return True

    def stalled_for_s(self) -> float:
        """Seconds since the frame id last advanced."""
        return max(0.0, self.clock() - self.last_progress_mono)

    def is_alive(self, timeout_s: Optional[float] = None) -> bool:
        """True while the frame id has advanced within *timeout_s*.

        :attr:`stalls` counts falling *edges*, not calls, so a unit polled at 20 Hz
        during one five-second stall reports one stall rather than a hundred.
        """
        limit = self.stall_timeout_s if timeout_s is None else float(timeout_s)
        alive = self.stalled_for_s() < limit
        if not alive and not self._stalled:
            self.stalls += 1
            logger.error("pipeline stalled: no frame in %.1fs (limit %.1fs, last id %s)",
                         self.stalled_for_s(), limit, self.last_frame_id)
        self._stalled = not alive
        return alive

    def check_watchdog(self, timeout: float = 10.0) -> bool:
        """Deprecated alias of :meth:`is_alive`, kept for source compatibility."""
        return self.is_alive(timeout)


__all__ = [
    "DEFAULT_WINDOW",
    "STAGES",
    "LatencyWindow",
    "PerformanceMetrics",
    "SystemHealthMonitor",
]
