"""Health and metrics HTTP endpoint (stdlib only — no Flask, no FastAPI).

Routes
------
=============  ======  ====================================================
``/livez``     200     the process is running and the HTTP thread responds.
                       Never 503 — a liveness probe that fails on a dead
                       camera makes systemd kill a unit that is correctly
                       reporting a dead camera.
``/readyz``    200/503 200 only when every configured engine is loaded, the
                       source is delivering, the frame id is advancing and
                       the snapshot is fresh. This is the probe a fleet or
                       an ``ExecStartPost`` gate should key off.
``/healthz``   200/503 503 when ``ok`` is false or the snapshot is stale;
``/health``            200 with ``"degraded": true`` for a unit that is
``/``                  running in LIMITED — degraded is a mode, not an
                       outage, and paging on it teaches operators to ignore
                       the page.
``/metrics``   200     Prometheus text, always 200 — a scraper must be able
                       to scrape a broken unit; that is the point.
=============  ======  ====================================================

Any other path is 404; any method other than GET/HEAD is 405 with ``Allow``.

Status codes, not body fields
-----------------------------
Every standard prober — ``curl -f``, a systemd gate, an uptime monitor, a fleet agent
— keys off the HTTP status and never parses the body. So the verdict has to be in the
status line: ``/healthz`` returns 503 when the unit is not ok or its snapshot is
stale, and ``/readyz`` returns 503 until the pipeline has actually produced a frame.

Staleness is what makes this more than a liveness check. :meth:`HealthState.mark_update`
stamps a monotonic timestamp on every refresh and an advancing ``frame_id`` counts as
an implicit refresh, so a wedged frame loop turns into ``stale: true`` and a 503 within
``stale_after_s`` even though the HTTP thread is perfectly responsive.

Binding and exposure
--------------------
The default bind is ``127.0.0.1`` and a non-loopback bind is **refused** unless
``allow_remote=True`` is passed explicitly. ``/healthz`` publishes the vehicle's live
safety state, ego speed and lead-vehicle range; a config that quietly set
``bind: 0.0.0.0`` would publish that unauthenticated to the whole vehicle network.
When remote access is genuinely wanted, ``token`` enables ``Authorization: Bearer``
checking on everything except ``/livez``.

Resource limits: a 5 s socket timeout (a half-open connection would otherwise pin a
worker thread for the life of the process) and a hard cap on concurrent connections,
over which the server answers 503 and closes.

Units: ``fps`` frames per second, ``latency_ms`` milliseconds, ``speed_mps`` metres per
second, ``distance_m`` metres, ``ram_mb`` MiB, ``uptime_s``/``stale_after_s`` seconds.

Memory: ``ram_mb`` is this process's resident set size, re-read from
``/proc/self/statm`` at most once per ``process_refresh_s`` on every health refresh and
on every probe. On a 6.7 GiB board it is the number an operator watches, so it is
measured rather than defaulted: when /proc cannot be read it stays ``null`` in the JSON
and is exported as ``NaN`` on ``/metrics`` -- never as ``0``, which a scraper would
graph as a process holding no memory.

``engine_sha256`` lists the sha256 of every engine this process verified against
``models/MANIFEST.json`` (see :func:`adas.infer.trt_engine.verify_engine_file`). An
engine that could not be verified is absent from the map rather than present with an
empty digest.

Failure behaviour: :meth:`HealthServer.start` returns ``None`` and logs an error when
the port is taken or the bind is refused; pass ``required=True`` to make that raise
instead, which is what a production profile should do rather than run blind.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from adas.io import metrics

log = logging.getLogger("adas.health")

#: Addresses that keep the endpoint on the box.
LOOPBACK_BINDS = frozenset(("127.0.0.1", "::1", "localhost", "127.0.1.1"))

#: A snapshot older than this (seconds) means the pipeline is not refreshing health.
DEFAULT_STALE_AFTER_S = 5.0
#: No staleness or readiness verdict during startup; matches TimeoutStartSec=90.
DEFAULT_STARTUP_GRACE_S = 60.0
#: Lower bound between two ``/proc/self/statm`` reads. The frame loop refreshes health
#: at 20 Hz and a fleet scrapes at 1 Hz; one read per second covers both.
DEFAULT_PROCESS_REFRESH_S = 1.0
#: Default port. DMS owns 8088; ADAS takes 8090 so both can run on one board.
DEFAULT_PORT = 8090

ROUTES = ("/healthz", "/livez", "/readyz", "/metrics")


def is_loopback(bind: str) -> bool:
    """True when *bind* keeps the listener off the network.

    An empty bind or ``0.0.0.0``/``::`` means every interface and is not loopback.
    """
    b = (bind or "").strip()
    if not b:
        return False
    if b in LOOPBACK_BINDS:
        return True
    try:
        packed = socket.inet_pton(socket.AF_INET, b)
        return packed[0] == 127
    except (OSError, ValueError):
        pass
    try:
        socket.inet_pton(socket.AF_INET6, b)
        return b == "::1"
    except (OSError, ValueError):
        return False


class HealthState:
    """Mutable snapshot served at ``/healthz``. Thread-safe via :attr:`lock`.

    The pipeline thread mutates the plain attributes under ``with state.lock:`` and
    calls :meth:`mark_update`; the HTTP threads only ever read through :meth:`as_dict`,
    which takes the same lock.

    Everything here is a *report*, never a computation the pipeline depends on. If a
    field is unknown it stays ``None`` rather than being defaulted to something
    plausible: ``lead_distance_m: null`` means "no lead object reported", which is not
    the same claim as ``lead_distance_m: 250.0``.
    """

    def __init__(self, stale_after_s: float = DEFAULT_STALE_AFTER_S,
                 startup_grace_s: float = DEFAULT_STARTUP_GRACE_S,
                 clock=time.monotonic,
                 process_refresh_s: float = DEFAULT_PROCESS_REFRESH_S) -> None:
        self.lock = threading.RLock()
        self._clock = clock
        self.started_mono = clock()
        self.last_update_mono = 0.0
        self._last_frame_seen = 0
        self.stale_after_s = float(stale_after_s)
        self.startup_grace_s = float(startup_grace_s)
        self.process_refresh_s = float(process_refresh_s)
        self._last_process_refresh = 0.0

        # -- liveness / readiness
        self.ok = True
        self.degraded = False
        self.source = "starting"          # ok | starting | lost | eof | error
        self.source_type = "synthetic"    # synthetic | video | camera | csi
        self.source_uri = ""
        self.source_reconnects = 0
        self.engines: Dict[str, str] = {}   # name -> loaded | missing | failed | mock
        self.notes: List[str] = []

        # -- perception
        self.perception_ok = True
        self.perception_consecutive_failures = 0
        self.perception_reason = ""
        self.detector_backend = "unknown"
        self.lane_backend = "unknown"
        self.lane_is_mock = True

        # -- safety / planning
        self.safety_state = "nominal"
        self.plan_reason = ""
        self.safety_violations = 0
        self.safety_warnings = 0
        # -- what the ARBITER itself decided on the last frame.  The state label
        # alone answers "is it degraded?" and nothing else; these answer "why,
        # how hard, and did it actually change the command?", which is what an
        # operator looking at a vehicle that is braking needs.  All are plain
        # snapshots of the last ArbitrationResult -- reporting only, never read
        # back by the decision path.
        self.safety_reason = ""
        self.safety_demand_mps2 = 0.0
        self.safety_overrode_command = False
        self.safety_last_violations: list = []
        self.safety_transitions = 0
        self.safety_last_transition_frame: Optional[int] = None
        self.safety_lead_track_id: Optional[int] = None
        self.safety_rate_is_measured = False
        self.ego_speed_mps: Optional[float] = None
        self.ego_speed_valid = False
        self.lead_distance_m: Optional[float] = None

        # -- throughput
        self.fps = 0.0
        self.latency_ms: Dict[str, float] = {"e2e_p50": 0.0, "e2e_p95": 0.0}
        self.frame_id = 0
        self.frames_total = 0
        self.frames_dropped = 0

        # -- board and storage
        self.ram_mb: Optional[float] = None
        self.events: Dict[str, Any] = {"writable": None, "path": ""}
        self.watchdog: Dict[str, Any] = {}
        self.build: Dict[str, str] = {}

    # ------------------------------------------------------------------ update

    def mark_update(self) -> None:
        """Stamp the snapshot as fresh. Call at the end of every health refresh.

        Also re-reads the process facts (:meth:`refresh_process`), rate-limited to
        ``process_refresh_s``: this is the one call the frame loop already makes every
        frame, so hanging RSS off it is what keeps ``ram_mb`` from being ``null`` for
        the life of the run.
        """
        with self.lock:
            self.last_update_mono = self._clock()
        self.refresh_process()

    def _observe(self, now: float) -> None:
        """Treat an advancing frame id as an implicit refresh.

        The pipeline is expected to call :meth:`mark_update`, but a caller that only
        assigns ``frame_id`` still gets working staleness detection: an id that has not
        moved between two observations more than ``stale_after_s`` apart is a stalled
        pipeline whatever else the snapshot says. Frame id 0 does not count — before
        the first frame there is nothing to be stale about, which is what
        ``startup_grace_s`` covers.
        """
        if self.frame_id > 0 and self.frame_id != self._last_frame_seen:
            self._last_frame_seen = self.frame_id
            self.last_update_mono = now

    def set_build(self, **labels: str) -> Dict[str, str]:
        """Record and publish build identity (version, git_sha, trt_version, config)."""
        info = metrics.set_build_info(**labels)
        with self.lock:
            self.build = info
        return info

    def set_engine(self, name: str, status: str) -> None:
        """Report one engine's load state: ``loaded``/``missing``/``failed``/``mock``.

        ``mock`` means a deterministic stub is standing in for the model. It does not
        block readiness — a mock lane estimator is a documented configuration — but it
        does set ``degraded`` so nobody reads the unit as fully instrumented.
        """
        with self.lock:
            self.engines[name] = status
            metrics.MODEL_LOADED.set_bool(status == "loaded", model=name)
            if status == "mock":
                self.degraded = True

    def note(self, text: str) -> None:
        """Attach a one-line operator-visible note (e.g. 'lane estimator is a stub')."""
        with self.lock:
            if text not in self.notes:
                self.notes.append(text)

    def update_from_metrics(self, perf: Any) -> None:
        """Copy the throughput fields out of an :class:`adas.core.metrics.PerformanceMetrics`.

        Duck-typed on the attributes ``total_frames``, ``measured_fps``,
        ``latency_ms()`` and ``safety_violations``/``safety_warnings`` so the health
        module does not import the pipeline's metrics object and tests can pass a stub.
        """
        with self.lock:
            self.frames_total = int(getattr(perf, "total_frames", self.frames_total))
            fps = getattr(perf, "measured_fps", None)
            if fps is None:
                fps = getattr(perf, "avg_fps", None)
            if fps is not None:
                self.fps = float(fps)
            latency = getattr(perf, "latency_ms", None)
            if callable(latency):
                try:
                    self.latency_ms = dict(latency())
                except Exception:  # a reporting helper must not break the endpoint
                    log.debug("latency_ms() raised while refreshing health", exc_info=True)
            self.safety_violations = int(
                getattr(perf, "safety_violations", self.safety_violations)
            )
            self.safety_warnings = int(getattr(perf, "safety_warnings", self.safety_warnings))
            self._observe(self._clock())

    def refresh_process(self, force: bool = False) -> None:
        """Re-read cheap process-level facts (RSS) into the snapshot.

        Called from :meth:`mark_update` (the pipeline's per-frame health refresh) and
        from :meth:`as_dict` (every probe and every scrape), so ``ram_mb`` is live
        whichever of the two is running. One read of ``/proc/self/statm`` plus the
        gauge update measures 0.3 ms on this board, which is why it is rate-limited to
        ``process_refresh_s`` rather than run on all 20 frames of every second;
        ``force`` bypasses the rate limit.

        Leaves ``ram_mb`` at ``None`` when /proc is unreadable: "not measured" is not
        the same claim as "zero".
        """
        now = self._clock()
        with self.lock:
            fresh_enough = (
                not force
                and self.ram_mb is not None
                and (now - self._last_process_refresh) < self.process_refresh_s
            )
            if fresh_enough:
                return
            self._last_process_refresh = now
        rss = metrics.ram_mb()
        with self.lock:
            self.ram_mb = None if rss is None else round(rss, 1)

    # ----------------------------------------------------------------- verdict

    def _age_s(self, now: float) -> float:
        base = self.last_update_mono or self.started_mono
        return max(0.0, now - base)

    def _in_grace(self, now: float) -> bool:
        return self.last_update_mono == 0.0 and (now - self.started_mono) < self.startup_grace_s

    def is_stale(self) -> bool:
        """True when the pipeline has stopped refreshing the snapshot."""
        with self.lock:
            now = self._clock()
            self._observe(now)
            if self._in_grace(now):
                return False
            return self._age_s(now) > self.stale_after_s

    def is_ok(self) -> bool:
        """Liveness verdict behind ``/healthz``: the operator flag AND freshness."""
        with self.lock:
            return bool(self.ok) and not self.is_stale()

    def is_ready(self) -> bool:
        """Readiness verdict behind ``/readyz``.

        Requires: ``ok``, a source reporting ``ok``, no engine in a ``missing``/
        ``failed``/``error`` state, at least one processed frame, and a fresh snapshot.
        A ``mock`` engine is ready-but-degraded, deliberately: the reference pipeline
        with the mock lane estimator is a supported configuration, and refusing
        readiness for it would make ``/readyz`` useless in the lab.
        """
        with self.lock:
            if not self.ok or self.is_stale():
                return False
            if self.source != "ok":
                return False
            if self.frame_id <= 0:
                return False
            if not self.perception_ok:
                return False
            for status in self.engines.values():
                if str(status) in ("missing", "failed", "error"):
                    return False
            return True

    # ---------------------------------------------------------------- rendering

    def as_dict(self) -> Dict[str, Any]:
        """Full JSON body. Top-level keys are stable; new detail goes in sub-objects.

        Refreshes the process facts first (rate-limited), so a probe reports the RSS
        of the moment even if the frame loop has stopped calling :meth:`mark_update` --
        which is exactly when an operator is asking about memory.
        """
        self.refresh_process()
        with self.lock:
            now = self._clock()
            stale = self.is_stale()
            return {
                "ok": bool(self.ok) and not stale,
                "live": True,
                "ready": self.is_ready(),
                "degraded": bool(self.degraded),
                "stale": stale,
                "safety_state": self.safety_state,
                "plan_reason": self.plan_reason,
                "source": self.source,
                "source_type": self.source_type,
                "source_uri": self.source_uri,
                "source_reconnects": int(self.source_reconnects),
                "engines": dict(self.engines),
                "perception": {
                    "ok": bool(self.perception_ok),
                    "consecutive_failures": int(self.perception_consecutive_failures),
                    "reason": self.perception_reason,
                    "detector": self.detector_backend,
                    "lane": self.lane_backend,
                    "lane_is_mock": bool(self.lane_is_mock),
                },
                "fps": round(self.fps, 2),
                "latency_ms": {k: round(float(v), 2) for k, v in self.latency_ms.items()},
                "frame_id": int(self.frame_id),
                "frames_total": int(self.frames_total),
                "frames_dropped": int(self.frames_dropped),
                "safety_violations": int(self.safety_violations),
                "safety_warnings": int(self.safety_warnings),
                "safety": {
                    "state": self.safety_state,
                    "reason": self.safety_reason,
                    "demand_mps2": round(float(self.safety_demand_mps2), 3),
                    "overrode_command": bool(self.safety_overrode_command),
                    "violations": list(self.safety_last_violations),
                    "transitions": int(self.safety_transitions),
                    "last_transition_frame": self.safety_last_transition_frame,
                    "lead_track_id": self.safety_lead_track_id,
                    "lead_rate_is_measured": bool(self.safety_rate_is_measured),
                },
                "ego_speed_mps": self.ego_speed_mps,
                "ego_speed_valid": bool(self.ego_speed_valid),
                "lead_distance_m": self.lead_distance_m,
                "ram_mb": self.ram_mb,
                "engine_sha256": verified_engine_digests(),
                "events": dict(self.events),
                "watchdog": dict(self.watchdog),
                "build": dict(self.build),
                "notes": list(self.notes),
                "uptime_s": round(now - self.started_mono, 1),
                "snapshot_age_s": round(self._age_s(now), 2),
                "stale_after_s": self.stale_after_s,
                "schema": 1,
            }

    def publish_metrics(self) -> None:
        """Copy the snapshot into the metric registry. Called before every scrape."""
        d = self.as_dict()
        metrics.UP.set(1.0)
        metrics.UPTIME_SECONDS.set(d["uptime_s"])
        metrics.FPS.set(d["fps"])
        metrics.FRAME_ID.set(d["frame_id"])
        metrics.DEGRADED.set_bool(d["degraded"])
        metrics.HEALTH_STALE.set_bool(d["stale"])
        metrics.SOURCE_UP.set_bool(d["source"] == "ok")
        metrics.PERCEPTION_OK.set_bool(d["perception"]["ok"])
        metrics.PERCEPTION_CONSECUTIVE_FAILURES.set(d["perception"]["consecutive_failures"])
        metrics.LANE_IS_MOCK.set_bool(d["perception"]["lane_is_mock"])
        metrics.set_safety_state(d["safety_state"])
        if d["ram_mb"] is not None:
            metrics.RAM_MB.set(d["ram_mb"])
        else:
            # An unset gauge renders as 0, and a scraper cannot tell "0 MiB resident"
            # from "never measured". NaN is the exposition format's way of saying the
            # latter, and Prometheus stores it as a real absence of value.
            metrics.RAM_MB.set(float("nan"))
        if d["ego_speed_mps"] is not None:
            metrics.EGO_SPEED_MPS.set(d["ego_speed_mps"])
        metrics.EGO_SPEED_VALID.set_bool(d["ego_speed_valid"])
        if d["lead_distance_m"] is not None:
            metrics.LEAD_DISTANCE_M.set(d["lead_distance_m"])
        if d["events"].get("writable") is not None:
            metrics.EVENTS_WRITABLE.set_bool(d["events"]["writable"])
        # The pipeline owns monotonic totals as plain ints; mirror the increments into
        # the counters so a fleet gets rate() without the application having to know
        # the metric registry exists.
        _advance(metrics.FRAMES_TOTAL, d["frames_total"])
        _advance(metrics.FRAMES_PROCESSED_TOTAL, d["frame_id"])
        _advance(metrics.FRAMES_DROPPED_TOTAL, d["frames_dropped"], reason="capture")
        _advance(metrics.SOURCE_RECONNECTS_TOTAL, d["source_reconnects"])
        for key, value in d["latency_ms"].items():
            stage, quantile = _split_latency_key(key)
            try:
                metrics.STAGE_LATENCY_MS.set(float(value), stage=stage, quantile=quantile)
            except (TypeError, ValueError):
                continue

    def metrics_text(self) -> str:
        """Prometheus exposition body for ``/metrics``."""
        self.publish_metrics()
        return metrics.render()


def verified_engine_digests() -> Dict[str, str]:
    """``{engine filename: sha256}`` for every engine verified in this process.

    Reads :data:`adas.infer.trt_engine.VERIFIED_ENGINES`, which
    :func:`adas.perception.factory.verify_engine` fills in as each engine is checked
    against ``models/MANIFEST.json``. Empty when no TensorRT engine has been verified
    (mock backends, or a manifest with no entry for the file) -- it reports what was
    checked, never what was assumed. Imported lazily so the health endpoint keeps
    working on a host with no numpy/CUDA stack.
    """
    try:
        from adas.infer.trt_engine import VERIFIED_ENGINES
    except Exception:  # pragma: no cover - only on a stripped install
        return {}
    # dict() copies atomically; sorting the live mapping could race a startup
    # thread still verifying engines while a scrape iterates it.
    snapshot = dict(VERIFIED_ENGINES)
    return {os.path.basename(path): digest for path, digest in sorted(snapshot.items())}


def _advance(counter: Any, total: float, **labels: Any) -> None:
    """Raise *counter* to *total*, ignoring a total that went backwards (a restart)."""
    have = counter.value(**labels)
    if total > have:
        counter.inc(total - have, **labels)


def _split_latency_key(key: str):
    """``'e2e_p95' -> ('e2e', 'p95')``; anything else keeps the key as the stage."""
    head, sep, tail = str(key).rpartition("_")
    if sep and tail.startswith("p") and tail[1:].isdigit():
        return head or "e2e", tail
    return str(key), "value"


# ------------------------------------------------------------------------ server


def _consteq(a: str, b: str) -> bool:
    result = 0
    for x, y in zip(a, b):
        result |= ord(x) ^ ord(y)
    return result == 0


def _handler(state: HealthState, token: str):
    class Handler(BaseHTTPRequestHandler):
        # Explicit, not inherited: HTTP/1.0 closes every connection, which is what we
        # want from a probe endpoint, and the 5 s timeout stops one half-open socket
        # from holding a worker thread for the life of the process.
        protocol_version = "HTTP/1.0"
        timeout = 5.0
        server_version = "adas-health"
        sys_version = ""

        def log_message(self, fmt, *args):
            # Requests are counted in adas_health_requests_total; per-request stdout
            # lines would flood journald at fleet scrape rates.
            return

        def _send(self, code, body, content_type, extra_headers=()):
            data = body.encode("utf-8") if isinstance(body, str) else body
            try:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                for key, value in extra_headers:
                    self.send_header(key, value)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(data)
            except (OSError, ValueError) as exc:  # client hung up mid-response
                log.debug("health response aborted: %s", exc)

        def _json(self, code, obj, route):
            metrics.HEALTH_REQUESTS_TOTAL.inc(route=route, code=str(code))
            self._send(code, json.dumps(obj), "application/json")

        def _authorised(self, route):
            if not token or route == "/livez":
                return True
            header = self.headers.get("Authorization", "")
            expected = "Bearer " + token
            if len(header) == len(expected) and _consteq(header, expected):
                return True
            metrics.HEALTH_REQUESTS_TOTAL.inc(route=route, code="401")
            self._send(
                401,
                json.dumps({"error": "unauthorised"}),
                "application/json",
                (("WWW-Authenticate", 'Bearer realm="adas"'),),
            )
            return False

        def _route(self):
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path in ("/health", "/"):
                path = "/healthz"
            if not self._authorised(path):
                return
            if path == "/livez":
                self._json(200, {"live": True, "pid": os.getpid()}, path)
                return
            if path == "/readyz":
                ready = state.is_ready()
                self._json(200 if ready else 503, state.as_dict(), path)
                return
            if path == "/healthz":
                body = state.as_dict()
                # 503 when not ok or stale. A degraded-but-driving unit stays 200:
                # LIMITED is a mode, not an outage.
                self._json(200 if body["ok"] else 503, body, path)
                return
            if path == "/metrics":
                text = state.metrics_text()
                metrics.HEALTH_REQUESTS_TOTAL.inc(route=path, code="200")
                self._send(200, text, "text/plain; version=0.0.4; charset=utf-8")
                return
            self._json(404, {"error": "not found", "routes": sorted(ROUTES)}, "unknown")

        def do_GET(self):
            self._route()

        def do_HEAD(self):
            self._route()

        def _reject_method(self):
            metrics.HEALTH_REQUESTS_TOTAL.inc(route="_", code="405")
            self._send(
                405,
                json.dumps({"error": "method not allowed"}),
                "application/json",
                (("Allow", "GET, HEAD"),),
            )

        do_POST = _reject_method
        do_PUT = _reject_method
        do_DELETE = _reject_method
        do_PATCH = _reject_method
        do_OPTIONS = _reject_method

    return Handler


class _BoundedServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a hard cap on simultaneous connections."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, max_connections: int = 16) -> None:
        self._max_connections = int(max_connections)
        self._active = 0
        self._count_lock = threading.Lock()
        ThreadingHTTPServer.__init__(self, address, handler)

    def process_request(self, request, client_address):
        with self._count_lock:
            over = self._active >= self._max_connections
            if not over:
                self._active += 1
        if over:
            log.warning("health connection refused: %d already active", self._max_connections)
            metrics.HEALTH_REQUESTS_TOTAL.inc(route="_", code="503")
            with contextlib.suppress(OSError):
                # The client may already be gone; the point is to close, not to reply.
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\n"
                    b"Connection: close\r\n\r\n"
                )
            self.close_request(request)
            return
        ThreadingHTTPServer.process_request(self, request, client_address)

    def shutdown_request(self, request):
        try:
            ThreadingHTTPServer.shutdown_request(self, request)
        finally:
            with self._count_lock:
                self._active = max(0, self._active - 1)


class HealthServer:
    """Owns the listening socket and the serving thread.

    Args:
        state: the :class:`HealthState` to serve.
        bind: interface. Anything other than loopback needs ``allow_remote=True``.
        port: TCP port; 0 asks the kernel for a free one (tests).
        allow_remote: explicit opt-in to a non-loopback bind. Logged at WARNING every
            time it is used, because it publishes vehicle state to the network.
        token: when set, ``Authorization: Bearer <token>`` is required on every route
            except ``/livez``. Read it from a root-owned file, never from the config.
        max_connections: concurrent connection cap.
        required: raise instead of returning ``None`` when the listener cannot start.
    """

    def __init__(self, state: HealthState, bind: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 allow_remote: bool = False, token: str = "", max_connections: int = 16,
                 required: bool = False) -> None:
        self.state = state
        self.bind = bind
        self.port = int(port)
        self.allow_remote = bool(allow_remote)
        self.token = token or ""
        self.max_connections = int(max_connections)
        self.required = bool(required)
        self.error = ""
        self._httpd: Optional[_BoundedServer] = None
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def from_config(cls, state: HealthState, cfg: Any, **kwargs: Any) -> "HealthServer":
        """Build from ``cfg.health``, reading optional keys with getattr defaults.

        Works against a :class:`~adas.core.config.RuntimeConfig` that has no ``health``
        section yet (everything falls back to the loopback defaults) and against one
        that has grown ``enabled``/``bind``/``port``/``allow_remote``/``token_file``.
        """
        hc = getattr(cfg, "health", None)
        token_file = str(getattr(hc, "token_file", "") or "")
        secret = ""
        if token_file:
            try:
                with open(token_file) as f:
                    secret = f.read().strip()
            except OSError as exc:
                log.error("health token_file %s unreadable: %s", token_file, exc)
        return cls(
            state,
            bind=str(getattr(hc, "bind", "127.0.0.1")),
            port=int(getattr(hc, "port", DEFAULT_PORT)),
            allow_remote=bool(getattr(hc, "allow_remote", False)),
            token=secret,
            max_connections=int(getattr(hc, "max_connections", 16)),
            required=bool(getattr(hc, "required", False)),
            **kwargs
        )

    def _refuse(self, reason: str) -> Optional[int]:
        """Record and log a refusal. Returns None so callers can ``return self._refuse(...)``."""
        self.error = reason
        log.error("HEALTH_BIND_REFUSED %s", reason)
        if self.required:
            raise RuntimeError("health endpoint refused: %s" % reason)
        return None

    def start(self) -> Optional[int]:
        """Bind and serve on a daemon thread. Returns the bound port, or None."""
        if not is_loopback(self.bind) and not self.allow_remote:
            return self._refuse(
                "bind=%s is not loopback and health.allow_remote is false; /healthz "
                "publishes live vehicle safety state" % self.bind
            )
        if not is_loopback(self.bind):
            if self.token:
                log.warning("HEALTH_REMOTE_BIND bind=%s port=%s — bearer token required",
                            self.bind, self.port)
            else:
                log.warning(
                    "HEALTH_REMOTE_BIND bind=%s port=%s with NO token: live safety state, "
                    "ego speed and lead range are readable by anything on this network",
                    self.bind, self.port,
                )
        try:
            self._httpd = _BoundedServer(
                (self.bind, self.port), _handler(self.state, self.token), self.max_connections
            )
        except OSError as exc:
            self._httpd = None
            return self._refuse("bind %s:%s failed: %s" % (self.bind, self.port, exc))
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="adas-health", daemon=True
        )
        self._thread.start()
        actual = self._httpd.server_address[1]
        self.port = actual
        self.error = ""
        log.info("health listening http://%s:%s/healthz (routes: %s)", self.bind, actual,
                 " ".join(ROUTES))
        return actual

    def stop(self) -> None:
        """Stop serving and close the socket. Idempotent."""
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        metrics.UP.set(0.0)

    def __enter__(self) -> "HealthServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.stop()
        return False


__all__ = [
    "DEFAULT_PORT",
    "DEFAULT_PROCESS_REFRESH_S",
    "DEFAULT_STALE_AFTER_S",
    "HealthServer",
    "HealthState",
    "ROUTES",
    "is_loopback",
    "verified_engine_digests",
]
