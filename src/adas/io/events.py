"""Durable append-only JSONL of safety-relevant ADAS events.

Lab default ``data/events.jsonl``; production ``/var/lib/adas/events.jsonl``.

What belongs in here
--------------------
Only things an investigator would want after an incident, and nothing that happens
every frame:

* every :class:`~adas.core.models.SafetyState` transition, in particular into
  ``LIMITED`` / ``MIN_RISK_MANEUVER`` / ``DISENGAGE`` and back out again;
* perception dropouts — the edge where consecutive perception failures crossed the
  configured threshold, and the edge where perception recovered;
* engine load/inference failures, by engine name;
* capture-source loss and reconnection;
* startup and shutdown, so a restart loop is visible in the record itself.

Per-frame telemetry does **not** belong here. It goes to ``/metrics``.

Why this file is written the way it is
--------------------------------------
This is a vehicle. Ignition cut, a 12 V brown-out and an OOM kill all drop whatever is
still in the page cache, and the records that get dropped are the last ones before the
incident — exactly the ones that matter. ``flush()`` alone only moves bytes from
Python's buffer into the kernel, so this module adds an explicit :func:`os.fsync`
barrier whose cost is bounded by policy (see *Durability*).

The second failure this module exists to survive is a full disk. An ENOSPC that
propagates out of the frame loop kills the process, systemd restarts it, it fills the
disk again, and after five restarts the ADAS unit sits in ``failed`` — driver
assistance gone because a log file grew. Here ENOSPC is a *state*: writing stops,
``adas_disk_full`` goes to 1, ``/healthz`` reports ``events.writable: false``, and the
writer retries on a backoff. Nothing raises.

Record schema (``v`` = 2)
-------------------------
=============  ==========================================================
``v``          schema version integer; bump on any incompatible change
``ts``         wall clock, Unix epoch seconds (float). May jump at NTP sync;
               this is the field to correlate against journald
``t``          ``time.monotonic()`` seconds, stamped inside the writer at the
               moment of the append. One clock, one meaning: it is a
               *machine uptime* reading, comparable only between records that
               carry the same ``boot``, and never supplied by a caller
``boot``       boot id (``/proc/sys/kernel/random/boot_id``) or a per-process
               uuid4 off-board; says which records' ``t`` may be compared
``run``        12 hex digits identifying one :class:`EventLog` instance, so
               records from two runs (or two writers) in one file are
               distinguishable
``seq``        record counter, continued from the last record already in the
               file at startup, so it is the authoritative order **within a
               file** and survives a restart that appends to it
``kind``       one of :data:`EVENT_KINDS`
``type``       event type string, e.g. ``"safety_state"``
``severity``   ``info`` | ``warn`` | ``critical``
``frame_id``   pipeline frame id the event was raised on, or null
``detail``     free-form JSON object; keys depend on ``type``
=============  ==========================================================

Ordering
--------
Sort by ``seq``. It is stamped by the writer, it is dense within a run, and it
continues across a restart that appends to the same file. ``t`` is monotonic
within one ``boot`` and says nothing across a reboot (the kernel's monotonic
clock restarts at zero); ``ts`` is wall clock and can step backwards at an NTP
correction. Neither is a total order on its own, and no field orders records
that were written into *different* files -- for that, compare ``ts`` and accept
its accuracy.

Units: ``ts``/``t`` seconds, ``max_bytes`` bytes, ``min_free_mb`` MiB.

Durability policy (``fsync`` argument)
--------------------------------------
``"critical"`` (default) fsyncs every CRITICAL record immediately and everything else
at most once per ``fsync_interval_s``; ``"always"`` fsyncs every record; ``"interval"``
only rate-limits; ``"never"`` flushes to the kernel and stops there (lab use). Safety
transitions are rare and an fsync of one short line measures well under a millisecond
on this board's NVMe, so ``"critical"`` costs nothing measurable and bounds worst-case
loss to one second of non-critical records.

Rotation
--------
Two independent mechanisms, because logrotate only runs from a daily timer and a burst
of state transitions can outrun it:

* *in-process* — at ``max_bytes`` the live file is renamed to ``.1`` (shifting older
  generations up to ``backups``) and reopened. This is the real bound.
* *external* — logrotate ``create`` renames the file out from under the open fd. The
  writer re-stats the path every ``stat_interval_s`` and reopens when the inode
  changes, so no ``copytruncate`` (and no lost lines) is needed. :meth:`EventLog.reopen`
  is also the SIGHUP entry point for a ``postrotate`` trigger.

``backups`` here and ``rotate`` in ``deploy/adas.logrotate`` must stay consistent.

Failure behaviour
-----------------
No method raises during operation. :meth:`EventLog.write` returns ``True`` on an
append that reached the kernel and ``False`` on any refusal (closed, disk full,
unwritable), counting the reason in :mod:`adas.io.metrics`. Construction degrades to a
fallback path and finally to a disabled writer, unless ``strict=True``, which makes an
unwritable primary path fatal at startup — before READY, which is where a production
unit should fail.
"""
from __future__ import annotations

import atexit
import contextlib
import errno
import json
import logging
import os
import shutil
import time
import uuid
from typing import Any, Callable, Dict, Iterator, Optional, TextIO, Tuple

from adas.io import metrics

log = logging.getLogger("adas.events")

#: Schema version written as ``v`` on every record. v2 added ``run`` and pinned
#: ``t`` to the writer's own ``time.monotonic()``; in v1 ``t`` came from the
#: injectable clock, so two writers on one file could log two clock domains in it.
SCHEMA_VERSION = 2

#: Stable ``kind`` vocabulary. A consumer may switch on these without a lookup table.
EVENT_KINDS = (
    "lifecycle",       # startup, shutdown, config reload
    "safety_state",    # SafetyState transition
    "safety_event",    # violation or warning raised by the safety monitor
    "perception",      # perception dropout / recovery
    "engine",          # model load or inference failure
    "source",          # capture source loss / reconnect
    "storage",         # the event log's own disk-full edges
)

#: Severity vocabulary, ordered.
SEVERITIES = ("info", "warn", "critical")

#: Errnos that mean "the filesystem is out of room", as opposed to a broken path.
_FULL_ERRNOS = frozenset((errno.ENOSPC, errno.EDQUOT))

#: Accepted values for the ``fsync`` policy argument.
FSYNC_POLICIES = ("never", "interval", "critical", "always")

_DEFAULT_MAX_BYTES = 32 * 1024 * 1024
_DEFAULT_MIN_FREE_MB = 64.0
_FILE_MODE = 0o640
_DIR_MODE = 0o750

#: SafetyStates whose entry is, by itself, a critical record.
_CRITICAL_STATES = frozenset(("min_risk_maneuver", "disengage"))


def default_events_path(dev: bool = True, configured: str = "") -> str:
    """Configured path wins; otherwise lab writes under ``data/``, production /var/lib."""
    if configured:
        return configured
    if dev:
        return os.path.join("data", "events.jsonl")
    return "/var/lib/adas/events.jsonl"


def default_fallback_path(dev: bool = True) -> str:
    """Where to go when the configured path cannot be opened.

    Lab falls back to ``data/events.jsonl`` next to the checkout. Production falls back
    to ``/run/adas/events.jsonl``: ``RuntimeDirectory=adas`` already exists, it is
    tmpfs so it can never fill the root filesystem, and ``deploy/adas.logrotate``
    covers it. It is *not* durable across a reboot, which is why the fallback is
    reported on ``/healthz`` as ``events.fallback: true`` rather than being silent.
    """
    if dev:
        return os.path.join("data", "events.jsonl")
    return "/run/adas/events.jsonl"


def _boot_id() -> str:
    """Kernel boot id, or a per-process uuid4 when /proc is unavailable (off-board)."""
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            return f.read().strip()
    except OSError:
        return str(uuid.uuid4())


def read_events(path: str, skip_bad: bool = True) -> Iterator[Dict[str, Any]]:
    """Iterate records from a JSONL, tolerating a torn or corrupt line.

    A power cut can leave a partial line behind, and a filesystem fault can leave
    binary garbage. With ``skip_bad`` (the default) such a line is logged at WARNING
    and skipped, so one bad record cannot make the whole incident history unreadable;
    with ``skip_bad=False`` a :class:`ValueError` propagates for a tool that wants to
    know. Undecodable bytes are replaced rather than raising, for the same reason.
    """
    with open(path, "r", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                if not skip_bad:
                    raise
                log.warning("events %s:%d is not valid JSON; skipped", path, lineno)
                continue
            if isinstance(obj, dict):
                yield obj
            elif skip_bad:
                log.warning("events %s:%d is not a JSON object; skipped", path, lineno)
            else:
                raise ValueError("%s:%d is not a JSON object" % (path, lineno))


class EventLog:
    """Append-only JSONL writer with fsync, rotation, quota and ENOSPC survival.

    Args:
        path: configured destination, e.g. ``/var/lib/adas/events.jsonl``.
        fsync: one of :data:`FSYNC_POLICIES`.
        fsync_interval_s: upper bound between fsync barriers for non-critical records.
        max_bytes: in-process rotation threshold in bytes. 0 disables in-process
            rotation and leaves the bound entirely to logrotate.
        backups: how many ``.1 .. .N`` generations to keep. Must match ``rotate`` in
            ``deploy/adas.logrotate`` so the two do not fight.
        min_free_mb: refuse to write below this much free space, so this process is
            never the one that fills the disk out from under the rest of the system.
        strict: raise instead of falling back / degrading when the path is unusable.
        allow_fallback: try :func:`default_fallback_path` when the primary fails.
        dev: lab flag; only used to choose the fallback path.
        on_disk_full: called with ``(edge, detail_dict)`` on each disk-full edge, so the
            health snapshot can show it. Exceptions from the callback are logged and
            swallowed — a health hook must never break the writer.
        retry_s: how long to stay in the disk-full state before retrying a write.
        stat_interval_s: how often to re-stat the path to notice an external rotation.
            0 checks on every write; a negative value disables the check.
        clock: injectable ``time.monotonic`` for the writer's own timers (fsync
            interval, disk-full backoff, re-stat cadence) in tests. It does **not**
            reach the ``t`` field: that is stamped from :func:`time.monotonic`
            inside :meth:`_emit`, so one file can never hold two clock domains.
    """

    def __init__(
        self,
        path: str,
        fsync: str = "critical",
        fsync_interval_s: float = 1.0,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        backups: int = 3,
        min_free_mb: float = _DEFAULT_MIN_FREE_MB,
        strict: bool = False,
        allow_fallback: bool = True,
        dev: bool = True,
        on_disk_full: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        retry_s: float = 5.0,
        stat_interval_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if fsync not in FSYNC_POLICIES:
            raise ValueError("fsync must be one of %s, got %r" % (list(FSYNC_POLICIES), fsync))
        self.configured_path = path
        self.path = path
        self.fsync_policy = fsync
        self.fsync_interval_s = float(fsync_interval_s)
        self.max_bytes = int(max_bytes)
        self.backups = max(0, int(backups))
        self.min_free_mb = float(min_free_mb)
        self.strict = bool(strict)
        self.retry_s = float(retry_s)
        self.stat_interval_s = float(stat_interval_s)
        self._clock = clock
        self._on_disk_full = on_disk_full

        self.boot_id = _boot_id()
        #: Identifies this writer instance in the ``run`` field of every record.
        self.run_id = uuid.uuid4().hex[:12]
        #: Sequence number of the last record written. Continued from the file at
        #: startup (see :meth:`_resume_seq`), so it is not a count of what this
        #: process wrote -- that is :attr:`written`.
        self.seq = 0
        #: Records this instance appended successfully.
        self.written = 0
        self.fallback = False
        self.writable = False
        self.disk_full = False
        self.open_error = ""
        self.dropped = 0
        self.rotations = 0
        self.write_failures = 0

        self._fh: Optional[TextIO] = None
        self._ino: Optional[Tuple[int, int]] = None
        self._closed = False
        self._last_fsync = clock()
        self._last_stat = 0.0
        self._retry_at = 0.0
        self._full_since = 0.0
        self._full_dropped = 0
        self._warned_closed = False

        self._open_initial(path, allow_fallback=allow_fallback, dev=dev)
        if self.writable:
            self._resume_seq()
        atexit.register(self.close)
        metrics.EVENTS_WRITABLE.set_bool(self.writable)

    # ----------------------------------------------------------------- opening

    def _open_initial(self, path: str, allow_fallback: bool, dev: bool) -> None:
        try:
            self._open(path)
            return
        except OSError as exc:
            first = exc
        fallback = default_fallback_path(dev)
        same = os.path.abspath(fallback) == os.path.abspath(path)
        if self.strict or not allow_fallback or same:
            self.open_error = "%s: %s" % (path, first)
            if self.strict:
                # Fail closed, before READY: a production unit that cannot record
                # safety transitions must not pretend it is recording them.
                raise first
            log.error("EVENTS_UNWRITABLE %s (%s); safety events will not be recorded",
                      path, first)
            return
        log.warning(
            "EVENTS_FALLBACK configured=%s unwritable (%s); using %s — logrotate covers "
            "both paths but the fallback does not survive a reboot",
            path, first, fallback,
        )
        try:
            self._open(fallback)
            self.fallback = True
        except OSError as exc:
            self.open_error = "%s: %s (fallback %s: %s)" % (path, first, fallback, exc)
            log.error("EVENTS_UNWRITABLE %s", self.open_error)

    def _resume_seq(self) -> None:
        """Continue ``seq`` from the last record already in the open file.

        A restart appends to the same file, and a counter that restarted at 1 would
        give that file duplicate ``seq`` values and no field that orders it -- while
        the module advertises the restart loop as being visible in the record itself.
        Costs one 64 KiB read of the tail, once, at startup. A file whose tail is
        unreadable or holds no parsable ``seq`` leaves the counter at 0.
        """
        try:
            size = os.path.getsize(self.path)
            if size <= 0:
                return
            with open(self.path, "rb") as probe:
                probe.seek(max(0, size - 65536))
                tail = probe.read().decode("utf-8", "replace")
        except OSError as exc:
            log.debug("could not read the tail of %s to resume seq: %s", self.path, exc)
            return
        for line in reversed(tail.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            last = obj.get("seq") if isinstance(obj, dict) else None
            if isinstance(last, int) and not isinstance(last, bool) and last > 0:
                self.seq = last
                log.info("events %s already holds %d records; continuing seq at %d",
                         self.path, last, last + 1)
                return

    def _open(self, path: str) -> None:
        """Open one path for append, 0640, creating its directory 0750."""
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, mode=_DIR_MODE, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, _FILE_MODE)
        try:
            # os.open's mode is masked by umask (022 under systemd unless UMask= is
            # set), so a 0640 request can land as 0644. Fix it explicitly.
            try:
                os.fchmod(fd, _FILE_MODE)
            except OSError as exc:  # foreign owner, or a filesystem without modes
                log.debug("could not chmod %s to 0%o: %s", path, _FILE_MODE, exc)
            fh = os.fdopen(fd, "a")
        except Exception:
            os.close(fd)
            raise
        self._fh = fh
        self.path = path
        self.writable = True
        self.open_error = ""
        st = os.fstat(fh.fileno())
        self._ino = (st.st_dev, st.st_ino)
        self._last_stat = self._clock()
        self._ensure_newline(st.st_size)
        log.info("events jsonl %s (fsync=%s max_bytes=%s)", self.path, self.fsync_policy,
                 self.max_bytes)

    def _ensure_newline(self, size: int) -> None:
        """Terminate a torn final line so the next record cannot be glued onto it."""
        if size <= 0 or self._fh is None:
            return
        try:
            with open(self.path, "rb") as probe:
                probe.seek(-1, os.SEEK_END)
                last = probe.read(1)
        except OSError as exc:
            log.debug("could not probe tail of %s: %s", self.path, exc)
            return
        if last != b"\n":
            log.warning("events %s ends mid-record; terminating the torn line", self.path)
            try:
                self._fh.write("\n")
                self._fh.flush()
            except OSError as exc:
                log.warning("could not terminate torn line in %s: %s", self.path, exc)

    def reopen(self) -> bool:
        """Close and reopen the current path. SIGHUP / logrotate ``postrotate`` hook."""
        if self._closed:
            return False
        self._close_fh()
        try:
            self._open(self.path)
        except OSError as exc:
            self.writable = False
            self.open_error = "%s: %s" % (self.path, exc)
            log.error("EVENTS_REOPEN_FAILED %s", self.open_error)
            metrics.EVENTS_WRITABLE.set_bool(False)
            return False
        metrics.EVENTS_WRITABLE.set_bool(True)
        return True

    # ---------------------------------------------------------------- rotation

    def _check_external_rotation(self, now: float) -> None:
        """Reopen if something renamed or replaced the file under our fd."""
        if self.stat_interval_s < 0:
            return
        if self.stat_interval_s > 0 and now - self._last_stat < self.stat_interval_s:
            return
        self._last_stat = now
        try:
            st = os.stat(self.path)
            ident: Optional[Tuple[int, int]] = (st.st_dev, st.st_ino)
        except OSError:
            ident = None
        if ident != self._ino:
            log.info("events %s was rotated externally; reopening", self.path)
            metrics.EVENT_ROTATIONS_TOTAL.inc(reason="external")
            self.rotations += 1
            self.reopen()

    def _rotate(self) -> None:
        """Shift ``.N-1 -> .N`` and the live file to ``.1``, then reopen. Never raises."""
        self._close_fh()
        try:
            for i in range(self.backups, 0, -1):
                src = self.path if i == 1 else "%s.%d" % (self.path, i - 1)
                dst = "%s.%d" % (self.path, i)
                if os.path.exists(src):
                    os.replace(src, dst)
            if self.backups == 0 and os.path.exists(self.path):
                os.unlink(self.path)
        except OSError as exc:
            log.error("events rotation failed for %s: %s", self.path, exc)
        metrics.EVENT_ROTATIONS_TOTAL.inc(reason="size")
        self.rotations += 1
        try:
            self._open(self.path)
        except OSError as exc:
            self.writable = False
            self.open_error = "%s: %s" % (self.path, exc)
            log.error("EVENTS_UNWRITABLE after rotation: %s", self.open_error)
            metrics.EVENTS_WRITABLE.set_bool(False)

    def _maybe_rotate(self, extra_bytes: int) -> None:
        if self.max_bytes <= 0 or self._fh is None:
            return
        try:
            size = self._fh.tell()
        except (OSError, ValueError):
            return
        if size + extra_bytes > self.max_bytes:
            self._rotate()

    # --------------------------------------------------------------- disk full

    def _free_mb(self) -> Optional[float]:
        d = os.path.dirname(os.path.abspath(self.path)) or "."
        try:
            usage = shutil.disk_usage(d)
        except OSError:
            return None
        return usage.free / (1024.0 * 1024.0)

    def _enter_disk_full(self, detail: str, now: float) -> None:
        if self.disk_full:
            return
        self.disk_full = True
        self.writable = False
        self._full_since = now
        self._full_dropped = 0
        self._retry_at = now + self.retry_s
        metrics.DISK_FULL.set(1.0)
        metrics.EVENTS_WRITABLE.set_bool(False)
        log.error("DISK_FULL events %s: %s — event recording suspended", self.path, detail)
        self._notify_disk_full("enter", {"path": self.path, "detail": detail})

    def _exit_disk_full(self, now: float) -> None:
        if not self.disk_full:
            return
        seconds = max(0.0, now - self._full_since)
        dropped = self._full_dropped
        self.disk_full = False
        self.writable = self._fh is not None
        metrics.DISK_FULL.set(0.0)
        metrics.EVENTS_WRITABLE.set_bool(self.writable)
        log.warning("DISK_FULL cleared for %s after %.1f s; %d records were lost",
                    self.path, seconds, dropped)
        self._notify_disk_full("exit", {"seconds": round(seconds, 3), "dropped": dropped})
        # Record the gap itself, so the JSONL says why it has a hole in it.
        self._emit(
            {
                "kind": "storage",
                "type": "disk_full",
                "severity": "warn",
                "frame_id": None,
                "detail": {"edge": "exit", "seconds": round(seconds, 3), "dropped": dropped},
            },
            critical=True,
            now=now,
        )

    def _notify_disk_full(self, edge: str, detail: Dict[str, Any]) -> None:
        if self._on_disk_full is None:
            return
        try:
            self._on_disk_full(edge, dict(detail))
        except Exception:  # a health callback must never break the writer
            log.exception("on_disk_full callback raised (edge=%s)", edge)

    # ------------------------------------------------------------------- write

    def write(self, kind: str, type: str, severity: str = "info",
              frame_id: Optional[int] = None, **detail: Any) -> bool:
        """Append one event. Returns True when the bytes reached the kernel.

        Args:
            kind: one of :data:`EVENT_KINDS`. An unknown kind is written anyway (a
                dropped safety record is worse than an off-vocabulary one) but logged.
            type: event type within the kind, e.g. ``"transition"``.
            severity: one of :data:`SEVERITIES`.
            frame_id: pipeline frame id, or None outside the frame loop.
            **detail: JSON-serialisable extras. Values that are not JSON-serialisable
                are stringified rather than losing the record.

        Never raises.
        """
        if kind not in EVENT_KINDS:
            log.warning("event kind %r is not in EVENT_KINDS; writing it anyway", kind)
        if severity not in SEVERITIES:
            log.warning("event severity %r is not in SEVERITIES; recorded as 'info'", severity)
            severity = "info"
        metrics.EVENTS_BY_TYPE_TOTAL.inc(type=str(type), severity=severity)
        record = {
            "kind": str(kind),
            "type": str(type),
            "severity": severity,
            "frame_id": None if frame_id is None else int(frame_id),
            "detail": dict(detail),
        }
        return self._emit(record, critical=(severity == "critical"), now=self._clock())

    # -- typed helpers ---------------------------------------------------------

    def safety_state(self, previous: Any, current: Any, frame_id: Optional[int] = None,
                     reason: str = "", **detail: Any) -> bool:
        """Record a SafetyState transition.

        Entering ``min_risk_maneuver`` or ``disengage`` is CRITICAL and is fsynced
        immediately; every other transition is a warning, except a return to
        ``nominal``, which is info.
        """
        prev = str(getattr(previous, "value", previous))
        cur = str(getattr(current, "value", current))
        if cur in _CRITICAL_STATES:
            severity = "critical"
        elif cur == "nominal":
            severity = "info"
        else:
            severity = "warn"
        return self.write("safety_state", "transition", severity, frame_id,
                          previous=prev, current=cur, reason=reason, **detail)

    def perception_dropout(self, consecutive_failures: int, reason: str,
                           frame_id: Optional[int] = None, **detail: Any) -> bool:
        """Record the edge where perception stopped producing usable frames."""
        return self.write("perception", "dropout", "critical", frame_id,
                          consecutive_failures=int(consecutive_failures),
                          reason=reason, **detail)

    def perception_recovered(self, outage_frames: int, frame_id: Optional[int] = None,
                             **detail: Any) -> bool:
        """Record the edge where perception started producing usable frames again."""
        return self.write("perception", "recovered", "warn", frame_id,
                          outage_frames=int(outage_frames), **detail)

    def engine_failure(self, engine: str, kind: str, error: str,
                       frame_id: Optional[int] = None, **detail: Any) -> bool:
        """Record a model load or inference failure and count it in the registry.

        Args:
            engine: engine file stem, e.g. ``yolox_nano``.
            kind: ``load`` | ``shape`` | ``execution`` | ``hang``.
        """
        metrics.ENGINE_ERRORS_TOTAL.inc(engine=str(engine), kind=str(kind))
        return self.write("engine", "failure", "critical", frame_id,
                          engine=str(engine), failure=str(kind), error=str(error), **detail)

    def source_event(self, type: str, uri: str = "", frame_id: Optional[int] = None,
                     **detail: Any) -> bool:
        """Record a capture-source loss (``type="lost"``) or reopen (``"reconnected"``)."""
        severity = "critical" if type == "lost" else "warn"
        return self.write("source", type, severity, frame_id, uri=uri, **detail)

    def lifecycle(self, type: str, **detail: Any) -> bool:
        """Record ``started`` / ``stopping`` / ``reloaded``. Always fsynced."""
        return self.write("lifecycle", type, "critical", None, **detail)

    # -- emit ------------------------------------------------------------------

    def _emit(self, record: Dict[str, Any], critical: bool, now: float) -> bool:
        if self._closed or self._fh is None:
            if not self._warned_closed:
                log.error("events write after close/open-failure (%s); records are being "
                          "dropped", self.open_error or self.path)
                self._warned_closed = True
            self.dropped += 1
            metrics.EVENT_DROPPED_TOTAL.inc(reason="closed")
            return False
        if self.disk_full:
            if now < self._retry_at:
                self.dropped += 1
                self._full_dropped += 1
                metrics.EVENT_DROPPED_TOTAL.inc(reason="disk_full")
                return False
            self._retry_at = now + self.retry_s
            free = self._free_mb()
            if free is not None and free < self.min_free_mb:
                self.dropped += 1
                self._full_dropped += 1
                metrics.EVENT_DROPPED_TOTAL.inc(reason="disk_full")
                return False
            self._exit_disk_full(now)

        self._check_external_rotation(now)

        full: Dict[str, Any] = {
            "v": SCHEMA_VERSION,
            "ts": time.time(),
            # Always the writer's own monotonic clock, never `now` (which may be an
            # injected test clock or any other caller's time base): one field, one
            # clock domain, or the file cannot be ordered at all.
            "t": round(time.monotonic(), 6),
            "boot": self.boot_id,
            "run": self.run_id,
            "seq": self.seq + 1,
        }
        full.update(record)
        line = json.dumps(full, sort_keys=False, default=str) + "\n"
        self._maybe_rotate(len(line))
        if self._fh is None:
            self.dropped += 1
            metrics.EVENT_DROPPED_TOTAL.inc(reason="closed")
            return False

        free = self._free_mb()
        if free is not None:
            metrics.DISK_FREE_MB.set(
                free, path=os.path.dirname(os.path.abspath(self.path)) or "/"
            )
            if free < self.min_free_mb:
                self._enter_disk_full(
                    "free=%.1f MiB < min_free_mb=%.1f" % (free, self.min_free_mb), now
                )
                self.dropped += 1
                self._full_dropped += 1
                metrics.EVENT_DROPPED_TOTAL.inc(reason="disk_full")
                return False

        try:
            self._fh.write(line)
            self._fh.flush()
        except OSError as exc:
            return self._on_write_error(exc, now)
        self.seq += 1
        self.written += 1
        metrics.EVENT_WRITES_TOTAL.inc()
        with contextlib.suppress(OSError, ValueError):
            # tell() on a file whose fd was pulled out from under us; the size gauge
            # is cosmetic and must not fail the write that already succeeded.
            metrics.EVENTS_BYTES.set(self._fh.tell())
        if self._should_fsync(critical, now):
            try:
                os.fsync(self._fh.fileno())
                self._last_fsync = now
                metrics.EVENT_FSYNCS_TOTAL.inc()
            except OSError as exc:
                return self._on_write_error(exc, now)
        return True

    def _should_fsync(self, critical: bool, now: float) -> bool:
        if self.fsync_policy == "never":
            return False
        if self.fsync_policy == "always":
            return True
        if self.fsync_policy == "critical" and critical:
            return True
        return (now - self._last_fsync) >= self.fsync_interval_s

    def _on_write_error(self, exc: OSError, now: float) -> bool:
        name = errno.errorcode.get(exc.errno or 0, str(exc.errno))
        self.write_failures += 1
        metrics.EVENT_WRITE_FAILURES_TOTAL.inc(errno=name)
        if exc.errno in _FULL_ERRNOS:
            self._enter_disk_full("%s writing %s" % (name, self.path), now)
        else:
            log.error("events write failed on %s: %s", self.path, exc)
            self.writable = False
            metrics.EVENTS_WRITABLE.set_bool(False)
        self.dropped += 1
        if self.disk_full:
            # The record that hit ENOSPC is lost too; count it in the gap summary.
            self._full_dropped += 1
        return False

    # ------------------------------------------------------------------- state

    def stats(self) -> Dict[str, Any]:
        """Snapshot for ``/healthz``. Cheap; safe to call on every health refresh."""
        size = None
        try:
            if self._fh is not None:
                size = self._fh.tell()
        except (OSError, ValueError):
            size = None
        return {
            "path": self.path,
            "configured_path": self.configured_path,
            "writable": bool(self.writable and not self.disk_full),
            "fallback": self.fallback,
            "disk_full": self.disk_full,
            "bytes": size,
            "max_bytes": self.max_bytes,
            "run": self.run_id,
            "seq": self.seq,
            "written": self.written,
            "dropped": self.dropped,
            "rotations": self.rotations,
            "write_failures": self.write_failures,
            "fsync": self.fsync_policy,
            "schema_version": SCHEMA_VERSION,
            "error": self.open_error,
        }

    # --------------------------------------------------------------- lifecycle

    def _close_fh(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            fh.flush()
            os.fsync(fh.fileno())
        except (OSError, ValueError) as exc:
            log.debug("fsync during close failed: %s", exc)
        try:
            fh.close()
        except OSError as exc:
            log.debug("close failed: %s", exc)

    def close(self) -> None:
        """Flush, fsync and close. Idempotent, and registered with :mod:`atexit`."""
        if self._closed:
            return
        self._closed = True
        self._close_fh()
        self.writable = False
        metrics.EVENTS_WRITABLE.set_bool(False)

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.close()
        return False


def from_config(cfg: Any, override_path: str = "",
                on_disk_full: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> EventLog:
    """Build an :class:`EventLog` from a :class:`adas.core.config.RuntimeConfig`.

    Reads the optional keys ``events.path``, ``events.fsync``, ``events.fsync_interval_s``,
    ``events.max_mb``, ``events.backups``, ``events.min_free_mb`` and ``events.strict``
    with :func:`getattr` defaults, so this works against today's schema (which has no
    ``events`` section yet) and against one that has grown it. ``source.type ==
    "synthetic"`` is treated as lab and selects the ``data/`` paths.
    """
    ev = getattr(cfg, "events", None)
    source_type = str(getattr(getattr(cfg, "source", None), "type", "synthetic"))
    dev = source_type in ("synthetic", "video")
    path = override_path or default_events_path(dev, str(getattr(ev, "path", "") or ""))
    max_mb = float(getattr(ev, "max_mb", 32.0) or 0.0)
    return EventLog(
        path,
        fsync=str(getattr(ev, "fsync", "critical")),
        fsync_interval_s=float(getattr(ev, "fsync_interval_s", 1.0)),
        max_bytes=int(max_mb * 1024 * 1024),
        backups=int(getattr(ev, "backups", 3)),
        min_free_mb=float(getattr(ev, "min_free_mb", _DEFAULT_MIN_FREE_MB)),
        strict=bool(getattr(ev, "strict", False)),
        dev=dev,
        on_disk_full=on_disk_full,
    )


__all__ = [
    "EVENT_KINDS",
    "EventLog",
    "FSYNC_POLICIES",
    "SCHEMA_VERSION",
    "SEVERITIES",
    "default_events_path",
    "default_fallback_path",
    "from_config",
    "read_events",
]
