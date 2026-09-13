"""Structured logging for the ADAS system.

One handler, on the root logger, installed once
-----------------------------------------------
The previous version of this module gave *every* module logger an explicit level of
INFO at import time. A child logger with an explicit level ignores its ancestors, so
``--log-level DEBUG`` (which only sets the root level) enabled nothing and
``--log-level ERROR`` suppressed nothing: the documented control was inert in both
directions, and an operator could not quiet a service that logs per frame.

The rule here is the standard one: **module loggers carry no level and no handler**,
the root logger carries exactly one handler and the level, and the level is decided
once by :func:`configure_logging`. :func:`setup_logger` is kept because every module
in the tree calls it, but it now only returns ``logging.getLogger(name)`` after making
sure a root handler exists.

Formats
-------
``text`` (default) is the human format for a terminal. ``json`` emits one JSON object
per line — ``ts``, ``level``, ``logger``, ``msg``, plus any structured fields passed
through ``extra=`` and the ``event`` vocabulary below — which is what a vehicle unit
should use, because journald keeps the line as ``MESSAGE`` and ``journalctl -u adas -o
cat | jq`` then works directly.

Event vocabulary
----------------
Free-text log lines cannot be alerted on. Anything an operator or a fleet rule needs to
match on goes through :func:`log_event` with a name from :data:`EVENTS`, which renders
as ``event=<name>`` in text mode and as a top-level ``"event"`` key in JSON mode. The
vocabulary is deliberately short and deliberately closed: adding a name is a
deliberate act, and an unknown name is logged (once) as a defect rather than silently
becoming a new alertable string.

Throttling
----------
:class:`Throttle` bounds a repeating message to one line per interval and reports how
many were suppressed. Per-frame logging at 20 Hz is ~45 journald lines a second from a
service meant to run for hours; it costs CPU and eMMC writes, and it buries the two
lines a week that matter.

Failure behaviour: nothing in this module raises. An unwritable log file is reported
once on stderr and logging continues to the stream handler.
"""
from __future__ import annotations

import contextlib
import json
import logging
import logging.handlers
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

#: Closed vocabulary of alertable events. Keep this list short; every entry is a thing
#: an operator or a fleet rule is expected to match on by name.
EVENTS = (
    "startup",                 # process started, config resolved
    "ready",                   # first frame processed, READY=1 sent
    "shutdown",                # deliberate stop
    "config_loaded",           # configuration file accepted
    "engine_loaded",           # a TensorRT engine deserialised
    "engine_failed",           # an engine failed to load or execute
    "model_stub",              # a deterministic stub is standing in for a model
    "source_opened",           # capture source opened
    "source_lost",             # capture source stopped delivering
    "source_reconnected",      # capture source reopened after a loss
    "perception_dropout",      # perception failed enough times to be unusable
    "perception_recovered",    # perception produced a usable frame again
    "safety_state",            # SafetyState transition
    "safety_violation",        # a hard limit was breached and the command was vetoed
    "safety_warning",          # a soft limit was breached; the command stood
    "watchdog_withheld",       # a watchdog deadline passed with no frame progress
    "events_unwritable",       # the safety JSONL cannot be written
    "disk_full",               # ENOSPC on a writer
    "health_bind_refused",     # the health endpoint could not listen
    "summary",                 # the periodic throughput summary
)

#: Fields that :class:`logging.LogRecord` owns; anything else on the record was put
#: there by ``extra=`` and is ours to emit.
_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}

_TEXT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_TEXT_DATEFMT = "%Y-%m-%dT%H:%M:%S"

_configured = False
_lock = threading.Lock()


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, msg, plus structured extras.

    ``ts`` is Unix epoch seconds (float, wall clock — it may jump at NTP sync, which is
    why durations elsewhere use ``time.monotonic``). Exceptions are rendered into a
    ``traceback`` string field rather than trailing newlines, so one record stays one
    line and ``jq`` never chokes.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info
        try:
            return json.dumps(payload, default=str)
        except (TypeError, ValueError):
            # A non-serialisable extra must not lose the log line.
            return json.dumps({"ts": payload["ts"], "level": payload["level"],
                               "logger": payload["logger"], "msg": payload["msg"],
                               "note": "extras dropped: not JSON-serialisable"})


class TextFormatter(logging.Formatter):
    """Human format, with structured extras appended as ``key=value`` pairs."""

    def __init__(self) -> None:
        logging.Formatter.__init__(self, _TEXT_FORMAT, _TEXT_DATEFMT)

    def format(self, record: logging.LogRecord) -> str:
        base = logging.Formatter.format(self, record)
        extras = [
            "%s=%s" % (k, v)
            for k, v in sorted(record.__dict__.items())
            if k not in _RESERVED and not k.startswith("_")
        ]
        return base + (" | " + " ".join(extras) if extras else "")


def configure_logging(level: Any = logging.INFO, fmt: str = "text",
                      stream: Any = None, file_path: str = "",
                      max_bytes: int = 8 * 1024 * 1024, backups: int = 3,
                      force: bool = True) -> logging.Logger:
    """Install exactly one root handler and set the root level. Call once, early.

    Args:
        level: ``logging`` level, or its name (``"DEBUG"``, ``"INFO"``, ...). An
            unknown name falls back to INFO and logs a warning rather than raising —
            a typo in a config file must not stop a safety service from starting.
        fmt: ``"text"`` or ``"json"``. Anything else falls back to ``"text"``.
        stream: stream handler destination. Defaults to ``sys.stdout`` so journald
            captures it as ``MESSAGE`` and a container's ``docker logs`` shows it.
        file_path: optional rotating file, in addition to the stream. Rotation here is
            in-process and independent of ``deploy/adas.logrotate``, which covers the
            event JSONL; a service that logs to journald does not need this.
        max_bytes: rotation threshold for *file_path*, bytes.
        backups: generations kept for *file_path*.
        force: replace any handlers already on the root logger. True by default so a
            second call (a test, a reload) does not double every line.

    Returns:
        The root logger.

    Never raises: an unopenable *file_path* is reported on stderr and the stream
    handler still gets installed, because losing logging entirely is worse than losing
    the file copy.
    """
    global _configured
    resolved = _resolve_level(level)
    formatter = JsonFormatter() if str(fmt).lower() == "json" else TextFormatter()
    root = logging.getLogger()
    with _lock:
        if force:
            for handler in list(root.handlers):
                root.removeHandler(handler)
                if getattr(handler, "_adas_owned", False):
                    # Closing our own handler can raise if its stream is already
                    # gone (a closed pytest capture, a rotated file). Reconfiguring
                    # logging must not fail because of that.
                    with contextlib.suppress(Exception):
                        handler.close()
        stream_handler = logging.StreamHandler(sys.stdout if stream is None else stream)
        stream_handler.setFormatter(formatter)
        stream_handler._adas_owned = True  # type: ignore[attr-defined]
        root.addHandler(stream_handler)
        if file_path:
            try:
                directory = os.path.dirname(file_path)
                if directory:
                    os.makedirs(directory, mode=0o750, exist_ok=True)
                file_handler = logging.handlers.RotatingFileHandler(
                    file_path, maxBytes=int(max_bytes), backupCount=int(backups)
                )
                file_handler.setFormatter(formatter)
                file_handler._adas_owned = True  # type: ignore[attr-defined]
                root.addHandler(file_handler)
            except OSError as exc:
                sys.stderr.write(
                    "adas: log file %s is unwritable (%s); logging to stream only\n"
                    % (file_path, exc)
                )
        root.setLevel(resolved)
        _configured = True
    return root


def _resolve_level(level: Any) -> int:
    if isinstance(level, int):
        return level
    name = str(level).upper()
    value = logging.getLevelName(name)
    if isinstance(value, int):
        return value
    logging.getLogger("adas.logger").warning(
        "unknown log level %r; using INFO", level
    )
    return logging.INFO


def setup_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return the module logger for *name*, ensuring root logging is configured.

    **This does not set a level on the returned logger.** That is the whole point: a
    module logger with an explicit level cannot be controlled from the command line.
    The *level* argument is accepted for source compatibility with the previous
    signature and, when given, is applied to the *root* logger — which is what the
    caller always meant.

    Modules should simply do ``logger = setup_logger(__name__)`` at import; the first
    call installs a default root handler so a library import never logs to the
    "no handlers could be found" fallback.
    """
    if not _configured:
        configure_logging(
            _resolve_level(os.environ.get("ADAS_LOG_LEVEL", "INFO")),
            fmt=os.environ.get("ADAS_LOG_FORMAT", "text"),
            force=not logging.getLogger().handlers,
        )
    if level is not None:
        logging.getLogger().setLevel(_resolve_level(level))
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, msg: str = "",
              level: int = logging.INFO, **fields: Any) -> None:
    """Log one entry from the closed :data:`EVENTS` vocabulary with structured fields.

    In JSON mode the fields become top-level keys; in text mode they are appended as
    ``key=value``. An event name outside :data:`EVENTS` is still logged (dropping a
    safety-relevant line would be worse) but a warning names the offender so it is
    fixed rather than becoming a second, undocumented vocabulary.
    """
    if event not in EVENTS:
        logging.getLogger("adas.logger").warning(
            "log_event called with unknown event %r; add it to adas.core.logger.EVENTS",
            event,
        )
    logger.log(level, msg or event, extra=dict(fields, event=event))


def log_safety_event(logger: logging.Logger, event: str, severity: str = "WARNING",
                     **kwargs: Any) -> None:
    """Log a safety-relevant event. Kept for source compatibility.

    ``severity`` is the textual level (``INFO``/``WARNING``/``CRITICAL``); the event is
    emitted under the ``safety_violation`` vocabulary entry for CRITICAL and
    ``safety_warning`` otherwise, so fleet rules can match on the name rather than on
    a substring of the message.
    """
    sev = str(severity).upper()
    level = {"CRITICAL": logging.CRITICAL, "WARNING": logging.WARNING}.get(sev, logging.INFO)
    name = "safety_violation" if level >= logging.CRITICAL else "safety_warning"
    log_event(logger, name, event, level=level, severity=sev, **kwargs)


def log_performance(logger: logging.Logger, operation: str, duration_ms: float) -> None:
    """Log one timing at DEBUG. Kept for source compatibility.

    This used to log at INFO and was called once per frame, which is ~20 lines a second
    forever. Latency belongs in ``adas_stage_duration_ms`` (a histogram a fleet can
    aggregate), not in a log line per frame, so callers should prefer
    :meth:`adas.core.metrics.PerformanceMetrics.stage`. DEBUG keeps the call harmless.
    """
    logger.debug("PERF %s %.2fms", operation, duration_ms,
                 extra={"operation": operation, "duration_ms": round(duration_ms, 3)})


class Throttle:
    """Rate-limit a repeating log line to one emission per *interval_s*.

    Suppressed occurrences are counted and reported on the next line that gets through
    (``suppressed=N``), so throttling never silently hides a growing problem.

    Args:
        interval_s: minimum seconds between emissions.
        clock: injectable ``time.monotonic`` for tests.

    Example:
        >>> t = Throttle(10.0)
        >>> if t.ready():
        ...     logger.warning("detector slow", extra=t.fields())
    """

    def __init__(self, interval_s: float = 10.0, clock=time.monotonic) -> None:
        self.interval_s = float(interval_s)
        self._clock = clock
        self._last = -1e18
        self._suppressed = 0
        self.last_suppressed = 0

    def ready(self) -> bool:
        """True when the caller should emit; counts the call as suppressed otherwise."""
        now = self._clock()
        if now - self._last < self.interval_s:
            self._suppressed += 1
            return False
        self._last = now
        self.last_suppressed = self._suppressed
        self._suppressed = 0
        return True

    def fields(self) -> Dict[str, int]:
        """``{"suppressed": N}`` for the emission :meth:`ready` just authorised."""
        return {"suppressed": self.last_suppressed}


__all__ = [
    "EVENTS",
    "JsonFormatter",
    "TextFormatter",
    "Throttle",
    "configure_logging",
    "log_event",
    "log_performance",
    "log_safety_event",
    "setup_logger",
]
