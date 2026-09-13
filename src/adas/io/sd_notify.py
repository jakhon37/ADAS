"""systemd readiness/watchdog notification over ``$NOTIFY_SOCKET``.

Written by hand against the sd_notify protocol rather than linking ``libsystemd`` or
installing ``python-systemd``: the board's dependency policy forbids adding packages,
and the protocol is a few lines of ``AF_UNIX``/``SOCK_DGRAM``.

The socket path may be a filesystem path or, when it starts with ``@``, a Linux
abstract socket (``\\0``-prefixed). Both are handled.

Units: :func:`watchdog_interval_s` returns seconds; ``$WATCHDOG_USEC`` is microseconds.

Failure behaviour: every function is a no-op outside systemd (no ``$NOTIFY_SOCKET``)
and logs — never raises — when the datagram cannot be delivered. A failed notification
must not take down a safety service; systemd's own watchdog will notice the silence,
which is the correct escalation path.

The watchdog contract, and why :class:`WatchdogPinger` exists
-------------------------------------------------------------
``WatchdogSec=`` in the unit means *systemd kills the service if it does not hear
``WATCHDOG=1`` within that interval*. A bare timer that pings every ``WatchdogSec/2``
satisfies systemd while the frame loop is wedged on a hung engine — the unit stays
"healthy" and blind, which is the exact failure the watchdog was added to catch.

:class:`WatchdogPinger` therefore gates on **progress**: it pings only when the frame
id has advanced since the last ping. When the pipeline stalls, the pings stop, systemd
kills the unit at ``WatchdogSec``, and ``Restart=always`` brings it back. Every skipped
deadline is counted in ``adas_watchdog_skipped_total`` so a marginal unit is visible
before it is killed.
"""
from __future__ import annotations

import logging
import os
import socket
from typing import Optional

from adas.io import metrics

log = logging.getLogger("adas.sd_notify")

_warned = False


def notify_socket_path() -> Optional[str]:
    """The socket address from the environment, translated for abstract sockets.

    Returns ``None`` outside systemd. Only the process systemd started (or its
    children, if the unit sets ``NotifyAccess=all``) may notify.
    """
    path = os.environ.get("NOTIFY_SOCKET")
    if not path:
        return None
    if path.startswith("@"):
        return "\0" + path[1:]
    return path


def under_systemd() -> bool:
    """True when a notification would actually go somewhere."""
    return notify_socket_path() is not None


def sd_notify(message: str) -> bool:
    """Send one sd_notify datagram. Returns True when it was delivered.

    A fresh socket per call: notifications happen at most a few times a second, and a
    long-lived fd would have to be recreated after a systemd restart anyway.
    """
    global _warned
    addr = notify_socket_path()
    if not addr:
        return False
    sock = None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC)
        sock.settimeout(1.0)
        sock.connect(addr)
        sock.sendall(message.encode("utf-8"))
        return True
    except OSError as exc:
        if not _warned:
            _warned = True
            log.warning("NOTIFY_SOCKET send failed (%s); further failures at debug", exc)
        else:
            log.debug("NOTIFY_SOCKET send failed: %s", exc)
        return False
    finally:
        if sock is not None:
            sock.close()


def notify_ready(status: str = "") -> bool:
    """``READY=1`` — the unit is up.

    Send exactly once, and only after the engines have deserialised *and* the first
    frame has been processed. Sending it at process start makes ``TimeoutStartSec``
    meaningless and tells the fleet a blind unit is ready.
    """
    msg = "READY=1"
    if status:
        msg += "\nSTATUS=" + status.replace("\n", " ")
    return sd_notify(msg)


def notify_status(status: str) -> bool:
    """``STATUS=`` — the one line ``systemctl status`` shows. Cheap; call freely."""
    return sd_notify("STATUS=" + status.replace("\n", " "))


def notify_watchdog() -> bool:
    """``WATCHDOG=1`` — pet the watchdog. Counted in ``adas_watchdog_pings_total``.

    Prefer :class:`WatchdogPinger`, which will not let you ping a stalled pipeline.
    """
    ok = sd_notify("WATCHDOG=1")
    if ok:
        metrics.WATCHDOG_PINGS_TOTAL.inc()
    return ok


def notify_stopping(status: str = "") -> bool:
    """``STOPPING=1`` — a deliberate shutdown, so systemd does not call it a crash."""
    msg = "STOPPING=1"
    if status:
        msg += "\nSTATUS=" + status.replace("\n", " ")
    return sd_notify(msg)


def notify_reloading() -> bool:
    """``RELOADING=1`` — used by a SIGHUP handler around a config or log reopen."""
    return sd_notify("RELOADING=1")


def notify_error(errno_value: int, status: str = "") -> bool:
    """``ERRNO=`` — report a startup failure to systemd before exiting non-zero."""
    msg = "ERRNO=%d" % int(errno_value)
    if status:
        msg += "\nSTATUS=" + status.replace("\n", " ")
    return sd_notify(msg)


def watchdog_interval_s() -> Optional[float]:
    """``WatchdogSec`` as seconds, or ``None`` when the watchdog is not armed.

    Honours ``$WATCHDOG_PID``: systemd sets it so a forked child does not mistakenly
    believe the watchdog is its responsibility.
    """
    raw = os.environ.get("WATCHDOG_USEC")
    if not raw:
        return None
    pid = os.environ.get("WATCHDOG_PID")
    if pid and pid.strip().isdigit() and int(pid) != os.getpid():
        return None
    try:
        usec = int(raw)
    except ValueError:
        log.warning("WATCHDOG_USEC=%r is not an integer; watchdog disabled", raw)
        return None
    if usec <= 0:
        return None
    return usec / 1e6


def watchdog_ping_interval_s(default: float = 5.0) -> float:
    """How often to ping: half the systemd deadline, or *default* outside systemd.

    Half is the interval systemd's own documentation recommends; it leaves one whole
    missed ping of margin before the unit is killed.
    """
    deadline = watchdog_interval_s()
    if deadline is None:
        return float(default)
    return max(0.5, deadline / 2.0)


class WatchdogPinger:
    """Progress-gated ``WATCHDOG=1`` sender.

    Call :meth:`tick` from the frame loop with the id of the frame that was just
    processed. A ping goes out only when both are true:

    * at least ``interval_s`` has elapsed since the last ping, and
    * ``frame_id`` has advanced since the last ping.

    A stalled pipeline therefore stops petting the watchdog and systemd kills the unit
    at ``WatchdogSec``. Deadlines that pass without progress are counted in
    ``adas_watchdog_skipped_total``.

    Outside systemd (:func:`under_systemd` false) :meth:`tick` still keeps its
    bookkeeping and returns ``False`` without sending anything, so the same code path
    runs in tests and in ``PYTHONPATH=src python3 -m adas.cli``.

    Args:
        interval_s: ping period. Defaults to :func:`watchdog_ping_interval_s`.
        clock: injectable ``time.monotonic`` for tests.
        notify: injectable sender, for tests. Must return True on delivery.
    """

    def __init__(self, interval_s: Optional[float] = None, clock=None, notify=None) -> None:
        import time as _time

        self._clock = clock or _time.monotonic
        self._notify = notify or notify_watchdog
        self.interval_s = float(
            watchdog_ping_interval_s() if interval_s is None else interval_s
        )
        self.armed = under_systemd() and watchdog_interval_s() is not None
        self.last_ping_mono = self._clock()
        self.last_ping_frame: Optional[int] = None
        self.pings = 0
        self.skipped = 0

    def tick(self, frame_id: int) -> bool:
        """Maybe ping. Returns True only when a datagram was actually sent."""
        now = self._clock()
        if now - self.last_ping_mono < self.interval_s:
            return False
        if self.last_ping_frame is not None and frame_id <= self.last_ping_frame:
            # The deadline came round and the pipeline has not moved. Say nothing and
            # let systemd act; log once per deadline so the reason is in the journal.
            self.skipped += 1
            metrics.WATCHDOG_SKIPPED_TOTAL.inc()
            log.error(
                "WATCHDOG_WITHHELD frame_id=%s has not advanced in %.1f s; not pinging "
                "systemd (skipped=%d)", frame_id, now - self.last_ping_mono, self.skipped
            )
            self.last_ping_mono = now
            return False
        self.last_ping_mono = now
        self.last_ping_frame = frame_id
        if not self.armed:
            return False
        sent = bool(self._notify())
        if sent:
            self.pings += 1
        return sent

    def stats(self) -> dict:
        """Snapshot for ``/healthz``."""
        return {
            "armed": self.armed,
            "interval_s": round(self.interval_s, 2),
            "pings": self.pings,
            "skipped": self.skipped,
            "last_frame": self.last_ping_frame,
        }


__all__ = [
    "WatchdogPinger",
    "notify_error",
    "notify_ready",
    "notify_reloading",
    "notify_socket_path",
    "notify_status",
    "notify_stopping",
    "notify_watchdog",
    "sd_notify",
    "under_systemd",
    "watchdog_interval_s",
    "watchdog_ping_interval_s",
]
