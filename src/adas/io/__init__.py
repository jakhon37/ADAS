"""Operations surface: metric registry, health/metrics HTTP, durable events, sd_notify.

This package is what makes an ADAS process supervisable: an operator or a fleet agent
can ask *is it processing frames right now* (``/readyz``), *what is it doing*
(``/healthz``), *how has it behaved over time* (``/metrics``), and *what happened
before the incident* (the JSONL). systemd can kill and restart it when the frame loop
stops advancing (:class:`adas.io.sd_notify.WatchdogPinger`).

Import order matters. :mod:`adas.io.metrics` owns the process-wide registry and every
other module here writes to it, so it is imported first; importing a sibling before it
would try to resolve ``adas.io.metrics`` through a half-initialised package.

Nothing here imports OpenCV, TensorRT or numpy, at module scope or otherwise, so the
whole package imports and unit-tests on a machine with none of them.
"""
from adas.io import metrics  # must be imported first: everything else writes to it
from adas.io.events import (
    EVENT_KINDS,
    SCHEMA_VERSION,
    EventLog,
    default_events_path,
    read_events,
)
from adas.io.health import (
    DEFAULT_PORT,
    ROUTES,
    HealthServer,
    HealthState,
    is_loopback,
)
from adas.io.sd_notify import (
    WatchdogPinger,
    notify_ready,
    notify_status,
    notify_stopping,
    notify_watchdog,
    sd_notify,
    under_systemd,
    watchdog_ping_interval_s,
)

__all__ = [
    "DEFAULT_PORT",
    "EVENT_KINDS",
    "EventLog",
    "HealthServer",
    "HealthState",
    "ROUTES",
    "SCHEMA_VERSION",
    "WatchdogPinger",
    "default_events_path",
    "is_loopback",
    "metrics",
    "notify_ready",
    "notify_status",
    "notify_stopping",
    "notify_watchdog",
    "read_events",
    "sd_notify",
    "under_systemd",
    "watchdog_ping_interval_s",
]
