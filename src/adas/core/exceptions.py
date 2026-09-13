"""Domain-specific exceptions for the ADAS system.

Every exception here derives from :class:`ADASException` and carries an optional
structured *context* mapping alongside its message. The context is what ends up in the
JSON log line and in the safety event record: ``PerceptionError("detector failed",
engine="yolox_nano", frame_id=1204)`` is greppable and machine-matchable, whereas the
same facts interpolated into the message string are not.

Backwards compatible: ``raise PerceptionError("something went wrong")`` still works and
``str(exc)`` is still exactly that message.

Which exception means what
--------------------------
The distinction that matters operationally is **fail-open vs fail-closed**:

* :class:`SafetyViolation` and :class:`ActuationRefused` mean *a command was or must be
  vetoed*. A caller that catches one and then actuates anyway has defeated the safety
  layer; catch them only where you can substitute a safe command.
* :class:`PerceptionError`, :class:`SensorError` and :class:`EngineError` mean *the
  world model is unavailable*. An empty detection list is **not** an equivalent
  outcome: "nothing is there" and "I cannot see" must never be conflated, which is why
  these are exceptions rather than empty returns.
* :class:`ConfigurationError` and :class:`ValidationError` are startup/programming
  faults. They should stop a unit before it drives, not be swallowed at runtime.
"""
from __future__ import annotations

from typing import Any, Dict


class ADASException(Exception):
    """Base exception for all ADAS errors, with optional structured context.

    Args:
        message: human-readable description. Keep it stable — it is what an operator
            greps for; put the varying values in *context*, not in the string.
        **context: JSON-serialisable key/value pairs describing the occurrence
            (``engine``, ``frame_id``, ``limit``, ``value``, ...).

    Attributes:
        message: the message as given.
        context: the context mapping (empty dict when none was supplied).
    """

    def __init__(self, message: str = "", **context: Any) -> None:
        Exception.__init__(self, message)
        self.message = message
        self.context: Dict[str, Any] = dict(context)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        if self.context:
            pairs = ", ".join("%s=%r" % kv for kv in sorted(self.context.items()))
            return "%s(%r, %s)" % (type(self).__name__, self.message, pairs)
        return "%s(%r)" % (type(self).__name__, self.message)

    def as_dict(self) -> Dict[str, Any]:
        """``{"error": <class name>, "message": ..., **context}`` for a log or a record."""
        payload: Dict[str, Any] = {"error": type(self).__name__, "message": self.message}
        payload.update(self.context)
        return payload


class ValidationError(ADASException):
    """Input validation failed. A programming or configuration fault, not a road event."""


class ConfigurationError(ADASException):
    """Configuration is invalid, unreadable, or refers to something that is not there."""


class SensorError(ADASException):
    """A sensor produced no data, or data the pipeline cannot interpret.

    Raised by capture. Distinct from end-of-file on a replay clip, which is a normal
    termination and not an error.
    """


class SafetyViolation(ADASException):
    """A hard safety limit was breached.

    Raising this is a *veto*: the caller must not actuate the command that triggered
    it. Catching it to log and then proceeding is the fail-open bug this class exists
    to make visible.
    """


class ActuationRefused(SafetyViolation):
    """The safety arbiter replaced or suppressed a command before actuation.

    Carries the substituted command in ``context["command"]`` when there is one, so the
    caller does not have to guess what to send instead.
    """


class PerceptionError(ADASException):
    """Perception could not produce a world model for this frame.

    Not the same as "no objects detected". A caller that turns this into an empty
    detection list has told the planner the road is clear.
    """


class EngineError(PerceptionError):
    """A TensorRT engine failed to load, bind, or execute.

    ``context["engine"]`` names the engine and ``context["kind"]`` is one of ``load``,
    ``shape``, ``execution``, ``hang`` — the same vocabulary as the ``adas_engine_errors_total``
    metric label, so a log line and a counter can be correlated.
    """


class TrackingError(ADASException):
    """Object tracking failed for this frame."""


class PlanningError(ADASException):
    """Motion planning could not produce a plan."""


class ControlError(ADASException):
    """Control command generation failed."""


class HealthEndpointError(ADASException):
    """The health/metrics endpoint could not bind or serve.

    Only raised when the endpoint is configured as ``required``; otherwise the failure
    is logged and the process runs without it, because losing observability is not a
    reason to stop monitoring the road.
    """


class EventLogError(ADASException):
    """The durable event log could not be opened.

    Only raised when the log is configured ``strict``; otherwise the writer degrades to
    a fallback path or to a disabled state and reports it on ``/healthz``.
    """


__all__ = [
    "ADASException",
    "ActuationRefused",
    "ConfigurationError",
    "ControlError",
    "EngineError",
    "EventLogError",
    "HealthEndpointError",
    "PerceptionError",
    "PlanningError",
    "SafetyViolation",
    "SensorError",
    "TrackingError",
    "ValidationError",
]
