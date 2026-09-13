"""Control modules.

Public surface:

* :class:`PIDLikeLongitudinalController` -- stateful PI speed control with
  deadband, anti-windup, jerk and pedal-rate limits.
* :class:`SafetyMonitor` -- holds the authoritative arbiter and the legacy
  advisory checks.  ``SafetyMonitor.arbitrate`` is the only safety entry point
  whose result is binding.
* :class:`SafetyArbiter` / :class:`ArbiterLimits` / :class:`SafetyContext` --
  the independent arbitration engine and its inputs.
"""

from adas.control.arbiter import ArbiterLimits, LeadAssessment, SafetyArbiter, SafetyContext
from adas.control.controller import PIDLikeLongitudinalController
from adas.control.safety import SafetyLimits, SafetyMonitor

__all__ = [
    "ArbiterLimits",
    "LeadAssessment",
    "PIDLikeLongitudinalController",
    "SafetyArbiter",
    "SafetyContext",
    "SafetyLimits",
    "SafetyMonitor",
]
