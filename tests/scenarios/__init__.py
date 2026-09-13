"""Executable safety specification for the ADAS longitudinal path.

This package is the acceptance harness described in ``docs/SAFETY_SPEC.md``.
It is deliberately independent of :mod:`adas.control.arbiter`: it imports the
production stack only through its public interfaces (``BehaviorPlanner``,
``PIDLikeLongitudinalController``, ``SafetyMonitor``, ``SafetyContext``) and
judges the result against kinematics.

Layout:

``plant``
    The deterministic vehicle and world model.  Ground truth.
``oracle``
    Pure kinematics: what should have happened, from the true state history.
``scenario``
    The scenario format, the closed loop, and the pass/fail judgement.
``library``
    The scenarios themselves, each with its physics justification.
``report``
    A readable table and a machine-readable JSON artifact.

Nothing here needs a GPU, a TensorRT engine, a camera or a recording: the plant
replaces perception entirely, so the whole suite runs in CI and off-board.

Run it directly with::

    python -m tests.scenarios.report

or under pytest via ``tests/test_scenarios.py``.
"""

from __future__ import annotations

import os
import sys

# Allow ``python -m tests.scenarios.report`` from a source checkout that has not
# been pip-installed.  Under pytest this is a no-op, because pyproject.toml
# already puts ``src`` on the path.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    try:  # pragma: no cover - import bootstrap
        import adas  # noqa: F401
    except ImportError:  # pragma: no cover - import bootstrap
        sys.path.insert(0, _SRC)
if _ROOT not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, _ROOT)
