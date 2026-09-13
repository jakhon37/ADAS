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


# --------------------------------------------------------------------------- #
# Baseline provenance
# --------------------------------------------------------------------------- #

ARBITER_RELPATH = os.path.join("src", "adas", "control", "arbiter.py")
"""The system under test, relative to the repository root.

Recorded in ``baseline.json`` by fingerprint only.  The harness still imports
the production stack through its public interfaces and never reads this file
to decide what to expect.
"""


def _md5_file(path):
    """The md5 of a file, or ``None`` when it is not there."""
    import hashlib

    try:
        with open(path, "rb") as handle:
            return hashlib.md5(handle.read()).hexdigest()
    except (IOError, OSError):  # pragma: no cover - only when the tree is odd
        return None


def _md5_names(names):
    """The md5 of a sorted name list, so a rename shows up as a difference."""
    import hashlib

    return hashlib.md5("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def baseline_header(scenario_names, known_failures=None):
    """Provenance for ``baseline.json``: what was measured, and when.

    A baseline is only meaningful against the pair (system under test, corpus)
    it was recorded from.  Both moved under the committed file once already --
    the corpus grew from 37 to 52 scenarios and several finding codes were
    renamed -- and nothing in the file said so, so every comparison against it
    was quietly wrong.  This makes that detectable: re-measure the three
    fingerprints, and if any differs the file is stale.

    Args:
        scenario_names: Every scenario name in the corpus that was run.
        known_failures: The recorded ``{name: [codes]}`` mapping, used to
            fingerprint the set of diagnosis codes in play.  Optional.

    Returns:
        The ``generated`` block to store at the top level of the document.
    """
    import datetime
    import subprocess

    arbiter = os.path.join(_ROOT, ARBITER_RELPATH)
    digest = _md5_file(arbiter)
    names = sorted(scenario_names)
    corpus_md5 = _md5_names(names)
    codes = sorted({c for v in (known_failures or {}).values() for c in v})
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT
        ).decode().strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", ARBITER_RELPATH], cwd=_ROOT
            ).decode().strip()
        )
    except Exception:  # pragma: no cover - no git, or not a checkout
        head, dirty = None, None
    return {
        "generated_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": head,
        "system_under_test": {
            "path": ARBITER_RELPATH,
            "md5": digest,
            "md5_short": (digest or "")[:8] or None,
            "clean_at_head": (None if dirty is None else not dirty),
        },
        "corpus": {
            "scenario_count": len(names),
            "scenario_names_md5": corpus_md5,
            "diagnosis_codes_md5": _md5_names(codes) if codes else None,
        },
        "staleness_check": (
            "This baseline describes the arbiter with md5 %s judged by a corpus "
            "of %d scenarios (names md5 %s). If any of those three differ from "
            "what you measure now, the file is STALE and the new/worsened/fixed "
            "split below is meaningless: re-record it. Regenerating with "
            "report.py --write-baseline does NOT rewrite this header -- "
            "report.build_baseline() should call "
            "tests.scenarios.baseline_header() so it cannot silently go missing."
        ) % ((digest or "unknown")[:8], len(names), corpus_md5[:8]),
    }
