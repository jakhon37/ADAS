"""The harness's own acceptance test, run under pytest.

``tests/scenarios/backtest.py`` runs this specification against arbiter versions
whose failures were measured on real video and recorded in the repository's
history.  This file makes that a committed regression instead of something an
agent did once and described in a summary.

Why it has to be committed
--------------------------
The first version of this harness was REJECTED by exactly this check.  It ran,
it printed a confident table, and against both known-broken commits it never
once emitted ``collided``, ``clearance``, ``missed_intervention``,
``late_intervention`` or ``no_response`` -- the entire collision half of the
specification was decoration, and nothing in the repository would have said so.
Its scenarios also sat off the failure boundary: four cases that NAMED round-2
missed braking as their guard reported 2.25-5.82 m of margin and PASSED on the
commit that collides.

Both of those are the kind of defect that comes back.  A future change to the
oracle's margins, to a scenario's initial gap, or to the finding taxonomy can
silently stop catching either commit, and the suite would go GREENER, not
redder.  This file is the tripwire: if the specification stops detecting a
defect somebody already measured, this test goes red and names it.

What it costs and when it skips
-------------------------------
Each commit means one ``git worktree add`` and one full harness run -- about
15 s per commit on the target board.  The working tree is never touched: no
checkout, no reset, no stash, and the worktree is removed in a ``finally``.

It SKIPS, rather than fails, when the git history is not available -- a source
tarball, a shallow clone, no ``git`` on PATH, an old git without ``worktree``.
The harness is still perfectly correct in those environments and an assertion
that fails for want of history is an assertion people delete.  Set
``ADAS_SKIP_BACKTEST=1`` to skip it deliberately.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, _ROOT)

from tests.scenarios import backtest as bt  # noqa: E402

_UNAVAILABLE = bt.unavailable_reason()


pytestmark = pytest.mark.skipif(
    bool(os.environ.get("ADAS_SKIP_BACKTEST")) or _UNAVAILABLE is not None,
    reason=_UNAVAILABLE or "ADAS_SKIP_BACKTEST is set",
)


@pytest.fixture(scope="module")
def outcomes():
    """Backtest every known-broken commit once and share the results."""
    return {o.commit.sha: o for o in bt.backtest_all()}


def test_known_broken_commits_are_still_in_history():
    """The commits this harness is calibrated against must still be reachable.

    A rebase, a squash or a force-push that removes them does not just break
    this file -- it removes the only recorded evidence of what the two failure
    modes actually looked like.
    """
    for commit in bt.KNOWN_BROKEN:
        resolved = bt.resolve_commit(bt.REPO_ROOT, commit.sha)
        assert resolved is not None, "%s is no longer in this clone" % commit.sha
        assert bt.committed_arbiter_md5(bt.REPO_ROOT, resolved) == commit.arbiter_md5, (
            "%s no longer has arbiter md5 %s; history was rewritten, and this backtest "
            "is no longer testing the code it claims to test"
            % (commit.sha, commit.arbiter_md5)
        )


@pytest.mark.parametrize("sha", [c.sha for c in bt.KNOWN_BROKEN], ids=lambda s: s)
def test_harness_catches_known_defect(outcomes, sha):
    """The harness must still fail the right family with the right diagnosis.

    * ``25e3ba5`` -- phantom braking -- must produce ``phantom_intervention``
      (or its sub-emergency sibling ``unwarranted_brake``) on the
      ``constant_range`` family.
    * ``1ce4886`` -- missed braking -- must produce a collision diagnosis on the
      ``lead_brakes_6mps2`` family.

    If this fails, do not adjust ``KNOWN_BROKEN``.  Adjusting it is how a
    specification stops being one.  The change that made this go red either
    moved the scenarios off the failure boundary or broke a detector, and the
    fix is in the harness.
    """
    outcome = outcomes[sha]
    assert outcome.satisfied, "\n" + outcome.describe()


def test_the_harness_actually_ran_against_the_old_code(outcomes):
    """Guard against the backtest quietly grading HEAD twice.

    An installed ``adas``, a stray ``.pth`` or a mangled ``PYTHONPATH`` would
    make every assertion above pass for the wrong reason.  The md5 of the module
    that actually imported is checked against the recorded one, and the module
    must live inside the temporary worktree.
    """
    for commit in bt.KNOWN_BROKEN:
        run = outcomes[commit.sha].run
        assert run.loaded_arbiter_md5 == commit.arbiter_md5, (
            "%s: the arbiter that loaded has md5 %s, expected %s"
            % (commit.sha, run.loaded_arbiter_md5, commit.arbiter_md5)
        )
        assert "adas-backtest-" in run.loaded_arbiter_path, run.loaded_arbiter_path


def test_backtest_leaves_no_worktrees_behind(outcomes):
    """The working tree and the worktree list must be exactly as we found them.

    A leaked worktree is a lock in ``.git/worktrees`` that the next
    ``git worktree add`` trips over, and this suite runs alongside other work.
    """
    listing = subprocess.run(
        ["git", "-C", bt.REPO_ROOT, "worktree", "list", "--porcelain"],
        stdout=subprocess.PIPE,
        universal_newlines=True,
    ).stdout
    assert "adas-backtest-" not in listing, listing


def test_backtest_does_not_touch_the_working_tree(outcomes):
    """Nothing here may stage, commit, stash or check anything out.

    Asserted structurally rather than by diffing the tree, because concurrent
    edits by a human or another agent would make a diff-based check flaky while
    proving nothing about this module.
    """
    source = open(os.path.join(_HERE, "scenarios", "backtest.py")).read()
    for forbidden in (
        '"checkout"',
        '"reset"',
        '"stash"',
        '"commit"',
        '"add", "-A"',
        '"clean"',
    ):
        assert forbidden not in source, (
            "backtest.py contains a mutating git command (%s); it must be read-only "
            "apart from 'git worktree add/remove/prune'" % forbidden
        )


def test_the_collision_half_of_the_spec_fires_on_the_missed_braking_commit(outcomes):
    """The specific defect that rejected version 1 of this harness.

    Version 1 never emitted ``collided``, ``collided_unavoidable``,
    ``clearance``, ``missed_intervention``, ``late_intervention``,
    ``no_response`` or ``lane_departure`` against EITHER broken commit, while
    ``1ce4886`` is known to make CONTACT at d0 = 15/20/25/30 m against a lead
    braking at 6 m/s^2.  Half the specification was unreachable.

    This test is narrower than the family check above on purpose: it demands
    that the collision codes fire, not merely that something fails.
    """
    run = outcomes["1ce4886"].run
    fired = {code for codes in run.findings.values() for code in codes}
    assert fired & bt.COLLISION_CODES, (
        "no collision diagnosis fired anywhere in the library against the commit that "
        "collides at 15, 20, 25 and 30 m. Everything that did fire: %s" % sorted(fired)
    )


def test_the_phantom_family_sits_on_the_phantom_boundary(outcomes):
    """``25e3ba5`` phantom brakes between 12 and 40 m, so the family must too.

    Version 1's ``constant_range_motorway`` sat at 52 m and PASSED on the
    phantom commit while claiming in its own metadata to guard round-1 phantom
    AEB.  A scenario off the failure boundary is not a regression test; it is a
    green tick with a comment on it.  At least one member of the family must be
    inside the region where the defect actually occurs.
    """
    outcome = outcomes["25e3ba5"]
    assert outcome.caught, (
        "the constant_range family caught nothing on the phantom commit:\n%s"
        % outcome.describe()
    )
    clean = [n for n in outcome.family if not outcome.run.findings[n]]
    assert len(clean) < len(outcome.family), (
        "every constant_range scenario passed on the phantom-braking commit; the whole "
        "family is off the boundary"
    )
