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

Proved by mutation, not asserted
--------------------------------
The previous revision of this file did NOT have that property, and a family-only
check is why.  Three mutations were applied to the working tree, ``pytest
tests/test_backtest.py`` was run against each, and the result recorded:

===================================================  ==================  ==================
mutation                                             before these tests  after them
===================================================  ==================  ==================
move the constant_range family off the phantom
boundary (12/20/30/40 -> 55/60/65/68 m)              8 passed            FAILED
weaken a diagnosis: drop ``collided`` from the
codes the missed-braking commit must report          8 passed            FAILED
delete a scenario: remove ``constant_range_12m``     8 passed            FAILED
===================================================  ==================  ==================

Each mutation was reverted afterwards and the suite returned to its recorded
state.  The verbatim output of all three runs is in the workstream report; the
point of the table is that every row's left-hand column is the reason the
right-hand column had to be built.

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


@pytest.mark.parametrize("sha", [c.sha for c in bt.KNOWN_BROKEN], ids=lambda s: s)
def test_named_scenarios_still_produce_their_named_diagnosis(outcomes, sha):
    """The pins: THIS scenario must report THIS diagnosis on THIS commit.

    The family check above is satisfied by any one member reporting any one of a
    set of codes, and that is too weak to be a regression test.  Verified by
    mutation before these pins existed: moving the whole ``constant_range``
    family off the phantom boundary to 55/60/65/68 m left this file at
    "8 passed", because at 55 m the phantom commit still tripped the softer
    ``unwarranted_brake`` and the family count of one was met.

    What is pinned, and why each one:

    * ``25e3ba5`` -- ``constant_range_12m/20m/30m/40m`` must each report
      ``phantom_intervention``.  Four cases spanning 12 to 40 m, so a boundary
      that moves in by any amount fails at least one of them by name.
    * ``25e3ba5`` -- ``noisy_range_40m_030m_noise`` must too: the +/-0.30 m
      range-noise case is the one the arbiter's corroboration window was written
      against, and a harness that stops reproducing that false positive cannot
      certify a fix for it.
    * ``25e3ba5`` -- ``out_of_lane_vehicle_at_12m`` must too: braking at
      5.0 m/s^2 for a car one lane over is the in-path gate failing, and it is
      the only case in the library that says so.
    * ``1ce4886`` -- ``lead_brakes_6mps2_ego20_at_16m/20m/26m`` and
      ``lead_brakes_6mps2_ego25_at_30m`` must each report ``collided``.  Not
      "some collision code": ``collided``.  A run that ends 0.4 m short is a
      different measurement from one that ends in contact, and the whole point
      of these four is that they end in contact.

    Do not adjust the pins to make this pass.  A pin failing means a scenario
    was renamed, deleted, or moved off the boundary it was placed on.
    """
    outcome = outcomes[sha]
    assert not outcome.pinned_failures, "\n" + outcome.describe()
    assert len(outcome.pinned_ok) == len(outcome.commit.required_by_scenario), (
        "\n" + outcome.describe()
    )


@pytest.mark.parametrize("sha", [c.sha for c in bt.KNOWN_BROKEN], ids=lambda s: s)
def test_the_far_side_of_each_boundary_is_still_clean(outcomes, sha):
    """The other edge: the cases OUTSIDE the region must NOT report it.

    A failure region has two edges and the previous revision of this file
    asserted only one.  "The phantom fires up to 43 m at 20 m/s" is a claim
    about ``constant_range_52m`` and ``constant_range_70m`` PASSING on
    ``25e3ba5`` exactly as much as it is a claim about ``constant_range_40m``
    failing, and a family that fails at every range has measured the width of
    the grid rather than the width of the defect.

    This is also what makes the family impossible to move quietly.  Push the
    constant-range cases outwards and the pins above go red; pull them inwards
    and these go red.
    """
    outcome = outcomes[sha]
    assert not outcome.straddle_failures, "\n" + outcome.describe()


def test_the_pins_actually_pin_something(outcomes):
    """Guard against the pins being emptied instead of satisfied.

    A pin list that is empty passes every assertion above.  This is the
    ratchet: both commits must carry pins, and between them they must cover both
    edges of both boundaries.
    """
    for commit in bt.KNOWN_BROKEN:
        assert commit.required_by_scenario, (
            "%s carries no per-scenario pins. Deleting them turns this file back into "
            "the family-only check that survived moving a whole family off its "
            "boundary." % commit.sha
        )
        assert commit.forbidden_by_scenario, (
            "%s carries no straddle scenarios, so only one edge of its boundary is "
            "asserted and the region has no measured width." % commit.sha
        )
        assert commit.min_caught >= 2, (
            "%s requires only %d family member(s) to catch the defect; one is what the "
            "old check accepted and it was not enough."
            % (commit.sha, commit.min_caught)
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
