"""Backtest this harness against arbiter versions with KNOWN, MEASURED failures.

This is the harness's own acceptance test, and the reason it exists is that the
first version of the harness failed it.  A specification that cannot detect a
defect somebody already measured on the road is not a specification; it is a
set of assertions that happen to be true.

The method
----------
For each known-broken commit:

1. ``git worktree add --detach`` that commit into a temporary directory.  The
   working tree is never touched -- no checkout, no reset, no stash -- and the
   worktree is removed again in a ``finally``.
2. Run **this** harness (the one in the working tree, ``tests/scenarios``)
   against **that** commit's ``src`` by putting the worktree's ``src`` first on
   ``PYTHONPATH``.  The judgement layer is today's; the system under test is the
   old one.  Running the old commit's own tests would prove nothing, because the
   old commits have no harness.
3. Confirm the ``adas`` package actually loaded from the worktree, and that the
   md5 of the arbiter that loaded matches the recorded one.  Without this check
   a stale ``PYTHONPATH``, an installed ``adas`` or a ``.pth`` file would let the
   backtest quietly grade HEAD twice and report success.
4. Assert that the RIGHT scenarios fail with the RIGHT diagnosis.

What is asserted, and why it is a family and not a scenario name
-----------------------------------------------------------------
The requirement is stated over a FAMILY of scenarios matched by name prefix,
not over one hard-coded name:

* ``25e3ba5`` (arbiter md5 ``d85a8a2b``) -- PHANTOM BRAKING.  On real video,
  42 of 400 frames entered ``MIN_RISK_MANEUVER`` and 6 commanded ``brake=1.00``
  for a lead at CONSTANT range.  The ``constant_range`` family must therefore
  report ``phantom_intervention``.
* ``1ce4886`` (arbiter md5 ``d517e2aa``) -- MISSED BRAKING.  Against a lead
  braking at 6 m/s^2 from a steady 20 m/s follow it makes contact at
  d0 = 15/20/25/30 m.  The ``lead_brakes`` family must therefore report a
  COLLISION diagnosis: ``collided``, ``clearance``, ``missed_intervention``,
  ``late_intervention`` or ``no_response``.

A family survives the scenario renames and re-parameterisations that moving a
case onto its failure boundary requires; a hard-coded name turns this file into
something people delete.

But a family ON ITS OWN IS NOT ENOUGH, and this file used to contain only
families.  Verified by mutation: moving the whole ``constant_range`` family off
the phantom boundary to 55/60/65/68 m left ``pytest tests/test_backtest.py`` at
"8 passed", because the family check is satisfied by ANY member reporting ANY of
the required codes, and at 55 m the phantom commit still tripped the softer
``unwarranted_brake``.  A regression test that survives the exact regression it
exists to prevent is decoration.

So every commit now also carries PINS:

``required_by_scenario``
    named scenario -> the diagnosis it must report.  Pins the case AND the
    diagnosis, so renaming, deleting or moving one fails by name.
``forbidden_by_scenario``
    named scenario -> diagnoses it must NOT report.  This is the other edge of
    the boundary, and it was missing entirely.  "The phantom stops at 43 m" is a
    statement about ``constant_range_52m`` PASSING just as much as about
    ``constant_range_40m`` failing, and a family that fails everywhere has
    measured the grid rather than the defect.
``min_caught``
    how many family members must catch it, not merely one.

The three mutations this arrangement was verified against, each run as a real
edit followed by ``pytest tests/test_backtest.py``, are recorded in
``tests/test_backtest.py``.

Run it directly::

    python -m tests.scenarios.backtest
    python -m tests.scenarios.backtest --keep-worktrees --verbose

or under pytest via ``tests/test_backtest.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""The working tree this file lives in.  Never modified by anything here."""

ARBITER_REL_PATH = "src/adas/control/arbiter.py"
"""The file whose md5 identifies an arbiter version."""

COLLISION_CODES = frozenset(
    {
        "collided",
        "collided_unavoidable",
        "clearance",
        "missed_intervention",
        "late_intervention",
        "no_response",
    }
)
"""Every diagnosis that means "it hit something, or came far too close".

The missed-braking commit must produce at least one of these on the lead-braking
family.  Which one is not pinned, and deliberately: whether a run ends in
contact or in a 0.4 m miss depends on the exact gap the scenario starts from,
and pinning ``collided`` specifically would make this test a hostage to the
boundary the scenarios are parked on.  What must not happen is that all six stay
silent, which is precisely what the first version of this harness did.
"""

PHANTOM_CODES = frozenset({"phantom_intervention", "unwarranted_brake"})
"""Diagnoses that mean "it braked for something that was not there".

``unwarranted_brake`` is included because the sub-emergency band is the same
defect wearing a smaller number: a system dragged from 20 m/s to 12 m/s behind a
lead at constant range has phantom braked whether or not it ever crossed the
3.5 m/s^2 line.
"""

DEFAULT_TIMEOUT_S = 900.0
"""Wall clock allowed for one harness run.  The full library is ~11 s at HEAD."""


@dataclass(frozen=True)
class BrokenCommit:
    """One arbiter version with a measured, documented failure.

    Attributes:
        sha: Commit to check out into a temporary worktree.
        arbiter_md5: md5 of :data:`ARBITER_REL_PATH` at that commit.  Both the
            committed blob and the file that actually loads are checked against
            it, which is what makes "we really did test the old code" a fact
            rather than a hope.
        label: Short name for the failure mode.
        measured: The measurement this commit is known for, in the words of the
            report that produced it.  Printed on failure so that whoever has to
            fix this file can see what it is protecting.
        family_prefix: Scenario-name prefix selecting the family that must fail.
        required_any: The family must report at least one of these codes.
        min_family_size: Fewest scenarios the family must contain.  An empty or
            gutted family is a FAILURE: it means the coverage was deleted.
        min_caught: Fewest family members that must report one of
            ``required_any``.  The family check alone accepted ONE member
            catching something, which is why moving the whole family off its
            boundary survived it: a single case still tripped a softer
            diagnosis and the count of one was met.
        required_by_scenario: ``(scenario name, codes)`` pairs. Each NAMED
            scenario must report at least one of those codes. This is the
            assertion a prefix cannot make -- it pins the case AND the
            diagnosis, so a scenario that is renamed, deleted or moved off the
            boundary fails here by name rather than being silently covered for
            by a sibling.
        forbidden_by_scenario: ``(scenario name, codes)`` pairs. Each named
            scenario must report NONE of those codes. This is the other half of
            a boundary and the half that was missing: a family that catches the
            defect everywhere is not measuring a boundary, it is measuring the
            grid. ``constant_range_52m`` passing on the phantom commit is as
            much a part of "the phantom stops at 43 m" as
            ``constant_range_40m`` failing.

    Measured on this plant, and the reason the pins are the pins they are:

    * ``25e3ba5`` -- ``constant_range`` at 12/20/30/40 m all report
      ``phantom_intervention``; at 52 m and 70 m both PASS outright.
    * ``1ce4886`` -- ``lead_brakes_6mps2_ego20_at`` 16/20/26 m all report
      ``collided``; 32 m reports ``clearance`` and does NOT collide.
    """

    sha: str
    arbiter_md5: str
    label: str
    measured: str
    family_prefix: str
    required_any: FrozenSet[str]
    min_family_size: int = 2
    min_caught: int = 1
    required_by_scenario: Tuple[Tuple[str, FrozenSet[str]], ...] = ()
    forbidden_by_scenario: Tuple[Tuple[str, FrozenSet[str]], ...] = ()


KNOWN_BROKEN: Tuple[BrokenCommit, ...] = (
    BrokenCommit(
        sha="25e3ba5",
        arbiter_md5="d85a8a2bf8da8f03ca090d3d360b106f",
        label="phantom braking",
        measured=(
            "On real video 42/400 frames entered MIN_RISK_MANEUVER and 6 commanded "
            "brake=1.00 for a lead at CONSTANT range. In this plant, on a constant-range "
            "follow at 20 m/s, gaps of 12/20/30/40 m produced peak decelerations of "
            "8.00/8.00/8.00/6.39 m/s^2 with a worst state of MRM, dragging the ego from "
            "20 m/s down to 13.80/14.83/15.86/17.27 m/s. At 52 m and 70 m there was no "
            "intervention at all."
        ),
        family_prefix="constant_range",
        required_any=PHANTOM_CODES,
        min_caught=4,
        required_by_scenario=(
            ("constant_range_12m", frozenset({"phantom_intervention"})),
            ("constant_range_20m", frozenset({"phantom_intervention"})),
            ("constant_range_30m", frozenset({"phantom_intervention"})),
            ("constant_range_40m", frozenset({"phantom_intervention"})),
            ("noisy_range_40m_030m_noise", frozenset({"phantom_intervention"})),
            ("out_of_lane_vehicle_at_12m", frozenset({"phantom_intervention"})),
        ),
        forbidden_by_scenario=(
            ("constant_range_52m", PHANTOM_CODES),
            ("constant_range_70m", PHANTOM_CODES),
        ),
    ),
    BrokenCommit(
        sha="1ce4886",
        arbiter_md5="d517e2aa2662bf251b9cc0bd108d206b",
        label="missed braking",
        measured=(
            "Against a lead braking at 6 m/s^2 from a steady 20 m/s follow it makes "
            "CONTACT at d0 = 15/20/25/30 m ('YES @3.6 m/s', 'YES @4.7 m/s', "
            "'YES @5.1 m/s', 'YES @3.4 m/s'), where 25e3ba5 stopped with "
            "3.51 / 5.43 / 6.16 / 10.02 m to spare."
        ),
        family_prefix="lead_brakes_6mps2",
        required_any=COLLISION_CODES,
        min_caught=5,
        required_by_scenario=(
            ("lead_brakes_6mps2_ego20_at_16m", frozenset({"collided"})),
            ("lead_brakes_6mps2_ego20_at_20m", frozenset({"collided"})),
            ("lead_brakes_6mps2_ego20_at_26m", frozenset({"collided"})),
            ("lead_brakes_6mps2_ego25_at_30m", frozenset({"collided"})),
            # The sensitivity control, and the only pin whose diagnosis depends
            # on a THRESHOLD rather than on contact.  1ce4886 does not collide
            # at 32 m -- it clears by 2.22 m in the sweep and still finishes
            # inside the 2.0 m this specification requires -- so it must report
            # ``clearance`` and must not report ``collided``.  Weakening
            # REQUIRED_CLEARANCE_M is therefore a mutation this backtest can
            # see, which is the property it lacked before.
            ("lead_brakes_6mps2_ego20_at_32m", frozenset({"clearance"})),
        ),
        forbidden_by_scenario=(
            ("lead_brakes_6mps2_ego20_at_32m", frozenset({"collided"})),
        ),
    ),
)
"""The commits this harness must still catch.  Do not shorten this tuple."""


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #


def unavailable_reason(root: str = REPO_ROOT) -> Optional[str]:
    """Why the backtest cannot run here, or None when it can.

    Every one of these is a reason to SKIP and not to fail: a source tarball, a
    shallow clone, a machine without git and a sandbox that forbids subprocesses
    are all environments in which the harness itself is still perfectly correct.
    An assertion that fails for want of git history teaches nobody anything and
    gets deleted within a week.
    """
    if shutil.which("git") is None:
        return "git is not on PATH"
    if not os.path.isdir(os.path.join(root, ".git")) and not os.path.exists(
        os.path.join(root, ".git")
    ):
        return "%s is not a git working tree" % root
    try:
        _git(root, "rev-parse", "--is-inside-work-tree")
    except (OSError, subprocess.SubprocessError) as exc:
        return "git is unusable here: %s" % exc
    try:
        _git(root, "worktree", "list")
    except subprocess.SubprocessError:
        return "this git does not support 'git worktree'"
    missing = [c.sha for c in KNOWN_BROKEN if resolve_commit(root, c.sha) is None]
    if missing:
        return "commit(s) not in this clone's history: %s" % ", ".join(missing)
    return None


def _git(root: str, *args: str) -> str:
    """Run a READ-ONLY-or-worktree git command in ``root`` and return stdout.

    Nothing in this module ever runs checkout, reset, stash, add or commit.
    """
    proc = subprocess.run(
        ["git", "-C", root] + list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if proc.returncode != 0:
        raise subprocess.SubprocessError(
            "git %s failed (%d): %s" % (" ".join(args), proc.returncode, proc.stderr.strip())
        )
    return proc.stdout


def resolve_commit(root: str, sha: str) -> Optional[str]:
    """Full 40-character sha for ``sha``, or None when it is not in this clone."""
    try:
        return _git(root, "rev-parse", "--verify", "--quiet", "%s^{commit}" % sha).strip() or None
    except subprocess.SubprocessError:
        return None


def committed_arbiter_md5(root: str, sha: str) -> str:
    """md5 of :data:`ARBITER_REL_PATH` as committed at ``sha``.

    Read with ``git cat-file`` rather than from a checkout, so the identity of
    the code under test is established before anything is written to disk.
    """
    proc = subprocess.run(
        ["git", "-C", root, "cat-file", "-p", "%s:%s" % (sha, ARBITER_REL_PATH)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise subprocess.SubprocessError(
            "cannot read %s at %s: %s" % (ARBITER_REL_PATH, sha, proc.stderr.decode().strip())
        )
    return hashlib.md5(proc.stdout).hexdigest()


# --------------------------------------------------------------------------- #
# Running the harness against an old commit
# --------------------------------------------------------------------------- #


@dataclass
class HarnessRun:
    """The result of running this harness against one commit's ``src``."""

    sha: str
    resolved_sha: str
    loaded_arbiter_path: str
    loaded_arbiter_md5: str
    returncode: int
    table: str
    findings: Dict[str, List[str]] = field(default_factory=dict)
    """Scenario name -> its finding codes.  Empty list means it passed."""
    metrics: Dict[str, Dict[str, object]] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        """Whether the run produced a usable verdict.

        Note that the harness EXITS NON-ZERO when scenarios fail, which on these
        commits is the expected outcome, so the return code is not the health
        signal -- the presence of parsed scenarios is.
        """
        return bool(self.findings) and not self.error

    def family(self, prefix: str) -> List[str]:
        """Scenario names in this run whose name starts with ``prefix``."""
        return sorted(n for n in self.findings if n.startswith(prefix))


def _subprocess_env(worktree: str, root: str) -> Dict[str, str]:
    """Environment that loads ``adas`` from ``worktree`` and ``tests`` from ``root``.

    ``PYTHONPATH`` order is the whole trick: the worktree's ``src`` comes first,
    so ``import adas`` resolves to the OLD code, while ``root`` supplies today's
    ``tests.scenarios``.  ``PYTHONDONTWRITEBYTECODE`` keeps the temporary
    worktree free of ``__pycache__`` so removing it cannot fail on stray files.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(worktree, "src"), root])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    return env


def _verify_loaded_arbiter(env: Dict[str, str], root: str, worktree: str) -> Tuple[str, str]:
    """Import ``adas`` in a child process and report where it came from.

    Returns:
        ``(module path, md5 of the arbiter source file that loaded)``.

    Raises:
        RuntimeError: when ``adas`` did not load from ``worktree``.  That would
            mean the backtest was about to grade the wrong code, which is a far
            worse outcome than a loud failure.
    """
    code = (
        "import hashlib, json, adas.control.arbiter as a;"
        "p = a.__file__.replace('.pyc', '.py');"
        "print(json.dumps([p, hashlib.md5(open(p, 'rb').read()).hexdigest()]))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("could not import adas from %s: %s" % (worktree, proc.stderr.strip()))
    path, digest = json.loads(proc.stdout.strip().splitlines()[-1])
    if not os.path.abspath(path).startswith(os.path.abspath(worktree) + os.sep):
        raise RuntimeError(
            "adas loaded from %s, not from the worktree %s -- an installed copy or a .pth "
            "file is shadowing it, and this backtest would have graded the wrong code"
            % (path, worktree)
        )
    return path, digest


def run_harness_against(
    commit: BrokenCommit,
    root: str = REPO_ROOT,
    only: Optional[Sequence[str]] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    keep_worktree: bool = False,
) -> HarnessRun:
    """Check ``commit`` out into a temporary worktree and run the harness on it.

    Args:
        commit: The version under test.
        root: The working tree holding this harness.  Never modified.
        only: Scenario names to restrict the run to, or None for the whole
            library.
        timeout_s: Wall clock for the harness subprocess.
        keep_worktree: Leave the worktree on disk for inspection.  Off by
            default: a leaked worktree is a lock file in ``.git/worktrees`` that
            somebody else's ``git worktree add`` will trip over.

    Returns:
        A :class:`HarnessRun`.  Never raises for a failing scenario -- failing
        scenarios are the point -- but does raise if the wrong code loaded.
    """
    resolved = resolve_commit(root, commit.sha)
    if resolved is None:
        return HarnessRun(
            sha=commit.sha,
            resolved_sha="",
            loaded_arbiter_path="",
            loaded_arbiter_md5="",
            returncode=-1,
            table="",
            error="commit %s is not in this clone" % commit.sha,
        )

    committed_md5 = committed_arbiter_md5(root, resolved)
    if committed_md5 != commit.arbiter_md5:
        return HarnessRun(
            sha=commit.sha,
            resolved_sha=resolved,
            loaded_arbiter_path="",
            loaded_arbiter_md5=committed_md5,
            returncode=-1,
            table="",
            error=(
                "%s has arbiter md5 %s, expected %s. Either history was rewritten or "
                "KNOWN_BROKEN is stale; in both cases this backtest is no longer testing "
                "what it claims to test." % (commit.sha, committed_md5, commit.arbiter_md5)
            ),
        )

    holder = tempfile.mkdtemp(prefix="adas-backtest-%s-" % commit.sha)
    worktree = os.path.join(holder, "tree")
    try:
        _git(root, "worktree", "add", "--detach", worktree, resolved)
        env = _subprocess_env(worktree, root)
        loaded_path, loaded_md5 = _verify_loaded_arbiter(env, root, worktree)
        if loaded_md5 != commit.arbiter_md5:
            raise RuntimeError(
                "the arbiter that loaded (%s) has md5 %s, expected %s"
                % (loaded_path, loaded_md5, commit.arbiter_md5)
            )

        json_path = os.path.join(holder, "acceptance.json")
        argv = [
            sys.executable,
            "-m",
            "tests.scenarios.report",
            "--json",
            json_path,
            "--log-level",
            "CRITICAL",
        ]
        for name in only or ():
            argv += ["--only", name]
        proc = subprocess.run(
            argv,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout_s,
        )
        run = HarnessRun(
            sha=commit.sha,
            resolved_sha=resolved,
            loaded_arbiter_path=loaded_path,
            loaded_arbiter_md5=loaded_md5,
            returncode=proc.returncode,
            table=proc.stdout,
        )
        if not os.path.exists(json_path):
            run.error = "the harness produced no JSON artifact:\n%s" % (
                proc.stderr[-4000:] or proc.stdout[-4000:]
            )
            return run
        with open(json_path) as handle:
            payload = json.load(handle)
        for entry in payload["scenarios"]:
            run.findings[entry["name"]] = [f["code"] for f in entry["findings"]]
            run.metrics[entry["name"]] = entry["metrics"]
        return run
    finally:
        if not keep_worktree:
            try:
                _git(root, "worktree", "remove", "--force", worktree)
            except subprocess.SubprocessError:
                shutil.rmtree(worktree, ignore_errors=True)
            try:
                _git(root, "worktree", "prune")
            except subprocess.SubprocessError:
                pass
            shutil.rmtree(holder, ignore_errors=True)


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #


@dataclass
class BacktestOutcome:
    """Whether the harness caught what this commit is known to be broken for."""

    commit: BrokenCommit
    run: HarnessRun
    family: List[str] = field(default_factory=list)
    caught: Dict[str, List[str]] = field(default_factory=dict)
    """Family member -> the required codes it reported."""
    missed: List[str] = field(default_factory=list)
    """Family members that reported none of the required codes."""
    pinned_ok: List[str] = field(default_factory=list)
    """Named scenarios that reported the exact diagnosis pinned for them."""
    pinned_failures: List[str] = field(default_factory=list)
    """Named scenarios that did NOT: gone, renamed, or no longer diagnosing."""
    straddle_failures: List[str] = field(default_factory=list)
    """Named scenarios OUTSIDE the failure region that reported it anyway."""
    problem: str = ""

    @property
    def satisfied(self) -> bool:
        """True when the harness caught this commit's known defect."""
        return not self.problem

    def describe(self) -> str:
        """A full explanation, printable by the CLI and by pytest alike."""
        lines = [
            "commit %s (%s) -- %s" % (self.commit.sha, self.commit.arbiter_md5[:8], self.commit.label),
            "  arbiter under test: %s" % (self.run.loaded_arbiter_path or "<not loaded>"),
            "  family %r: %d scenario(s)%s"
            % (
                self.commit.family_prefix,
                len(self.family),
                (" -- " + ", ".join(self.family)) if self.family else "",
            ),
            "  required diagnosis (any of): %s" % ", ".join(sorted(self.commit.required_any)),
        ]
        if self.caught:
            for name in sorted(self.caught):
                lines.append("    CAUGHT  %-34s %s" % (name, ", ".join(self.caught[name])))
        for name in self.missed:
            lines.append(
                "    missed  %-34s %s"
                % (name, ", ".join(self.run.findings.get(name, [])) or "PASSED")
            )
        if self.commit.required_by_scenario:
            lines.append(
                "  pinned scenarios (each must report its own diagnosis, by name):"
            )
            for name in self.pinned_ok:
                lines.append("    PINNED  %s" % name)
            for detail in self.pinned_failures:
                lines.append("    BROKEN  %s" % detail)
        if self.commit.forbidden_by_scenario:
            lines.append(
                "  straddle scenarios (outside the region; must NOT report it):"
            )
            for name, codes in self.commit.forbidden_by_scenario:
                bad = [d for d in self.straddle_failures if d.startswith(name)]
                lines.append(
                    "    %s  %s"
                    % ("BROKEN " if bad else "CLEAN  ", bad[0] if bad else name)
                )
        if self.problem:
            lines += ["", "  PROBLEM: %s" % self.problem, "", "  what this commit is known for:"]
            lines.append("    " + self.commit.measured)
        return "\n".join(lines)


def backtest_commit(
    commit: BrokenCommit,
    root: str = REPO_ROOT,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    keep_worktree: bool = False,
) -> BacktestOutcome:
    """Run the harness against ``commit`` and judge whether it caught the defect."""
    run = run_harness_against(
        commit, root=root, timeout_s=timeout_s, keep_worktree=keep_worktree
    )
    outcome = BacktestOutcome(commit=commit, run=run)
    if run.error:
        outcome.problem = run.error
        return outcome
    if not run.ok:
        outcome.problem = "the harness produced no scenarios for %s" % commit.sha
        return outcome

    outcome.family = run.family(commit.family_prefix)
    if len(outcome.family) < commit.min_family_size:
        outcome.problem = (
            "the %r family has %d scenario(s), fewer than the %d this backtest requires. "
            "The coverage that catches %s has been deleted or renamed out from under it; "
            "restore it, or change family_prefix here and say why in the commit message."
            % (
                commit.family_prefix,
                len(outcome.family),
                commit.min_family_size,
                commit.label,
            )
        )
        return outcome

    for name in outcome.family:
        hit = sorted(set(run.findings[name]) & commit.required_any)
        if hit:
            outcome.caught[name] = hit
        else:
            outcome.missed.append(name)

    # --- the pins ---------------------------------------------------------- #
    # Checked before the family count, because they are the specific statement
    # and the family count is the fallback.  A failure here names the scenario
    # and the diagnosis, which is the whole reason the pins exist: "the family
    # caught something" survived moving the entire family off its boundary.
    for name, codes in commit.required_by_scenario:
        if name not in run.findings:
            outcome.pinned_failures.append(
                "%s is NOT IN THE LIBRARY at all (renamed or deleted); it was pinned "
                "here because it is the case that reports %s on %s"
                % (name, "/".join(sorted(codes)), commit.sha)
            )
            continue
        hit = sorted(set(run.findings[name]) & codes)
        if hit:
            outcome.pinned_ok.append(name)
        else:
            outcome.pinned_failures.append(
                "%s reported %s but NOT any of %s"
                % (
                    name,
                    ", ".join(sorted(run.findings[name])) or "nothing (it PASSED)",
                    "/".join(sorted(codes)),
                )
            )
    for name, codes in commit.forbidden_by_scenario:
        if name not in run.findings:
            outcome.straddle_failures.append(
                "%s is NOT IN THE LIBRARY at all; it is the case OUTSIDE the failure "
                "region, and without it the region has no measured upper edge" % name
            )
            continue
        hit = sorted(set(run.findings[name]) & codes)
        if hit:
            outcome.straddle_failures.append(
                "%s reported %s, but it sits OUTSIDE the measured failure region and "
                "must not" % (name, ", ".join(hit))
            )

    problems: List[str] = []
    if outcome.pinned_failures:
        problems.append(
            "%d pinned scenario(s) no longer produce the diagnosis they were pinned "
            "for on %s (%s). A prefix-matched family cannot say this: it is satisfied "
            "by ANY member catching ANYTHING, which is how moving the whole "
            "constant_range family from 12/20/30/40 m to 55/60/65/68 m once left this "
            "backtest green. Do not adjust the pins; the change that made this red "
            "either moved a scenario off its boundary or broke a detector.\n      %s"
            % (
                len(outcome.pinned_failures),
                commit.sha,
                commit.label,
                "\n      ".join(outcome.pinned_failures),
            )
        )
    if outcome.straddle_failures:
        problems.append(
            "%d scenario(s) OUTSIDE the measured failure region reported it anyway. A "
            "boundary needs both edges: a family that fails everywhere has measured "
            "the grid, not the defect.\n      %s"
            % (len(outcome.straddle_failures), "\n      ".join(outcome.straddle_failures))
        )
    if len(outcome.caught) < commit.min_caught:
        problems.append(
            "only %d of the %d scenario(s) in the %r family reported any of %s, and %d "
            "are required. The harness has stopped being a specification: either the "
            "scenarios have drifted off the failure boundary, or the detector for these "
            "diagnoses no longer fires."
            % (
                len(outcome.caught),
                len(outcome.family),
                commit.family_prefix,
                ", ".join(sorted(commit.required_any)),
                commit.min_caught,
            )
        )
    outcome.problem = "\n  PROBLEM: ".join(problems)
    return outcome


def backtest_all(
    root: str = REPO_ROOT,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    keep_worktrees: bool = False,
) -> List[BacktestOutcome]:
    """Backtest every commit in :data:`KNOWN_BROKEN`, in order."""
    return [
        backtest_commit(c, root=root, timeout_s=timeout_s, keep_worktree=keep_worktrees)
        for c in KNOWN_BROKEN
    ]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print the backtest verdict for every known-broken commit."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verbose", action="store_true", help="print each harness table too")
    parser.add_argument(
        "--keep-worktrees", action="store_true", help="leave the temporary worktrees on disk"
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds per harness run"
    )
    args = parser.parse_args(argv)

    reason = unavailable_reason()
    if reason is not None:
        print("SKIP: %s" % reason)
        return 0

    print("=" * 100)
    print("HARNESS BACKTEST -- does this specification still catch the defects we measured?")
    print("=" * 100)
    outcomes = backtest_all(timeout_s=args.timeout, keep_worktrees=args.keep_worktrees)
    for outcome in outcomes:
        if args.verbose:
            print(outcome.run.table)
        print(outcome.describe())
        print("  VERDICT: %s" % ("caught" if outcome.satisfied else "NOT CAUGHT"))
        print("-" * 100)
    bad = [o for o in outcomes if not o.satisfied]
    print("%d of %d known defects still caught" % (len(outcomes) - len(bad), len(outcomes)))
    return 1 if bad else 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
