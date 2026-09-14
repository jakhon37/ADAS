"""Longitudinal safety envelope sweep, judged against the harness truth oracle.

What this is
------------
An executable specification for the longitudinal behaviour of the safety
arbiter over its whole operating envelope.  Single scenarios missed every one of
the three historical blockers.  A sweep found all three, because the interesting
output of a sweep is not a pass count but a *boundary*: "it fires for any
in-path track closer than 26 m at 15 m/s" is a diagnosis; "312/336 passed" is
not.

Nothing here reads the arbiter's own thresholds.  The physics comes from
:mod:`tests.scenarios.plant` (the vehicle) and :mod:`tests.scenarios.oracle`
(what should have happened), which are shared with the scenario suite so that
there is exactly one specification and not two.  This module adds only what is
specific to a sweep: the envelope grid, the two-pass execution, the region
classification, and the reduction of failing cells to region *boundaries*.

The three instants a cell is graded against
-------------------------------------------
``warrant``
    The first frame at which the situation can no longer be recovered by
    ``oracle.COMFORT_DECEL_MPS2`` (3.0 m/s^2) while preserving
    ``oracle.REQUIRED_CLEARANCE_M`` (2.0 m).  Before this frame an *emergency*
    intervention is not warranted: ordinary following control still has the
    situation.  Braking hard before it is EARLY; braking hard when it never
    arrives at all is a PHANTOM.  This uses the oracle's causal assumption --
    the lead holds its current acceleration -- so it never demands clairvoyance.
``mandate``
    The last frame from which full braking, **through the real plant including
    its 0.15 s brake rise time**, still preserves that 2 m clearance.  This is
    the last defensible moment to intervene, and it is a fact about the vehicle
    rather than a tuning constant: intervening after it means the car can no
    longer stop with room to spare.  Intervening later than this is LATE.
``lost``
    The first frame from which even full braking through the real plant cannot
    avoid contact.  Past it the collision is arithmetic.

Splitting the requirement from the vehicle this way is deliberate and follows
:mod:`tests.scenarios.oracle`: ``warrant`` is idealised kinematics, because it
asks what the *situation* demands; ``mandate`` and ``lost`` run the actual
actuator, because they ask what the *car* can still do.

Three independent gradings, not one
-----------------------------------
``verdict``
    The EMERGENCY grading: did the arbiter apply emergency authority when and
    only when the kinematics demanded it, and did the closed loop avoid contact?
``headway_verdict``
    The SOFT grading against :func:`safe_following_gap_m`.
``band_verdict``
    The SUB-EMERGENCY grading, added after the backtest found the band
    unpoliced.  Everything between ``comfort_decel_mps2`` (3.0 m/s^2) and
    ``emergency_decel_mps2`` (3.5 m/s^2) is invisible to the emergency grading
    by construction -- the emergency test starts at 3.5 -- and invisible to the
    headway grading, which only asks whether *some* response happened.  A system
    can therefore hold 3.4 m/s^2 for a whole run against a lead that never moved
    and score CORRECT / SOFT_OK on every cell.  Measured on the two broken
    commits: 1ce4886 holds 3.0-3.3 m/s^2 on a constant-range follow at every gap
    from 25 m to 45 m at 20 m/s, which is 0.2-0.3 m/s^2 under the emergency
    threshold and entirely unreported before this grading existed.  A cell is
    ``BAND_UNWARRANTED`` when a command in the band lands on a frame at which
    neither an emergency nor an unsafe headway has yet arisen.

The headway requirement is graded separately
--------------------------------------------
Following a lead at 5 m at 30 m/s with a matched speed is unsafe headway.  The
correct response is to open the gap, and answering it with full-authority
braking is the exact confusion that produced the phantom AEB.  So a second,
independent verdict grades the *soft* response against
:func:`safe_following_gap_m`, and the emergency verdict never punishes a cell
for a gentle deceleration.

Two passes per cell
-------------------
OPEN LOOP
    The ego holds ``ego_speed_mps``; the lead follows its script.  The scene
    evolves identically whatever the arbiter says, so the arbiter's *decision*
    is measured against the oracle with the plant out of the way.  This is what
    detects PHANTOM / EARLY / LATE / MISSED.
CLOSED LOOP
    The ego obeys the arbitrated command through :class:`tests.scenarios.plant.Plant`,
    brake lag and all.  This is what detects COLLISION, and it is the pass that
    would have caught fix round 2: an arbiter can look merely LATE open loop and
    still put the car into the lead once the plant is in the circuit.

A cell that collides closed loop is reported as COLLISION whatever the
open-loop verdict was; a collision is the only outcome that cannot be argued
with.

The incoming command
--------------------
Every frame is handed ``ControlCommand(throttle=0.4, brake=0.0)``: a plain
cruise request with **no brake in it at all**.  Any brake in the arbitrated
output is therefore the arbiter's own, which is what makes the
demand-versus-command distinction (see :mod:`tests.scenarios.instrument`)
unambiguous here.  On real footage, where the planner is also braking, that
distinction has to be made explicitly; see :mod:`tests.scenarios.footage`.

Determinism
-----------
No wall clock, no randomness, no I/O, no GPU.  Cells are generated in sorted
order and every cell builds a fresh arbiter, so a cell's result cannot depend on
the cell before it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from tests.scenarios.oracle import (
    COMFORT_DECEL_MPS2,
    CONTACT_GAP_M,
    EMERGENCY_DECEL_MPS2,
    REQUIRED_CLEARANCE_M,
    full_braking_min_gap,
    min_gap_under_constant_decel,
    required_decel_mps2,
    travel_m,
)
from tests.scenarios.plant import (
    DEFAULT_PLANT,
    DT_S,
    MAX_BRAKE_DECEL_MPS2,
    LeadSpec,
    Plant,
    WorldState,
    lead_brakes,
    lead_constant_speed,
    render_box,
)

__all__ = [
    "SweepSpec",
    "SweepGrid",
    "PROFILES",
    "CellSpec",
    "CellResult",
    "Verdict",
    "safe_following_gap_m",
    "required_decel_mps2",
    "warranted_now",
    "run_sweep",
    "classify_cell",
    "classify_headway",
    "classify_band",
    "BAND_FAILURES",
    "BAND_CHARS",
    "region_boundaries",
    "render_grids",
    "summarise",
]


# --------------------------------------------------------------------------- #
# The specification constants
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SweepSpec:
    """Sweep-specific thresholds.  The physics constants come from the core.

    Every default here either *is* a core constant (re-stated so the report can
    print it) or is specific to sweeping an envelope.  None of them is read from
    :class:`adas.control.arbiter.ArbiterLimits`; the arbiter is the thing under
    test, not the source of the specification.
    """

    standstill_gap_m: float = REQUIRED_CLEARANCE_M
    """Clearance a correct intervention preserves.  ``oracle.REQUIRED_CLEARANCE_M``."""
    comfort_decel_mps2: float = COMFORT_DECEL_MPS2
    """Boundary between headway keeping and collision avoidance.  Core constant."""
    emergency_decel_mps2: float = EMERGENCY_DECEL_MPS2
    """At or above this an arbitrated command counts as an emergency
    intervention.  ``oracle.EMERGENCY_DECEL_MPS2`` (3.5 m/s^2), half a metre
    above comfort so a comfort-limited ramp cannot be mistaken for an AEB."""
    brake_authority_mps2: float = MAX_BRAKE_DECEL_MPS2
    accel_authority_mps2: float = DEFAULT_PLANT.max_accel_mps2

    reaction_s: float = 0.6
    """Reaction latency used by the HEADWAY requirement only.

    Hazard-observable to full brake authority for a 20 Hz pipeline: one frame to
    perceive, two or three to corroborate a range rate, ~0.15 s for the brake to
    build, plus margin.  It is also the classical driver reaction figure, which
    is what a following-distance rule is written against.  It deliberately does
    NOT appear in the emergency tests: there, latency is modelled by running the
    real actuator (see ``mandate`` in the module docstring), not by a constant.
    """

    band_frames_min: int = 3
    """Frames of sub-emergency braking before the band grading calls it a
    policy rather than a transient.

    Three frames is 0.15 s, the plant's own brake rise time: a command cannot
    produce its deceleration faster than that, so anything shorter is the edge
    of a ramp and anything longer is a deliberate hold.
    """

    soft_decel_mps2: float = 0.4
    """An arbitrated deceleration above this counts as *some* longitudinal
    intervention: 0.05 of brake against 8 m/s^2 of authority."""

    dt_s: float = DT_S
    horizon_s: float = 12.0
    """How long the ARBITER is exercised for, seconds."""
    oracle_horizon_s: float = 150.0
    """How far the truth oracle looks ahead when deciding whether an emergency
    EVER arises, seconds.  Deliberately much longer than ``horizon_s``: whether
    an emergency exists is a fact about the scene, not about how long we chose
    to run the arbiter.  Truncating it would turn a slow closure into a fake
    "no emergency exists" and every early brake into a fake PHANTOM."""

    allow_oncoming: bool = False
    """Cells whose implied lead speed is negative are head-on encounters, which
    are outside a highway-following ODD; they are marked INFEASIBLE."""

    hard_states: Tuple[str, ...] = ("min_risk_maneuver", "disengage")
    """States that are an emergency intervention in their own right, whatever
    the pedal says.  A minimum-risk manoeuvre is not a graded response."""

    @property
    def max_frames(self) -> int:
        """Arbitration episode length in frames."""
        return int(round(self.horizon_s / self.dt_s))

    @property
    def oracle_frames_cap(self) -> int:
        """Oracle look-ahead in frames."""
        return int(round(self.oracle_horizon_s / self.dt_s))


# --------------------------------------------------------------------------- #
# The one requirement the core oracle does not carry
# --------------------------------------------------------------------------- #


def safe_following_gap_m(spec: SweepSpec, ego_speed_mps: float, lead_speed_mps: float) -> float:
    """RSS-style safe standing gap for a speed pair, metres.

    The gap at which the ego could still stop behind a lead that begins braking
    at full authority this instant::

        d_safe = v_e * tau + v_e^2 / (2 a) - v_l^2 / (2 a)

    Both vehicles are given the same braking capability, which is the
    conservative-but-standard RSS choice: if the lead can out-brake the ego, no
    finite gap is safe and the quantity is meaningless.

    This is the *headway* requirement.  It is deliberately not an emergency
    test, and it is the only place a reaction-time constant appears: a following
    rule has to reserve the distance covered before anything happens, whereas the
    emergency tests model that latency by running the real actuator instead.
    """
    a = spec.brake_authority_mps2
    v_e = max(0.0, ego_speed_mps)
    v_l = max(0.0, lead_speed_mps)
    return max(
        spec.standstill_gap_m,
        v_e * spec.reaction_s + (v_e * v_e) / (2.0 * a) - (v_l * v_l) / (2.0 * a),
    )


def warranted_now(
    spec: SweepSpec,
    gap_m: float,
    ego_v_mps: float,
    lead_v_mps: float,
    lead_a_mps2: float,
) -> bool:
    """Is an emergency (beyond-comfort) intervention warranted at this instant?

    Equivalent to ``oracle.required_decel_mps2(...) >= comfort_decel_mps2`` but
    without the bisection: it asks the oracle's min-gap function the single
    question that matters, "does braking at the comfort limit still keep the
    clearance?", which is one closed-form evaluation instead of forty-eight.
    The sweep asks this at every frame of every cell, so the difference is the
    difference between a two-minute run and an hour.
    """
    if not math.isfinite(gap_m):
        return False
    closing = ego_v_mps - lead_v_mps
    if closing <= 1e-6 and lead_a_mps2 >= -1e-9:
        return False
    target = min(spec.standstill_gap_m, max(0.0, gap_m - 0.05))
    worst = min_gap_under_constant_decel(
        gap_m, ego_v_mps, lead_v_mps, lead_a_mps2, spec.comfort_decel_mps2
    )
    return worst < target


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SweepGrid:
    """The axes of the envelope.

    ``relative_rate_mps`` is the *initial* closing rate: negative closes.
    ``lead_decel_mps2`` is the lead's own braking, applied from frame 0.  The
    combination ``(rate = 0, decel = 6)`` is the scenario fix round 2 collided
    on, and ``(rate = 0, decel = 0)`` is the constant-range scene fix round 1
    phantom-braked for; both must be in every profile.
    """

    name: str
    ego_speeds_mps: Tuple[float, ...]
    ranges_m: Tuple[float, ...]
    relative_rates_mps: Tuple[float, ...]
    lead_decels_mps2: Tuple[float, ...] = (0.0, 6.0)
    horizon_s: Optional[float] = None

    def cells(self) -> List["CellSpec"]:
        """Every cell, in a fixed sorted order."""
        out: List[CellSpec] = []
        for v in sorted(self.ego_speeds_mps):
            for d in sorted(self.ranges_m):
                for rate in sorted(self.relative_rates_mps):
                    for decel in sorted(self.lead_decels_mps2):
                        out.append(
                            CellSpec(
                                ego_speed_mps=float(v),
                                range_m=float(d),
                                relative_rate_mps=float(rate),
                                lead_decel_mps2=float(decel),
                            )
                        )
        return out


#: Three resolutions, all placed against the boundaries this envelope is known
#: to contain.  Every profile keeps at least one range either side of each
#: measured boundary, because a grid whose sample points all sit inside (or all
#: outside) a failure region reports its width as zero or as the width of the
#: grid, and both are wrong.
#:
#: Measured boundaries the range axes straddle (see ``docs/SAFETY_SWEEP.md``).
#: All of them were RE-MEASURED after the ``velocity_mps`` contract fix (see
#: :func:`closing_mps`); every one is unchanged, because none of the three
#: committed arbiters reads that field.
#:
#: * constant-range phantom on 25e3ba5 -- fires up to 43 m at 20 m/s, 26 m at
#:   15 m/s, 66 m at 25 m/s, and not at all beyond; hence 26/32/40/44/52/70.
#: * lead braking at 6 m/s^2 on 1ce4886 -- avoidable contact for 11-37 m at
#:   20 m/s (16-26 m closed loop through the full stack); hence 12/16/26/32/40.
#: * stationary lead -- the plant's own full-authority stop is 14.70 m at
#:   15 m/s, 25.85 m at 20 m/s and 40.13 m at 25 m/s, so contact closer than
#:   that is arithmetic; hence 12/16/26/40/44.
#: * the FAR edge, added in this revision because the tool itself reported two
#:   regions running off the top of the axis ("EARLY for range 16-70 at ego
#:   20 m/s ... the real boundary is beyond 70 m").  Both were measured, and
#:   both are real boundaries rather than artefacts:
#:
#:   - EARLY against a lead 8 m/s slower, no lead braking, 8 s window: the
#:     arbiter stops firing inside the window at **82 m at 20 m/s and 89 m at
#:     25 m/s** (identical on both broken commits); hence 80 inside and 90
#:     outside on the ``fast`` axis.  At the 12 s window of ``standard`` the
#:     same edge moves out to 120 m at 20 m/s and between 120 and 140 m at
#:     25-30 m/s, so ``standard`` carries 120 inside and 150 outside.
#:   - LATE at ego 25 m/s against a lead braking at 6 m/s^2, 8 s window: the
#:     last late cell is **76 m at rate +0 and 70 m at rate -8**; hence 70
#:     inside and 80 outside.
#:   - BAND_UNWARRANTED at ego 10 m/s against a lead braking at 6 m/s^2, 12 s
#:     window: held from 16 m out to **between 120 and 140 m**; hence 120
#:     inside and 150 outside on ``standard``.
#:
#: Nothing on any axis now runs off the top of the grid on either broken
#: commit, so every failure region the tool reports is bounded on both sides
#: and its width is a measurement rather than the width of the grid.
#:
#: ``fast`` is the CI resolution: 120 cells, about fifteen seconds, and it
#: contains a straddling pair for every boundary above.  ``standard`` is the
#: default and adds the low and high ends of the envelope and the intermediate
#: closing rates.  ``dense`` halves the ego and range steps for boundary-finding
#: by hand.
PROFILES: Dict[str, "SweepGrid"] = {
    "fast": SweepGrid(
        name="fast",
        ego_speeds_mps=(15.0, 20.0, 25.0),
        ranges_m=(12.0, 16.0, 26.0, 32.0, 40.0, 44.0, 52.0, 70.0, 80.0, 90.0),
        relative_rates_mps=(0.0, -8.0),
        lead_decels_mps2=(0.0, 6.0),
        horizon_s=8.0,
    ),
    "standard": SweepGrid(
        name="standard",
        ego_speeds_mps=(5.0, 10.0, 15.0, 20.0, 25.0, 30.0),
        ranges_m=(
            5.0, 8.0, 12.0, 16.0, 20.0, 26.0, 30.0, 32.0, 40.0, 43.0, 44.0,
            52.0, 60.0, 70.0, 80.0, 90.0, 120.0, 150.0,
        ),
        relative_rates_mps=(3.0, 0.0, -1.0, -2.0, -4.0, -8.0, -15.0),
        lead_decels_mps2=(0.0, 6.0),
    ),
    "dense": SweepGrid(
        name="dense",
        ego_speeds_mps=(5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0),
        ranges_m=(
            4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0,
            25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0, 35.0, 40.0, 42.0, 43.0,
            44.0, 45.0, 52.0, 60.0, 66.0, 70.0, 80.0, 90.0, 100.0, 120.0, 150.0,
        ),
        relative_rates_mps=(3.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -11.0, -15.0),
        lead_decels_mps2=(0.0, 3.0, 6.0),
    ),
}


@dataclass(frozen=True)
class CellSpec:
    """One point of the envelope."""

    ego_speed_mps: float
    range_m: float
    relative_rate_mps: float
    lead_decel_mps2: float

    @property
    def lead_speed_mps(self) -> float:
        """Implied initial lead speed."""
        return self.ego_speed_mps + self.relative_rate_mps

    @property
    def key(self) -> Tuple[float, float, float, float]:
        """Hashable, sortable identity."""
        return (
            self.ego_speed_mps,
            self.range_m,
            self.relative_rate_mps,
            self.lead_decel_mps2,
        )

    def lead_spec(self) -> LeadSpec:
        """The :class:`tests.scenarios.plant.LeadSpec` this cell describes."""
        speed = max(0.0, self.lead_speed_mps)
        accel_fn = (
            lead_brakes(self.lead_decel_mps2)
            if self.lead_decel_mps2 > 0.0
            else lead_constant_speed()
        )
        return LeadSpec(
            initial_gap_m=self.range_m,
            initial_speed_mps=speed,
            accel_fn=accel_fn,
            label="sweep rate %+g decel %g" % (self.relative_rate_mps, self.lead_decel_mps2),
        )

    def label(self) -> str:
        """One-line human identity."""
        return "ego %5.1f  range %5.1f  rate %+5.1f  lead_decel %.1f" % (
            self.ego_speed_mps,
            self.range_m,
            self.relative_rate_mps,
            self.lead_decel_mps2,
        )


class Verdict(object):
    """Cell outcome codes.  Plain strings so results serialise as-is."""

    CORRECT = "CORRECT"
    PHANTOM = "PHANTOM"
    EARLY = "EARLY"
    LATE = "LATE"
    MISSED = "MISSED"
    COLLISION = "COLLISION"
    COLLISION_UNAVOIDABLE = "COLLISION_UNAVOIDABLE"
    """Contact that was ALREADY unavoidable on the cell's first frame.

    A statement about the CELL, not about the system, and the exact counterpart
    of :data:`tests.scenarios.scenario.SCENARIO_DEFECT_CODES`'s
    ``collided_unavoidable`` in the readable suite -- which the sweep lacked, so
    the sweep's own floor was non-zero and ``--gate`` could not return 0 for any
    design whatever.

    The test is the oracle's own: ``lost_frame == 0``, i.e.
    :func:`tests.scenarios.oracle.full_braking_min_gap` from frame 0 already
    contacts.  That counterfactual is an OMNISCIENT controller -- it commits full
    authority on the first frame, before any measurement of the lead's motion
    could exist -- driving the same plant with the same 0.15 s brake rise and the
    same 20 m/s^3 jerk ceiling this specification's ``excess_jerk`` requirement
    imposes.  Nothing a compliant system can do beats it, so grading such a cell
    as a failure is the sweep manufacturing a bug report out of its own
    arithmetic, which is precisely what section 3a/B3 of the specification
    forbids for the scenario library.

    It is NOT counted as a failure, and it is reported separately rather than
    folded into CORRECT, so that a design change which moves a cell across this
    boundary is visible in the counts.
    """
    INFEASIBLE = "INFEASIBLE"

    ORDER = (
        COLLISION,
        MISSED,
        PHANTOM,
        LATE,
        EARLY,
        COLLISION_UNAVOIDABLE,
        CORRECT,
        INFEASIBLE,
    )
    CHARS = {
        CORRECT: ".",
        PHANTOM: "P",
        EARLY: "E",
        LATE: "L",
        MISSED: "M",
        COLLISION: "X",
        COLLISION_UNAVOIDABLE: "u",
        INFEASIBLE: " ",
    }
    FAILURES = (COLLISION, MISSED, PHANTOM, LATE, EARLY)


@dataclass
class CellResult:
    """Everything the sweep learned about one cell."""

    cell: CellSpec
    verdict: str = Verdict.CORRECT

    # ---- oracle -----------------------------------------------------------
    warrant_frame: Optional[int] = None
    mandate_frame: Optional[int] = None
    lost_frame: Optional[int] = None
    headway_unsafe_frame: Optional[int] = None
    headway_unsafe_at_start: bool = False
    headway_verdict: str = "SOFT_OK"
    """Secondary grading of the *soft* response against
    :func:`safe_following_gap_m`.  Deliberately separate from ``verdict``: an
    unsafe headway calls for opening the gap, never for an AEB."""
    initial_required_decel_mps2: float = 0.0

    # ---- open loop --------------------------------------------------------
    hard_frame: Optional[int] = None
    soft_frame: Optional[int] = None
    open_frames: int = 0
    max_decel_open_mps2: float = 0.0
    """Peak deceleration the arbiter COMMANDED, m/s^2 (``brake * authority``)."""
    max_demand_open_mps2: float = -1.0
    """Peak deceleration the arbiter DEMANDED of itself, m/s^2.  Needs the
    instrument; ``-1.0`` when the demand hook could not attach."""
    first_hard_state: str = ""
    first_hard_findings: List[str] = field(default_factory=list)
    hard_with_inferred_rate: bool = False
    """True when the first hard intervention happened on a frame where the
    arbiter's closing rate was still the seeded prior, not a measurement."""
    states_seen: Dict[str, int] = field(default_factory=dict)

    # ---- sub-emergency band ------------------------------------------------
    band_frames: int = 0
    """Frames whose commanded deceleration lay in
    ``[comfort_decel_mps2, emergency_decel_mps2)`` without a hard state."""
    first_band_frame: Optional[int] = None
    """First such frame, or ``None``."""
    max_band_decel_mps2: float = 0.0
    """Largest deceleration commanded inside the band."""
    band_verdict: str = "BAND_OK"
    """Grading of the sub-emergency band; see :func:`classify_band`."""

    # ---- closed loop ------------------------------------------------------
    collided: bool = False
    min_gap_m: float = float("inf")
    final_ego_speed_mps: float = 0.0
    closed_frames: int = 0
    stopped: bool = False
    """The ego reached a standstill.  A stop on a cell where no emergency ever
    arises is the plant-level signature of a phantom brake."""

    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """JSON-ready."""
        return {
            "ego_speed_mps": self.cell.ego_speed_mps,
            "range_m": self.cell.range_m,
            "relative_rate_mps": self.cell.relative_rate_mps,
            "lead_decel_mps2": self.cell.lead_decel_mps2,
            "verdict": self.verdict,
            "headway_verdict": self.headway_verdict,
            "warrant_frame": self.warrant_frame,
            "mandate_frame": self.mandate_frame,
            "lost_frame": self.lost_frame,
            "headway_unsafe_frame": self.headway_unsafe_frame,
            "hard_frame": self.hard_frame,
            "soft_frame": self.soft_frame,
            "headway_unsafe_at_start": self.headway_unsafe_at_start,
            "initial_required_decel_mps2": _finite(self.initial_required_decel_mps2),
            "max_decel_open_mps2": self.max_decel_open_mps2,
            "max_demand_open_mps2": self.max_demand_open_mps2,
            "first_hard_state": self.first_hard_state,
            "first_hard_findings": self.first_hard_findings,
            "hard_with_inferred_rate": self.hard_with_inferred_rate,
            "states_seen": self.states_seen,
            "band_verdict": self.band_verdict,
            "band_frames": self.band_frames,
            "first_band_frame": self.first_band_frame,
            "max_band_decel_mps2": self.max_band_decel_mps2,
            "collided": self.collided,
            "min_gap_m": _finite(self.min_gap_m),
            "final_ego_speed_mps": self.final_ego_speed_mps,
            "closed_frames": self.closed_frames,
            "stopped": self.stopped,
            "notes": self.notes,
        }


def _finite(value: float) -> Optional[float]:
    """``inf``/``nan`` -> ``None``, so the JSON is valid without ``Infinity``."""
    return None if (value is None or not math.isfinite(value)) else value


# --------------------------------------------------------------------------- #
# The oracle, applied to a cell
# --------------------------------------------------------------------------- #


def open_loop_states(spec: SweepSpec, cell: CellSpec, frames: int) -> List[WorldState]:
    """The TRUE state at each frame with the ego holding speed.

    Closed form, using :func:`tests.scenarios.oracle.travel_m` for both bodies,
    so the open-loop truth cannot drift from the oracle's own arithmetic.  The
    list stops at contact.
    """
    out: List[WorldState] = []
    v_e = cell.ego_speed_mps
    v_l0 = max(0.0, cell.lead_speed_mps)
    a_l = -cell.lead_decel_mps2
    for i in range(frames):
        t = i * spec.dt_s
        ego_x = v_e * t
        lead_x = cell.range_m + travel_m(v_l0, a_l, t)
        gap = lead_x - ego_x
        if gap <= CONTACT_GAP_M:
            break
        lead_v = max(0.0, v_l0 + a_l * t)
        out.append(
            WorldState(
                frame=i,
                t_s=t,
                ego_x_m=ego_x,
                ego_v_mps=v_e,
                ego_a_mps2=0.0,
                lead_present=True,
                lead_x_m=lead_x,
                lead_v_mps=lead_v,
                lead_a_mps2=(a_l if lead_v > 0.0 else 0.0),
            )
        )
    return out


def _last_true(predicate: Callable[[int], bool], n: int) -> Optional[int]:
    """Largest ``i < n`` with ``predicate(i)``, assuming it is monotone.

    The predicates used here ("full braking from frame i still preserves the
    clearance", "... still avoids contact") can only get worse as the ego closes
    on the lead in an open-loop episode, so a binary search is exact and turns a
    per-frame plant simulation into eight of them.  The endpoints are checked
    explicitly, so a cell that never satisfies the predicate returns ``None``
    rather than a wrong index.
    """
    if n <= 0 or not predicate(0):
        return None
    lo, hi = 0, n - 1
    if predicate(hi):
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if predicate(mid):
            lo = mid
        else:
            hi = mid
    return lo


def oracle_frames(
    spec: SweepSpec, cell: CellSpec, states: Sequence[WorldState]
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """``(warrant, mandate, lost, headway_unsafe)`` frame indices for a cell.

    ``warrant`` and ``headway_unsafe`` are searched over the whole physical
    future (``spec.oracle_horizon_s``), because whether an emergency ever arises
    is a fact about the scene, not about how long the arbiter was run.
    ``mandate`` and ``lost`` are searched over the frames the arbiter was
    actually exercised on, because they exist to grade its timing, and they run
    the REAL plant so that the brake rise time is part of the answer.
    """
    v_e = cell.ego_speed_mps
    v_l0 = max(0.0, cell.lead_speed_mps)
    a_l = -cell.lead_decel_mps2

    warrant: Optional[int] = None
    headway: Optional[int] = None
    for i in range(spec.oracle_frames_cap):
        t = i * spec.dt_s
        gap = cell.range_m + travel_m(v_l0, a_l, t) - v_e * t
        if gap <= CONTACT_GAP_M:
            break
        lead_v = max(0.0, v_l0 + a_l * t)
        if headway is None and gap < safe_following_gap_m(spec, v_e, lead_v):
            headway = i
        if warrant is None and warranted_now(spec, gap, v_e, lead_v, a_l if lead_v > 0 else 0.0):
            warrant = i
        if warrant is not None and headway is not None:
            break
        if a_l >= 0.0 and lead_v >= v_e:
            # Constant-speed lead no slower than the ego: the gap never
            # decreases, so nothing further can ever be lost.  Frame 0 settled it.
            break
        if v_e <= 0.0 and lead_v <= 0.0:
            break

    lead = cell.lead_spec()
    n = len(states)

    def keeps_clearance(i: int) -> bool:
        return full_braking_min_gap(states[i], lead) >= spec.standstill_gap_m

    def avoids_contact(i: int) -> bool:
        return full_braking_min_gap(states[i], lead) > CONTACT_GAP_M

    mandate = _last_true(keeps_clearance, n)
    last_avoidable = _last_true(avoids_contact, n)
    lost = None
    if last_avoidable is None:
        lost = 0 if n else None
    elif last_avoidable < n - 1:
        lost = last_avoidable + 1
    return warrant, mandate, lost, headway


# --------------------------------------------------------------------------- #
# Running a cell
# --------------------------------------------------------------------------- #


class _Adas(object):
    """Lazy holder for the ``adas`` symbols, so this module imports GPU-free."""

    def __init__(self) -> None:
        from adas.control.arbiter import SafetyArbiter, SafetyContext
        from adas.core.models import (
            ControlCommand,
            EgoState,
            MotionPlan,
            PerceptionStatus,
            TrackedObject,
        )

        self.SafetyArbiter = SafetyArbiter
        self.SafetyContext = SafetyContext
        self.ControlCommand = ControlCommand
        self.EgoState = EgoState
        self.MotionPlan = MotionPlan
        self.PerceptionStatus = PerceptionStatus
        self.TrackedObject = TrackedObject


def closing_mps(ego_v: float, lead_v: float) -> float:
    """The quantity ``TrackedObject.velocity_mps`` is defined to carry.

    The production contract is stated at ``adas.tracking.tracker`` (see
    ``MultiObjectTracker._to_tracked_object`` and ``time_to_collision_s``):
    ``velocity_mps`` is a **range rate with the positive-when-closing
    convention** -- the negated derivative of the measured range -- and NOT the
    lead's absolute speed over the ground.  A camera measures range; it has no
    way to know a lead's ground speed at all, and every consumer downstream
    divides this number into a gap to get a time.

    Until this function existed the sweep put ``lead_v`` (the lead's absolute
    speed) straight into that field, so on the cell that matters most -- a
    matched-speed follow, true closing rate exactly zero -- the arbiter was
    handed "closing at 20 m/s" and a TTC of one second.  That is not a
    measurement of the scene the oracle graded; it is a different scene, and
    the numbers the gate printed for it were partly a measurement of the
    harness.  The plant's own sensor model has always reported the same
    quantity (``Sensor._tracks_injected`` differentiates measured range); this
    makes the sweep agree with it.

    Args:
        ego_v: Ego ground speed, m/s.
        lead_v: Lead ground speed, m/s.

    Returns:
        ``ego_v - lead_v``: positive while the gap shrinks, negative while it
        opens.
    """
    return float(ego_v) - float(lead_v)


def _ttc_s(gap_m: float, closing: float) -> float:
    """Contact time for a gap and a positive-when-closing rate, seconds.

    The same degenerate cases the production tracker declares in
    ``time_to_collision_s``: an opening or negligible closure is ``inf`` ("not
    closing"), never a huge finite number that reads as a real time to a
    downstream threshold, and a gap already at zero is ``0.0``.  A pure contact
    time, with no standstill gap subtracted -- consumers that want one subtract
    it themselves.
    """
    if closing <= 1e-3:
        return float("inf")
    if gap_m <= 0.0:
        return 0.0
    return gap_m / closing


def _context(api: _Adas, spec: SweepSpec, frame: int, ego_v: float, gap: float, lead_v: float):
    """One :class:`SafetyContext` for a single in-path lead.

    The bounding box comes from :func:`tests.scenarios.plant.render_box`, the
    same projection the scenario suite uses, so the arbiter's image-space
    in-path gate sees geometry consistent with the range it is given.

    ``velocity_mps`` and ``ttc_s`` are filled from :func:`closing_mps` and
    :func:`_ttc_s`, so the three numbers on the track (range, range rate, time
    to contact) are mutually consistent and all three mean what the production
    stack defines them to mean.  ``lead_v`` is the lead's ground speed and is
    converted here; it is deliberately not passed through raw.
    """
    closing = closing_mps(ego_v, lead_v)
    track = api.TrackedObject(
        track_id=1,
        box=render_box(gap),
        velocity_mps=closing,
        distance_m=max(0.3, gap),
        age_frames=frame + 1,
        hits=frame + 1,
        time_since_update=0,
        ttc_s=_ttc_s(gap, closing),
        in_ego_lane=True,
    )
    return api.SafetyContext(
        ego=api.EgoState(speed_mps=ego_v, valid=True, timestamp_s=frame * spec.dt_s),
        tracks=[track],
        perception=api.PerceptionStatus(ok=True),
        dt_s=spec.dt_s,
        timestamp_s=frame * spec.dt_s,
        frame_width_px=1280,
        frame_height_px=720,
        ego_lane_half_width_frac=0.30,
    )


def _run_open_loop(
    api: _Adas,
    spec: SweepSpec,
    cell: CellSpec,
    states: Sequence[WorldState],
    instrument,
) -> Dict[str, object]:
    """Ego holds speed; measure only what the arbiter decides."""
    arb = api.SafetyArbiter()
    plan = api.MotionPlan(cell.ego_speed_mps, 0.0, "cruise")
    cmd = api.ControlCommand(0.4, 0.0, 0.0)

    hard_frame: Optional[int] = None
    soft_frame: Optional[int] = None
    band_frames = 0
    first_band_frame: Optional[int] = None
    max_band = 0.0
    max_decel = 0.0
    max_demand = -1.0
    first_state = ""
    first_findings: List[str] = []
    hard_inferred = False
    seen: Dict[str, int] = {}
    if instrument is not None:
        instrument.reset()

    for i, state in enumerate(states):
        ctx = _context(api, spec, i, state.ego_v_mps, state.gap_m, state.lead_v_mps)
        result = arb.arbitrate(plan, cmd, ctx)
        name = result.state.value
        seen[name] = seen.get(name, 0) + 1
        decel = result.command.brake * spec.brake_authority_mps2
        max_decel = max(max_decel, decel)
        hard = name in spec.hard_states or decel >= spec.emergency_decel_mps2
        soft = hard or name == "limited" or decel > spec.soft_decel_mps2
        if (not hard) and spec.comfort_decel_mps2 <= decel < spec.emergency_decel_mps2:
            band_frames += 1
            max_band = max(max_band, decel)
            if first_band_frame is None:
                first_band_frame = i
        if soft and soft_frame is None:
            soft_frame = i
        if hard and hard_frame is None:
            hard_frame = i
            first_state = name
            first_findings = list(result.violations)[:6]
            if instrument is not None and instrument.frames:
                lead = instrument.frames[-1].lead
                if lead is not None:
                    hard_inferred = not lead.rate_is_measured
        if instrument is not None and instrument.frames:
            demand = instrument.frames[-1].demanded_decel_mps2
            if demand is not None:
                max_demand = max(max_demand, demand)

    return {
        "hard_frame": hard_frame,
        "soft_frame": soft_frame,
        "frames": len(states),
        "max_decel": max_decel,
        "max_demand": max_demand,
        "first_state": first_state,
        "first_findings": first_findings,
        "hard_inferred": hard_inferred,
        "states": seen,
        "band_frames": band_frames,
        "first_band_frame": first_band_frame,
        "max_band": max_band,
    }


def _run_closed_loop(api: _Adas, spec: SweepSpec, cell: CellSpec, frames: int) -> Dict[str, object]:
    """Ego obeys the arbitrated command through the real plant.

    :class:`tests.scenarios.plant.Plant` is the same vehicle the scenario suite
    and the oracle's avoidability test use, brake rise time included, so a
    collision reported here is a collision by the same physics that says whether
    it was avoidable.
    """
    arb = api.SafetyArbiter()
    plant = Plant(ego_speed_mps=cell.ego_speed_mps, lead=cell.lead_spec())
    plan = api.MotionPlan(cell.ego_speed_mps, 0.0, "cruise")
    cmd_in = api.ControlCommand(0.4, 0.0, 0.0)
    state = plant.state
    min_gap = state.gap_m
    collided = False
    stopped = False
    n = 0
    for i in range(frames):
        if state.gap_m <= CONTACT_GAP_M:
            collided = True
            break
        ctx = _context(api, spec, i, state.ego_v_mps, state.gap_m, state.lead_v_mps)
        out = arb.arbitrate(plan, cmd_in, ctx).command
        state = plant.step(out.throttle, out.brake, 0.0)
        n += 1
        min_gap = min(min_gap, state.gap_m)
        if state.ego_v_mps <= 1e-6:
            stopped = True
        if state.gap_m <= CONTACT_GAP_M:
            collided = True
            break
        if state.gap_m > 200.0:
            break
        if state.ego_v_mps <= 1e-6 and state.lead_v_mps <= 1e-6:
            break
    return {
        "collided": collided,
        "min_gap": min_gap,
        "final_speed": state.ego_v_mps,
        "frames": n,
        "stopped": stopped,
    }


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #


def classify_cell(spec: SweepSpec, result: CellResult) -> str:
    """Grade one cell's EMERGENCY behaviour against the oracle.

    The order is the order of severity:

    ``COLLISION``  the closed-loop pass put the ego into the lead, and the
                   oracle says it was AVOIDABLE.  Overrides every open-loop
                   opinion; an avoidable collision cannot be argued with.
    ``COLLISION_UNAVOIDABLE``
                   the closed-loop pass put the ego into the lead, and full
                   authority committed on frame 0 -- before any measurement of
                   the lead's motion could exist -- would have hit it too.  A
                   statement about the CELL; not a failure.  See
                   :attr:`Verdict.COLLISION_UNAVOIDABLE`.
    ``PHANTOM``    an emergency-grade intervention when no emergency ever arises
                   in this scene at all, however long you wait.
    ``EARLY``      an emergency-grade intervention before the frame at which
                   comfort braking stopped being enough.  Not a loss of safety;
                   an unnecessary emergency, which carries its own rear-end and
                   trust cost, and which is what a phantom looks like on a cell
                   where a real hazard does eventually appear.
    ``MISSED``     an emergency arose inside the exercised window and no
                   emergency-grade intervention ever followed.
    ``LATE``       intervened, but after the last frame from which full braking
                   through the real actuator still preserved the 2 m clearance.
    ``CORRECT``    intervened inside ``[warrant, mandate]``, or correctly did
                   nothing.

    A cell whose emergency arises only *after* the arbitration window is not
    graded for timing: ``CORRECT`` with a note, never ``MISSED``.
    """
    if result.collided:
        if result.lost_frame == 0:
            result.notes.append(
                "contact was ALREADY unavoidable on frame 0: full authority "
                "committed on the first frame, through the same plant and under "
                "the same 20 m/s^3 jerk ceiling, still contacts. Not graded as a "
                "failure -- no compliant system can pass this cell, and grading "
                "it would put the gate's floor above zero."
            )
            return Verdict.COLLISION_UNAVOIDABLE
        return Verdict.COLLISION
    if result.warrant_frame is None:
        # No emergency ever exists in this scene.  Hard braking here is a
        # phantom -- ordinary following control was never out of its depth.
        return Verdict.PHANTOM if result.hard_frame is not None else Verdict.CORRECT
    if result.hard_frame is not None and result.hard_frame < result.warrant_frame:
        return Verdict.EARLY
    if result.hard_frame is None:
        if result.warrant_frame >= result.open_frames:
            result.notes.append(
                "emergency arises at frame %d, after the %d-frame arbitration "
                "window; intervention timing not graded"
                % (result.warrant_frame, result.open_frames)
            )
            return Verdict.CORRECT
        return Verdict.MISSED
    if result.mandate_frame is None:
        result.notes.append(
            "full braking could not preserve the %.1f m clearance from frame 0, so "
            "there is no defensible latest moment; lateness not graded"
            % spec.standstill_gap_m
        )
        return Verdict.CORRECT
    if result.hard_frame > result.mandate_frame:
        return Verdict.LATE
    return Verdict.CORRECT


def classify_headway(spec: SweepSpec, result: CellResult) -> str:
    """Grade the SOFT response against the safe-following-gap requirement.

    ``SOFT_PHANTOM``  some longitudinal intervention on a scene whose headway is
                      safe throughout and in which no emergency ever arises.
    ``SOFT_MISSED``   the headway was unsafe from frame 0 and nothing at all
                      happened inside the arbitration window.
    ``SOFT_OK``       otherwise.
    """
    unsafe = result.headway_unsafe_frame
    if unsafe is None and result.warrant_frame is None:
        return "SOFT_PHANTOM" if result.soft_frame is not None else "SOFT_OK"
    if unsafe == 0 and result.soft_frame is None and result.open_frames > 0:
        return "SOFT_MISSED"
    return "SOFT_OK"


def classify_band(spec: SweepSpec, result: CellResult) -> str:
    """Grade the SUB-EMERGENCY band: braking above comfort but below emergency.

    This band is structurally invisible to the other two gradings.
    :func:`classify_cell` only looks at commands at or above
    ``emergency_decel_mps2``, so 3.4 m/s^2 is not an intervention to it; and
    :func:`classify_headway` only asks whether *any* response happened, so 3.4
    m/s^2 satisfies it. Between them a system can brake harder than any
    passenger would tolerate, for the whole run, against a lead that never
    moved, and be graded CORRECT on every cell. That is what the backtest of the
    scenario library found, and it is what this grading closes.

    ``BAND_UNWARRANTED``
        A command in the band, held for at least ``spec.band_frames_min``
        frames, first appearing on a frame at which NEITHER an emergency has
        arisen (``warrant_frame``) NOR the headway has become unsafe
        (``headway_unsafe_frame``). Nothing in the scene asks for more than
        comfort at that instant, so the excess is the system's own.
    ``BAND_OK``
        Everything else, including a band command that follows a genuine
        headway deficit -- opening a gap firmly is not a fault.

    Args:
        spec: The specification constants.
        result: A cell whose oracle frames and open-loop pass have both run.

    Returns:
        One of the two strings above.
    """
    first = result.first_band_frame
    if first is None or result.band_frames < spec.band_frames_min:
        return "BAND_OK"
    if result.warrant_frame is not None and first >= result.warrant_frame:
        return "BAND_OK"
    if result.headway_unsafe_frame is not None and first >= result.headway_unsafe_frame:
        return "BAND_OK"
    return "BAND_UNWARRANTED"


def run_sweep(
    grid: SweepGrid,
    spec: Optional[SweepSpec] = None,
    use_instrument: bool = True,
    progress=None,
) -> List[CellResult]:
    """Run every cell of ``grid`` and grade it.

    Args:
        grid: The envelope, e.g. ``PROFILES["standard"]``.
        spec: Specification constants; defaults to :class:`SweepSpec`.
        use_instrument: Attach
            :class:`~tests.scenarios.instrument.ArbiterInstrument` so the
            arbiter's own demanded deceleration and its measured-versus-seeded
            rate flag are recorded.  Costs roughly 40 % more time.
        progress: Optional ``callable(done, total)`` for a progress line.

    Returns:
        One :class:`CellResult` per cell, in grid order.
    """
    spec = spec or SweepSpec()
    if grid.horizon_s is not None:
        spec = SweepSpec(**dict(spec.__dict__, horizon_s=grid.horizon_s))
    frames = spec.max_frames
    api = _Adas()

    instrument = None
    if use_instrument:
        from tests.scenarios.instrument import ArbiterInstrument

        instrument = ArbiterInstrument(capture_tracks=False).attach()

    results: List[CellResult] = []
    cells = grid.cells()
    try:
        for index, cell in enumerate(cells):
            results.append(_run_cell(api, spec, cell, frames, instrument))
            if progress:
                progress(index + 1, len(cells))
    finally:
        if instrument is not None:
            instrument.detach()
    return results


def _run_cell(api: _Adas, spec: SweepSpec, cell: CellSpec, frames: int, instrument) -> CellResult:
    """Oracle, open loop, closed loop and grading for one cell."""
    res = CellResult(cell=cell)
    if cell.lead_speed_mps < 0.0 and not spec.allow_oncoming:
        res.verdict = Verdict.INFEASIBLE
        res.notes.append(
            "implied lead speed %.1f m/s is oncoming; outside the following ODD"
            % cell.lead_speed_mps
        )
        return res

    states = open_loop_states(spec, cell, frames)
    (
        res.warrant_frame,
        res.mandate_frame,
        res.lost_frame,
        res.headway_unsafe_frame,
    ) = oracle_frames(spec, cell, states)
    res.headway_unsafe_at_start = res.headway_unsafe_frame == 0
    res.initial_required_decel_mps2 = required_decel_mps2(
        cell.range_m,
        cell.ego_speed_mps,
        max(0.0, cell.lead_speed_mps),
        -cell.lead_decel_mps2,
    )

    op = _run_open_loop(api, spec, cell, states, instrument)
    res.hard_frame = op["hard_frame"]
    res.soft_frame = op["soft_frame"]
    res.open_frames = op["frames"]
    res.max_decel_open_mps2 = op["max_decel"]
    res.max_demand_open_mps2 = op["max_demand"]
    res.first_hard_state = op["first_state"]
    res.first_hard_findings = op["first_findings"]
    res.hard_with_inferred_rate = op["hard_inferred"]
    res.states_seen = op["states"]
    res.band_frames = op["band_frames"]
    res.first_band_frame = op["first_band_frame"]
    res.max_band_decel_mps2 = op["max_band"]

    cl = _run_closed_loop(api, spec, cell, frames)
    res.collided = cl["collided"]
    res.min_gap_m = cl["min_gap"]
    res.final_ego_speed_mps = cl["final_speed"]
    res.closed_frames = cl["frames"]
    res.stopped = cl["stopped"]

    res.verdict = classify_cell(spec, res)
    res.headway_verdict = classify_headway(spec, res)
    res.band_verdict = classify_band(spec, res)
    return res


# --------------------------------------------------------------------------- #
# Reporting: regions, not counts
# --------------------------------------------------------------------------- #


def _compress(values: Sequence[float]) -> str:
    """``[5, 8, 12]`` -> ``'5-12'``; a single value stays as itself."""
    if not values:
        return "-"
    vals = sorted(values)
    if len(vals) == 1:
        return "%g" % vals[0]
    return "%g-%g" % (vals[0], vals[-1])


def _count(values: Iterable[str]) -> Dict[str, int]:
    """Tally strings into a plain dict."""
    out: Dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


BAND_FAILURES = ("BAND_UNWARRANTED",)
"""Sub-emergency band gradings that count as a failure."""

BAND_CHARS = {"BAND_OK": ".", "BAND_UNWARRANTED": "B"}
"""Grid characters for the band grading."""


def region_boundaries(
    results: Iterable[CellResult],
    grid: SweepGrid,
    attr: str = "verdict",
    classes: Sequence[str] = Verdict.FAILURES,
) -> Dict[str, List[Dict[str, object]]]:
    """Reduce failing cells to the *edges* of each failure region.

    For every failure class and every ``(ego, rate, lead_decel)`` slice, report
    the span of ranges over which the class occurs and whether that span is
    bounded above by the grid itself -- in which case the true boundary is
    outside the sweep and the grid needs extending, which the statement says in
    so many words rather than leaving the reader to notice.

    Args:
        results: The graded cells.
        grid: The grid they came from, for the range axis.
        attr: Which grading to reduce -- ``"verdict"`` (the emergency grading),
            ``"band_verdict"`` (the sub-emergency band) or
            ``"headway_verdict"``.  Each is an independent view of the same
            cells, and a region in one says nothing about the others.
        classes: Which values of that grading count as failures.

    Returns a dict keyed by that grading's value, each entry a list of region
    records with a ready-made ``statement``.
    """
    all_ranges = sorted(grid.ranges_m)
    by_class: Dict[str, Dict[Tuple[float, float, float], List[CellResult]]] = {}
    for res in results:
        value = getattr(res, attr)
        if value not in classes:
            continue
        key = (res.cell.ego_speed_mps, res.cell.relative_rate_mps, res.cell.lead_decel_mps2)
        by_class.setdefault(value, {}).setdefault(key, []).append(res)

    out: Dict[str, List[Dict[str, object]]] = {}
    for verdict in classes:
        slices = by_class.get(verdict)
        if not slices:
            continue
        rows: List[Dict[str, object]] = []
        for (ego, rate, decel), cells in sorted(slices.items()):
            ranges = sorted(c.cell.range_m for c in cells)
            top = max(ranges)
            open_ended = top >= all_ranges[-1]
            contiguous_from_bottom = ranges == [r for r in all_ranges if r <= top]
            if contiguous_from_bottom and not open_ended:
                statement = (
                    "%s for every in-path range <= %g m at ego %g m/s "
                    "(rate %+g, lead_decel %g); correct from %g m out"
                    % (verdict, top, ego, rate, decel,
                       min(r for r in all_ranges if r > top))
                )
            elif open_ended:
                statement = (
                    "%s for range %s at ego %g m/s (rate %+g, lead_decel %g) -- reaches "
                    "the top of the swept range axis, the real boundary is beyond %g m"
                    % (verdict, _compress(ranges), ego, rate, decel, all_ranges[-1])
                )
            else:
                statement = "%s at ego %g m/s (rate %+g, lead_decel %g) for ranges %s" % (
                    verdict, ego, rate, decel, ", ".join("%g" % r for r in ranges)
                )
            rows.append(
                {
                    "ego_speed_mps": ego,
                    "relative_rate_mps": rate,
                    "lead_decel_mps2": decel,
                    "ranges_m": ranges,
                    "range_span": _compress(ranges),
                    "max_range_m": top,
                    "min_range_m": min(ranges),
                    "open_ended": open_ended,
                    "n_cells": len(cells),
                    "statement": statement,
                }
            )
        out[verdict] = rows
    return out


def render_grids(
    results: Sequence[CellResult],
    grid: SweepGrid,
    attr: str = "verdict",
    chars: Optional[Dict[str, str]] = None,
) -> str:
    """ASCII grids -- rows are range, columns are ego speed, one per slice.

    Failure *regions* are contiguous blocks of the same letter, which is the
    whole point: a count cannot show you that everything closer than 26 m fires.

    Args:
        results: The graded cells.
        grid: The grid they came from.
        attr: Which grading to draw; see :func:`region_boundaries`.
        chars: Value-to-character map, defaulting to the emergency grading's.
    """
    lines: List[str] = []
    chars = chars or Verdict.CHARS
    egos = sorted(grid.ego_speeds_mps)
    ranges = sorted(grid.ranges_m, reverse=True)
    index = {r.cell.key: r for r in results}
    legend = "  ".join(
        "%s=%s" % (chars[v], v) for v in sorted(chars) if chars[v].strip()
    )
    lines.append("legend: " + legend + "   (blank = INFEASIBLE)")
    for decel in sorted(grid.lead_decels_mps2):
        for rate in sorted(grid.relative_rates_mps):
            lines.append("")
            lines.append("  relative rate %+g m/s, lead braking at %g m/s^2" % (rate, decel))
            lines.append("  range \\ ego | " + " ".join("%5g" % e for e in egos))
            lines.append("  " + "-" * (13 + 6 * len(egos)))
            for rng in ranges:
                row = []
                for ego in egos:
                    res = index.get((ego, rng, rate, decel))
                    if res is None:
                        row.append("%5s" % "?")
                    elif res.verdict == Verdict.INFEASIBLE:
                        row.append("%5s" % " ")
                    else:
                        row.append("%5s" % chars.get(getattr(res, attr), "?"))
                lines.append("  %10g  | " % rng + " ".join(row))
    return "\n".join(lines)


def summarise(
    results: Sequence[CellResult], grid: SweepGrid, spec: Optional[SweepSpec] = None
) -> Dict[str, object]:
    """Counts, regions and the notable individual cells, as plain data."""
    spec = spec or SweepSpec()
    graded = [r for r in results if r.verdict != Verdict.INFEASIBLE]
    failures = [r for r in graded if r.verdict in Verdict.FAILURES]
    phantom_inferred = [
        r for r in graded
        if r.verdict in (Verdict.PHANTOM, Verdict.EARLY) and r.hard_with_inferred_rate
    ]
    return {
        "profile": grid.name,
        "spec": {k: v for k, v in spec.__dict__.items()},
        "cells_total": len(results),
        "cells_graded": len(graded),
        "counts": _count(r.verdict for r in results),
        "failures": len(failures),
        "headway_counts": _count(r.headway_verdict for r in graded),
        "band_counts": _count(r.band_verdict for r in graded),
        "band_failures": sum(1 for r in graded if r.band_verdict in BAND_FAILURES),
        "band_regions": region_boundaries(results, grid, "band_verdict", BAND_FAILURES),
        "worst_band": [
            r.to_dict()
            for r in sorted(
                (r for r in graded if r.band_verdict in BAND_FAILURES),
                key=lambda r: (-r.max_band_decel_mps2, r.cell.key),
            )[:20]
        ],
        "unwarranted_on_inferred_rate": len(phantom_inferred),
        "regions": region_boundaries(results, grid),
        "worst_collisions": [
            r.to_dict()
            for r in sorted(
                (r for r in graded if r.verdict == Verdict.COLLISION),
                key=lambda r: (r.min_gap_m, r.cell.key),
            )[:20]
        ],
        "worst_phantoms": [
            r.to_dict()
            for r in sorted(
                (r for r in graded if r.verdict == Verdict.PHANTOM),
                key=lambda r: (-r.max_decel_open_mps2, r.cell.key),
            )[:20]
        ],
        "worst_missed": [
            r.to_dict()
            for r in sorted(
                (r for r in graded if r.verdict == Verdict.MISSED),
                key=lambda r: (r.min_gap_m, r.cell.key),
            )[:20]
        ],
    }
