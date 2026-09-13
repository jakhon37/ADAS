"""Deterministic longitudinal (and minimal lateral) vehicle plant.

This module is the GROUND TRUTH of the acceptance harness.  Nothing in it is
derived from :mod:`adas.control.arbiter`; every constant below is a statement
about a vehicle and a road, and each one carries the reasoning that produced
it.  The oracle in :mod:`tests.scenarios.oracle` judges the system against
*this* model, so the model's honesty matters more than its sophistication.

Modelling decisions, stated up front so that a reviewer can disagree with them
explicitly rather than discover them in the arithmetic:

* **Point masses on a straight, flat, dry road.**  No grade, no rolling
  resistance, no aerodynamic drag.  Over the 5-20 s that a longitudinal safety
  scenario lasts, drag on a passenger car is under 0.4 m/s^2 and grade is zero
  by assumption; both are small next to the 8 m/s^2 braking authority and
  omitting them keeps the oracle's closed-form kinematics exact.
* **Bumper-to-bumper gap.**  ``gap_m`` is the clear distance between the ego's
  front bumper and the lead's rear bumper.  Vehicle lengths therefore never
  appear: ``gap_m <= 0`` IS contact.
* **First-order actuator lag.**  A commanded brake does not produce its
  deceleration instantly.  This is the difference between a system that
  "intervenes in time" on paper and one that does not on the road, so it is
  modelled rather than assumed away.
* **Fixed timestep, no wall clock, no unseeded randomness.**  Every run is
  reproducible bit for bit.

The plant is deliberately *more* capable than the vehicle the system believes
it is driving in one respect only: it never refuses a command.  Saturation and
authority limits are applied here, in the plant, so a system that asks for 20
m/s^2 gets 8 m/s^2 and the harness can say so.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from adas.core.models import (
    BoundingBox,
    EgoState,
    LaneModel,
    PerceptionStatus,
    TrackedObject,
)

# --------------------------------------------------------------------------- #
# Physical constants.  Each one is a claim about the world, not about the code.
# --------------------------------------------------------------------------- #

DT_S = 0.05
"""Simulation timestep, seconds (20 Hz).

Chosen to match the pipeline's nominal frame period so that one plant step is
one perception frame and "frame N" means the same thing to the plant, the
oracle and the report.  It is a fixed constant: the harness never reads a
clock, so a slow CI machine and a fast one produce identical results.
"""

MAX_BRAKE_DECEL_MPS2 = 8.0
"""Maximum braking authority, m/s^2.

Tyre-road friction limit on dry asphalt: a = mu * g with mu ~= 0.82 and
g = 9.81 gives 8.0 m/s^2.  This is the number the oracle uses when it asks
"could this collision have been avoided?", so it is deliberately the *optimistic*
end of a real car's capability -- a harness that assumed a weaker vehicle would
excuse late braking that a real car could have survived.
"""

MAX_ACCEL_MPS2 = 2.5
"""Maximum forward acceleration, m/s^2.

A mid-size passenger car at part throttle: 0-100 km/h in roughly 11 s.  The ego
never needs more than this in any scenario here; it exists so that recovery
after an intervention takes a realistic amount of time instead of being
instantaneous.
"""

BRAKE_RISE_TIME_S = 0.15
"""Time from a brake demand to 90% of the demanded deceleration, seconds.

Hydraulic pressure build-up plus pad bite.  Euro NCAP AEB test protocols and
ISO 15622 both work with 0.1-0.2 s for a modern electro-hydraulic system; 0.15 s
is the middle of that band.  Modelled as a first-order lag, so the time
constant is ``BRAKE_RISE_TIME_S / ln(10)``.
"""

THROTTLE_RISE_TIME_S = 0.35
"""Time from a throttle demand to 90% of the demanded acceleration, seconds.

Powertrain torque response is slower than brake response.  This only affects
how quickly the ego recovers speed after an intervention, which is exactly what
the recovery assertions measure, so it is modelled rather than idealised.
"""

WHEELBASE_M = 2.8
"""Front-to-rear axle distance, metres.  Mid-size passenger car."""

MAX_ROAD_WHEEL_RAD = 0.436
"""Road-wheel angle at full steering command, radians (25 degrees).

The controller emits a steering command normalised to +/-1 against a 25 degree
full-scale, so a command of 1.0 means 25 degrees at the road wheel.  25 degrees
is a normal maximum for a passenger car at the road wheel (the steering wheel
turns much further; the ratio is not modelled).
"""

STEERING_RISE_TIME_S = 0.10
"""Time from a steering demand to 90% of the demanded road-wheel angle."""

CAMERA_FOCAL_PX = 910.0
"""Vertical focal length in pixels for a 1280x720 image.

Used only to render a plausible bounding box from the true range, so that the
system's in-path gate has something geometrically consistent to look at.  This
is the focal length the project's own camera default documents for this image
size; using a different one here would make the rendered boxes disagree with
every other consumer of a 1280x720 frame in the repo.
"""

CAMERA_HEIGHT_M = 1.2
"""Camera mount height above the road, metres.  Typical windscreen mounting."""

HORIZON_PX = 360.0
"""Row of the horizon in a 1280x720 image with a level camera: the centre row."""

CAR_HEIGHT_M = 1.5
"""Nominal height of a passenger car, metres.  Renders the box height."""

FRAME_WIDTH_PX = 1280
FRAME_HEIGHT_PX = 720

LOOKAHEAD_M = 20.0
"""Distance ahead at which the lane centre is measured, metres.

Used to turn a metric lateral offset into a lane-centre column.  20 m is about
1 s of preview at motorway speed, which is the range a lane detector actually
fits over.
"""


def _lag_alpha(dt_s: float, rise_time_s: float) -> float:
    """First-order lag coefficient reaching 90% of a step in ``rise_time_s``.

    A first-order lag reaches 90% after ``ln(10) * tau``, so ``tau`` is
    ``rise_time_s / ln(10)`` and the per-step blend is ``1 - exp(-dt/tau)``.
    """
    if rise_time_s <= 0.0:
        return 1.0
    tau = rise_time_s / math.log(10.0)
    return 1.0 - math.exp(-dt_s / tau)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlantConfig:
    """Vehicle and integration parameters.  Defaults are the constants above."""

    dt_s: float = DT_S
    max_brake_decel_mps2: float = MAX_BRAKE_DECEL_MPS2
    max_accel_mps2: float = MAX_ACCEL_MPS2
    brake_rise_time_s: float = BRAKE_RISE_TIME_S
    throttle_rise_time_s: float = THROTTLE_RISE_TIME_S
    wheelbase_m: float = WHEELBASE_M
    max_road_wheel_rad: float = MAX_ROAD_WHEEL_RAD
    steering_rise_time_s: float = STEERING_RISE_TIME_S


DEFAULT_PLANT = PlantConfig()


# --------------------------------------------------------------------------- #
# The lead vehicle
# --------------------------------------------------------------------------- #

#: A lead acceleration law.  Arguments are ``(t_s, lead_speed_mps,
#: ego_speed_mps, gap_m)`` and the return value is the lead's commanded
#: acceleration in m/s^2.  It is a *script*: the lead is ground truth and obeys
#: it exactly, with no actuator lag of its own.
LeadAccelFn = Callable[[float, float, float, float], float]


@dataclass(frozen=True)
class LeadSpec:
    """A scripted lead vehicle.

    Attributes:
        initial_gap_m: Bumper-to-bumper clear distance at t=0, metres.
        initial_speed_mps: Lead speed at t=0, m/s.
        accel_fn: The acceleration law; see :data:`LeadAccelFn`.
        appears_at_s: Before this time the lead is not in the ego's lane at all
            (a cut-in).  It still travels its script, so the gap at the moment
            it appears is a physical consequence of the script, not a jump.
        vanishes_at_s: After this time the lead has left the lane.  Used for the
            "the hazard is over, now recover" half of the recovery assertions.
        label: Human-readable description for the report.
    """

    initial_gap_m: float
    initial_speed_mps: float
    accel_fn: LeadAccelFn
    appears_at_s: float = 0.0
    vanishes_at_s: float = float("inf")
    label: str = ""

    def present_at(self, t_s: float) -> bool:
        """Whether the lead occupies the ego lane at ``t_s``."""
        return self.appears_at_s - 1e-9 <= t_s < self.vanishes_at_s


def lead_stationary() -> LeadAccelFn:
    """A parked or stopped vehicle: zero speed, zero acceleration, forever."""

    def fn(t_s: float, lead_v: float, ego_v: float, gap_m: float) -> float:
        return 0.0

    return fn


def lead_constant_speed() -> LeadAccelFn:
    """Cruise: the lead holds whatever speed it started with."""

    def fn(t_s: float, lead_v: float, ego_v: float, gap_m: float) -> float:
        return 0.0

    return fn


def lead_brakes(decel_mps2: float, start_s: float = 0.0, stop_s: float = float("inf")) -> LeadAccelFn:
    """The lead brakes at a constant rate between ``start_s`` and ``stop_s``.

    Deceleration stops when the lead reaches standstill; the plant clamps speed
    at zero, so ``decel_mps2`` is applied only while the lead is still moving.
    """

    def fn(t_s: float, lead_v: float, ego_v: float, gap_m: float) -> float:
        if start_s - 1e-9 <= t_s < stop_s and lead_v > 0.0:
            return -abs(decel_mps2)
        return 0.0

    return fn


def lead_accelerates(accel_mps2: float, cap_mps: float) -> LeadAccelFn:
    """The lead pulls away at ``accel_mps2`` until it reaches ``cap_mps``."""

    def fn(t_s: float, lead_v: float, ego_v: float, gap_m: float) -> float:
        return abs(accel_mps2) if lead_v < cap_mps else 0.0

    return fn


def lead_brake_then_release(
    decel_mps2: float, start_s: float, duration_s: float, resume_accel_mps2: float, resume_to_mps: float
) -> LeadAccelFn:
    """Brake for a while, then accelerate back up.  The recovery test case."""

    def fn(t_s: float, lead_v: float, ego_v: float, gap_m: float) -> float:
        if start_s - 1e-9 <= t_s < start_s + duration_s and lead_v > 0.0:
            return -abs(decel_mps2)
        if t_s >= start_s + duration_s and lead_v < resume_to_mps:
            return abs(resume_accel_mps2)
        return 0.0

    return fn


# --------------------------------------------------------------------------- #
# The road
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RoadSpec:
    """Road geometry ahead of the ego.

    Attributes:
        curvature_1pm: Signed curvature (1/radius) as a function of time,
            positive for a left-hand bend.  A constant 1/230 is a 230 m radius
            bend, which at 20 m/s is 1.74 m/s^2 of lateral acceleration -- a
            brisk but entirely ordinary motorway curve.
        label: Human-readable description.
    """

    curvature_1pm: Callable[[float], float] = lambda t_s: 0.0
    label: str = "straight"


def straight_road() -> RoadSpec:
    return RoadSpec(curvature_1pm=lambda t_s: 0.0, label="straight")


def constant_bend(radius_m: float, sign: float = 1.0) -> RoadSpec:
    """A bend of constant radius.  ``sign`` is +1 for left, -1 for right."""
    kappa = sign / float(radius_m)
    return RoadSpec(
        curvature_1pm=lambda t_s: kappa,
        label="bend R=%.0f m (a_lat=%.2f m/s^2 at 20 m/s)" % (radius_m, 400.0 / radius_m),
    )


# --------------------------------------------------------------------------- #
# World state
# --------------------------------------------------------------------------- #


@dataclass
class WorldState:
    """The complete true state of the world at one frame.

    Every field is ground truth.  The system under test never sees this object;
    it sees whatever :class:`Sensor` chooses to report about it.
    """

    frame: int = 0
    t_s: float = 0.0

    ego_x_m: float = 0.0
    ego_v_mps: float = 0.0
    ego_a_mps2: float = 0.0

    lead_present: bool = False
    lead_x_m: float = float("inf")
    lead_v_mps: float = 0.0
    lead_a_mps2: float = 0.0

    lateral_offset_m: float = 0.0
    """Ego offset from the lane centre, metres, positive to the right."""

    heading_err_rad: float = 0.0
    """Ego heading relative to the lane tangent, radians, positive to the left."""

    road_wheel_rad: float = 0.0
    """Road-wheel angle, radians, positive to the RIGHT -- the same sign
    convention as ``ControlCommand.steering``."""

    road_curvature_1pm: float = 0.0

    throttle: float = 0.0
    brake: float = 0.0
    steering: float = 0.0

    @property
    def gap_m(self) -> float:
        """True bumper-to-bumper clear distance, or +inf when there is no lead."""
        if not self.lead_present:
            return float("inf")
        return self.lead_x_m - self.ego_x_m

    @property
    def closing_mps(self) -> float:
        """True closing rate: positive when the gap is shrinking."""
        if not self.lead_present:
            return 0.0
        return self.ego_v_mps - self.lead_v_mps


# --------------------------------------------------------------------------- #
# The plant
# --------------------------------------------------------------------------- #


class Plant:
    """A deterministic vehicle-and-world simulator.

    Usage::

        plant = Plant(ego_speed_mps=20.0, lead=lead_spec, road=road_spec)
        state = plant.state              # frame 0, before any command
        while ...:
            state = plant.step(command)  # advance one dt with that command

    ``step`` applies the command through the actuator lags, integrates one
    timestep and returns the new true state.  It is pure with respect to
    everything outside the instance: no clock, no global RNG.
    """

    def __init__(
        self,
        ego_speed_mps: float,
        lead: Optional[LeadSpec] = None,
        road: Optional[RoadSpec] = None,
        config: PlantConfig = DEFAULT_PLANT,
        initial_lateral_offset_m: float = 0.0,
    ) -> None:
        self.config = config
        self.lead = lead
        self.road = road or straight_road()
        self._state = WorldState(
            frame=0,
            t_s=0.0,
            ego_x_m=0.0,
            ego_v_mps=float(ego_speed_mps),
            ego_a_mps2=0.0,
            lead_present=lead.present_at(0.0) if lead is not None else False,
            lead_x_m=(lead.initial_gap_m if lead is not None else float("inf")),
            lead_v_mps=(lead.initial_speed_mps if lead is not None else 0.0),
            lead_a_mps2=0.0,
            lateral_offset_m=float(initial_lateral_offset_m),
            heading_err_rad=0.0,
            road_wheel_rad=0.0,
            road_curvature_1pm=self.road.curvature_1pm(0.0),
        )
        # The lead's absolute position is measured on the same axis as the ego's,
        # so lead_x_m starts at the initial gap because ego_x_m starts at 0.
        self._brake_alpha = _lag_alpha(config.dt_s, config.brake_rise_time_s)
        self._throttle_alpha = _lag_alpha(config.dt_s, config.throttle_rise_time_s)
        self._steer_alpha = _lag_alpha(config.dt_s, config.steering_rise_time_s)

    @classmethod
    def from_state(
        cls,
        state: WorldState,
        lead: Optional[LeadSpec] = None,
        road: Optional[RoadSpec] = None,
        config: PlantConfig = DEFAULT_PLANT,
    ) -> "Plant":
        """A plant resumed from an existing true state.

        Used by the oracle to ask counterfactual questions ("what if the ego had
        braked flat out from here?") against the same physics the run itself
        used, including the actuator lag already built up.
        """
        plant = cls(ego_speed_mps=state.ego_v_mps, lead=lead, road=road, config=config)
        plant._state = WorldState(**vars(state))
        return plant

    @property
    def state(self) -> WorldState:
        """The current true state.  Do not mutate."""
        return self._state

    # ---------------------------------------------------------------- stepping

    def step(self, throttle: float, brake: float, steering: float) -> WorldState:
        """Advance the world by one timestep under the given actuator command.

        Args:
            throttle: Pedal fraction in [0, 1].
            brake: Pedal fraction in [0, 1].
            steering: Normalised steering in [-1, 1]; 1.0 is
                :data:`MAX_ROAD_WHEEL_RAD` at the road wheel.

        Returns:
            The new :class:`WorldState`.
        """
        cfg = self.config
        dt = cfg.dt_s
        s = self._state

        throttle = min(1.0, max(0.0, float(throttle)))
        brake = min(1.0, max(0.0, float(brake)))
        steering = min(1.0, max(-1.0, float(steering)))

        # --- longitudinal ------------------------------------------------- #
        # Demanded acceleration.  Brake wins over throttle by superposition:
        # a real car with both pedals down decelerates, and the friction limit
        # caps the result either way.
        demand = cfg.max_accel_mps2 * throttle - cfg.max_brake_decel_mps2 * brake
        demand = min(cfg.max_accel_mps2, max(-cfg.max_brake_decel_mps2, demand))

        # Lag: use the brake time constant when the demand is a deceleration,
        # the throttle one otherwise.  Braking builds faster than torque does.
        alpha = self._brake_alpha if demand < s.ego_a_mps2 else self._throttle_alpha
        a_ego = s.ego_a_mps2 + (demand - s.ego_a_mps2) * alpha

        v_new = s.ego_v_mps + a_ego * dt
        if v_new < 0.0:
            # The car cannot reverse under braking.  Truncate the step and
            # record the acceleration actually realised over it.
            a_ego = -s.ego_v_mps / dt if dt > 0 else 0.0
            v_new = 0.0
        # Trapezoidal integration, which is exact for a constant acceleration
        # over the step.
        ego_x = s.ego_x_m + 0.5 * (s.ego_v_mps + v_new) * dt

        # --- lead ----------------------------------------------------------- #
        lead_present = False
        lead_x = s.lead_x_m
        lead_v = s.lead_v_mps
        a_lead = 0.0
        t_next = s.t_s + dt
        if self.lead is not None:
            a_lead = float(self.lead.accel_fn(s.t_s, s.lead_v_mps, s.ego_v_mps, s.gap_m))
            lv_new = s.lead_v_mps + a_lead * dt
            if lv_new < 0.0:
                a_lead = -s.lead_v_mps / dt if dt > 0 else 0.0
                lv_new = 0.0
            lead_x = s.lead_x_m + 0.5 * (s.lead_v_mps + lv_new) * dt
            lead_v = lv_new
            lead_present = self.lead.present_at(t_next)

        # --- lateral ---------------------------------------------------------- #
        # Bicycle model, small-angle.  The road turns under the ego at
        # v * kappa rad/s; the ego turns at v * tan(delta) / L rad/s.  The
        # difference is the heading error rate, and the heading error carries
        # the ego across the lane at v * sin(heading) m/s.
        # ``steering`` and ``road_wheel_rad`` are POSITIVE TO THE RIGHT, matching
        # the control command's own convention (adas.planning.lateral: a positive
        # steering angle means the lane centre is to the right, so steer right).
        # The bicycle model below works in absolute headings, which are positive
        # to the LEFT, hence the single negation.
        delta_demand = steering * cfg.max_road_wheel_rad
        delta = s.road_wheel_rad + (delta_demand - s.road_wheel_rad) * self._steer_alpha
        kappa = float(self.road.curvature_1pm(t_next))
        v_mid = 0.5 * (s.ego_v_mps + v_new)
        yaw_rate = -v_mid * math.tan(delta) / cfg.wheelbase_m
        heading = s.heading_err_rad + (yaw_rate - v_mid * kappa) * dt
        lateral = s.lateral_offset_m - v_mid * math.sin(heading) * dt

        self._state = WorldState(
            frame=s.frame + 1,
            t_s=t_next,
            ego_x_m=ego_x,
            ego_v_mps=v_new,
            ego_a_mps2=a_ego,
            lead_present=lead_present,
            lead_x_m=lead_x,
            lead_v_mps=lead_v,
            lead_a_mps2=a_lead,
            lateral_offset_m=lateral,
            heading_err_rad=heading,
            road_wheel_rad=delta,
            road_curvature_1pm=kappa,
            throttle=throttle,
            brake=brake,
            steering=steering,
        )
        return self._state


# --------------------------------------------------------------------------- #
# Perception model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PerceptionSpec:
    """How the (simulated) perception stack reports the true world.

    Everything here is a *sensor* characteristic.  It never changes the world,
    only what the system is told about it, which is what makes the difference
    between "the system was wrong" and "the system was lied to" measurable.

    Attributes:
        max_range_m: Beyond this true range the lead is not reported at all.
            80 m is a realistic detection horizon for a 1280x720 mono camera
            against a car-sized target.
        range_noise_m: Standard deviation of zero-mean Gaussian range noise,
            metres.  Drawn from a seeded generator.
        range_bias_frac: Systematic multiplicative range error (0.1 = 10% far).
        miss_frames: Frame indices on which the detector produces nothing for
            the lead, while perception itself still reports healthy.  This is a
            detection miss, not a sensor failure.
        failed_frames: Frame indices on which perception reports unhealthy
            (``PerceptionStatus.ok = False``).  This is a sensor/pipeline
            failure; the track list is empty AND the system is told why.
        source_lost_from_frame: From this frame onward perception is failed for
            the rest of the run.  Models a camera that stops delivering.
        reid_frames: Frame indices on which the tracker assigns a NEW track id
            to the same physical vehicle.  Everything downstream sees a
            brand-new object with no history.
        range_jump_at: ``(frame, delta_m)`` -- a one-off step in the reported
            range at that frame, persisting afterwards.  Models a range
            re-estimate after a re-identification.
        lane_center_bias_px: Constant error added to the reported lane-centre
            column.  A large value models a mock/failed lane estimate.
        lane_is_mock: Whether the reported lane model is flagged as mock.
        lane_confidence: Confidence reported with the lane model.
        seed: Seed for the noise generator.  Fixed, so runs are reproducible.
    """

    max_range_m: float = 80.0
    range_noise_m: float = 0.0
    range_bias_frac: float = 0.0
    miss_frames: Tuple[int, ...] = ()
    failed_frames: Tuple[int, ...] = ()
    source_lost_from_frame: Optional[int] = None
    reid_frames: Tuple[int, ...] = ()
    range_jump_at: Optional[Tuple[int, float]] = None
    lane_center_bias_px: float = 0.0
    lane_is_mock: bool = False
    lane_confidence: float = 0.85
    seed: int = 20240914


PERFECT_PERCEPTION = PerceptionSpec()


@dataclass
class Observation:
    """What the system under test is given for one frame."""

    frame: int
    t_s: float
    ego: EgoState
    tracks: List[TrackedObject]
    perception: PerceptionStatus
    lane_center_px: Optional[float]
    lane: Optional[LaneModel]
    frame_width_px: int = FRAME_WIDTH_PX
    frame_height_px: int = FRAME_HEIGHT_PX
    lateral_offset_m: Optional[float] = None


def render_box(distance_m: float, lateral_offset_m: float = 0.0) -> BoundingBox:
    """A geometrically consistent bounding box for a car at ``distance_m``.

    Flat-ground pinhole: the rear bumper contacts the road at
    ``horizon + f * h_cam / d`` and the roof is ``f * h_car / d`` pixels above
    it.  The horizontal centre is the image centre displaced by the object's
    lateral offset, ``f * y / d`` pixels.

    This exists so that the system's own in-path gate -- which works in image
    space -- sees a box that is consistent with the range it is also given.
    A harness that put a fixed box at a varying range would be testing a
    contradiction.
    """
    d = max(1.0, float(distance_m))
    half_w_px = 0.5 * 1.8 * CAMERA_FOCAL_PX / d  # 1.8 m wide car
    h_px = CAR_HEIGHT_M * CAMERA_FOCAL_PX / d
    cx = FRAME_WIDTH_PX / 2.0 + lateral_offset_m * CAMERA_FOCAL_PX / d
    y2 = HORIZON_PX + CAMERA_HEIGHT_M * CAMERA_FOCAL_PX / d
    y2 = min(float(FRAME_HEIGHT_PX - 1), y2)
    y1 = max(0.0, y2 - h_px)
    return BoundingBox(cx - half_w_px, y1, cx + half_w_px, y2, 0.93, "car")


def lane_center_px_for(lateral_offset_m: float, curvature_1pm: float) -> float:
    """The lane-centre column a correct lane detector would report.

    At a lookahead ``L`` the lane centre is displaced laterally by
    ``0.5 * kappa * L**2`` from the ego's heading, and the ego's own offset
    ``y`` moves it the other way.  Metres convert to pixels at ``f / L``.
    Positive curvature is a left-hand bend, which puts the lane centre left of
    the image centre, i.e. at a smaller column.
    """
    bend_m = 0.5 * curvature_1pm * LOOKAHEAD_M * LOOKAHEAD_M
    metres_right_of_ego = -bend_m - lateral_offset_m
    return FRAME_WIDTH_PX / 2.0 + metres_right_of_ego * CAMERA_FOCAL_PX / LOOKAHEAD_M


class Sensor:
    """Turns true :class:`WorldState` into an :class:`Observation`.

    Stateful only in the ways a real perception stack is: a seeded noise
    generator, a track-id counter that changes on a re-identification, and a
    dropout counter that feeds ``PerceptionStatus.consecutive_failures``.
    """

    def __init__(self, spec: PerceptionSpec = PERFECT_PERCEPTION) -> None:
        self.spec = spec
        self._rng = random.Random(spec.seed)
        self._track_id = 1
        self._consecutive_failures = 0
        self._last_good_t_s = 0.0
        self._age = 0
        self._hits = 0
        self._range_offset_m = 0.0

    def observe(self, state: WorldState) -> Observation:
        """Report ``state`` through the configured perception characteristics."""
        spec = self.spec
        frame = state.frame

        failed = frame in spec.failed_frames or (
            spec.source_lost_from_frame is not None and frame >= spec.source_lost_from_frame
        )

        if spec.range_jump_at is not None and frame == spec.range_jump_at[0]:
            self._range_offset_m += float(spec.range_jump_at[1])
        if frame in spec.reid_frames:
            self._track_id += 1
            self._age = 0
            self._hits = 0

        if failed:
            self._consecutive_failures += 1
            perception = PerceptionStatus(
                ok=False,
                consecutive_failures=self._consecutive_failures,
                last_good_timestamp_s=self._last_good_t_s,
                detector_ok=False,
                lane_ok=False,
                reason="source_lost" if spec.source_lost_from_frame is not None else "detector",
            )
            return Observation(
                frame=frame,
                t_s=state.t_s,
                ego=EgoState(speed_mps=state.ego_v_mps, valid=True, timestamp_s=state.t_s),
                tracks=[],
                perception=perception,
                lane_center_px=None,
                lane=None,
                lateral_offset_m=None,
            )

        self._consecutive_failures = 0
        self._last_good_t_s = state.t_s
        perception = PerceptionStatus(ok=True, last_good_timestamp_s=state.t_s)

        tracks: List[TrackedObject] = []
        visible = (
            state.lead_present
            and state.gap_m <= spec.max_range_m
            and frame not in spec.miss_frames
        )
        if visible:
            self._age += 1
            self._hits += 1
            true_gap = state.gap_m
            measured = true_gap * (1.0 + spec.range_bias_frac) + self._range_offset_m
            if spec.range_noise_m > 0.0:
                measured += self._rng.gauss(0.0, spec.range_noise_m)
            measured = max(0.3, measured)
            closing = state.closing_mps
            ttc = measured / closing if closing > 1e-3 else float("inf")
            tracks.append(
                TrackedObject(
                    track_id=self._track_id,
                    box=render_box(measured, lateral_offset_m=-state.lateral_offset_m),
                    velocity_mps=closing,
                    distance_m=measured,
                    age_frames=self._age,
                    hits=self._hits,
                    time_since_update=0,
                    lateral_offset_m=-state.lateral_offset_m,
                    ttc_s=ttc,
                    in_ego_lane=True,
                )
            )
        else:
            self._age += 1

        centre = lane_center_px_for(state.lateral_offset_m, state.road_curvature_1pm)
        centre += spec.lane_center_bias_px
        lane = LaneModel(
            left_coeffs=(0.0, 0.0, 0.0),
            right_coeffs=(0.0, 0.0, 0.0),
            lane_center_px=centre,
            curvature_m=(1.0 / state.road_curvature_1pm) if state.road_curvature_1pm else 0.0,
            confidence=spec.lane_confidence,
            is_mock=spec.lane_is_mock,
        )
        return Observation(
            frame=frame,
            t_s=state.t_s,
            ego=EgoState(speed_mps=state.ego_v_mps, valid=True, timestamp_s=state.t_s),
            tracks=tracks,
            perception=perception,
            lane_center_px=centre,
            lane=lane,
            lateral_offset_m=None,
        )
