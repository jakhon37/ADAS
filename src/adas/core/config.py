"""Configuration objects and strict JSON loading.

Every runtime knob the ADAS process has lives here, in one dataclass per
subsystem, and every one of them is range-checked at construction.  Three rules
are load-bearing and are enforced by :func:`load_config`:

1. **Unknown keys are an error.**  A typo in a vehicle configuration used to be
   silently ignored, which meant a safety limit could be "set" in JSON and never
   reach the code (ADAS-OPS-17).  Unknown keys now raise
   :class:`~adas.core.exceptions.ConfigurationError` naming the key and the
   section.  A small, explicit list of *deprecated* keys is accepted with a
   WARNING instead, so old files keep loading while telling the operator what to
   delete.
2. **Sections cross-validate.**  A configuration where the planner may steer
   further than the safety monitor allows, or drive faster than the safety
   monitor's ceiling, is rejected at load time rather than discovered by the
   arbiter clamping every frame (ADAS-DEC-18).
3. **Nothing is derived silently.**  Where one number must equal another
   (``safety.max_road_wheel_rad`` is the road-wheel angle at ``steering=1.0``,
   i.e. ``radians(controller.max_steering_angle_deg)``) the default is ``0.0``
   meaning "derive it", and an explicitly supplied value that disagrees is an
   error.

Units are stated in every field's comment.  Angles are degrees in the planner
and controller (they are human-facing setpoints) and radians in the safety
limits (they are compared against physical quantities); the conversion happens
exactly once, in :meth:`RuntimeConfig.__post_init__`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

from adas.core.exceptions import ConfigurationError, ValidationError
from adas.core.logger import setup_logger
from adas.core.validation import validate_config_value

logger = setup_logger(__name__)

#: Bumped when a key is renamed or removed in an incompatible way. A config file
#: may carry ``"schema_version"``; a *newer* one than this build understands is
#: refused, an older one is accepted (every removal is handled by DEPRECATED_KEYS).
SCHEMA_VERSION = 2

#: section -> {dead key: human-readable replacement}. Accepted with a WARNING.
DEPRECATED_KEYS: Dict[str, Dict[str, str]] = {
    "lane": {
        "input_width": "derived from lane.dataset",
        "input_height": "derived from lane.dataset",
        "crop_ratio": "derived from lane.dataset",
        "num_row": "derived from lane.dataset",
        "num_col": "derived from lane.dataset",
    },
}

DETECTOR_BACKENDS = ("mock", "tensorrt", "yolov5", "yolox")
LANE_BACKENDS = ("mock", "ufld", "yolop", "twinlite")
DEPTH_BACKENDS = ("off", "midas")
EGO_SOURCES = ("none", "config", "simulated", "file")
SOURCE_TYPES = ("synthetic", "video", "camera")
DETECTOR_LAYOUTS = ("auto", "yolov5", "yolov8", "yolox")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _coerce(name: str, value: Any, default: Any) -> Any:
    """Coerce a JSON value to the type of the dataclass default.

    ``None`` defaults and ``class_ids`` (str or list) are pass-through: they are
    validated by the owning dataclass instead.
    """
    if default is None:
        return value
    if isinstance(default, bool):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "yes", "1", "on"):
                return True
            if lowered in ("false", "no", "0", "off"):
                return False
            raise ConfigurationError("%s must be a boolean, got %r" % (name, value))
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("%s must be an integer, got %r" % (name, value)) from exc
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("%s must be a number, got %r" % (name, value)) from exc
    if isinstance(default, str) and not isinstance(value, str):
        raise ConfigurationError("%s must be a string, got %r" % (name, value))
    if isinstance(default, (list, tuple)) and not isinstance(value, (list, tuple)):
        raise ConfigurationError("%s must be a list, got %r" % (name, value))
    return value


def _fill(cls: type, data: Any, defaults: Any, section: str) -> Any:
    """Build a dataclass from one config section, strictly.

    Args:
        cls: the dataclass to build.
        data: the JSON object for this section (``{}`` when absent).
        defaults: an instance supplying the default for every missing key.
        section: dotted name used in error messages.

    Raises:
        ConfigurationError: the section is not an object, or it carries a key
            that is neither a field of *cls* nor listed in ``DEPRECATED_KEYS``.
    """
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigurationError("config section %r must be an object, got %r" % (section, data))

    known = {f.name for f in fields(cls) if f.init}
    dead = DEPRECATED_KEYS.get(section, {})
    for key in data:
        if key in known:
            continue
        if key in dead:
            logger.warning(
                "config: %s.%s is deprecated and ignored (%s). Remove it from the file.",
                section,
                key,
                dead[key],
            )
            continue
        near = sorted(k for k in known if k.startswith(key[:3]))
        hint = (" Did you mean one of %s?" % ", ".join(near)) if near else ""
        raise ConfigurationError(
            "unknown key %r in config section %r.%s Known keys: %s"
            % (key, section, hint, ", ".join(sorted(known)))
        )

    kwargs = {}
    for f in fields(cls):
        if not f.init:
            continue
        if f.name in data and data[f.name] is not None:
            kwargs[f.name] = _coerce("%s.%s" % (section, f.name), data[f.name], getattr(defaults, f.name))
        else:
            kwargs[f.name] = getattr(defaults, f.name)
    return cls(**kwargs)


def _one_of(section: str, name: str, value: str, allowed: Sequence[str]) -> str:
    lowered = str(value).lower()
    if lowered not in allowed:
        raise ConfigurationError(
            "%s.%s must be one of %s, got %r" % (section, name, "|".join(allowed), value)
        )
    return lowered


@dataclass
class DetectorConfig:
    """Object detection configuration.

    ``backend``:
        ``mock`` fabricates one box and is refused unless ``allow_mock`` is set
        (see :func:`adas.perception.factory.build_detector`); ``yolox`` and
        ``yolov5`` pin the decode layout; ``tensorrt`` reads the layout from the
        engine's own output shape.
    """

    backend: str = "yolox"
    model_path: str = "models/yolox_nano.engine"
    confidence_threshold: float = 0.35  # min obj*class score, [0, 1]
    iou_threshold: float = 0.5  # class-wise NMS IoU, [0, 1]
    max_detections: int = 100  # cap after NMS
    input_size: int = 416  # advisory; the engine's own size always wins
    class_ids: Union[str, List[int]] = "road_users"  # CLASS_SETS name or COCO ids
    layout: str = "auto"  # auto | yolov5 | yolov8 | yolox
    pre_nms_topk: int = 300  # candidate cap entering NMS

    def __post_init__(self) -> None:
        self.backend = _one_of("detector", "backend", self.backend, DETECTOR_BACKENDS)
        self.layout = _one_of("detector", "layout", self.layout, DETECTOR_LAYOUTS)
        validate_config_value("detector.confidence_threshold", self.confidence_threshold, 0.0, 1.0)
        validate_config_value("detector.iou_threshold", self.iou_threshold, 0.0, 1.0)
        validate_config_value("detector.max_detections", self.max_detections, 1, 1000)
        validate_config_value("detector.input_size", self.input_size, 32, 2048)
        validate_config_value("detector.pre_nms_topk", self.pre_nms_topk, 1, 10000)
        if isinstance(self.class_ids, (list, tuple)):
            ids = []
            for item in self.class_ids:
                if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                    raise ConfigurationError(
                        "detector.class_ids entries must be non-negative integers, got %r" % (item,)
                    )
                ids.append(int(item))
            if not ids:
                raise ConfigurationError("detector.class_ids must not be an empty list")
            self.class_ids = ids
        elif not isinstance(self.class_ids, str) or not self.class_ids.strip():
            raise ConfigurationError(
                "detector.class_ids must be a CLASS_SETS name or a list of COCO ids, got %r"
                % (self.class_ids,)
            )
        if self.backend != "mock" and not self.model_path:
            raise ConfigurationError("detector.model_path is required for backend %r" % self.backend)


@dataclass
class LaneConfig:
    """Lane estimation configuration.

    ``every_n_frames`` exists because YOLOP costs ~59 ms end to end and cannot run
    per frame at 20 Hz.  The pipeline reuses the previous model on skipped frames
    and marks it stale; a stale model older than ``max_age_frames`` is dropped
    rather than presented as current.
    """

    backend: str = "ufld"
    model_path: str = "models/ufldv2_culane_res18.engine"
    dataset: str = "culane"  # culane | tusimple | curvelanes (UFLD only)
    min_confidence: float = 0.0  # below this estimate() returns None
    every_n_frames: int = 1  # 1 = every frame; >1 schedules a slow backend
    max_age_frames: int = 3  # frames a reused lane model stays usable
    allow_unavailable: bool = False  # twinlite: accept the honest no-engine stub

    def __post_init__(self) -> None:
        self.backend = _one_of("lane", "backend", self.backend, LANE_BACKENDS)
        self.dataset = str(self.dataset).lower()
        validate_config_value("lane.min_confidence", self.min_confidence, 0.0, 1.0)
        validate_config_value("lane.every_n_frames", self.every_n_frames, 1, 100)
        validate_config_value("lane.max_age_frames", self.max_age_frames, 0, 100)
        if self.backend != "mock" and not self.model_path:
            raise ConfigurationError("lane.model_path is required for backend %r" % self.backend)


@dataclass
class DepthConfig:
    """Monocular depth used as an INDEPENDENT range cross-check, not as a sensor.

    MiDaS emits inverse relative depth with an unknown per-frame affine scale and
    shift.  :class:`adas.perception.depth.DepthRangeChannel` aligns it against
    road-plane anchors projected through the camera homography, which is what
    makes the resulting per-object range independent of the detection box.  It is
    never published as a calibrated metric sensor: with no camera, or with too few
    anchors, every estimate is ``RangeSource.UNAVAILABLE``.
    """

    backend: str = "off"  # off | midas
    model_path: str = "models/midas_v21_small_256.engine"
    cadence_frames: int = 5  # run the net every N frames (5 @ 20 Hz = 4 Hz)
    require_engine: bool = False  # True: a missing engine is fatal, not a stub

    def __post_init__(self) -> None:
        self.backend = _one_of("depth", "backend", self.backend, DEPTH_BACKENDS)
        validate_config_value("depth.cadence_frames", self.cadence_frames, 1, 100)
        if self.backend != "off" and not self.model_path:
            raise ConfigurationError("depth.model_path is required for backend %r" % self.backend)


@dataclass
class CameraConfig:
    """Intrinsics + road-plane extrinsics.  Everything metric depends on these.

    ``calibrated`` is the honesty switch.  ``False`` (the default) means these
    numbers are an assumption: :mod:`adas.perception.geometry` caps every derived
    range confidence, the metric steering law refuses to engage, and the lane
    layer publishes pixels rather than metres.  Setting ``calibrated: true``
    without having actually measured the mount height and pitch is the single
    easiest way to make this system lie.
    """

    enabled: bool = True  # False: no camera model at all; pixel-only everywhere
    image_width: int = 1280  # resolution the intrinsics were measured at, px
    image_height: int = 720
    fx: float = 910.0  # focal length, px
    fy: float = 910.0
    cx: float = 640.0  # principal point, px
    cy: float = 360.0
    mount_height_m: float = 1.30  # optical centre above the road, m
    pitch_deg: float = 2.0  # downward tilt, deg (positive = nose down)
    calibrated: bool = False
    label: str = "uncalibrated-default"

    def __post_init__(self) -> None:
        validate_config_value("camera.image_width", self.image_width, 16, 7680)
        validate_config_value("camera.image_height", self.image_height, 16, 4320)
        validate_config_value("camera.fx", self.fx, 1.0, 100000.0)
        validate_config_value("camera.fy", self.fy, 1.0, 100000.0)
        validate_config_value("camera.cx", self.cx, -10000.0, 10000.0)
        validate_config_value("camera.cy", self.cy, -10000.0, 10000.0)
        validate_config_value("camera.mount_height_m", self.mount_height_m, 0.1, 5.0)
        validate_config_value("camera.pitch_deg", self.pitch_deg, -60.0, 60.0)
        if not self.label:
            raise ConfigurationError("camera.label must not be empty")


@dataclass
class EgoConfig:
    """Where the measured ego speed comes from.  There is NO vehicle bus here.

    ``source``:
        ``none``
            No ego speed at all.  ``EgoState.valid`` is False every frame and the
            planner degrades to hold-and-ramp-down.  This is the honest default
            for a board with no CAN interface.
        ``config``
            A fixed ``speed_mps`` from this file.  ``valid`` is True, because the
            operator asserted it, but it is a *declared* speed and the health
            endpoint reports ``ego_speed_measured=false``.
        ``simulated``
            A closed-loop point-mass integrator driven by the actuated command,
            for replay and bench work.  ``valid`` is True (the planner needs a
            number to close the loop) but ``measured`` is False and both the log
            banner and ``/healthz`` say so.  NEVER for a vehicle.
        ``file``
            A recorded speed channel: a CSV or JSON sidecar of
            ``timestamp_s,speed_mps`` rows, replayed against the frame clock.
            Samples older than ``max_age_s`` mark the state invalid rather than
            being held forever.
    """

    source: str = "none"
    speed_mps: float = 0.0  # used by source="config" and as the initial value
    file: str = ""  # used by source="file": CSV or JSON sidecar
    max_age_s: float = 0.5  # a sample older than this invalidates the state
    accel_authority_mps2: float = 2.5  # simulated plant: accel at throttle=1.0
    brake_authority_mps2: float = 8.0  # simulated plant: decel at brake=1.0
    drag_per_s: float = 0.02  # simulated plant: linear speed decay, 1/s
    actuator_tau_s: float = 0.15
    """Simulated plant: first-order actuator lag, seconds. 0 makes the plant
    respond instantaneously, which produces infinite jerk on a stepped pedal
    command and puts the arbiter into a permanent LIMITED state -- correctly,
    because no real actuator does that."""

    def __post_init__(self) -> None:
        self.source = _one_of("ego", "source", self.source, EGO_SOURCES)
        validate_config_value("ego.speed_mps", self.speed_mps, 0.0, 100.0)
        validate_config_value("ego.max_age_s", self.max_age_s, 0.01, 60.0)
        validate_config_value("ego.accel_authority_mps2", self.accel_authority_mps2, 0.1, 20.0)
        validate_config_value("ego.brake_authority_mps2", self.brake_authority_mps2, 0.1, 20.0)
        validate_config_value("ego.drag_per_s", self.drag_per_s, 0.0, 1.0)
        validate_config_value("ego.actuator_tau_s", self.actuator_tau_s, 0.0, 5.0)
        if self.source == "file" and not self.file:
            raise ConfigurationError("ego.file is required when ego.source == 'file'")


@dataclass
class TrackerConfig:
    """Multi-object tracker configuration (Kalman range filter + Hungarian)."""

    max_missed_frames: int = 5
    association_threshold_px: float = 120.0  # hard ceiling on the gate, px
    focal_length_px: float = 910.0  # fallback pinhole focal length, px
    object_height_m: float = 1.5  # height prior for unknown labels, m
    min_box_height_px: float = 1.0  # floor before dividing, px (> 0)
    max_distance_m: float = 200.0  # ceiling on a reported range, m
    confirm_hits: int = 3  # M of the M-of-N confirmation
    confirm_window: int = 5  # N of the M-of-N confirmation
    min_range_box_height_px: float = 8.0  # below this the range is UNAVAILABLE
    box_sigma_px: float = 1.5  # assumed detector box-height noise, px
    jerk_psd: float = 1.0  # range-filter process noise, m^2/s^5
    max_detections: int = 64  # bound on assignment work per frame
    max_tracks: int = 64
    use_class_heights: bool = True  # per-class height priors
    use_geometry_range: bool = True  # inject geometry.estimate_range as range_fn

    def __post_init__(self) -> None:
        validate_config_value("tracker.max_missed_frames", self.max_missed_frames, 1, 100)
        validate_config_value(
            "tracker.association_threshold_px", self.association_threshold_px, 1.0, 1000.0
        )
        validate_config_value("tracker.focal_length_px", self.focal_length_px, 1.0, 5000.0)
        validate_config_value("tracker.object_height_m", self.object_height_m, 0.1, 10.0)
        # min_box_height_px: 0.0 used to load and then divide by zero.
        validate_config_value("tracker.min_box_height_px", self.min_box_height_px, 0.1, 100.0)
        validate_config_value("tracker.max_distance_m", self.max_distance_m, 1.0, 500.0)
        validate_config_value("tracker.confirm_hits", self.confirm_hits, 1, 20)
        validate_config_value("tracker.confirm_window", self.confirm_window, 1, 50)
        validate_config_value(
            "tracker.min_range_box_height_px", self.min_range_box_height_px, 1.0, 200.0
        )
        validate_config_value("tracker.box_sigma_px", self.box_sigma_px, 0.1, 50.0)
        validate_config_value("tracker.jerk_psd", self.jerk_psd, 0.0001, 1000.0)
        validate_config_value("tracker.max_detections", self.max_detections, 1, 256)
        validate_config_value("tracker.max_tracks", self.max_tracks, 1, 256)
        if self.confirm_window < self.confirm_hits:
            raise ConfigurationError(
                "tracker.confirm_window (%d) must be >= tracker.confirm_hits (%d)"
                % (self.confirm_window, self.confirm_hits)
            )


@dataclass
class PlannerConfig:
    """Behaviour planner: constant-time-gap ACC plus a lateral law."""

    cruise_speed_mps: float = 15.0  # ~54 km/h
    min_follow_distance_m: float = 12.0  # d0 of the spacing policy, m
    max_steering_deg: float = 22.0  # road-wheel setpoint limit, deg
    time_gap_s: float = 2.0  # T of the spacing policy, s
    max_decel_mps2: float = 3.0  # comfort deceleration limit
    lane_center_gain: float = 1.0
    ego_lane_half_width_frac: float = 0.2  # 0 disables the in-path gate
    max_accel_mps2: float = 2.0
    standstill_gap_m: float = 4.0
    k_distance: float = 0.4  # spacing-error gain, 1/s
    k_speed: float = 0.6  # relative-speed gain, dimensionless
    aeb_ttc_s: float = 0.9
    warn_ttc_s: float = 1.6
    emergency_decel_mps2: float = 8.0
    mrm_decel_mps2: float = 3.5
    wheelbase_m: float = 2.8
    max_lateral_accel_mps2: float = 3.0
    steering_speed_ref_mps: float = 12.0
    allow_uncalibrated_metric_steering: bool = False  # bench only; never in a vehicle

    def __post_init__(self) -> None:
        validate_config_value("planner.cruise_speed_mps", self.cruise_speed_mps, 0.1, 50.0)
        validate_config_value("planner.min_follow_distance_m", self.min_follow_distance_m, 0.0, 100.0)
        validate_config_value("planner.max_steering_deg", self.max_steering_deg, 0.1, 45.0)
        validate_config_value("planner.time_gap_s", self.time_gap_s, 0.5, 5.0)
        validate_config_value("planner.max_decel_mps2", self.max_decel_mps2, 0.1, 15.0)
        validate_config_value("planner.lane_center_gain", self.lane_center_gain, 0.0, 100.0)
        validate_config_value(
            "planner.ego_lane_half_width_frac", self.ego_lane_half_width_frac, 0.0, 1.0
        )
        validate_config_value("planner.max_accel_mps2", self.max_accel_mps2, 0.1, 10.0)
        validate_config_value("planner.standstill_gap_m", self.standstill_gap_m, 0.0, 50.0)
        validate_config_value("planner.k_distance", self.k_distance, 0.0, 10.0)
        validate_config_value("planner.k_speed", self.k_speed, 0.0, 10.0)
        validate_config_value("planner.aeb_ttc_s", self.aeb_ttc_s, 0.1, 10.0)
        validate_config_value("planner.warn_ttc_s", self.warn_ttc_s, 0.1, 20.0)
        validate_config_value("planner.emergency_decel_mps2", self.emergency_decel_mps2, 0.1, 15.0)
        validate_config_value("planner.mrm_decel_mps2", self.mrm_decel_mps2, 0.1, 15.0)
        validate_config_value("planner.wheelbase_m", self.wheelbase_m, 0.5, 20.0)
        validate_config_value(
            "planner.max_lateral_accel_mps2", self.max_lateral_accel_mps2, 0.1, 10.0
        )
        validate_config_value("planner.steering_speed_ref_mps", self.steering_speed_ref_mps, 0.1, 60.0)
        if self.warn_ttc_s < self.aeb_ttc_s:
            raise ConfigurationError(
                "planner.warn_ttc_s (%.2f) must be >= planner.aeb_ttc_s (%.2f)"
                % (self.warn_ttc_s, self.aeb_ttc_s)
            )
        if self.emergency_decel_mps2 < self.max_decel_mps2:
            raise ConfigurationError(
                "planner.emergency_decel_mps2 (%.2f) must be >= planner.max_decel_mps2 (%.2f)"
                % (self.emergency_decel_mps2, self.max_decel_mps2)
            )


@dataclass
class ControllerConfig:
    """Longitudinal PI + pedal map + rate/jerk shaping.

    ``kp_speed``/``ki_speed`` produce an ACCELERATION (1/s and 1/s^2), not a pedal
    fraction; ``accel_authority_mps2``/``brake_authority_mps2`` are what convert
    that acceleration into pedal travel.  If the real vehicle does not deliver
    ``brake_authority_mps2`` at ``brake = 1.0`` the whole stack under-brakes by
    exactly that ratio and nothing in software can detect it.
    """

    kp_speed: float = 0.15  # 1/s
    ki_speed: float = 0.05  # 1/s^2
    max_throttle: float = 1.0
    max_brake: float = 1.0
    max_steering_angle_deg: float = 25.0  # road-wheel angle at steering = 1.0
    steering_deadband_deg: float = 0.5
    accel_authority_mps2: float = 2.5
    brake_authority_mps2: float = 8.0
    speed_deadband_mps: float = 0.3
    speed_hysteresis_mps: float = 0.15
    max_jerk_mps3: float = 4.0
    max_jerk_emergency_mps3: float = 15.0
    throttle_rate_per_s: float = 2.0
    brake_apply_rate_per_s: float = 5.0
    brake_release_rate_per_s: float = 8.0
    integral_limit_mps2: float = 1.0

    def __post_init__(self) -> None:
        validate_config_value("controller.kp_speed", self.kp_speed, 0.0001, 10.0)
        validate_config_value("controller.ki_speed", self.ki_speed, 0.0, 10.0)
        validate_config_value("controller.max_throttle", self.max_throttle, 0.0, 1.0)
        validate_config_value("controller.max_brake", self.max_brake, 0.0, 1.0)
        # max_steering_angle_deg: 1.0 used to load and silently saturate to lock.
        validate_config_value(
            "controller.max_steering_angle_deg", self.max_steering_angle_deg, 1.0, 45.0
        )
        validate_config_value(
            "controller.steering_deadband_deg", self.steering_deadband_deg, 0.0, 10.0
        )
        validate_config_value(
            "controller.accel_authority_mps2", self.accel_authority_mps2, 0.1, 20.0
        )
        validate_config_value(
            "controller.brake_authority_mps2", self.brake_authority_mps2, 0.1, 20.0
        )
        validate_config_value("controller.speed_deadband_mps", self.speed_deadband_mps, 0.0, 10.0)
        validate_config_value(
            "controller.speed_hysteresis_mps", self.speed_hysteresis_mps, 0.0, 10.0
        )
        validate_config_value("controller.max_jerk_mps3", self.max_jerk_mps3, 0.1, 100.0)
        validate_config_value(
            "controller.max_jerk_emergency_mps3", self.max_jerk_emergency_mps3, 0.1, 200.0
        )
        validate_config_value("controller.throttle_rate_per_s", self.throttle_rate_per_s, 0.01, 100.0)
        validate_config_value(
            "controller.brake_apply_rate_per_s", self.brake_apply_rate_per_s, 0.01, 100.0
        )
        validate_config_value(
            "controller.brake_release_rate_per_s", self.brake_release_rate_per_s, 0.01, 100.0
        )
        validate_config_value("controller.integral_limit_mps2", self.integral_limit_mps2, 0.0, 20.0)
        if self.max_jerk_emergency_mps3 < self.max_jerk_mps3:
            raise ConfigurationError(
                "controller.max_jerk_emergency_mps3 must be >= controller.max_jerk_mps3"
            )


@dataclass
class SafetyConfig:
    """Limits the arbiter enforces.  Field names match ``SafetyLimits`` exactly.

    These are the *outer* envelope: the planner's own comfort limits must be
    inside them, and :meth:`RuntimeConfig.__post_init__` refuses a configuration
    where they are not.
    """

    max_speed_mps: float = 33.0  # ~120 km/h
    max_acceleration_mps2: float = 3.0
    max_deceleration_mps2: float = 8.0
    max_steering_rate_rad_s: float = 0.5
    max_steering_angle_rad: float = 0.52  # ~29.8 deg
    min_following_distance_m: float = 2.0  # ABSOLUTE floor, not a headway policy
    max_lateral_offset_m: float = 1.5  # enforced only with a metric lane geometry
    plan_horizon_s: float = 1.0
    max_jerk_mps3: float = 4.0
    max_jerk_emergency_mps3: float = 15.0
    max_lateral_accel_mps2: float = 4.5
    wheelbase_m: float = 2.8
    max_road_wheel_rad: float = 0.0  # 0 = derive from controller.max_steering_angle_deg
    brake_authority_mps2: float = 8.0
    accel_authority_mps2: float = 2.5
    standstill_gap_m: float = 4.0
    reaction_time_s: float = 0.6
    ego_brake_capability_mps2: float = 6.0
    lead_brake_capability_mps2: float = 8.0
    ttc_brake_s: float = 0.9
    ttc_warn_s: float = 1.6
    comfort_decel_mps2: float = 3.0
    mrm_decel_mps2: float = 3.5
    limited_after_dropouts: int = 1
    mrm_after_dropouts: int = 3
    disengage_after_frames: int = 40
    recovery_frames: int = 10

    def __post_init__(self) -> None:
        validate_config_value("safety.max_speed_mps", self.max_speed_mps, 0.1, 100.0)
        validate_config_value("safety.max_acceleration_mps2", self.max_acceleration_mps2, 0.1, 10.0)
        validate_config_value("safety.max_deceleration_mps2", self.max_deceleration_mps2, 0.1, 15.0)
        validate_config_value(
            "safety.max_steering_rate_rad_s", self.max_steering_rate_rad_s, 0.01, 10.0
        )
        validate_config_value(
            "safety.max_steering_angle_rad", self.max_steering_angle_rad, 0.01, 1.5
        )
        validate_config_value(
            "safety.min_following_distance_m", self.min_following_distance_m, 0.0, 100.0
        )
        validate_config_value("safety.max_lateral_offset_m", self.max_lateral_offset_m, 0.1, 10.0)
        validate_config_value("safety.plan_horizon_s", self.plan_horizon_s, 0.05, 5.0)
        validate_config_value("safety.max_jerk_mps3", self.max_jerk_mps3, 0.1, 100.0)
        validate_config_value(
            "safety.max_jerk_emergency_mps3", self.max_jerk_emergency_mps3, 0.1, 200.0
        )
        validate_config_value(
            "safety.max_lateral_accel_mps2", self.max_lateral_accel_mps2, 0.1, 15.0
        )
        validate_config_value("safety.wheelbase_m", self.wheelbase_m, 0.5, 20.0)
        validate_config_value("safety.max_road_wheel_rad", self.max_road_wheel_rad, 0.0, 1.5)
        validate_config_value("safety.brake_authority_mps2", self.brake_authority_mps2, 0.1, 20.0)
        validate_config_value("safety.accel_authority_mps2", self.accel_authority_mps2, 0.1, 20.0)
        validate_config_value("safety.standstill_gap_m", self.standstill_gap_m, 0.0, 50.0)
        validate_config_value("safety.reaction_time_s", self.reaction_time_s, 0.0, 5.0)
        validate_config_value(
            "safety.ego_brake_capability_mps2", self.ego_brake_capability_mps2, 0.1, 20.0
        )
        validate_config_value(
            "safety.lead_brake_capability_mps2", self.lead_brake_capability_mps2, 0.1, 20.0
        )
        validate_config_value("safety.ttc_brake_s", self.ttc_brake_s, 0.1, 10.0)
        validate_config_value("safety.ttc_warn_s", self.ttc_warn_s, 0.1, 20.0)
        validate_config_value("safety.comfort_decel_mps2", self.comfort_decel_mps2, 0.1, 15.0)
        validate_config_value("safety.mrm_decel_mps2", self.mrm_decel_mps2, 0.1, 15.0)
        validate_config_value("safety.limited_after_dropouts", self.limited_after_dropouts, 1, 1000)
        validate_config_value("safety.mrm_after_dropouts", self.mrm_after_dropouts, 1, 1000)
        validate_config_value("safety.disengage_after_frames", self.disengage_after_frames, 1, 10000)
        validate_config_value("safety.recovery_frames", self.recovery_frames, 0, 10000)
        if self.mrm_after_dropouts < self.limited_after_dropouts:
            raise ConfigurationError(
                "safety.mrm_after_dropouts must be >= safety.limited_after_dropouts"
            )


@dataclass
class SourceConfig:
    """Frame source configuration.

    ``reconnect_attempts`` applies to ``video``/``camera`` only.  End of file on a
    replay clip is not a fault and never triggers a reconnect; a mid-stream read
    failure does.
    """

    type: str = "synthetic"  # synthetic | video | camera
    uri: str = ""
    width: int = 1280
    height: int = 720
    loop: bool = False  # video only: restart at EOF (bench / soak testing)
    reconnect_attempts: int = 3  # 0 disables reconnection
    reconnect_delay_s: float = 1.0

    def __post_init__(self) -> None:
        self.type = _one_of("source", "type", self.type, SOURCE_TYPES)
        validate_config_value("source.width", self.width, 16, 7680)
        validate_config_value("source.height", self.height, 16, 4320)
        validate_config_value("source.reconnect_attempts", self.reconnect_attempts, 0, 100)
        validate_config_value("source.reconnect_delay_s", self.reconnect_delay_s, 0.0, 60.0)
        if self.type == "video" and not self.uri:
            raise ConfigurationError("source.uri is required when source.type == 'video'")


@dataclass
class HealthConfig:
    """Health/metrics HTTP endpoint (``adas.io.health``).

    The body carries live safety state, ego speed and lead range, so a
    non-loopback bind needs ``allow_remote`` set on purpose.
    """

    enabled: bool = True
    bind: str = "127.0.0.1"
    port: int = 8090  # DMS uses 8088
    allow_remote: bool = False
    token_file: str = ""
    max_connections: int = 16
    required: bool = False  # True: failing to bind is fatal

    def __post_init__(self) -> None:
        validate_config_value("health.port", self.port, 0, 65535)
        validate_config_value("health.max_connections", self.max_connections, 1, 1024)
        if not self.bind:
            raise ConfigurationError("health.bind must not be empty")


@dataclass
class EventsConfig:
    """Durable safety-event JSONL (``adas.io.events``).

    ``backups`` must stay consistent with ``rotate`` in ``deploy/adas.logrotate``.
    """

    enabled: bool = True
    path: str = ""  # "" selects the default for the deployment kind
    fsync: str = "critical"  # never | interval | critical | always
    fsync_interval_s: float = 1.0
    max_mb: float = 32.0
    backups: int = 3
    min_free_mb: float = 64.0
    strict: bool = False  # True: an unopenable log is fatal

    def __post_init__(self) -> None:
        if self.fsync not in ("never", "interval", "critical", "always"):
            raise ConfigurationError("events.fsync must be never|interval|critical|always")
        validate_config_value("events.fsync_interval_s", self.fsync_interval_s, 0.0, 3600.0)
        validate_config_value("events.max_mb", self.max_mb, 0.0, 10240.0)
        validate_config_value("events.backups", self.backups, 0, 100)
        validate_config_value("events.min_free_mb", self.min_free_mb, 0.0, 1000000.0)


@dataclass
class RuntimeConfig:
    """Complete runtime configuration.

    :meth:`__post_init__` performs the cross-section checks.  They exist because
    each individual limit can be legal while the *combination* is not: a planner
    allowed to steer 30 deg under a safety monitor that vetoes above 25 deg is a
    system that clamps on every curve and logs a violation each time.
    """

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    lane: LaneConfig = field(default_factory=LaneConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    ego: EgoConfig = field(default_factory=EgoConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    fps: int = 20
    log_level: str = "INFO"
    log_format: str = "text"  # text | json
    allow_mock: bool = False
    """Explicit opt-in to a fabricating backend. Without it a ``mock`` detector or
    lane backend is refused at load time (ADAS-PERC-24: the mock used to be the
    silent default, and nothing downstream could tell it apart from perception)."""
    schema_version: int = SCHEMA_VERSION
    config_path: str = ""  # provenance; filled by load_config

    def __post_init__(self) -> None:
        validate_config_value("fps", self.fps, 1, 120)
        if self.log_level not in LOG_LEVELS:
            raise ConfigurationError(
                "log_level must be one of %s, got %r" % ("|".join(LOG_LEVELS), self.log_level)
            )
        if self.log_format not in ("text", "json"):
            raise ConfigurationError("log_format must be 'text' or 'json', got %r" % self.log_format)
        if self.schema_version > SCHEMA_VERSION:
            raise ConfigurationError(
                "config schema_version %d is newer than this build understands (%d)"
                % (self.schema_version, SCHEMA_VERSION)
            )

        # --- derive the one value that must not be allowed to disagree --------
        derived_road_wheel = math.radians(self.controller.max_steering_angle_deg)
        if self.safety.max_road_wheel_rad == 0.0:
            self.safety.max_road_wheel_rad = derived_road_wheel
        elif abs(self.safety.max_road_wheel_rad - derived_road_wheel) > 1e-3:
            raise ConfigurationError(
                "safety.max_road_wheel_rad (%.4f rad) must equal "
                "radians(controller.max_steering_angle_deg) (%.4f rad), or be 0 to derive it. "
                "The arbiter converts a normalised steering command to a physical angle "
                "with this number; a mismatch silently mis-scales every steering check."
                % (self.safety.max_road_wheel_rad, derived_road_wheel)
            )

        for message in self.cross_validation_errors():
            raise ConfigurationError(message)

    def cross_validation_errors(self) -> List[str]:
        """Return every cross-section inconsistency, as human-readable strings."""
        errors: List[str] = []
        if self.planner.cruise_speed_mps > self.safety.max_speed_mps:
            errors.append(
                "planner.cruise_speed_mps (%.1f) exceeds safety.max_speed_mps (%.1f)"
                % (self.planner.cruise_speed_mps, self.safety.max_speed_mps)
            )
        planner_rad = math.radians(self.planner.max_steering_deg)
        if planner_rad > self.safety.max_steering_angle_rad + 1e-9:
            errors.append(
                "planner.max_steering_deg (%.1f deg = %.3f rad) exceeds "
                "safety.max_steering_angle_rad (%.3f rad)"
                % (self.planner.max_steering_deg, planner_rad, self.safety.max_steering_angle_rad)
            )
        if self.controller.max_steering_angle_deg < self.planner.max_steering_deg - 1e-9:
            errors.append(
                "controller.max_steering_angle_deg (%.1f) is below planner.max_steering_deg "
                "(%.1f): the planner's setpoint could never be actuated"
                % (self.controller.max_steering_angle_deg, self.planner.max_steering_deg)
            )
        if self.planner.min_follow_distance_m < self.safety.min_following_distance_m - 1e-9:
            errors.append(
                "planner.min_follow_distance_m (%.1f) is below safety.min_following_distance_m "
                "(%.1f)" % (self.planner.min_follow_distance_m, self.safety.min_following_distance_m)
            )
        if self.planner.max_accel_mps2 > self.safety.max_acceleration_mps2 + 1e-9:
            errors.append(
                "planner.max_accel_mps2 (%.1f) exceeds safety.max_acceleration_mps2 (%.1f)"
                % (self.planner.max_accel_mps2, self.safety.max_acceleration_mps2)
            )
        if self.planner.emergency_decel_mps2 > self.safety.max_deceleration_mps2 + 1e-9:
            errors.append(
                "planner.emergency_decel_mps2 (%.1f) exceeds safety.max_deceleration_mps2 (%.1f)"
                % (self.planner.emergency_decel_mps2, self.safety.max_deceleration_mps2)
            )
        if self.controller.brake_authority_mps2 != self.safety.brake_authority_mps2:
            errors.append(
                "controller.brake_authority_mps2 (%.2f) and safety.brake_authority_mps2 (%.2f) "
                "must agree: they are the same physical property of the vehicle and the "
                "arbiter's fail-safe brake fraction would not match the controller's"
                % (self.controller.brake_authority_mps2, self.safety.brake_authority_mps2)
            )
        mocks = [
            "%s.backend=mock" % name
            for name in ("detector", "lane")
            if getattr(self, name).backend == "mock"
        ]
        if mocks and not self.allow_mock:
            errors.append(
                "%s selected but allow_mock is false. A mock backend FABRICATES "
                "geometry that nothing downstream can distinguish from a measurement; "
                "pass --allow-mock (or set ADAS_ALLOW_MOCK=1, or \"allow_mock\": true) "
                "to accept that for bench work, or choose a real backend."
                % " and ".join(mocks)
            )
        if self.depth.backend != "off" and not self.camera.enabled:
            errors.append(
                "depth.backend is %r but camera.enabled is false: the depth channel needs a "
                "camera model to anchor its scale on the road plane, and without one every "
                "estimate would be UNAVAILABLE" % self.depth.backend
            )
        return errors

    # ------------------------------------------------------------------ helpers

    def describe(self) -> str:
        """One-line provenance summary for the startup banner and the event log."""
        return (
            "detector=%s(%s) lane=%s(%s) depth=%s camera=%s ego=%s source=%s fps=%d"
            % (
                self.detector.backend,
                Path(self.detector.model_path).name if self.detector.model_path else "-",
                self.lane.backend,
                Path(self.lane.model_path).name if self.lane.model_path else "-",
                self.depth.backend,
                "calibrated" if (self.camera.enabled and self.camera.calibrated) else
                ("assumed" if self.camera.enabled else "off"),
                self.ego.source,
                self.source.type,
                self.fps,
            )
        )


#: Sections of the JSON document, in the order they are built.
_SECTIONS: Tuple[Tuple[str, type], ...] = (
    ("detector", DetectorConfig),
    ("tracker", TrackerConfig),
    ("planner", PlannerConfig),
    ("controller", ControllerConfig),
    ("safety", SafetyConfig),
    ("lane", LaneConfig),
    ("depth", DepthConfig),
    ("camera", CameraConfig),
    ("ego", EgoConfig),
    ("source", SourceConfig),
    ("health", HealthConfig),
    ("events", EventsConfig),
)

_SCALARS = ("fps", "log_level", "log_format", "allow_mock", "schema_version")


def default_config() -> RuntimeConfig:
    """A fresh default configuration.

    A function, not a module constant: ``RuntimeConfig`` is mutable and the CLI
    applies overrides to it, so a shared singleton would leak between runs (and
    between tests).
    """
    return RuntimeConfig()


#: Kept for backwards compatibility with code that imported the old constant.
#: Prefer :func:`default_config`; this object is rebuilt on every import only.
DEFAULT_CONFIG = default_config()


def load_config(path: Union[str, Path, None] = None) -> RuntimeConfig:
    """Load, coerce, validate and cross-validate a configuration.

    Args:
        path: JSON file, or None for the built-in defaults.

    Returns:
        A fully validated :class:`RuntimeConfig`.

    Raises:
        ConfigurationError: file missing, invalid JSON, unknown key, out-of-range
            value, or a cross-section inconsistency.  Every message names the
            dotted key so an operator can fix it without reading the source.
    """
    if path is None:
        logger.debug("Using default configuration")
        return default_config()

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError("Configuration file not found: %s" % path)

    try:
        payload = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigurationError("Invalid JSON in %s: %s" % (path, exc)) from exc
    except OSError as exc:
        raise ConfigurationError("Could not read %s: %s" % (path, exc)) from exc

    if not isinstance(payload, dict):
        raise ConfigurationError("Configuration root must be a JSON object, got %s" % type(payload).__name__)

    known_top = {name for name, _ in _SECTIONS} | set(_SCALARS) | {"comment", "_comment"}
    unknown = sorted(set(payload) - known_top)
    if unknown:
        raise ConfigurationError(
            "unknown top-level config key(s): %s. Known: %s"
            % (", ".join(unknown), ", ".join(sorted(known_top)))
        )

    version = payload.get("schema_version", SCHEMA_VERSION)
    if not isinstance(version, int) or isinstance(version, bool):
        raise ConfigurationError("schema_version must be an integer, got %r" % (version,))

    defaults = default_config()
    try:
        sections = {
            name: _fill(cls, payload.get(name, {}), getattr(defaults, name), name)
            for name, cls in _SECTIONS
        }
        config = RuntimeConfig(
            fps=_coerce("fps", payload.get("fps", defaults.fps), defaults.fps),
            log_level=str(payload.get("log_level", defaults.log_level)).upper(),
            log_format=str(payload.get("log_format", defaults.log_format)).lower(),
            allow_mock=_coerce("allow_mock", payload.get("allow_mock", defaults.allow_mock), False),
            schema_version=version,
            config_path=str(config_path),
            **sections
        )
    except ValidationError as exc:
        # A range check is a configuration problem from the operator's point of
        # view; re-raising it as one keeps every load failure a single type.
        raise ConfigurationError("%s: %s" % (config_path, exc)) from exc
    logger.debug("Configuration loaded from %s: %s", config_path, config.describe())
    return config


__all__ = [
    "CameraConfig",
    "ControllerConfig",
    "DEFAULT_CONFIG",
    "DEPRECATED_KEYS",
    "DEPTH_BACKENDS",
    "DETECTOR_BACKENDS",
    "DepthConfig",
    "DetectorConfig",
    "EGO_SOURCES",
    "EgoConfig",
    "EventsConfig",
    "HealthConfig",
    "LANE_BACKENDS",
    "LaneConfig",
    "PlannerConfig",
    "RuntimeConfig",
    "SCHEMA_VERSION",
    "SafetyConfig",
    "SourceConfig",
    "TrackerConfig",
    "default_config",
    "load_config",
]
