"""CLI entry point: build the pipeline, wire the ops layer, run the frame loop.

This module is the integration surface.  Everything the other workstreams built
is assembled here exactly once, in this order:

1. **Logging first.**  The level is resolved from ``--log-level``, then
   ``ADAS_LOG_LEVEL``, then the config file, and :func:`configure_logging` runs
   *before* the configuration is loaded, so ``--log-level ERROR`` actually
   suppresses the loader's own output (ADAS-OPS-07).  Nothing later calls
   ``logging.getLogger().setLevel`` behind the flag's back.
2. **Configuration**, strictly validated and cross-validated.
3. **Camera model**, rescaled to the live frame size.
4. **Perception backends** through :mod:`adas.perception.factory`, which refuses
   a fabricating stub unless the operator opted in.
5. **Tracker, planner, controller, safety monitor**, every config value wired --
   the previous version silently ignored roughly half of them.
6. **Ops layer**: health endpoint, event log, systemd readiness and watchdog.
   Without this a ``Type=notify`` unit never receives ``READY=1`` and systemd
   fails the start at ``TimeoutStartSec``.
7. **Signals**: SIGTERM/SIGINT stop the loop cooperatively, SIGHUP reopens the
   event log for ``logrotate``.

Environment variables actually read here: ``ADAS_CONFIG_PATH``,
``ADAS_LOG_LEVEL``, ``ADAS_LOG_FORMAT``, ``ADAS_ALLOW_MOCK``.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
from typing import Any, Optional, Tuple

import adas
from adas.control import PIDLikeLongitudinalController, SafetyLimits, SafetyMonitor
from adas.core.config import RuntimeConfig, load_config
from adas.core.exceptions import (
    ADASException,
    ConfigurationError,
    PerceptionError,
    ValidationError,
)
from adas.core.logger import configure_logging, setup_logger
from adas.perception.factory import (
    allow_mock_from_env,
    build_camera,
    build_depth_channel,
    build_detector,
    build_lane_estimator,
)
from adas.infer.trt_engine import EngineError as TrtEngineError
from adas.planning import BehaviorPlanner
from adas.runtime import ADASPipeline, PipelineRunner, RunnerHooks
from adas.runtime.capture import open_ego_source, parse_source_arg
from adas.tracking import MultiObjectTracker

logger = setup_logger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def build_pipeline(
    config_path: Optional[str] = None,
    config: Optional[RuntimeConfig] = None,
) -> Tuple[ADASPipeline, RuntimeConfig]:
    """Build a fully wired :class:`ADASPipeline`.

    Args:
        config_path: JSON configuration file, or ``None`` for the defaults.
        config: an already-loaded configuration, which takes precedence.

    Returns:
        ``(pipeline, config)``.

    Raises:
        ConfigurationError: an invalid configuration, or a mock backend without
            ``allow_mock``.
        PerceptionError: a required engine is missing or will not load.
    """
    if config is None:
        config = load_config(config_path)

    logger.info("Building ADAS pipeline: %s", config.describe())

    camera = build_camera(config.camera, config.source.width, config.source.height)

    # Load the LARGEST engine first. On the Xavier NX's unified memory a single
    # contiguous cudaMalloc is what fails, not the total: UFLD-v2 needs 413 MB in
    # one block and reliably fails to deserialise if YOLOX's context, cuDNN and
    # cuBLAS workspaces have already fragmented the carveout. Measured on this
    # board: lane-then-detector loads with ~1.1 GB available, detector-then-lane
    # does not.
    lane_estimator = build_lane_estimator(
        config.lane,
        camera=camera,
        allow_mock=config.allow_mock,
        allow_fallback=config.allow_mock,
    )
    detector = build_detector(
        config.detector, allow_mock=config.allow_mock, allow_fallback=config.allow_mock
    )
    depth_channel = build_depth_channel(config.depth)

    range_fn = None
    if camera is not None and config.tracker.use_geometry_range:
        from adas.perception.geometry import estimate_range

        def range_fn(box, frame_w, frame_h):  # noqa: E306 - small local adapter
            """Ground-plane + pinhole fused range, the only channel that can
            correct a vertically truncated box (ADAS-DEC-10b)."""
            return estimate_range(box, camera, frame_w, frame_h)

    tracker = MultiObjectTracker(
        max_missed=config.tracker.max_missed_frames,
        association_threshold_px=config.tracker.association_threshold_px,
        focal_length_px=config.tracker.focal_length_px,
        object_height_m=config.tracker.object_height_m,
        min_box_height_px=config.tracker.min_box_height_px,
        max_distance_m=config.tracker.max_distance_m,
        confirm_hits=config.tracker.confirm_hits,
        confirm_window=config.tracker.confirm_window,
        nominal_dt_s=1.0 / float(config.fps),
        box_sigma_px=config.tracker.box_sigma_px,
        jerk_psd=config.tracker.jerk_psd,
        min_range_box_height_px=config.tracker.min_range_box_height_px,
        use_class_heights=config.tracker.use_class_heights,
        max_detections=config.tracker.max_detections,
        max_tracks=config.tracker.max_tracks,
        frame_width_px=config.source.width,
        frame_height_px=config.source.height,
        camera=camera,
        range_fn=range_fn,
    )

    planner_camera = None
    if camera is not None:
        from adas.planning.lateral import CameraGeometry

        planner_camera = CameraGeometry.from_camera_config(
            camera, allow_uncalibrated=config.planner.allow_uncalibrated_metric_steering
        )

    planner = BehaviorPlanner(
        cruise_speed_mps=config.planner.cruise_speed_mps,
        min_follow_distance_m=config.planner.min_follow_distance_m,
        max_steering_deg=config.planner.max_steering_deg,
        time_gap_s=config.planner.time_gap_s,
        max_decel_mps2=config.planner.max_decel_mps2,
        lane_center_gain=config.planner.lane_center_gain,
        ego_lane_half_width_frac=config.planner.ego_lane_half_width_frac,
        max_accel_mps2=config.planner.max_accel_mps2,
        standstill_gap_m=config.planner.standstill_gap_m,
        k_distance=config.planner.k_distance,
        k_speed=config.planner.k_speed,
        aeb_ttc_s=config.planner.aeb_ttc_s,
        warn_ttc_s=config.planner.warn_ttc_s,
        emergency_decel_mps2=config.planner.emergency_decel_mps2,
        mrm_decel_mps2=config.planner.mrm_decel_mps2,
        wheelbase_m=config.planner.wheelbase_m,
        max_lateral_accel_mps2=config.planner.max_lateral_accel_mps2,
        steering_speed_ref_mps=config.planner.steering_speed_ref_mps,
        camera=planner_camera,
    )

    controller = PIDLikeLongitudinalController(
        kp_speed=config.controller.kp_speed,
        ki_speed=config.controller.ki_speed,
        max_throttle=config.controller.max_throttle,
        max_brake=config.controller.max_brake,
        max_steering_angle_deg=config.controller.max_steering_angle_deg,
        steering_deadband_deg=config.controller.steering_deadband_deg,
        accel_authority_mps2=config.controller.accel_authority_mps2,
        brake_authority_mps2=config.controller.brake_authority_mps2,
        speed_deadband_mps=config.controller.speed_deadband_mps,
        speed_hysteresis_mps=config.controller.speed_hysteresis_mps,
        max_jerk_mps3=config.controller.max_jerk_mps3,
        max_jerk_emergency_mps3=config.controller.max_jerk_emergency_mps3,
        throttle_rate_per_s=config.controller.throttle_rate_per_s,
        brake_apply_rate_per_s=config.controller.brake_apply_rate_per_s,
        brake_release_rate_per_s=config.controller.brake_release_rate_per_s,
        integral_limit_mps2=config.controller.integral_limit_mps2,
    )

    safety_monitor = SafetyMonitor(limits=build_safety_limits(config))

    pipeline = ADASPipeline(
        detector=detector,
        lane_estimator=lane_estimator,
        tracker=tracker,
        planner=planner,
        controller=controller,
        safety_monitor=safety_monitor,
        depth_channel=depth_channel,
        camera=camera,
        ego_lane_half_width_frac=config.planner.ego_lane_half_width_frac,
        lane_every_n_frames=config.lane.every_n_frames,
        lane_max_age_frames=config.lane.max_age_frames,
    )

    logger.info(
        "Pipeline built: detector=%s lane=%s depth=%s camera=%s metric_steering=%s",
        type(detector).__name__,
        type(lane_estimator).__name__,
        "off" if depth_channel is None else ("real" if depth_channel.available else "stub"),
        "none" if camera is None else camera.label,
        bool(planner_camera is not None and planner_camera.is_configured()),
    )
    return pipeline, config


def build_safety_limits(config: RuntimeConfig) -> SafetyLimits:
    """Project ``SafetyConfig`` onto the arbiter's limit object.

    ``max_road_wheel_rad`` is set explicitly from the controller rather than left
    at its default: it is the constant that converts a normalised steering
    command into a physical road-wheel angle, and a disagreement silently
    mis-scales every steering check.  :meth:`RuntimeConfig.__post_init__` has
    already derived or verified it.
    """
    s = config.safety
    return SafetyLimits(
        max_speed_mps=s.max_speed_mps,
        max_acceleration_mps2=s.max_acceleration_mps2,
        max_deceleration_mps2=s.max_deceleration_mps2,
        max_steering_rate_rad_s=s.max_steering_rate_rad_s,
        max_steering_angle_rad=s.max_steering_angle_rad,
        min_following_distance_m=s.min_following_distance_m,
        max_lateral_offset_m=s.max_lateral_offset_m,
        plan_horizon_s=s.plan_horizon_s,
        max_jerk_mps3=s.max_jerk_mps3,
        max_jerk_emergency_mps3=s.max_jerk_emergency_mps3,
        max_lateral_accel_mps2=s.max_lateral_accel_mps2,
        wheelbase_m=s.wheelbase_m,
        max_road_wheel_rad=s.max_road_wheel_rad or math.radians(config.controller.max_steering_angle_deg),
        brake_authority_mps2=s.brake_authority_mps2,
        accel_authority_mps2=s.accel_authority_mps2,
        standstill_gap_m=s.standstill_gap_m,
        reaction_time_s=s.reaction_time_s,
        ego_brake_capability_mps2=s.ego_brake_capability_mps2,
        lead_brake_capability_mps2=s.lead_brake_capability_mps2,
        ttc_brake_s=s.ttc_brake_s,
        ttc_warn_s=s.ttc_warn_s,
        comfort_decel_mps2=s.comfort_decel_mps2,
        mrm_decel_mps2=s.mrm_decel_mps2,
        limited_after_dropouts=s.limited_after_dropouts,
        mrm_after_dropouts=s.mrm_after_dropouts,
        disengage_after_frames=s.disengage_after_frames,
        recovery_frames=s.recovery_frames,
    )


# --------------------------------------------------------------------------- #
# Ops wiring
# --------------------------------------------------------------------------- #


class _Ops:
    """Health endpoint, event log and systemd watchdog for one run.

    Everything here is best-effort: a refused port or an unwritable log directory
    degrades the *observability* of the process, and must never stop it driving.
    ``health.required`` / ``events.strict`` turn either into a hard failure for a
    deployment that would rather not run blind.
    """

    def __init__(self, config: RuntimeConfig, pipeline: ADASPipeline) -> None:
        from adas.io import EventLog, HealthServer, HealthState, WatchdogPinger
        from adas.io.events import from_config as events_from_config

        self.config = config
        self.state = HealthState()
        self.state.set_build(
            version=adas.__version__,
            config=config.config_path or "<defaults>",
            detector=config.detector.backend,
            lane=config.lane.backend,
        )
        with self.state.lock:
            self.state.detector_backend = config.detector.backend
            self.state.lane_backend = config.lane.backend
            self.state.source_type = config.source.type
            self.state.source_uri = config.source.uri
            self.state.ego_speed_valid = False

        self.state.set_engine("detector", _engine_status(pipeline.detector, config.detector.backend))
        self.state.set_engine("lane", _engine_status(pipeline.lane_estimator, config.lane.backend))
        if pipeline.depth_channel is not None:
            self.state.set_engine(
                "depth", "loaded" if pipeline.depth_channel.available else "mock"
            )
        if config.allow_mock:
            self.state.note("allow_mock is set: a fabricating backend may be active")
        if not (config.camera.enabled and config.camera.calibrated):
            self.state.note("camera is not calibrated: metric outputs are assumptions")

        self.server: Any = None
        if config.health.enabled:
            self.server = HealthServer.from_config(self.state, config)
            port = self.server.start()
            if port:
                logger.info("Health endpoint on http://%s:%d/healthz", config.health.bind, port)

        self.events: Any = None
        if config.events.enabled:
            self.events = events_from_config(config)
            self.events.lifecycle(
                "started",
                version=adas.__version__,
                config=config.config_path or "<defaults>",
                summary=config.describe(),
            )
            with self.state.lock:
                self.state.events = dict(self.events.stats())

        self.watchdog = WatchdogPinger()

    def hooks(self, on_ready) -> RunnerHooks:
        return RunnerHooks(
            health=self.state,
            events=self.events,
            watchdog=self.watchdog,
            on_ready=on_ready,
        )

    def reopen_events(self) -> None:
        if self.events is not None:
            self.events.reopen()

    def close(self, reason: str) -> None:
        if self.events is not None:
            self.events.lifecycle("stopping", reason=reason)
            self.events.close()
        if self.server is not None:
            self.server.stop()


def _engine_status(component: Any, backend: str) -> str:
    """Map a built backend onto the health vocabulary."""
    if backend == "mock" or getattr(component, "is_mock", False):
        return "mock"
    return "loaded"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adas.cli",
        description="ADAS Core - camera-only ADAS reference pipeline for Jetson Xavier NX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment: ADAS_CONFIG_PATH, ADAS_LOG_LEVEL, ADAS_LOG_FORMAT, "
            "ADAS_ALLOW_MOCK.\n"
            "Mock backends FABRICATE perception and must be enabled explicitly."
        ),
    )
    parser.add_argument("--config", default=None, help="JSON configuration file")
    parser.add_argument(
        "--source",
        default=None,
        help="synthetic (default), a video path, camera, csi, or camera:N",
    )
    parser.add_argument(
        "--detector", choices=["mock", "tensorrt", "yolov5", "yolox"], default=None,
        help="Override detector backend",
    )
    parser.add_argument(
        "--lane", choices=["mock", "ufld", "yolop", "twinlite"], default=None,
        help="Override lane backend",
    )
    parser.add_argument(
        "--depth", choices=["off", "midas"], default=None,
        help="Override the independent range channel",
    )
    parser.add_argument(
        "--ego-source", choices=["none", "config", "simulated", "file"], default=None,
        help="Where ego speed comes from. There is no vehicle bus on this board.",
    )
    parser.add_argument(
        "--ego-speed", type=float, default=None,
        help="Ego speed in m/s for --ego-source config, or the initial value for simulated",
    )
    parser.add_argument(
        "--ego-file", default=None, help="Recorded ego speed channel (CSV or JSON)"
    )
    parser.add_argument("--synthetic", action="store_true", help="Force the synthetic source")
    parser.add_argument(
        "--frames", type=int, default=60,
        help="Frames to process; 0 or negative runs until stopped (SIGTERM/SIGINT)",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Override the pacing target. 0 runs unpaced, for throughput measurement.",
    )
    parser.add_argument("--loop", action="store_true", help="Restart a video file at EOF")
    parser.add_argument(
        "--allow-mock", action="store_true",
        help="Permit FABRICATING backends. Bench use only; never in a vehicle.",
    )
    parser.add_argument("--no-health", action="store_true", help="Do not start the health endpoint")
    parser.add_argument("--no-events", action="store_true", help="Do not write the event log")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default=None
    )
    parser.add_argument("--log-format", choices=["text", "json"], default=None)
    parser.add_argument(
        "--print-config", action="store_true",
        help="Validate the configuration, print the resolved summary and exit",
    )
    return parser


def apply_overrides(config: RuntimeConfig, args: argparse.Namespace) -> RuntimeConfig:
    """Apply CLI overrides and re-run validation.

    Overrides are applied to the dataclasses and then the whole
    :class:`RuntimeConfig` is rebuilt, so an override that breaks a cross-section
    invariant (``--detector mock`` without ``--allow-mock``, a lane backend whose
    engine does not exist) is refused here rather than half-applied.
    """
    if args.allow_mock or allow_mock_from_env():
        config.allow_mock = True
    if args.detector:
        config.detector.backend = args.detector
        if args.detector == "yolox" and "yolox" not in config.detector.model_path:
            config.detector.model_path = "models/yolox_nano.engine"
        elif args.detector == "yolov5" and "yolov5" not in config.detector.model_path:
            config.detector.model_path = "models/yolov5n.engine"
    if args.lane:
        config.lane.backend = args.lane
        defaults = {
            "ufld": "models/ufldv2_culane_res18.engine",
            "yolop": "models/yolop_640.engine",
            "twinlite": "models/twinlite_360x640.engine",
        }
        if args.lane in defaults:
            config.lane.model_path = defaults[args.lane]
            if args.lane == "yolop" and config.lane.every_n_frames == 1:
                # 27 ms of GPU and ~59 ms end to end: it cannot run per frame.
                config.lane.every_n_frames = 4
    if args.depth:
        config.depth.backend = args.depth
    if args.ego_source:
        config.ego.source = args.ego_source
    if args.ego_speed is not None:
        config.ego.speed_mps = args.ego_speed
    if args.ego_file:
        config.ego.file = args.ego_file
        config.ego.source = "file"
    if args.source:
        source_type, uri = parse_source_arg(args.source)
        config.source.type = source_type
        config.source.uri = uri
    elif args.synthetic:
        config.source.type = "synthetic"
        config.source.uri = ""
    if args.loop:
        config.source.loop = True
    if args.fps is not None and args.fps >= 1.0:
        # fps 0 means "do not pace"; it is a runner setting, not a config value,
        # because every filter's nominal period is still derived from config.fps.
        config.fps = int(round(args.fps))
    if args.no_health:
        config.health.enabled = False
    if args.no_events:
        config.events.enabled = False
    if args.log_level:
        config.log_level = args.log_level
    if args.log_format:
        config.log_format = args.log_format

    # Re-run every dataclass validator plus the cross-section checks.
    for section in (
        config.detector, config.lane, config.depth, config.camera, config.ego,
        config.tracker, config.planner, config.controller, config.safety,
        config.source, config.health, config.events,
    ):
        section.__post_init__()
    config.__post_init__()
    return config


def resolve_log_settings(args: argparse.Namespace) -> Tuple[str, str]:
    """Resolve the logging level and format BEFORE the config file is read."""
    level = args.log_level or os.environ.get("ADAS_LOG_LEVEL") or "INFO"
    fmt = args.log_format or os.environ.get("ADAS_LOG_FORMAT") or "text"
    return str(level).upper(), str(fmt).lower()


def main(argv: Optional[list] = None) -> int:
    """Run the ADAS pipeline.  Returns a process exit code."""
    args = build_parser().parse_args(argv)

    level, fmt = resolve_log_settings(args)
    configure_logging(level, fmt=fmt)

    config_path = args.config or os.environ.get("ADAS_CONFIG_PATH") or None

    try:
        config = load_config(config_path)
        config = apply_overrides(config, args)
    except (ConfigurationError, ValidationError) as exc:
        logger.error("Configuration error: %s", exc)
        return EXIT_CONFIG

    # A config-file level only applies when neither the flag nor the env var did.
    if not args.log_level and not os.environ.get("ADAS_LOG_LEVEL"):
        configure_logging(config.log_level, fmt=config.log_format)

    if args.print_config:
        print(config.describe())
        for name in ("detector", "lane", "depth", "camera", "ego", "source", "health", "events"):
            print("%-10s %s" % (name, getattr(config, name)))
        return EXIT_OK

    if config.allow_mock:
        logger.warning(
            "ALLOW_MOCK IS SET. A fabricating backend may be active; this process is "
            "NOT FOR VEHICLE USE."
        )

    pipeline: Optional[ADASPipeline] = None
    ops: Optional[_Ops] = None
    runner: Optional[PipelineRunner] = None
    reason = "completed"
    exit_code = EXIT_OK

    try:
        pipeline, config = build_pipeline(config=config)
        ego_source = open_ego_source(config.ego)
        ops = _Ops(config, pipeline)

        needs_image = config.detector.backend != "mock" or config.lane.backend != "mock"
        ready_sent = []

        def on_ready() -> None:
            from adas.io import notify_ready

            if not ready_sent:
                ready_sent.append(True)
                notify_ready("frame 1 processed")

        target_fps = float(config.fps) if args.fps is None else max(0.0, float(args.fps))
        runner = PipelineRunner(
            pipeline,
            target_fps=target_fps,
            ego_source=ego_source,
            hooks=ops.hooks(on_ready),
        )
        _install_signal_handlers(runner, ops)

        logger.info(
            "Starting ADAS: %s (fps=%d, frames=%s)",
            config.describe(),
            config.fps,
            "continuous" if args.frames <= 0 else args.frames,
        )
        summary = runner.run(
            source_type=config.source.type,
            uri=config.source.uri,
            max_frames=args.frames,
            width=config.source.width,
            height=config.source.height,
            as_image=needs_image,
            loop=config.source.loop,
            reconnect_attempts=config.source.reconnect_attempts,
            reconnect_delay_s=config.source.reconnect_delay_s,
        )
        reason = summary.stopped_reason
        print(summary.describe())
        pipeline.metrics.log_summary()

    except KeyboardInterrupt:
        reason = "interrupted"
        logger.info("Interrupted by user")
    except (ConfigurationError, ValidationError) as exc:
        reason = "configuration error"
        logger.error("Configuration error: %s", exc)
        exit_code = EXIT_CONFIG
    except (PerceptionError, TrtEngineError) as exc:
        # An engine that will not load is an operator problem (a missing file, or
        # not enough free memory), not a bug: report it in one line, no traceback.
        reason = "perception backend unavailable"
        logger.error("Cannot start: %s", exc)
        logger.error(
            "Check that the engine exists and that enough memory is free "
            "(`free -m`); models/README.md lists each engine's size."
        )
        exit_code = EXIT_ERROR
    except ADASException as exc:
        reason = "fatal pipeline error"
        logger.error("Fatal error: %s", exc, exc_info=True)
        exit_code = EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - top level; must report, not traceback-dump
        reason = "fatal error"
        logger.error("Fatal error: %s", exc, exc_info=True)
        exit_code = EXIT_ERROR
    finally:
        if ops is not None:
            try:
                from adas.io import notify_stopping

                notify_stopping("shutting down: %s" % reason)
            except Exception as exc:  # noqa: BLE001
                logger.debug("notify_stopping failed: %s", exc)
            ops.close(reason)
        if pipeline is not None:
            pipeline.close()
        logger.info("ADAS shutdown complete (%s)", reason)

    return exit_code


def _install_signal_handlers(runner: PipelineRunner, ops: Optional[_Ops]) -> None:
    """SIGTERM/SIGINT stop the loop; SIGHUP reopens the event log.

    Without these, ``systemctl stop`` and ``docker stop`` SIGKILL the process
    mid-frame: the source is never released (on a real Jetson the Argus daemon
    keeps the sensor), the metrics summary is lost and the event log's last
    records may not have reached the disk.
    """

    def _stop(signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name
        logger.info("Received %s; stopping after the current frame", name)
        runner.request_stop("signal %s" % name)

    def _hup(_signum: int, _frame: Any) -> None:
        if ops is not None:
            ops.reopen_events()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _stop)
        except (ValueError, OSError) as exc:  # not the main thread, or unsupported
            logger.debug("Could not install handler for %s: %s", sig, exc)
    if hasattr(signal, "SIGHUP"):
        try:
            signal.signal(signal.SIGHUP, _hup)
        except (ValueError, OSError) as exc:
            logger.debug("Could not install SIGHUP handler: %s", exc)


if __name__ == "__main__":
    sys.exit(main())
