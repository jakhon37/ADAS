"""Integration tests: configuration, factory, ego sources, the runner and the CLI.

Everything here runs without a GPU.  The tests that need a real TensorRT engine
are marked ``engine`` and skip when the file is absent, so the suite is green on
a laptop and exercises the real models on the board::

    PYTHONPATH=src python3 -m pytest tests/test_integration.py -q
    PYTHONPATH=src python3 -m pytest tests/test_integration.py -q -m engine
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

from adas.core.config import (
    DetectorConfig,
    EgoConfig,
    LaneConfig,
    RuntimeConfig,
    SourceConfig,
    default_config,
    load_config,
)
from adas.core.exceptions import ConfigurationError, PerceptionError
from adas.core.models import ControlCommand, SafetyState
from adas.perception.factory import (
    build_camera,
    build_depth_channel,
    build_detector,
    build_lane_estimator,
    resolve_model_path,
)
from adas.runtime import PipelineRunner
from adas.runtime.capture import (
    ConstantEgoSpeed,
    FileEgoSpeed,
    NoEgoSpeed,
    SimulatedEgoSpeed,
    SyntheticSource,
    open_ego_source,
    open_source,
    parse_source_arg,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "config.example.json"
EXAMPLE_VIDEO = REPO_ROOT / "Ultra-Fast-Lane-Detection-v2" / "example.mp4"
YOLOX_ENGINE = REPO_ROOT / "models" / "yolox_nano.engine"


def mock_config(**overrides) -> RuntimeConfig:
    """A fully mock configuration, with the mock opt-in set explicitly."""
    config = default_config()
    config.detector.backend = "mock"
    config.lane.backend = "mock"
    config.allow_mock = True
    config.fps = 20
    for key, value in overrides.items():
        setattr(config, key, value)
    config.__post_init__()
    return config


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_the_shipped_example_config_loads_and_validates():
    """config.example.json is the BENCH profile: mock backends, synthetic source.

    It has to run on a machine with no engines, so it opts into the mocks
    explicitly. The vehicle profile is documented in README.md; the point of this
    test is that the shipped file parses, cross-validates and is honest about
    being a mock configuration.
    """
    config = load_config(EXAMPLE_CONFIG)
    assert config.allow_mock is True, "a mock example must declare itself"
    assert config.source.type == "synthetic"
    assert config.config_path.endswith("config.example.json")
    assert config.cross_validation_errors() == []


def test_unknown_keys_are_rejected(tmp_path):
    """ADAS-OPS-17: a typo used to be silently ignored."""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"safety": {"max_speed_mpsX": 10.0}}))
    with pytest.raises(ConfigurationError, match="unknown key"):
        load_config(path)


def test_unknown_top_level_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"planer": {}}))
    with pytest.raises(ConfigurationError, match="unknown top-level"):
        load_config(path)


def test_deprecated_lane_keys_are_accepted_with_a_warning(tmp_path, caplog):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"lane": {"backend": "mock", "input_width": 1600}, "allow_mock": True}))
    config = load_config(path)
    assert config.lane.backend == "mock"
    assert any("deprecated" in record.message for record in caplog.records)


def test_out_of_range_values_are_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"fps": 500}))
    with pytest.raises(ConfigurationError):
        load_config(path)


def test_a_newer_schema_version_is_refused(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"schema_version": 9999}))
    with pytest.raises(ConfigurationError, match="newer than this build"):
        load_config(path)


def test_min_box_height_of_zero_is_refused():
    """It used to load and then be divided by."""
    config = default_config()
    with pytest.raises(Exception):
        config.tracker.min_box_height_px = 0.0
        config.tracker.__post_init__()


def test_a_planner_faster_than_the_safety_ceiling_is_refused():
    config = default_config()
    config.planner.cruise_speed_mps = 40.0
    config.safety.max_speed_mps = 30.0
    with pytest.raises(ConfigurationError, match="cruise_speed_mps"):
        config.__post_init__()


def test_a_planner_that_steers_further_than_the_arbiter_allows_is_refused():
    config = default_config()
    config.planner.max_steering_deg = 40.0
    config.safety.max_steering_angle_rad = 0.3
    with pytest.raises(ConfigurationError, match="max_steering_deg"):
        config.__post_init__()


def test_max_road_wheel_rad_is_derived_from_the_controller():
    config = default_config()
    assert config.safety.max_road_wheel_rad == pytest.approx(
        math.radians(config.controller.max_steering_angle_deg)
    )


def test_a_disagreeing_road_wheel_angle_is_refused():
    config = default_config()
    config.safety.max_road_wheel_rad = 0.9
    with pytest.raises(ConfigurationError, match="max_road_wheel_rad"):
        config.__post_init__()


def test_a_mock_backend_without_allow_mock_is_refused():
    """ADAS-PERC-24: the mock must never be reachable by accident."""
    config = default_config()
    config.detector.backend = "mock"
    with pytest.raises(ConfigurationError, match="allow_mock"):
        config.__post_init__()


def test_describe_names_every_backend():
    text = mock_config().describe()
    for token in ("detector=", "lane=", "depth=", "camera=", "ego=", "source="):
        assert token in text


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def test_build_detector_refuses_the_mock_without_permission():
    with pytest.raises(ConfigurationError, match="allow_mock"):
        build_detector(DetectorConfig(backend="mock"), allow_mock=False)


def test_an_explicit_mock_request_is_honoured_programmatically():
    """The CONFIG layer is the gate; a direct call naming the mock gets it."""
    assert build_detector(DetectorConfig(backend="mock")).is_mock is True


def test_build_detector_returns_a_self_identifying_mock():
    detector = build_detector(DetectorConfig(backend="mock"), allow_mock=True)
    assert detector.is_mock is True


def test_build_lane_estimator_refuses_the_mock_without_permission():
    with pytest.raises(ConfigurationError, match="allow_mock"):
        build_lane_estimator(LaneConfig(backend="mock"), allow_mock=False)


def test_a_missing_engine_is_an_error_not_a_silent_downgrade():
    config = DetectorConfig(backend="yolox", model_path="models/does_not_exist.engine")
    with pytest.raises(PerceptionError, match="not found"):
        build_detector(config, allow_mock=False)


def test_a_missing_engine_falls_back_only_when_mocks_are_allowed(caplog):
    config = DetectorConfig(backend="yolox", model_path="models/does_not_exist.engine")
    with pytest.raises(PerceptionError):
        build_detector(config, allow_mock=True)  # allow_fallback still off
    detector = build_detector(config, allow_mock=True, allow_fallback=True)
    assert detector.is_mock is True
    assert any("FALLBACK TO MOCK" in record.message for record in caplog.records)


def test_resolve_model_path_finds_a_repo_relative_engine(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolved = resolve_model_path("config.example.json")
    assert resolved == REPO_ROOT / "config.example.json"


def test_build_camera_rescales_intrinsics_to_the_live_frame_size():
    """ADAS-PERC-26: fx measured at 1280 is 1.5x too small at 1920."""
    config = default_config()
    camera = build_camera(config.camera, 1920, 1080)
    assert camera is not None
    assert camera.image_width == 1920
    assert camera.fx == pytest.approx(config.camera.fx * 1920.0 / 1280.0)


def test_build_camera_can_be_disabled():
    config = default_config()
    config.camera.enabled = False
    assert build_camera(config.camera, 1280, 720) is None


def test_depth_backend_off_returns_none():
    config = default_config()
    assert build_depth_channel(config.depth) is None


def test_a_missing_depth_engine_is_an_honest_stub():
    config = default_config()
    config.depth.backend = "midas"
    config.depth.model_path = "models/no_such_depth.engine"
    channel = build_depth_channel(config.depth)
    assert channel is not None
    assert channel.available is False
    assert channel.is_mock is True


# --------------------------------------------------------------------------- #
# Ego speed sources
# --------------------------------------------------------------------------- #


def test_no_ego_source_reports_invalid_forever():
    source = NoEgoSpeed()
    state = source.state(1.0)
    assert state.valid is False
    assert source.measured is False


def test_a_configured_constant_is_valid_but_not_measured():
    source = ConstantEgoSpeed(12.0)
    state = source.state(1.0)
    assert state.valid is True
    assert state.speed_mps == pytest.approx(12.0)
    assert source.measured is False


def test_the_simulated_plant_responds_to_the_actuated_command():
    """With no actuator lag the plant is the textbook point mass."""
    source = SimulatedEgoSpeed(initial_speed_mps=10.0, drag_per_s=0.0, actuator_tau_s=0.0)
    source.apply_command(ControlCommand(throttle=1.0, brake=0.0, steering=0.0), 1.0)
    assert source.state(0.0).speed_mps == pytest.approx(12.5, abs=1e-6)
    source.apply_command(ControlCommand(throttle=0.0, brake=1.0, steering=0.0), 1.0)
    assert source.state(0.0).speed_mps == pytest.approx(4.5, abs=1e-6)
    assert source.measured is False


def test_the_actuator_lag_bounds_the_jerk_the_plant_can_produce():
    """A stepped pedal command must not become an infinite acceleration step.

    Without the lag the arbiter's jerk check -- which differences the measured
    ego speed -- sees 40-60 m/s^3 and holds the system in LIMITED forever.
    """
    dt = 0.05
    source = SimulatedEgoSpeed(
        initial_speed_mps=15.0, drag_per_s=0.0, actuator_tau_s=0.15
    )
    full_brake = ControlCommand(throttle=0.0, brake=1.0, steering=0.0)
    source.apply_command(full_brake, dt)
    first = source.last_accel_mps2
    assert abs(first) < 8.0, "the plant must not reach full authority in one frame"
    assert abs(first) / dt < 60.0

    for _ in range(40):
        source.apply_command(full_brake, dt)
    assert source.last_accel_mps2 == pytest.approx(-8.0, abs=0.05), "it must get there"


def test_the_simulated_plant_never_goes_negative():
    source = SimulatedEgoSpeed(initial_speed_mps=1.0)
    source.apply_command(ControlCommand(0.0, 1.0, 0.0), 5.0)
    assert source.state(0.0).speed_mps == 0.0


def test_a_recorded_ego_channel_is_measured_and_ages_out(tmp_path):
    path = tmp_path / "speed.csv"
    path.write_text("timestamp_s,speed_mps\n0.0,10.0\n0.1,11.0\n0.2,12.0\n")
    source = FileEgoSpeed(str(path), max_age_s=0.25)
    assert source.measured is True

    assert source.state(0.0).speed_mps == pytest.approx(10.0)
    # Interpolated between the 0.1 s and 0.2 s samples, not held at 11.0.
    assert source.state(0.15).speed_mps == pytest.approx(11.5)
    assert source.state(0.2).speed_mps == pytest.approx(12.0)

    stale = source.state(1.0)
    assert stale.valid is False, "a frozen speed must not look healthy"


def test_a_sparse_ego_channel_holds_rather_than_interpolating(tmp_path):
    """Across a gap wider than max_age_s, interpolation is not evidence."""
    path = tmp_path / "sparse.csv"
    path.write_text("timestamp_s,speed_mps\n0.0,10.0\n5.0,20.0\n")
    source = FileEgoSpeed(str(path), max_age_s=0.5)
    assert source.state(0.0).speed_mps == pytest.approx(10.0)
    assert source.state(1.0).valid is False


def test_a_recorded_ego_channel_reads_json(tmp_path):
    path = tmp_path / "speed.json"
    path.write_text(json.dumps([[0.0, 5.0], [1.0, 6.0]]))
    source = FileEgoSpeed(str(path), max_age_s=2.0)
    assert source.state(0.0).speed_mps == pytest.approx(5.0)


def test_a_missing_ego_file_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="not found"):
        FileEgoSpeed("/nonexistent/speed.csv")


def test_open_ego_source_dispatches_on_the_config():
    assert isinstance(open_ego_source(EgoConfig(source="none")), NoEgoSpeed)
    assert isinstance(open_ego_source(EgoConfig(source="config", speed_mps=5.0)), ConstantEgoSpeed)
    assert isinstance(open_ego_source(EgoConfig(source="simulated")), SimulatedEgoSpeed)


def test_ego_file_source_requires_a_path():
    with pytest.raises(ConfigurationError, match="ego.file"):
        EgoConfig(source="file")


# --------------------------------------------------------------------------- #
# Frame sources
# --------------------------------------------------------------------------- #


def test_parse_source_arg():
    assert parse_source_arg(None) == ("synthetic", "")
    assert parse_source_arg("csi") == ("camera", "csi:0")
    assert parse_source_arg("camera:2") == ("camera", "2")
    assert parse_source_arg("clip.mp4") == ("video", "clip.mp4")


def test_a_missing_video_file_is_reported_before_opencv_is_asked():
    from adas.core.exceptions import SensorError

    with pytest.raises(SensorError, match="not found"):
        open_source("video", uri="/nonexistent/clip.mp4")


def test_a_video_source_reports_eof_rather_than_failure():
    if not EXAMPLE_VIDEO.exists():
        pytest.skip("example.mp4 not present")
    cv2 = pytest.importorskip("cv2")
    source = open_source("video", uri=str(EXAMPLE_VIDEO))
    try:
        assert source.read() is not None
        assert source.eof is False
    finally:
        source.close()


def test_source_type_video_requires_a_uri():
    with pytest.raises(ConfigurationError, match="source.uri"):
        SourceConfig(type="video")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def build_mock_pipeline():
    from adas.cli import build_pipeline

    return build_pipeline(config=mock_config())


def test_runner_processes_a_bounded_number_of_frames():
    pipeline, config = build_mock_pipeline()
    runner = PipelineRunner(pipeline, target_fps=0.0, ego_source=SimulatedEgoSpeed(10.0))
    summary = runner.run(source_type="synthetic", max_frames=10)
    assert summary.frames == 10
    assert summary.failures == 0
    assert summary.measured_fps > 0.0


def test_runner_stops_when_asked():
    pipeline, _config = build_mock_pipeline()
    runner = PipelineRunner(pipeline, target_fps=0.0, ego_source=SimulatedEgoSpeed(10.0))

    stopped = {}

    def on_frame(frame_id, plan, command):
        if frame_id == 4:
            stopped["at"] = frame_id
            runner.request_stop("test")

    runner.hooks.on_frame = on_frame
    summary = runner.run(source_type="synthetic", max_frames=0)  # continuous
    assert stopped["at"] == 4
    assert summary.frames == 5
    assert "test" in summary.stopped_reason


def test_continuous_mode_is_not_bounded_by_max_frames():
    """ADAS-OPS-02: `--frames 0` must not exit immediately."""
    pipeline, _config = build_mock_pipeline()
    runner = PipelineRunner(pipeline, target_fps=0.0, ego_source=NoEgoSpeed())
    runner.hooks.on_frame = lambda fid, plan, cmd: (
        runner.request_stop("enough") if fid >= 20 else None
    )
    summary = runner.run(source_type="synthetic", max_frames=0)
    assert summary.frames == 21


def test_a_failing_pipeline_actuates_the_failsafe_and_then_stops():
    pipeline, _config = build_mock_pipeline()

    class Exploding:
        is_mock = False

        def infer(self, *args, **kwargs):
            return []

    def explode(*args, **kwargs):
        raise RuntimeError("scripted tracker failure")

    pipeline.tracker.update = explode
    commands = []
    runner = PipelineRunner(
        pipeline,
        target_fps=0.0,
        ego_source=SimulatedEgoSpeed(10.0),
        max_consecutive_failures=3,
    )
    runner.hooks.on_frame = lambda fid, plan, cmd: commands.append(cmd)
    summary = runner.run(source_type="synthetic", max_frames=20)

    assert summary.failures == 3
    assert summary.frames == 3, "the loop must stop, not spin on a broken pipeline"
    assert all(cmd.throttle == 0.0 for cmd in commands), "no throttle after a failure"
    assert any(cmd.brake > 0.0 for cmd in commands)


def test_on_ready_fires_once_after_the_first_frame():
    pipeline, _config = build_mock_pipeline()
    calls = []
    runner = PipelineRunner(pipeline, target_fps=0.0, ego_source=SimulatedEgoSpeed(10.0))
    runner.hooks.on_ready = lambda: calls.append(time.monotonic())
    runner.run(source_type="synthetic", max_frames=5)
    assert len(calls) == 1


def test_dt_is_clamped_into_a_sane_band():
    from adas.runtime.runner import DT_CLAMP_HIGH, DT_CLAMP_LOW

    pipeline, _config = build_mock_pipeline()
    runner = PipelineRunner(pipeline, target_fps=20.0)
    assert runner._clamp_dt(0.0, 0.05, 0) == pytest.approx(DT_CLAMP_LOW * 0.05)
    assert runner._clamp_dt(10.0, 0.05, 0) == pytest.approx(DT_CLAMP_HIGH * 0.05)
    assert runner._clamp_dt(0.06, 0.05, 0) == pytest.approx(0.06)


def test_the_runner_feeds_the_arbitrated_command_back_to_the_plant():
    """The plant must be driven by what was ACTUATED, not by what was planned."""
    pipeline, _config = build_mock_pipeline()
    plant = SimulatedEgoSpeed(initial_speed_mps=20.0, drag_per_s=0.0)
    runner = PipelineRunner(pipeline, target_fps=0.0, ego_source=plant)
    runner.run(source_type="synthetic", max_frames=15)
    # The mock detector fabricates a close lead, so the arbiter brakes and the
    # plant must have slowed - if the planner's wish were fed back instead, this
    # would not track the arbiter's veto.
    assert plant.speed_mps < 20.0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_print_config_exits_zero():
    from adas.cli import main

    assert main(["--detector", "mock", "--lane", "mock", "--allow-mock", "--print-config"]) == 0


def test_cli_refuses_a_mock_backend_without_the_flag():
    from adas.cli import EXIT_CONFIG, main

    assert main(["--detector", "mock", "--lane", "mock", "--frames", "1"]) == EXIT_CONFIG


def test_cli_runs_a_short_synthetic_session():
    from adas.cli import main

    code = main([
        "--detector", "mock", "--lane", "mock", "--allow-mock",
        "--ego-source", "simulated", "--ego-speed", "10",
        "--frames", "5", "--fps", "0", "--no-health", "--no-events",
    ])
    assert code == 0


def test_cli_reads_adas_config_path(monkeypatch, tmp_path):
    from adas.cli import main

    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({
        "detector": {"backend": "mock"},
        "lane": {"backend": "mock"},
        "allow_mock": True,
        "fps": 7,
    }))
    monkeypatch.setenv("ADAS_CONFIG_PATH", str(path))
    assert main(["--print-config"]) == 0


def test_cli_allow_mock_env_var(monkeypatch):
    from adas.cli import main

    monkeypatch.setenv("ADAS_ALLOW_MOCK", "1")
    assert main(["--detector", "mock", "--lane", "mock", "--print-config"]) == 0


def test_cli_reports_a_bad_config_as_exit_code_two(tmp_path):
    from adas.cli import EXIT_CONFIG, main

    path = tmp_path / "bad.json"
    path.write_text("{not json")
    assert main(["--config", str(path), "--frames", "1"]) == EXIT_CONFIG


# --------------------------------------------------------------------------- #
# Record / replay round trip
# --------------------------------------------------------------------------- #


def test_record_and_replay_round_trip(tmp_path):
    from adas.runtime.capture import SyntheticSource as _Synthetic  # noqa: F401
    from adas.tools import (
        DataRecorder,
        DataReplayer,
        RecordingConfig,
        RecordingPipeline,
        ReplayConfig,
        replay_with_pipeline,
    )

    pipeline, _config = build_mock_pipeline()
    recorder = DataRecorder(RecordingConfig(output_dir=str(tmp_path), recording_name="run"))
    recorder.start_recording()
    wrapper = RecordingPipeline(pipeline, recorder)

    runner = PipelineRunner(wrapper, target_fps=0.0, ego_source=SimulatedEgoSpeed(10.0))
    runner.run(source_type="synthetic", max_frames=8)
    recorder.stop_recording()

    assert recorder.errors == 0, "recording a lane used to raise AttributeError"
    replayer = DataReplayer(ReplayConfig(recording_dir=str(tmp_path / "run"), playback_speed=0.0))
    assert len(replayer) == 8
    assert replayer.get_perception_frame(0) is not None
    assert replayer.get_ego(0).valid is True

    pipeline2, _c2 = build_mock_pipeline()
    results = list(replay_with_pipeline(replayer, pipeline2))
    assert len(results) == 8
    for recorded, (_plan, command) in results:
        assert recorded["arbitration"]["command"]["brake"] == pytest.approx(
            command.brake, abs=1e-6
        )
    summary_path = replayer.export_summary()
    assert summary_path.exists()


# --------------------------------------------------------------------------- #
# Real engines (skipped when absent)
# --------------------------------------------------------------------------- #


@pytest.mark.engine
def test_yolox_detector_runs_on_a_real_frame():
    if not YOLOX_ENGINE.exists():
        pytest.skip("yolox_nano.engine not built")
    numpy = pytest.importorskip("numpy")

    config = default_config()
    config.detector.model_path = str(YOLOX_ENGINE)
    detector = build_detector(config.detector, allow_mock=False)
    try:
        blank = numpy.zeros((720, 1280, 3), dtype=numpy.uint8)
        assert detector.infer(blank, 1280, 720) == []
        assert detector.is_mock is False
    finally:
        detector.close()


@pytest.mark.engine
def test_full_pipeline_over_the_example_clip():
    if not (YOLOX_ENGINE.exists() and EXAMPLE_VIDEO.exists()):
        pytest.skip("engine or example clip not present")
    pytest.importorskip("cv2")
    from adas.cli import build_pipeline

    config = default_config()
    config.detector.model_path = str(YOLOX_ENGINE)
    config.lane.backend = "mock"
    config.allow_mock = True
    config.source.type = "video"
    config.source.uri = str(EXAMPLE_VIDEO)
    config.__post_init__()

    pipeline, _config = build_pipeline(config=config)
    try:
        runner = PipelineRunner(pipeline, target_fps=0.0, ego_source=SimulatedEgoSpeed(15.0))
        summary = runner.run(
            source_type="video", uri=str(EXAMPLE_VIDEO), max_frames=20, as_image=True
        )
        assert summary.frames == 20
        assert summary.failures == 0
        assert pipeline.last_arbitration is not None
        assert pipeline.last_arbitration.state in set(SafetyState)
    finally:
        pipeline.close()
