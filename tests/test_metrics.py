"""Tests for adas.io.metrics (the registry) and adas.core.metrics (the pipeline view).

These run anywhere: no GPU, no TensorRT, no numpy, no network.
"""
from __future__ import annotations

import logging
import math

import pytest

from adas.core.logger import EVENTS, JsonFormatter, Throttle, configure_logging, log_event
from adas.core.metrics import LatencyWindow, PerformanceMetrics, SystemHealthMonitor
from adas.io import metrics as reg


class FakeClock:
    """Monotonic clock under test control. Seconds."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


@pytest.fixture(autouse=True)
def _clean_registry():
    reg.reset()
    yield
    reg.reset()


# --------------------------------------------------------------------- registry


def test_counter_and_gauge_render_prometheus_text():
    reg.FRAMES_PROCESSED_TOTAL.inc()
    reg.FRAMES_PROCESSED_TOTAL.inc(3)
    reg.FPS.set(19.5)
    text = reg.render()
    assert "# TYPE adas_frames_processed_total counter" in text
    assert "adas_frames_processed_total 4" in text
    assert "# TYPE adas_fps gauge" in text
    assert "adas_fps 19.5" in text
    assert text.endswith("\n")


def test_counter_refuses_to_decrease():
    with pytest.raises(ValueError):
        reg.FRAMES_PROCESSED_TOTAL.inc(-1)


def test_unknown_label_is_a_loud_error_not_a_dropped_sample():
    with pytest.raises(ValueError):
        reg.FRAMES_DROPPED_TOTAL.inc(1, cause="typo")


def test_label_values_are_escaped():
    reg.ENGINE_ERRORS_TOTAL.inc(engine='we"ird\\one', kind="load")
    text = reg.render()
    assert 'engine="we\\"ird\\\\one"' in text


def test_histogram_buckets_are_cumulative_and_end_with_inf():
    reg.STAGE_DURATION_MS.observe(3.0, stage="detect")
    reg.STAGE_DURATION_MS.observe(40.0, stage="detect")
    reg.STAGE_DURATION_MS.observe(4000.0, stage="detect")
    text = reg.render()
    assert '# TYPE adas_stage_duration_ms histogram' in text
    assert 'adas_stage_duration_ms_bucket{stage="detect",le="5"} 1' in text
    assert 'adas_stage_duration_ms_bucket{stage="detect",le="50"} 2' in text
    assert 'adas_stage_duration_ms_bucket{stage="detect",le="+Inf"} 3' in text
    assert 'adas_stage_duration_ms_count{stage="detect"} 3' in text
    assert 'adas_stage_duration_ms_sum{stage="detect"} 4043' in text


def test_histogram_quantile_is_none_before_any_observation():
    assert reg.STAGE_DURATION_MS.quantile(0.95, stage="never_used") is None


def test_histogram_ignores_nan():
    reg.STAGE_DURATION_MS.observe(float("nan"), stage="plan")
    assert reg.STAGE_DURATION_MS.count(stage="plan") == 0


def test_record_stage_feeds_both_histogram_and_counter():
    reg.record_stage("track", 25.0)
    assert reg.STAGE_DURATION_MS.count(stage="track") == 1
    assert reg.STAGE_SECONDS_TOTAL.value(stage="track") == pytest.approx(0.025)


def test_record_stage_drops_a_negative_duration():
    reg.record_stage("track", -5.0)
    assert reg.STAGE_DURATION_MS.count(stage="track") == 0
    assert reg.STAGE_SECONDS_TOTAL.value(stage="track") == 0.0


def test_safety_state_is_one_hot_and_counts_transitions():
    reg.set_safety_state("limited", previous="nominal")
    assert reg.SAFETY_STATE.value(state="limited") == 1.0
    assert reg.SAFETY_STATE.value(state="nominal") == 0.0
    assert reg.SAFETY_STATE_TRANSITIONS_TOTAL.value(
        from_state="nominal", to_state="limited") == 1.0
    # No transition counted when the state did not change.
    reg.set_safety_state("limited", previous="limited")
    assert reg.SAFETY_STATE_TRANSITIONS_TOTAL.value(
        from_state="limited", to_state="limited") == 0.0


def test_build_info_never_omits_a_label():
    labels = reg.set_build_info(version="9.9.9", git_sha="", trt_version="8.5.2.2")
    assert labels["git_sha"] == "unknown"
    assert labels["detector"] == "unknown"
    assert 'adas_build_info{' in reg.render()


def test_duplicate_registration_with_a_different_type_raises():
    r = reg.Registry()
    r.counter("x_total", "doc")
    with pytest.raises(ValueError):
        r.gauge("x_total", "doc")


def test_illegal_metric_and_label_names_are_rejected_at_registration():
    r = reg.Registry()
    with pytest.raises(ValueError):
        r.counter("1bad", "doc")
    with pytest.raises(ValueError):
        r.counter("ok_total", "doc", ("__reserved",))


def test_histogram_rejects_the_reserved_le_label_and_unsorted_buckets():
    r = reg.Registry()
    with pytest.raises(ValueError):
        r.histogram("h_ms", "doc", ("le",))
    with pytest.raises(ValueError):
        r.histogram("h2_ms", "doc", (), buckets=(10.0, 1.0))


def test_ram_mb_is_a_real_reading_or_none():
    value = reg.ram_mb()
    assert value is None or value > 0.0


# ------------------------------------------------------------- LatencyWindow


def test_latency_window_quantiles():
    win = LatencyWindow(capacity=100)
    for i in range(1, 101):
        win.observe(float(i))
    assert win.count == 100
    assert win.p50 == pytest.approx(50.0, abs=1.0)
    assert win.p95 == pytest.approx(95.0, abs=1.0)
    assert win.max_ms == 100.0
    assert win.min_ms == 1.0


def test_latency_window_is_bounded_and_forgets_old_samples():
    win = LatencyWindow(capacity=5)
    for _ in range(100):
        win.observe(1.0)
    win.observe(999.0)
    assert len(win._values) == 5
    assert win.count == 101
    assert win.p50 == 1.0
    # max_ms is over the whole run, not the window: a spike is not forgotten.
    assert win.max_ms == 999.0


def test_latency_window_ignores_negative_and_nan():
    win = LatencyWindow()
    win.observe(-1.0)
    win.observe(float("nan"))
    assert win.count == 0
    assert win.p95 == 0.0


# ------------------------------------------------------- PerformanceMetrics


def test_measured_fps_is_not_the_configured_rate():
    """The bug this replaces: avg_fps was forced to equal target_fps.

    Here the caller *claims* 50 ms frames (20 fps) while the clock says 100 ms
    (10 fps). The measurement must follow the clock, not the claim.
    """
    clock = FakeClock()
    m = PerformanceMetrics(clock=clock)
    for _ in range(11):
        m.update_frame(frame_time=0.05, num_detections=1, num_tracks=1, has_lane=True)
        clock.advance(0.1)
    assert m.total_frames == 11
    assert m.reported_frame_time_s == pytest.approx(0.55)
    assert m.measured_fps == pytest.approx(10.0, rel=0.01)
    assert m.avg_fps == m.measured_fps
    assert m.avg_frame_time == pytest.approx(0.1, rel=0.01)


def test_fps_is_zero_not_a_guess_before_the_second_frame():
    m = PerformanceMetrics(clock=FakeClock())
    assert m.measured_fps == 0.0
    m.update_frame(frame_time=0.05)
    assert m.measured_fps == 0.0


def test_min_frame_time_is_never_inf():
    m = PerformanceMetrics(clock=FakeClock())
    assert m.min_frame_time == 0.0
    assert "inf" not in m.summary().lower()


def test_stage_context_manager_records_even_when_the_block_raises():
    clock = FakeClock()
    m = PerformanceMetrics(clock=clock)

    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom), m.stage("detect"):
        clock.advance(0.9)
        raise Boom()
    assert m.stages["detect"].count == 1
    assert m.stages["detect"].max_ms == pytest.approx(900.0)
    assert reg.STAGE_DURATION_MS.count(stage="detect") == 1


def test_latency_ms_only_reports_measured_stages():
    clock = FakeClock()
    m = PerformanceMetrics(clock=clock)
    with m.stage("plan"):
        clock.advance(0.002)
    out = m.latency_ms()
    assert "plan_p50" in out and "plan_p95" in out
    assert "detect_p95" not in out


def test_safety_event_kinds_are_counted_separately():
    m = PerformanceMetrics(clock=FakeClock())
    m.record_safety_event(is_violation=True, kind="following_distance")
    m.record_safety_event(is_violation=False, kind="high_accel")
    assert m.safety_violations == 1
    assert m.safety_warnings == 1
    assert reg.SAFETY_VIOLATIONS_TOTAL.value(kind="following_distance") == 1.0
    assert reg.SAFETY_WARNINGS_TOTAL.value(kind="high_accel") == 1.0


def test_backward_clock_is_counted_not_recorded():
    clock = FakeClock()
    m = PerformanceMetrics(clock=clock)
    m.update_frame()
    clock.advance(-5.0)
    m.update_frame()
    assert m.clock_regressions == 1
    assert "interval" not in m.stages


def test_snapshot_is_json_shaped_and_complete():
    clock = FakeClock()
    m = PerformanceMetrics(clock=clock)
    with m.stage("detect"):
        clock.advance(0.01)
    m.update_frame(num_detections=2, num_tracks=1, has_lane=True)
    snap = m.snapshot()
    assert snap["frames"] == 1
    assert snap["detections"] == 2
    assert snap["lane_rate_pct"] == 100.0
    assert snap["stages"]["detect"]["count"] == 1
    assert not math.isinf(snap["fps"])


def test_reset_clears_local_state_but_not_the_monotonic_registry():
    m = PerformanceMetrics(clock=FakeClock())
    m.update_frame()
    before = reg.FRAMES_PROCESSED_TOTAL.value()
    m.reset()
    assert m.total_frames == 0
    assert reg.FRAMES_PROCESSED_TOTAL.value() == before


# ------------------------------------------------------- SystemHealthMonitor


def test_health_monitor_liveness_follows_frame_progress_not_calls():
    clock = FakeClock()
    mon = SystemHealthMonitor(stall_timeout_s=2.0, clock=clock)
    mon.heartbeat(frame_id=1)
    clock.advance(1.0)
    assert mon.is_alive() is True
    # Calling heartbeat with the SAME frame id is not progress.
    mon.heartbeat(frame_id=1)
    clock.advance(2.0)
    assert mon.is_alive() is False
    assert mon.stalls == 1
    # Still stalled: the edge is counted once, not per poll.
    assert mon.is_alive() is False
    assert mon.stalls == 1
    mon.heartbeat(frame_id=2)
    assert mon.is_alive() is True


def test_check_watchdog_alias_still_works():
    clock = FakeClock()
    mon = SystemHealthMonitor(clock=clock)
    mon.heartbeat(frame_id=5)
    assert mon.check_watchdog(timeout=1.0) is True


# ------------------------------------------------------------------- logging


def test_log_level_reaches_module_loggers(capsys):
    """The ADAS-OPS-07 regression: per-module setLevel made --log-level a no-op."""
    import adas.runtime.pipeline  # noqa: F401  ensure the module logger exists

    configure_logging("DEBUG")
    assert logging.getLogger("adas.runtime.pipeline").isEnabledFor(logging.DEBUG)
    configure_logging("ERROR")
    assert not logging.getLogger("adas.runtime.pipeline").isEnabledFor(logging.INFO)
    configure_logging("INFO")


def test_unknown_log_level_falls_back_to_info_without_raising():
    configure_logging("NOT_A_LEVEL")
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_does_not_duplicate_handlers():
    configure_logging("INFO")
    first = len(logging.getLogger().handlers)
    configure_logging("INFO")
    assert len(logging.getLogger().handlers) == first


def test_json_formatter_emits_one_line_with_structured_fields():
    record = logging.LogRecord("adas.test", logging.WARNING, __file__, 1,
                               "lane lost", (), None)
    record.event = "perception_dropout"
    record.frame_id = 42
    line = JsonFormatter().format(record)
    assert "\n" not in line
    import json as _json
    payload = _json.loads(line)
    assert payload["msg"] == "lane lost"
    assert payload["event"] == "perception_dropout"
    assert payload["frame_id"] == 42
    assert payload["level"] == "WARNING"


def test_json_formatter_survives_an_unserialisable_extra():
    record = logging.LogRecord("adas.test", logging.INFO, __file__, 1, "x", (), None)
    record.blob = object()
    import json as _json
    payload = _json.loads(JsonFormatter().format(record))
    # default=str stringifies rather than losing the line.
    assert payload["msg"] == "x"


def test_log_event_rejects_nothing_but_warns_on_an_unknown_name(caplog):
    logger = logging.getLogger("adas.test.event")
    with caplog.at_level(logging.WARNING):
        log_event(logger, "not_a_real_event", "hello")
    assert any("unknown event" in r.message for r in caplog.records)
    assert "summary" in EVENTS


def test_throttle_bounds_a_repeating_line_and_reports_the_gap():
    clock = FakeClock()
    t = Throttle(interval_s=10.0, clock=clock)
    assert t.ready() is True
    for _ in range(5):
        assert t.ready() is False
    clock.advance(10.0)
    assert t.ready() is True
    assert t.fields() == {"suppressed": 5}
