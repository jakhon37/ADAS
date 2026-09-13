"""Tests for adas.io.health (the HTTP endpoint) and adas.io.sd_notify (the watchdog).

Every test binds port 0 on loopback, so they are safe to run concurrently with a live
service on the board and need no network beyond localhost.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import urllib.error
import urllib.request

import pytest

from adas.io import metrics as reg
from adas.io.health import DEFAULT_PORT, ROUTES, HealthServer, HealthState, is_loopback
from adas.io.sd_notify import (
    WatchdogPinger,
    notify_socket_path,
    sd_notify,
    under_systemd,
    watchdog_interval_s,
    watchdog_ping_interval_s,
)


class FakeClock:
    def __init__(self, start: float = 100.0) -> None:
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


@pytest.fixture
def state():
    return HealthState(stale_after_s=5.0, startup_grace_s=60.0, clock=FakeClock())


@pytest.fixture
def server(state):
    srv = HealthServer(state, bind="127.0.0.1", port=0)
    port = srv.start()
    assert port, "health server failed to bind loopback:0"
    yield srv
    srv.stop()


def get(srv, path, method="GET", headers=None):
    """Return (status, body). A 4xx/5xx is a result here, not an exception."""
    url = "http://127.0.0.1:%d%s" % (srv.port, path)
    req = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.getcode(), resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def make_ready(state):
    """Drive a state into the fully-ready condition."""
    with state.lock:
        state.ok = True
        state.source = "ok"
        state.frame_id = 1
        state.perception_ok = True
        state.engines = {"detector": "loaded"}
    state.mark_update()


# ------------------------------------------------------------------- binding


def test_is_loopback_accepts_only_local_addresses():
    assert is_loopback("127.0.0.1")
    assert is_loopback("127.0.1.1")
    assert is_loopback("localhost")
    assert is_loopback("::1")
    assert not is_loopback("0.0.0.0")
    assert not is_loopback("")
    assert not is_loopback("192.168.1.10")
    assert not is_loopback("::")


def test_non_loopback_bind_is_refused_without_an_explicit_opt_in(state):
    srv = HealthServer(state, bind="0.0.0.0", port=0)
    assert srv.start() is None
    assert "not loopback" in srv.error
    srv.stop()


def test_required_turns_a_refused_bind_into_a_startup_failure(state):
    srv = HealthServer(state, bind="0.0.0.0", port=0, required=True)
    with pytest.raises(RuntimeError):
        srv.start()


def test_a_taken_port_fails_soft_by_default(state):
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    taken = holder.getsockname()[1]
    try:
        srv = HealthServer(state, bind="127.0.0.1", port=taken)
        assert srv.start() is None
        assert "failed" in srv.error
    finally:
        holder.close()


def test_default_port_does_not_collide_with_dms():
    assert DEFAULT_PORT == 8090, "DMS owns 8088; both units must coexist on one board"


# -------------------------------------------------------------------- routes


def test_livez_is_always_200_even_when_everything_else_is_broken(server, state):
    with state.lock:
        state.ok = False
        state.source = "lost"
    code, body = get(server, "/livez")
    assert code == 200
    assert json.loads(body)["pid"] == os.getpid()


def test_healthz_is_503_when_not_ok(server, state):
    make_ready(state)
    assert get(server, "/healthz")[0] == 200
    with state.lock:
        state.ok = False
    code, body = get(server, "/healthz")
    assert code == 503
    assert json.loads(body)["ok"] is False


def test_healthz_stays_200_when_merely_degraded(server, state):
    """LIMITED is a mode, not an outage. Paging on it teaches operators to ignore
    the page."""
    make_ready(state)
    with state.lock:
        state.degraded = True
        state.safety_state = "limited"
    code, body = get(server, "/healthz")
    assert code == 200
    payload = json.loads(body)
    assert payload["degraded"] is True
    assert payload["safety_state"] == "limited"


def test_healthz_goes_503_when_the_pipeline_stops_advancing(server, state):
    make_ready(state)
    assert get(server, "/healthz")[0] == 200
    state._clock.advance(30.0)
    code, body = get(server, "/healthz")
    assert code == 503
    assert json.loads(body)["stale"] is True
    assert reg.HEALTH_STALE.value() in (0.0, 1.0)


def test_an_advancing_frame_id_counts_as_a_refresh(server, state):
    make_ready(state)
    assert get(server, "/healthz")[0] == 200   # observes frame_id 1
    state._clock.advance(30.0)
    assert get(server, "/healthz")[0] == 503
    with state.lock:
        state.frame_id = 2
    assert get(server, "/healthz")[0] == 200


def test_readyz_is_503_until_the_first_frame(server, state):
    with state.lock:
        state.ok = True
        state.source = "ok"
        state.engines = {"detector": "loaded"}
    state.mark_update()
    assert get(server, "/readyz")[0] == 503
    with state.lock:
        state.frame_id = 1
    assert get(server, "/readyz")[0] == 200


def test_readyz_is_503_while_an_engine_is_missing(server, state):
    make_ready(state)
    state.set_engine("lane", "missing")
    assert get(server, "/readyz")[0] == 503
    state.set_engine("lane", "loaded")
    assert get(server, "/readyz")[0] == 200


def test_a_mock_engine_is_ready_but_degraded(server, state):
    make_ready(state)
    state.set_engine("lane", "mock")
    code, body = get(server, "/readyz")
    assert code == 200
    assert json.loads(body)["degraded"] is True


def test_readyz_is_503_when_perception_is_down(server, state):
    make_ready(state)
    with state.lock:
        state.perception_ok = False
        state.perception_consecutive_failures = 7
    code, body = get(server, "/readyz")
    assert code == 503
    assert json.loads(body)["perception"]["consecutive_failures"] == 7


def test_metrics_is_prometheus_text_and_always_200(server, state):
    with state.lock:
        state.ok = False
        state.fps = 18.25
        state.frame_id = 99
    code, body = get(server, "/metrics")
    assert code == 200
    assert "adas_fps 18.25" in body
    assert "adas_frame_id 99" in body
    assert "# TYPE adas_up gauge" in body
    assert body.endswith("\n")


def test_metrics_mirrors_safety_state_and_latency(server, state):
    with state.lock:
        state.safety_state = "min_risk_maneuver"
        state.latency_ms = {"e2e_p95": 42.5, "detect_p50": 4.0}
    body = get(server, "/metrics")[1]
    assert 'adas_safety_state{state="min_risk_maneuver"} 1' in body
    assert 'adas_safety_state{state="nominal"} 0' in body
    assert 'adas_stage_latency_ms{stage="e2e",quantile="p95"} 42.5' in body
    assert 'adas_stage_latency_ms{stage="detect",quantile="p50"} 4' in body


def test_root_and_health_alias_to_healthz(server, state):
    make_ready(state)
    for path in ("/", "/health", "/healthz/"):
        code, body = get(server, path)
        assert code == 200, path
        assert json.loads(body)["schema"] == 1


def test_unknown_path_is_404_and_lists_the_routes(server):
    code, body = get(server, "/nope")
    assert code == 404
    assert sorted(ROUTES) == json.loads(body)["routes"]


def test_non_get_methods_are_405_with_allow(server):
    url = "http://127.0.0.1:%d/healthz" % server.port
    req = urllib.request.Request(url, method="POST", data=b"")
    try:
        urllib.request.urlopen(req, timeout=5)
        raise AssertionError("POST should not be accepted")
    except urllib.error.HTTPError as exc:
        assert exc.code == 405
        assert exc.headers.get("Allow") == "GET, HEAD"


def test_head_returns_the_status_with_no_body(server, state):
    make_ready(state)
    code, body = get(server, "/healthz", method="HEAD")
    assert code == 200
    assert body == ""


def test_requests_are_counted(server, state):
    get(server, "/livez")
    get(server, "/livez")
    assert reg.HEALTH_REQUESTS_TOTAL.value(route="/livez", code="200") == 2.0


def test_query_strings_are_ignored(server, state):
    make_ready(state)
    assert get(server, "/healthz?pretty=1")[0] == 200


# ---------------------------------------------------------------------- token


def test_a_token_gates_everything_except_livez(state):
    srv = HealthServer(state, bind="127.0.0.1", port=0, token="s3cret")
    assert srv.start()
    try:
        assert get(srv, "/livez")[0] == 200
        assert get(srv, "/healthz")[0] == 401
        code, _ = get(srv, "/healthz", headers={"Authorization": "Bearer s3cret"})
        assert code in (200, 503)
        assert get(srv, "/healthz", headers={"Authorization": "Bearer wrong!"})[0] == 401
    finally:
        srv.stop()


# ----------------------------------------------------------------- state model


def test_unknown_values_stay_none_rather_than_becoming_plausible(state):
    body = state.as_dict()
    assert body["lead_distance_m"] is None
    assert body["ego_speed_mps"] is None


def test_unmeasurable_ram_stays_none_rather_than_becoming_zero(state, monkeypatch):
    monkeypatch.setattr(reg, "ram_mb", lambda: None)
    state.refresh_process(force=True)
    assert state.as_dict()["ram_mb"] is None


def test_startup_grace_suppresses_the_staleness_verdict(state):
    assert state.is_stale() is False
    state._clock.advance(10.0)
    assert state.is_stale() is False       # still inside the 60 s grace
    state._clock.advance(60.0)
    assert state.is_stale() is True        # grace expired with no refresh at all


def test_update_from_metrics_pulls_real_numbers(state):
    from adas.core.metrics import PerformanceMetrics

    clock = FakeClock()
    perf = PerformanceMetrics(clock=clock)
    for _ in range(6):
        perf.update_frame(frame_time=0.05)
        clock.advance(0.1)
    with perf.stage("detect"):
        clock.advance(0.005)
    state.update_from_metrics(perf)
    body = state.as_dict()
    assert body["frames_total"] == 6
    assert body["fps"] == pytest.approx(10.0, rel=0.05)
    assert body["latency_ms"]["detect_p95"] == pytest.approx(5.0, rel=0.05)


def test_update_from_metrics_survives_a_broken_reporter(state):
    class Broken:
        total_frames = 3

        def latency_ms(self):
            raise RuntimeError("no")

    state.update_from_metrics(Broken())
    assert state.as_dict()["frames_total"] == 3


def test_notes_are_deduplicated(state):
    state.note("lane estimator is a stub")
    state.note("lane estimator is a stub")
    assert state.as_dict()["notes"] == ["lane estimator is a stub"]


def test_counters_do_not_go_backwards_across_a_reset(state):
    with state.lock:
        state.frames_total = 100
    state.publish_metrics()
    assert reg.FRAMES_TOTAL.value() == 100.0
    with state.lock:
        state.frames_total = 5   # as if the pipeline object was recreated
    state.publish_metrics()
    assert reg.FRAMES_TOTAL.value() == 100.0


def test_context_manager_starts_and_stops(state):
    with HealthServer(state, bind="127.0.0.1", port=0) as srv:
        assert srv.port
        assert get(srv, "/livez")[0] == 200
    assert reg.UP.value() == 0.0


def test_concurrent_scrapes_do_not_corrupt_the_snapshot(server, state):
    make_ready(state)
    errors = []

    def hammer():
        for _ in range(15):
            try:
                code, body = get(server, "/healthz")
                json.loads(body)
                if code != 200:
                    errors.append(code)
            except Exception as exc:  # the assertion below is "nothing raised"
                errors.append(repr(exc))

    def mutate():
        for i in range(200):
            with state.lock:
                state.frame_id = i
                state.fps = float(i)
            state.mark_update()

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    threads.append(threading.Thread(target=mutate))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert errors == []


# ------------------------------------------------------------------ sd_notify


def test_sd_notify_is_a_noop_without_the_socket(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    assert under_systemd() is False
    assert notify_socket_path() is None
    assert sd_notify("READY=1") is False


def test_abstract_socket_paths_are_translated(monkeypatch):
    monkeypatch.setenv("NOTIFY_SOCKET", "@/org/freedesktop/systemd1/notify")
    assert notify_socket_path() == "\0/org/freedesktop/systemd1/notify"


def test_sd_notify_delivers_a_real_datagram(tmp_path, monkeypatch):
    path = str(tmp_path / "notify.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    srv.bind(path)
    srv.settimeout(5.0)
    monkeypatch.setenv("NOTIFY_SOCKET", path)
    try:
        assert sd_notify("READY=1\nSTATUS=up") is True
        assert srv.recv(256) == b"READY=1\nSTATUS=up"
    finally:
        srv.close()


def test_sd_notify_does_not_raise_when_the_socket_is_gone(monkeypatch, tmp_path):
    monkeypatch.setenv("NOTIFY_SOCKET", str(tmp_path / "missing.sock"))
    assert sd_notify("WATCHDOG=1") is False


def test_watchdog_interval_reads_the_environment(monkeypatch):
    monkeypatch.delenv("WATCHDOG_USEC", raising=False)
    monkeypatch.delenv("WATCHDOG_PID", raising=False)
    assert watchdog_interval_s() is None
    assert watchdog_ping_interval_s(default=7.0) == 7.0
    monkeypatch.setenv("WATCHDOG_USEC", "30000000")
    assert watchdog_interval_s() == pytest.approx(30.0)
    assert watchdog_ping_interval_s() == pytest.approx(15.0)


def test_watchdog_is_not_this_process_when_watchdog_pid_differs(monkeypatch):
    monkeypatch.setenv("WATCHDOG_USEC", "30000000")
    monkeypatch.setenv("WATCHDOG_PID", str(os.getpid() + 1))
    assert watchdog_interval_s() is None


def test_a_garbage_watchdog_usec_disables_rather_than_crashes(monkeypatch):
    monkeypatch.setenv("WATCHDOG_USEC", "thirty seconds")
    monkeypatch.delenv("WATCHDOG_PID", raising=False)
    assert watchdog_interval_s() is None


# ------------------------------------------------------------ watchdog gating


def _pinger(clock, sent):
    p = WatchdogPinger(interval_s=5.0, clock=clock, notify=lambda: (sent.append(1), True)[1])
    p.armed = True   # pretend systemd armed us; the socket itself is not under test
    return p


def test_watchdog_pings_only_when_frames_advance():
    clock, sent = FakeClock(), []
    p = _pinger(clock, sent)
    # Inside the interval: nothing, however much progress there is.
    assert p.tick(1) is False
    clock.advance(5.0)
    assert p.tick(2) is True
    assert len(sent) == 1


def test_watchdog_is_withheld_when_the_pipeline_stalls():
    """The whole point: a bare timer would keep a wedged unit alive for ever."""
    clock, sent = FakeClock(), []
    p = _pinger(clock, sent)
    clock.advance(5.0)
    assert p.tick(10) is True
    for _ in range(4):
        clock.advance(5.0)
        assert p.tick(10) is False       # same frame id: no progress, no ping
    assert len(sent) == 1
    assert p.skipped == 4
    assert reg.WATCHDOG_SKIPPED_TOTAL.value() == 4.0
    # Recovery resumes pinging.
    clock.advance(5.0)
    assert p.tick(11) is True
    assert len(sent) == 2


def test_watchdog_never_pings_when_systemd_did_not_arm_it():
    clock, sent = FakeClock(), []
    p = WatchdogPinger(interval_s=5.0, clock=clock,
                       notify=lambda: (sent.append(1), True)[1])
    p.armed = False
    clock.advance(5.0)
    assert p.tick(1) is False
    assert sent == []
    # ...but the bookkeeping still runs, so the same code path is exercised in the lab.
    assert p.last_ping_frame == 1


def test_watchdog_stats_are_reportable():
    p = WatchdogPinger(interval_s=3.0, clock=FakeClock(), notify=lambda: True)
    stats = p.stats()
    assert stats["interval_s"] == 3.0
    assert stats["pings"] == 0
    assert set(stats) == {"armed", "interval_s", "pings", "skipped", "last_frame"}


# ------------------------------------------------------------ memory reporting
#
# Regression cover for: refresh_process() existed but was never called from
# anywhere, so /healthz reported "ram_mb": null and adas_ram_mb scraped as 0 for
# a process holding over a gigabyte.


def test_mark_update_measures_resident_memory(state):
    """The frame loop's own health refresh is what has to populate ram_mb."""
    assert state.ram_mb is None
    state.mark_update()
    assert state.ram_mb is not None
    assert state.ram_mb > 1.0
    assert state.as_dict()["ram_mb"] == state.ram_mb


def test_a_probe_measures_memory_even_if_the_pipeline_never_refreshed(state):
    body = state.as_dict()
    assert body["ram_mb"] is not None and body["ram_mb"] > 1.0


def test_healthz_reports_the_real_rss_of_this_process(server, state):
    make_ready(state)
    code, body = get(server, "/healthz")
    assert code == 200
    reported = json.loads(body)["ram_mb"]
    assert reported is not None
    with open("/proc/self/statm") as f:
        resident = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)
    assert abs(reported - resident) < max(32.0, resident * 0.25)


def test_metrics_exports_the_measured_rss(server, state):
    state.mark_update()
    code, body = get(server, "/metrics")
    assert code == 200
    line = [ln for ln in body.splitlines() if ln.startswith("adas_ram_mb ")][0]
    assert float(line.split()[1]) > 1.0


def test_unmeasured_ram_is_exported_as_nan_never_as_zero(server, state, monkeypatch):
    """A scraper must not be able to read "never measured" as "0 MiB resident"."""
    monkeypatch.setattr(reg, "ram_mb", lambda: None)
    state.refresh_process(force=True)
    code, body = get(server, "/metrics")
    assert code == 200
    line = [ln for ln in body.splitlines() if ln.startswith("adas_ram_mb ")][0]
    assert line == "adas_ram_mb NaN"


def test_process_facts_are_rate_limited(state, monkeypatch):
    calls = []

    def counted():
        calls.append(1)
        return 123.4

    monkeypatch.setattr(reg, "ram_mb", counted)
    for _ in range(50):
        state.mark_update()
        state.as_dict()
    assert len(calls) == 1  # the FakeClock has not advanced
    state._clock.advance(state.process_refresh_s + 0.01)
    state.mark_update()
    assert len(calls) == 2


def test_healthz_publishes_the_verified_engine_digests(server, state, monkeypatch):
    from adas.infer import trt_engine

    monkeypatch.setitem(trt_engine.VERIFIED_ENGINES, "/models/yolox_nano.engine", "ab" * 32)
    code, body = get(server, "/healthz")
    assert code == 200
    assert json.loads(body)["engine_sha256"] == {"yolox_nano.engine": "ab" * 32}


def test_engine_digests_are_absent_rather_than_empty_when_nothing_was_verified(state, monkeypatch):
    from adas.io import health as H

    monkeypatch.setattr(H, "verified_engine_digests", dict)
    assert state.as_dict()["engine_sha256"] == {}
