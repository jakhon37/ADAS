"""Tests for adas.io.events — the durable safety-event JSONL.

Covers the three failures the writer exists to survive: a full disk, a corrupt/torn
file, and rotation (both in-process and external). No GPU, no network.
"""
from __future__ import annotations

import errno
import json
import os
import stat

import pytest

from adas.io import metrics as reg
from adas.io.events import (
    EVENT_KINDS,
    SCHEMA_VERSION,
    EventLog,
    default_events_path,
    default_fallback_path,
    from_config,
    read_events,
)


class FakeClock:
    def __init__(self, start: float = 500.0) -> None:
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
def log(tmp_path):
    ev = EventLog(str(tmp_path / "events.jsonl"), fsync="always", dev=True)
    yield ev
    ev.close()


def _lines(path):
    return list(read_events(str(path)))


# ------------------------------------------------------------------ happy path


def test_write_produces_one_schema_stamped_record(log, tmp_path):
    assert log.write("lifecycle", "started", "info", None, version="0.2.0") is True
    records = _lines(tmp_path / "events.jsonl")
    assert len(records) == 1
    rec = records[0]
    assert rec["v"] == SCHEMA_VERSION
    assert rec["kind"] == "lifecycle"
    assert rec["type"] == "started"
    assert rec["severity"] == "info"
    assert rec["seq"] == 1
    assert rec["detail"] == {"version": "0.2.0"}
    assert isinstance(rec["ts"], float) and rec["ts"] > 0
    assert rec["boot"]


def test_sequence_numbers_are_dense_and_start_at_one(log, tmp_path):
    for i in range(5):
        log.write("perception", "dropout", "warn", frame_id=i)
    assert [r["seq"] for r in _lines(tmp_path / "events.jsonl")] == [1, 2, 3, 4, 5]


def test_safety_state_transition_severity_matches_the_state(log, tmp_path):
    log.safety_state("nominal", "limited", frame_id=10, reason="perception degraded")
    log.safety_state("limited", "min_risk_maneuver", frame_id=11, reason="lost")
    log.safety_state("min_risk_maneuver", "nominal", frame_id=40)
    log.safety_state("nominal", "disengage", frame_id=50)
    sev = [r["severity"] for r in _lines(tmp_path / "events.jsonl")]
    assert sev == ["warn", "critical", "info", "critical"]
    first = _lines(tmp_path / "events.jsonl")[0]
    assert first["detail"]["previous"] == "nominal"
    assert first["detail"]["current"] == "limited"


def test_safety_state_accepts_the_enum_not_just_the_string(log, tmp_path):
    from adas.core.models import SafetyState

    log.safety_state(SafetyState.NOMINAL, SafetyState.DISENGAGE, frame_id=3)
    rec = _lines(tmp_path / "events.jsonl")[0]
    assert rec["detail"] == {"previous": "nominal", "current": "disengage", "reason": ""}
    assert rec["severity"] == "critical"


def test_typed_helpers_cover_the_documented_event_kinds(log, tmp_path):
    log.perception_dropout(4, "detector raised", frame_id=7)
    log.perception_recovered(12, frame_id=19)
    log.engine_failure("yolox_nano", "execution", "cuda error", frame_id=7)
    log.source_event("lost", uri="/dev/video0", frame_id=7)
    log.source_event("reconnected", uri="/dev/video0", frame_id=8)
    log.lifecycle("stopping", reason="SIGTERM")
    kinds = [r["kind"] for r in _lines(tmp_path / "events.jsonl")]
    assert kinds == ["perception", "perception", "engine", "source", "source", "lifecycle"]
    assert set(kinds) <= set(EVENT_KINDS)
    assert reg.ENGINE_ERRORS_TOTAL.value(engine="yolox_nano", kind="execution") == 1.0


def test_an_unknown_kind_is_still_written(log, tmp_path, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        assert log.write("not_a_kind", "x") is True
    assert len(_lines(tmp_path / "events.jsonl")) == 1
    assert any("not in EVENT_KINDS" in r.message for r in caplog.records)


def test_an_unserialisable_detail_does_not_lose_the_record(log, tmp_path):
    assert log.write("lifecycle", "started", "info", None, obj=object()) is True
    rec = _lines(tmp_path / "events.jsonl")[0]
    assert isinstance(rec["detail"]["obj"], str)


def test_file_mode_is_0640(log):
    mode = stat.S_IMODE(os.stat(log.path).st_mode)
    assert mode == 0o640, "the event log is a safety record, not world-readable"


def test_metrics_are_published(log):
    log.write("lifecycle", "started", "critical")
    assert reg.EVENT_WRITES_TOTAL.value() == 1.0
    assert reg.EVENT_FSYNCS_TOTAL.value() >= 1.0
    assert reg.EVENTS_WRITABLE.value() == 1.0
    assert reg.EVENTS_BY_TYPE_TOTAL.value(type="started", severity="critical") == 1.0


# -------------------------------------------------------------- fsync policies


def test_fsync_policy_never_does_not_fsync(tmp_path):
    ev = EventLog(str(tmp_path / "e.jsonl"), fsync="never", dev=True)
    ev.write("lifecycle", "started", "critical")
    assert reg.EVENT_FSYNCS_TOTAL.value() == 0.0
    ev.close()


def test_fsync_policy_critical_fsyncs_only_critical_records(tmp_path):
    clock = FakeClock()
    ev = EventLog(str(tmp_path / "e.jsonl"), fsync="critical", fsync_interval_s=1e9,
                  dev=True, clock=clock)
    ev.write("perception", "dropout", "info")
    assert reg.EVENT_FSYNCS_TOTAL.value() == 0.0
    ev.write("safety_state", "transition", "critical")
    assert reg.EVENT_FSYNCS_TOTAL.value() == 1.0
    ev.close()


def test_an_invalid_fsync_policy_is_a_startup_error(tmp_path):
    with pytest.raises(ValueError):
        EventLog(str(tmp_path / "e.jsonl"), fsync="sometimes")


# ------------------------------------------------------------------- corruption


def test_read_events_skips_a_torn_final_line(tmp_path):
    path = tmp_path / "e.jsonl"
    path.write_text('{"v":1,"seq":1,"kind":"lifecycle"}\n{"v":1,"seq":2,"ki')
    records = list(read_events(str(path)))
    assert len(records) == 1
    assert records[0]["seq"] == 1


def test_read_events_can_be_made_strict(tmp_path):
    path = tmp_path / "e.jsonl"
    path.write_text('{"v":1}\nnot json at all\n')
    with pytest.raises(ValueError):
        list(read_events(str(path), skip_bad=False))


def test_read_events_survives_binary_garbage(tmp_path):
    path = tmp_path / "e.jsonl"
    path.write_bytes(b'{"v":1,"seq":1}\n\xff\xfe\x00garbage\n{"v":1,"seq":2}\n')
    records = list(read_events(str(path)))
    assert [r["seq"] for r in records] == [1, 2]


def test_read_events_skips_a_json_scalar_line(tmp_path):
    path = tmp_path / "e.jsonl"
    path.write_text('42\n{"v":1,"seq":1}\n')
    assert len(list(read_events(str(path)))) == 1


def test_opening_a_torn_file_terminates_the_line_before_appending(tmp_path):
    path = tmp_path / "e.jsonl"
    path.write_text('{"v":1,"seq":1,"kind":"lifecycle"}\n{"v":1,"seq":2,"ki')
    ev = EventLog(str(path), dev=True)
    ev.write("lifecycle", "started")
    ev.close()
    raw = path.read_text().splitlines()
    # The torn line was closed off, so the new record is on a line of its own and
    # parses; nothing got glued onto the partial record.
    assert raw[-1].startswith("{")
    assert json.loads(raw[-1])["type"] == "started"
    assert len(list(read_events(str(path)))) == 2  # torn line skipped, two good ones


# ------------------------------------------------------------------- disk full


def test_min_free_mb_stops_writing_before_the_disk_is_actually_full(tmp_path):
    """A quota the writer enforces on itself: it must never be the process that
    fills the filesystem out from under the rest of the system."""
    seen = []
    ev = EventLog(str(tmp_path / "e.jsonl"), dev=True,
                  min_free_mb=1e12,  # larger than any real filesystem
                  on_disk_full=lambda edge, detail: seen.append(edge))
    assert ev.write("safety_state", "transition", "critical") is False
    assert ev.disk_full is True
    assert ev.writable is False
    assert ev.stats()["writable"] is False
    assert seen == ["enter"]
    assert reg.DISK_FULL.value() == 1.0
    assert reg.EVENT_DROPPED_TOTAL.value(reason="disk_full") == 1.0
    ev.close()


def test_enospc_on_write_becomes_a_state_not_an_exception(tmp_path):
    clock = FakeClock()
    ev = EventLog(str(tmp_path / "e.jsonl"), dev=True, retry_s=5.0, clock=clock)

    class FullFile:
        def write(self, _data):
            raise OSError(errno.ENOSPC, "No space left on device")

        def flush(self):
            pass

        def tell(self):
            return 0

        def fileno(self):
            return -1

        def close(self):
            pass

    ev._fh = FullFile()
    assert ev.write("safety_state", "transition", "critical") is False
    assert ev.disk_full is True
    assert ev.write_failures == 1
    assert reg.EVENT_WRITE_FAILURES_TOTAL.value(errno="ENOSPC") == 1.0
    # Inside the backoff window nothing is even attempted.
    assert ev.write("safety_state", "transition", "critical") is False
    assert ev.write_failures == 1
    assert reg.EVENT_DROPPED_TOTAL.value(reason="disk_full") >= 1.0
    ev._fh = None
    ev.close()


def test_recovery_from_disk_full_records_the_gap(tmp_path):
    clock = FakeClock()
    edges = []
    ev = EventLog(str(tmp_path / "e.jsonl"), dev=True, retry_s=5.0, clock=clock,
                  min_free_mb=1e12, on_disk_full=lambda e, d: edges.append(e))
    assert ev.write("safety_state", "transition", "critical") is False
    assert ev.disk_full
    # Space comes back: relax the quota and step past the retry window.
    ev.min_free_mb = 0.0
    clock.advance(6.0)
    assert ev.write("safety_state", "transition", "critical") is True
    assert ev.disk_full is False
    assert edges == ["enter", "exit"]
    records = _lines(tmp_path / "e.jsonl")
    gap = [r for r in records if r["type"] == "disk_full"]
    assert len(gap) == 1, "the JSONL must say why it has a hole in it"
    assert gap[0]["detail"]["dropped"] == 1
    assert gap[0]["kind"] == "storage"
    ev.close()


def test_a_raising_disk_full_callback_cannot_break_the_writer(tmp_path):
    def boom(edge, detail):
        raise RuntimeError("health hook exploded")

    ev = EventLog(str(tmp_path / "e.jsonl"), dev=True, min_free_mb=1e12, on_disk_full=boom)
    assert ev.write("lifecycle", "started") is False  # refused, but did not raise
    ev.close()


# -------------------------------------------------------------------- rotation


def test_in_process_rotation_keeps_the_configured_generations(tmp_path):
    path = tmp_path / "e.jsonl"
    ev = EventLog(str(path), dev=True, max_bytes=400, backups=2)
    for i in range(40):
        ev.write("perception", "dropout", "warn", frame_id=i, padding="x" * 60)
    ev.close()
    assert path.exists()
    assert (tmp_path / "e.jsonl.1").exists()
    assert (tmp_path / "e.jsonl.2").exists()
    assert not (tmp_path / "e.jsonl.3").exists(), "backups=2 must bound the generations"
    assert ev.rotations >= 2
    assert reg.EVENT_ROTATIONS_TOTAL.value(reason="size") >= 2
    assert path.stat().st_size <= 400 + 512


def test_external_rotation_is_noticed_and_the_file_reopened(tmp_path):
    """logrotate `create` renames the file out from under our fd. No copytruncate,
    so we must notice the inode change and reopen, or every later record is written
    into a file nobody can find."""
    clock = FakeClock()
    path = tmp_path / "e.jsonl"
    ev = EventLog(str(path), dev=True, stat_interval_s=0.0, clock=clock)
    ev.write("lifecycle", "started")
    os.rename(str(path), str(tmp_path / "e.jsonl.1"))
    clock.advance(1.0)
    ev.write("lifecycle", "stopping")
    ev.close()
    assert path.exists()
    assert [r["type"] for r in _lines(path)] == ["stopping"]
    assert [r["type"] for r in _lines(tmp_path / "e.jsonl.1")] == ["started"]
    assert reg.EVENT_ROTATIONS_TOTAL.value(reason="external") >= 1.0


def test_reopen_is_the_sighup_entry_point(tmp_path):
    ev = EventLog(str(tmp_path / "e.jsonl"), dev=True)
    ev.write("lifecycle", "started")
    assert ev.reopen() is True
    assert ev.write("lifecycle", "stopping") is True
    ev.close()


def test_max_bytes_zero_disables_in_process_rotation(tmp_path):
    ev = EventLog(str(tmp_path / "e.jsonl"), dev=True, max_bytes=0)
    for i in range(50):
        ev.write("perception", "dropout", frame_id=i, padding="y" * 100)
    ev.close()
    assert not (tmp_path / "e.jsonl.1").exists()
    assert ev.rotations == 0


# ---------------------------------------------------------------- unwritable


def test_unwritable_primary_falls_back_and_says_so(tmp_path, monkeypatch):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    fallback = tmp_path / "fallback" / "events.jsonl"
    monkeypatch.setattr("adas.io.events.default_fallback_path", lambda dev: str(fallback))
    try:
        ev = EventLog(str(blocked / "events.jsonl"), dev=True)
        if not ev.writable:  # running as root: the mode does not block the open
            pytest.skip("cannot create an unwritable directory as this user")
        assert ev.fallback is True
        assert ev.path == str(fallback)
        assert ev.write("lifecycle", "started") is True
        assert ev.stats()["fallback"] is True
        ev.close()
    finally:
        blocked.chmod(0o700)


def test_strict_mode_fails_closed_at_startup(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        try:
            ev = EventLog(str(blocked / "events.jsonl"), dev=True, strict=True)
        except OSError:
            return  # the expected outcome
        ev.close()
        pytest.skip("cannot create an unwritable directory as this user")
    finally:
        blocked.chmod(0o700)


def test_writing_after_close_is_refused_and_counted(log):
    log.close()
    assert log.write("lifecycle", "stopping") is False
    assert log.dropped == 1
    assert reg.EVENT_DROPPED_TOTAL.value(reason="closed") == 1.0


def test_close_is_idempotent(log):
    log.close()
    log.close()
    assert log.stats()["writable"] is False


# ------------------------------------------------------------------ config glue


def test_default_paths_differ_between_lab_and_production():
    assert default_events_path(dev=True) == os.path.join("data", "events.jsonl")
    assert default_events_path(dev=False) == "/var/lib/adas/events.jsonl"
    assert default_events_path(dev=False, configured="/x/y.jsonl") == "/x/y.jsonl"
    assert default_fallback_path(dev=False) == "/run/adas/events.jsonl"


def test_from_config_works_against_a_config_with_no_events_section(tmp_path, monkeypatch):
    from adas.core.config import load_config

    monkeypatch.chdir(tmp_path)
    cfg = load_config(None)
    ev = from_config(cfg)
    try:
        assert ev.path == os.path.join("data", "events.jsonl")
        assert ev.fsync_policy == "critical"
        assert ev.write("lifecycle", "started") is True
    finally:
        ev.close()


# ------------------------------------------------------------------- ordering
#
# Regression cover for: `t` came from the injectable clock, so two writers on one
# file wrote two clock domains into one field, and `seq` restarted at 1 on every
# process, so a file spanning a restart had duplicate sequence numbers.


def test_t_is_the_writers_own_monotonic_clock_not_an_injected_one(tmp_path):
    import time as _time

    path = tmp_path / "e.jsonl"
    before = _time.monotonic()
    ev = EventLog(str(path), dev=True, fsync="never", clock=lambda: 1.5)
    try:
        ev.lifecycle("started")
        ev.write("perception", "dropout", "critical", frame_id=3)
    finally:
        ev.close()
    after = _time.monotonic()

    stamps = [r["t"] for r in _lines(path)]
    assert stamps == sorted(stamps)
    for value in stamps:
        assert before <= value <= after, stamps


def test_seq_orders_the_whole_file_across_a_restart_of_the_writer(tmp_path):
    """Mixed call sites, two writer runs, one file: seq must still be a total order."""
    path = tmp_path / "e.jsonl"

    first = EventLog(str(path), dev=True, fsync="always")
    first.lifecycle("started", version="0.3.0")
    first.safety_state("nominal", "limited", frame_id=11)
    first.close()

    second = EventLog(str(path), dev=True, fsync="always", clock=lambda: 1.0)
    second.engine_failure("yolox_nano", "load", "engine missing")
    second.perception_dropout(5, "no frames", frame_id=12)
    second.lifecycle("stopping", reason="sigterm")
    second.close()

    records = _lines(path)
    assert [r["seq"] for r in records] == [1, 2, 3, 4, 5]
    stamps = [r["t"] for r in records]
    assert stamps == sorted(stamps), stamps
    assert second.stats()["written"] == 3
    assert second.stats()["seq"] == 5


def test_a_run_id_tells_two_writers_in_one_file_apart(tmp_path):
    path = tmp_path / "e.jsonl"
    first = EventLog(str(path), dev=True, fsync="never")
    first.lifecycle("started")
    first.close()
    second = EventLog(str(path), dev=True, fsync="never")
    second.lifecycle("stopping")
    second.close()

    runs = [r["run"] for r in _lines(path)]
    assert len(runs) == 2
    assert runs[0] != runs[1]
    assert first.run_id == runs[0] and second.run_id == runs[1]
    assert all(len(run) == 12 for run in runs)


def test_a_fresh_file_still_starts_at_seq_one(tmp_path):
    ev = EventLog(str(tmp_path / "fresh.jsonl"), dev=True, fsync="never")
    try:
        ev.lifecycle("started")
    finally:
        ev.close()
    assert [r["seq"] for r in _lines(tmp_path / "fresh.jsonl")] == [1]


def test_a_torn_tail_does_not_stop_the_writer_from_resuming(tmp_path):
    path = tmp_path / "e.jsonl"
    path.write_text('{"v":2,"seq":7,"kind":"lifecycle"}\n{"v":2,"seq":8,"ki')
    ev = EventLog(str(path), dev=True, fsync="never")
    try:
        ev.lifecycle("started")
    finally:
        ev.close()
    assert [r["seq"] for r in _lines(path)] == [7, 8]  # the torn line is skipped on read
