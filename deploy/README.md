# Deploying ADAS on a Jetson Xavier NX

Everything an operator needs to install, supervise, probe and roll back the ADAS
service on L4T R35.6.5 / JetPack 5.1.6.

This directory is the counterpart of the DMS project's `deploy/`, and it is
deliberately shaped the same way so a technician who has seen one has seen both:
`Type=notify` unit with a progress-gated watchdog, a loopback health endpoint, a
durable JSONL of safety events, a logrotate rule that does not lose lines, and an
installer whose only irreversible step is a symlink swap.

| file | what it is |
|---|---|
| `adas.service` | systemd unit. `Type=notify`, `WatchdogSec=30`, full sandbox. |
| `adas.logrotate` | `/etc/logrotate.d/adas` for the safety-event JSONL. |
| `setup_jetson.sh` | idempotent installer with `--dry-run`. Needs root to apply. |
| `Dockerfile.jetson` | arm64 runtime image on the L4T base (CUDA 11.4 + TensorRT 8.5). |

---

## 1. Honest status

Read this before you deploy anything.

| claim | status |
|---|---|
| Health endpoint (`/healthz`, `/readyz`, `/livez`, `/metrics`) | implemented and unit-tested — `src/adas/io/health.py` |
| Durable safety-event JSONL with fsync, rotation, ENOSPC survival | implemented and unit-tested — `src/adas/io/events.py` |
| systemd readiness + progress-gated watchdog | implemented and unit-tested — `src/adas/io/sd_notify.py` |
| Prometheus metric registry (counters, gauges, histograms) | implemented and unit-tested — `src/adas/io/metrics.py` |
| `adas.service`, `adas.logrotate`, `setup_jetson.sh` | written; validated with `bash -n`, `shellcheck`, `--dry-run`, `systemd-analyze verify` in CI |
| The unit **starting** on a real board | **never executed.** No account in this repository has root on the target, so nothing here has been `systemctl start`ed. |
| `--frames 0` meaning "run until stopped" | **not yet true in the runner.** See §5. `setup_jetson.sh` checks for it and refuses to enable the unit until it is. |
| The CLI wiring the three primitives together | **not yet done.** See §5: `src/adas/cli.py` and `src/adas/runtime/runner.py` still have to construct `HealthState`/`HealthServer`, `EventLog` and `WatchdogPinger`. Until then the unit will start, run, and report nothing. |
| `Dockerfile.jetson` | base image tag verified to exist (`docker manifest inspect`, 2026-09-13); **not built** — the base is ~5 GB. |

Nothing in this directory should be described as "production ready" until the two
**not yet** rows above are closed and the unit has been started once on a board.

---

## 2. Install

```bash
# See exactly what it would do. No root needed, changes nothing.
bash deploy/setup_jetson.sh --dry-run

# Apply.
sudo bash deploy/setup_jetson.sh
```

What it does, in order — nothing before the last step is irreversible:

1. **Preconditions.** Root (unless dry run), the tree looks like ADAS, `/usr/bin/python3`
   is 3.8, `PYTHONPATH=src python3 -c "import adas"` works, `numpy` and `cv2` are
   importable as *system* packages, engines are present.
2. **Continuous-mode preflight.** Runs `adas.cli --frames 0` for 8 s. If it exits
   early the installer says so and installs the unit **disabled** (§5).
3. **Account and directories.** System user `adas` (no home, `nologin`), added to
   `video`. `/var/lib/adas` and `/var/log/adas` 0750 `adas:adas`; `/etc/adas` and
   `/opt/adas` 0755 `root:root`.
4. **Stage.** `rsync` into `/opt/adas/releases/<git-sha>`, excluding `.git`, `tests`,
   `docs`, `data`, `recordings`, `__pycache__`, `*.onnx` and the vendored UFLD
   checkout. Everything is chowned to `root:root` and made read-only to the service.
5. **Release stamp.** `/etc/adas/release` records the release id, git sha and install
   time. `adas.io.metrics` reads the sha from there, so `adas_build_info` has a real
   revision on a unit that has no `.git`.
6. **Config.** `/etc/adas/config.json` from `config.example.json` (JSON-validated
   first). An existing file is preserved unless `--force-config`.
7. **Unit and logrotate.** Rendered for `--prefix`, installed 0644 root-owned.
   `logrotate --debug` is run to prove the rule parses.
8. **Activate.** `ln -sfn releases/<id> current` — the one irreversible step —
   then `daemon-reload`, `enable`, `restart`, and a 60 s poll of `/readyz`.

### Rollback

One symlink:

```bash
ls /opt/adas/releases
sudo ln -sfn /opt/adas/releases/<previous> /opt/adas/current
sudo systemctl restart adas
```

---

## 3. Probing a running unit

The endpoint binds `127.0.0.1:8090` and **refuses** a non-loopback bind unless
`health.allow_remote` is explicitly set: `/healthz` publishes the live safety state,
ego speed and lead-vehicle range, and that should not be readable by everything on
the vehicle network.

| route | code | meaning |
|---|---|---|
| `/livez` | always 200 | the process is scheduled and the HTTP thread answers. Nothing more. |
| `/readyz` | 200 / 503 | 200 only when every engine is loaded, the source is delivering, perception is up, at least one frame is done and the snapshot is fresh. **This is the one to gate on.** |
| `/healthz` | 200 / 503 | 503 when not ok or stale; 200 with `"degraded": true` when the unit is in LIMITED. |
| `/metrics` | always 200 | Prometheus text. A scraper must be able to scrape a broken unit. |

```bash
curl -fsS http://127.0.0.1:8090/readyz  >/dev/null && echo READY
curl -fsS http://127.0.0.1:8090/healthz | python3 -m json.tool
curl -fsS http://127.0.0.1:8090/metrics | grep -E '^adas_(fps|safety_state|degraded)'
```

**Why staleness matters.** A wedged frame loop still answers HTTP. `/healthz` reports
`stale: true` and returns 503 when the snapshot has not been refreshed within
`stale_after_s` (5 s), and an advancing `frame_id` counts as a refresh. A probe that
only checked "does the port answer" would call a blind unit healthy.

### The watchdog

`WatchdogSec=30` in the unit. `adas.io.sd_notify.WatchdogPinger` pings at 15 s **and
only when the frame id has advanced since the last ping**. A stalled pipeline stops
pinging, systemd kills the unit at 30 s, and `Restart=always` brings it back.
Withheld pings are counted in `adas_watchdog_skipped_total` and logged at ERROR, so a
marginal unit is visible before it is killed.

Do not replace this with a bare timer. A bare timer keeps a blind unit alive for ever
and is worse than no watchdog, because it looks like one.

### Metrics worth alerting on

| series | alert when |
|---|---|
| `adas_up` | absent — the unit is gone |
| `adas_health_stale` | `== 1` for > 30 s |
| `rate(adas_frames_processed_total[1m])` | below ~80% of the configured fps |
| `adas_safety_state{state="min_risk_maneuver"}` | `== 1` |
| `rate(adas_safety_state_transitions_total{to_state="disengage"}[5m])` | `> 0` |
| `adas_perception_consecutive_failures` | `> 3` |
| `adas_events_writable` | `== 0` — safety events are not being recorded |
| `adas_disk_full` | `== 1` |
| `histogram_quantile(0.95, rate(adas_stage_duration_ms_bucket[5m]))` | above the frame budget |
| `adas_lane_is_mock` | `== 1` on a vehicle unit — the lane model is a stub |
| `count by (git_sha) (adas_build_info)` | more than one sha in a fleet you thought was uniform |

---

## 4. The safety-event log

`/var/lib/adas/events.jsonl`, one JSON object per line, schema `v: 1`. It records
SafetyState transitions, perception dropouts and recoveries, engine failures, source
loss and reconnection, disk-full edges and lifecycle events — and nothing per-frame.
Per-frame telemetry goes to `/metrics`.

```bash
tail -f /var/lib/adas/events.jsonl | jq -c 'select(.severity=="critical")'
jq -r 'select(.kind=="safety_state") | "\(.ts) \(.detail.previous) -> \(.detail.current) \(.detail.reason)"' \
  /var/lib/adas/events.jsonl
```

Durability: entering `MIN_RISK_MANEUVER` or `DISENGAGE`, every engine failure and
every lifecycle record is `fsync`ed immediately; everything else rides a one-second
barrier. An ignition cut therefore cannot lose the record of the event that preceded
it, which is the only record anyone will want.

A full disk is a *state*, not a crash: writing stops, `adas_disk_full` goes to 1,
`/healthz` shows `events.writable: false`, and when space returns the log records the
gap (`{"kind":"storage","type":"disk_full","detail":{"edge":"exit","dropped":N}}`) so
the hole in the file explains itself.

Rotation happens twice over, on purpose. In-process at 32 MB (`EventLog(max_bytes=)`)
because logrotate runs from a daily timer and cannot bound a burst; and externally by
`adas.logrotate`, which uses `create` rather than `copytruncate` — `copytruncate`
loses whatever is written between the copy and the truncate, which for this file is
an evidence gap. The writer re-stats its path every 2 s and reopens when the inode
changes, so nothing is lost even if the `postrotate` SIGHUP is missed.

---

## 5. Known gaps — read before enabling the unit

### Continuous mode (`--frames 0`)

`deploy/adas.service` runs `--frames 0` and `setup_jetson.sh` verifies that this
means "run until stopped" before it will `systemctl enable` anything. At the time of
writing `src/adas/runtime/runner.py` still does `for frame_id in range(max_frames)`,
so `0` yields zero iterations, the process exits 0 immediately, and `Restart=always`
becomes an infinite restart loop that looks from the outside like a working
deployment. The preflight exists precisely so nobody discovers that in a vehicle.

Owner: the runtime workstream (`runner.py`, `cli.py`). Until it lands, the installer
prints the reason and leaves the unit disabled; `--force-enable` overrides.

### The three primitives are not yet wired into the CLI

`src/adas/io/` is complete and tested, but `src/adas/cli.py` and
`src/adas/runtime/runner.py` do not construct any of it yet, so a unit started today
will run and report nothing: no `/healthz`, no `READY=1` (so `Type=notify` will hit
`TimeoutStartSec` and systemd will consider the start failed), and no event log.

The wiring is about thirty lines. The snippet below was **executed end to end as a
scratch script on this board** against the real mock pipeline (30 frames): `/healthz`
returned 200 with `degraded: true`, `/readyz` 200, `/metrics` carried
`adas_fps 43.89` (measured, not the configured 20), `adas_safety_state{state="limited"} 1`
and `adas_stage_duration_ms_count{stage="e2e"} 30`; the watchdog pinged four times and
then withheld five pings once the frame id stopped advancing; `/healthz` went to 503
with `stale: true` 3.5 s after the last refresh; and five records landed in the JSONL.
What is missing is only that this lives in a scratch file rather than in `cli.py`:

```python
from adas.io import EventLog, HealthServer, HealthState, WatchdogPinger, notify_ready,
                    notify_status, notify_stopping
from adas.io.events import from_config as events_from_config

state  = HealthState()
state.set_build(version=adas.__version__, config=config_path)
health = HealthServer.from_config(state, config)
health.start()                                  # fails soft; logs on refusal
events = events_from_config(config)
wd     = WatchdogPinger()

# after the engines load and the FIRST frame has been processed, once:
notify_ready("frame 1 processed")

# every frame:
state.frame_id = frame_id
state.update_from_metrics(pipeline.metrics)
state.mark_update()
wd.tick(frame_id)                               # only pings when frame_id advanced

# on every SafetyState change:
events.safety_state(previous, current, frame_id=frame_id, reason=plan.reason)

# on shutdown:
notify_stopping("shutting down")
pipeline.metrics.log_summary()
events.lifecycle("stopping", reason=signal_name)
health.stop(); events.close()
```

`Type=notify` is the reason this is not optional: without `READY=1` the unit never
finishes starting.

### No SIGHUP handler

`adas.logrotate` sends `SIGHUP` in `postrotate` to make the reopen immediate. Nothing
installs a handler yet, so the `|| true` swallows it and the writer falls back to
noticing the inode change within 2 s. Adding `signal.signal(SIGHUP, lambda *_:
events.reopen())` in `cli.main` closes the 2 s window.

### Validating the tight sandbox

`adas.service` ships with `DevicePolicy=closed`, the `DeviceAllow=` list and
`SystemCallFilter=@system-service` **commented out**. They are the right hardening,
but a wrong device node or one syscall the CUDA runtime needs turns into a unit that
will not start, and nothing here can be tested without root. To enable them safely:

```bash
sudo systemctl stop adas
sudoedit /etc/systemd/system/adas.service        # uncomment the block
sudo systemctl daemon-reload && sudo systemctl start adas
sudo systemd-analyze security adas.service       # exposure score should drop
journalctl -u adas -b --grep 'Operation not permitted|Permission denied'
curl -fsS http://127.0.0.1:8090/readyz >/dev/null && echo "still ready"
```

If anything fails, re-comment the block; the loss is hardening, not function.

---

## 6. Containers

`Dockerfile` (repository root) is the **CI image**: `python:3.8-slim`, no CUDA, no
TensorRT, no OpenCV. It runs the test suite and the mock pipeline. It cannot load an
engine and must never be described as the deployment path.

`deploy/Dockerfile.jetson` is the **runtime image**: `nvcr.io/nvidia/l4t-jetpack:r35.4.1`
(verified to exist, ~5 GB compressed; carries CUDA 11.4, cuDNN 8.6, TensorRT 8.5.2.2,
matching this R35 board). Engines are bind-mounted, never baked in — a `.engine` is
tied to the TensorRT build and GPU that produced it.

```bash
# CI image
docker build -t adas-core:ci . && docker run --rm adas-core:ci

# Jetson runtime image (on the board; ~5 GB base pull the first time)
docker build -f deploy/Dockerfile.jetson -t adas-core:jetson .
docker compose --profile jetson up
```

`docker compose` needs the NVIDIA container runtime. This board already has it as the
*default* runtime (`docker info | grep 'Default Runtime'` → `nvidia`); the compose
file names `runtime: nvidia` anyway so the requirement is explicit elsewhere.

Remove the `/dev/video0` device mapping on a board with no camera — this one has
none — or the container will refuse to start.

Bases that do **not** exist, checked so nobody wastes an afternoon on them:
`nvcr.io/nvidia/l4t-base:r35.4.1`, `nvcr.io/nvidia/l4t-tensorrt:r8.5.2.2-runtime`,
`nvcr.io/nvidia/l4t-jetpack:r35.6.x`. `nvcr.io/nvidia/l4t-tensorrt:r8.5.2.2-devel`
does exist (~4.9 GB) but has no L4T OpenCV. And `nvcr.io/nvidia/tensorrt:22.12-py3`,
which `DEPLOYMENT.md` currently recommends, is **x86_64 only** and cannot run here at
all.

---

## 7. Operating notes

```bash
systemctl status adas
journalctl -u adas -f -o cat | jq          # ADAS_LOG_FORMAT=json is set in the unit
journalctl -u adas -b -p warning
systemd-analyze security adas.service
```

* **Restart loop.** `StartLimitBurst=5` / `StartLimitIntervalSec=300`: five starts in
  five minutes and the unit lands in `failed`, where it is visible and alertable,
  rather than restarting silently for ever. `systemctl reset-failed adas` clears it.
* **Memory.** `MemoryHigh=2500M` / `MemoryMax=3G` on a 6.7 GiB board with ~2.5 GiB
  typically free. UFLDv2's engine alone is 413 MB resident. Without the cap this
  process OOMs the whole system instead of being restarted.
* **Power.** `setup_jetson.sh --apply-power` runs `nvpmodel -m 8` (MODE_20W_6CORE).
  `jetson_clocks` is never run: pinning clocks on a vehicle unit trades thermal
  headroom for a few percent and this board has no active cooling margin to spare.
* **Two units on one board.** DMS uses port 8088, ADAS 8090. Both write to
  `/var/lib/<name>` under their own accounts. They share the GPU, so if both run,
  budget the engines accordingly.
* **Clocks.** Xavier NX has no battery-backed RTC. It boots with a wrong wall clock
  and steps when NTP syncs. Every duration in the ops layer uses `time.monotonic`;
  the `ts` field in the event log is wall clock and may jump, which is why every
  record also carries monotonic `t` and a boot id.
