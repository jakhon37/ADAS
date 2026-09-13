# Deployment

> **The service in this repository has never been started, and the Jetson
> container image has never been built.** `sudo` is unavailable in the
> development environment, so `deploy/adas.service` is validated only by
> `systemd-analyze verify`, a hardening-directive assertion in CI, and review.
> Treat the first real start as a bring-up: watch it with
> `journalctl -u adas -b -f`.
>
> Read [README.md](README.md#what-is-not-production-ready) before deploying this
> anywhere near a vehicle.

Everything operational lives in [`deploy/`](deploy/README.md); this file is the
overview and the order of operations.

## Requirements

**Target board (what this was built and measured on):**

| Item | Value |
|---|---|
| Board | NVIDIA Jetson Xavier NX |
| L4T / JetPack | R35.6.5 / 5.1.6 |
| OS | Ubuntu 20.04, aarch64, 6 cores |
| Python | **3.8.10 only** |
| CUDA / TensorRT / cuDNN | 11.4 / 8.5.2.2 / 8.6 |
| OpenCV | 4.5.4 system build, GStreamer yes, CUDA no |
| RAM | 6.7 GiB total. Keep ≥ 1.5 GiB available before loading UFLD-v2 |
| Disk | ~1.5 GB for the model artefacts |

**Any other Linux host** can run the test suite and the mock pipeline with
Python 3.8+ and numpy. It cannot run an engine.

Do **not** `pip install` torch, tensorflow, onnx, onnxruntime, pycuda,
opencv-python, ultralytics or jetson-stats on the board. TensorRT, CUDA and
OpenCV come from JetPack; a pip OpenCV shadows the GStreamer build and the
GStreamer build is what reads the camera.

## Order of operations

```
1. install the source          →  /opt/adas/releases/<sha>, symlinked as current
2. fetch + build the engines   →  scripts/fetch_models.sh, scripts/build_engines.py
3. calibrate the camera        →  measure mount height and pitch; set calibrated: true
4. write the vehicle config    →  /etc/adas/config.json
5. preflight                   →  deploy/setup_jetson.sh --dry-run, then adas.cli --frames 0
6. install and enable the unit →  only after the preflight passes
```

Step 3 is not optional if you intend to use any metric output. The shipped
camera block is an assumption and is measurably wrong for the one clip we have;
see [README.md](README.md#calibration).

## 1. Install

```bash
git clone --recurse-submodules https://github.com/jakhon37/ADAS.git
cd ADAS
PYTHONPATH=src python3 -m pytest tests/ -q          # 859 pass, 1 known failure
```

Or use the staged installer, which copies into
`/opt/adas/releases/<git sha>` and only swaps the `current` symlink at the end:

```bash
sudo deploy/setup_jetson.sh --dry-run    # every read-only check, for real
sudo deploy/setup_jetson.sh
```

Its distinctive feature is a continuous-mode preflight: it runs
`adas.cli --frames 0` for 8 s and, if the process exits early, installs the unit
but **refuses to enable it**, printing why. That is what stops a broken build
becoming a restart loop in a vehicle.

## 2. Engines

Engines are **not** distributed: they are gitignored, excluded from the sdist,
and are version- and device-locked to the TensorRT and GPU they were built on.
Build them on the target.

```bash
bash scripts/fetch_models.sh                 # sha256-pinned downloads
python3 scripts/build_engines.py             # trtexec --fp16, then verify
python3 scripts/build_engines.py --list      # what is available
```

Sizes and measured GPU compute (from `models/README.md`):

| engine | file | GPU median |
|---|---|---|
| `yolox_nano.engine` | 3.2 MB | 4.69 ms |
| `midas_v21_small_256.engine` | 33.9 MB | 6.33 ms |
| `yolox_tiny.engine` | 12.7 MB | 6.43 ms |
| `yolov5n.engine` (AGPL) | 5.7 MB | 7.23 ms |
| `ufldv2_culane_res18.engine` | **413.4 MB** | 16.53 ms |
| `yolop_640.engine` | 20.2 MB | 26.98 ms |

**UFLD-v2 needs 413 MB in one contiguous allocation.** The pipeline loads the
largest engine first for exactly this reason (measured: detector-then-lane fails
with `Cuda Runtime (out of memory)` at ~1.1 GB available; lane-then-detector
succeeds). If you cannot guarantee the headroom, use `--lane yolop` instead and
accept a noisier lane fit, or run without lane perception.

Building an engine needs the board idle. Other projects share it:

```bash
flock /tmp/jetson-gpu.lock -c "python3 scripts/build_engines.py --only yolox_nano"
```

## 3. Configuration

`config.example.json` is the bench profile (mock backends, synthetic source,
`allow_mock: true`) and doubles as the complete key reference. The vehicle
profile is in [README.md](README.md#the-vehicle-profile). Validate before
deploying:

```bash
PYTHONPATH=src python3 -m adas.cli --config /etc/adas/config.json --print-config
```

An unknown key is a hard error naming the section, and cross-section rules
(planner ⊆ safety) are checked at load. `ADAS_CONFIG_PATH` is read when
`--config` is absent; both container images set it.

## 4. systemd

`deploy/adas.service` is `Type=notify` with `NotifyAccess=main`. The process
sends `READY=1` only after the **first frame has completed**, so a start that
succeeds means a frame went all the way through the pipeline.

```bash
sudo install -m 0644 deploy/adas.service   /etc/systemd/system/adas.service
sudo install -m 0644 deploy/adas.logrotate /etc/logrotate.d/adas
sudo systemctl daemon-reload
sudo systemctl start adas         # watch it: journalctl -u adas -b -f
sudo systemctl enable adas        # only once you have watched a start succeed
```

Key directives and why:

* `WatchdogSec=30` against `TimeoutStartSec=120`. `WatchdogPinger` pings only
  when the frame id has advanced, so a wedged frame loop is killed and restarted
  rather than sitting there looking alive.
* `Restart=always` with `StartLimitBurst=5` — a persistent failure stops
  restarting instead of thrashing.
* `MemoryHigh=2500M` / `MemoryMax=3G` / `OOMPolicy=stop`.
* `NoNewPrivileges`, `ProtectSystem=strict` with scoped `ReadWritePaths`,
  `ProtectHome`, `PrivateTmp`, an empty `CapabilityBoundingSet`,
  `RestrictAddressFamilies`.
* The tighter `DevicePolicy` / `SystemCallFilter` block ships **commented out**
  with the validation procedure written next to it: it cannot be tested without
  root, and a sandbox that hides a `/dev/nvhost-*` node the CUDA runtime needs
  would fail at load time, in the vehicle.
* `TimeoutStopSec=15` matches the CLI's SIGTERM handler, which stops the loop
  after the current frame, releases the source, logs the metrics summary and
  flushes the event log.

`deploy/adas.logrotate` uses `maxsize` + `create` (never `copytruncate`, which
loses records from an append-only evidence log) and `postrotate` sends SIGHUP;
the CLI reopens the event log on SIGHUP. Even without the signal the log notices
the inode change within `stat_interval_s` (2 s).

## 5. Containers

There are two images, and they do different things.

**`Dockerfile` — the CI image.** `python:3.8-slim`, numpy + pinned
`opencv-python-headless` + pytest + ruff, default `CMD` is the test suite. It
**cannot load a TensorRT engine** and has no `HEALTHCHECK`, because a test runner
has no steady state. Do not deploy it.

```bash
docker build -t adas-core:ci .
docker run --rm adas-core:ci                 # runs the tests
```

**`deploy/Dockerfile.jetson` — the runtime image.** Based on
`nvcr.io/nvidia/l4t-jetpack:r35.4.1` (existence and ~5 GB size verified with
`docker manifest inspect`; `l4t-base:r35.4.1` and `l4t-tensorrt:r8.5.2.2-runtime`
do **not** exist). System packages only, a build-time assertion that
`tensorrt`, `cv2` and `numpy` import, `tini` for SIGTERM forwarding, and a
`HEALTHCHECK` against `/healthz`. **It has never been built** — the first build
pulls ~5 GB and may need an apt package-name correction; the import assertion is
there so that fails the build rather than the vehicle.

```bash
docker compose --profile jetson up -d
```

The `jetson` profile sets `runtime: nvidia`, mounts the Argus socket, bind-mounts
`./models` **read-only** (engines are never baked into an image — they are
device-locked and large), uses a named volume for the event log, and publishes
the health port on loopback only. `docker compose` with no profile starts
nothing.

## 6. Verifying a deployment

```bash
systemctl is-active adas
curl -sf http://127.0.0.1:8090/readyz  && echo READY
curl -s  http://127.0.0.1:8090/healthz | python3 -m json.tool
curl -s  http://127.0.0.1:8090/metrics | grep -E 'adas_(safety_state|frames_processed|lane_is_mock|ego_speed_valid)'
tail -n 20 /var/lib/adas/events.jsonl
```

What to check, in order:

1. `readyz` returns 200. It requires a delivering source, no missing or failed
   engine, `perception.ok`, at least one completed frame, and a fresh snapshot.
2. `adas_lane_is_mock == 0` and `engines` contains no `"mock"`. A mock backend on
   a vehicle unit is an alerting condition.
2b. `engine_sha256` in `/healthz` lists every engine this process verified against
   `models/MANIFEST.json`, and `adas_ram_mb` is a real number (never `0`; an
   unmeasurable RSS exports `NaN`). An engine the manifest does not list loads with a
   WARNING and does **not** appear there — on a vehicle unit that is an alerting
   condition too.
3. `adas_ego_speed_valid == 1`. It is 1 only for an ego source that is both valid
   **and** a measurement — a simulated or declared speed reports 0.
4. `adas_safety_state{state="nominal"} == 1` in steady traffic. Persistent
   `limited` means the arbiter is intervening every frame; read the event log for
   the violation names before changing any threshold.
5. `adas_frames_processed_total` is advancing and `adas_watchdog_skipped_total`
   is 0.

## Health and metrics

| route | 200 when |
|---|---|
| `/livez` | the HTTP thread is alive. Always |
| `/readyz` | everything in (1) above |
| `/healthz` | ok and fresh. **503 when stale** even though HTTP answers; 200 when merely degraded, because `LIMITED` is a mode, not an outage |
| `/metrics` | always; Prometheus exposition format |

The endpoint binds to `127.0.0.1:8090` (DMS uses 8088). A non-loopback bind needs
`health.allow_remote: true` set deliberately — the body carries live safety
state, ego speed and lead range. `health.token_file` adds a bearer token on
everything except `/livez`; read it from a root-owned file, never from the
config.

## Troubleshooting

| symptom | cause | action |
|---|---|---|
| unit fails at `TimeoutStartSec` | no `READY=1`: the first frame never completed | run the same command by hand; the source or an engine is the usual cause |
| `Cuda Runtime (out of memory)` at start | UFLD-v2 needs 413 MB contiguous | `free -m`; stop the other GPU process; or `--lane yolop` |
| every frame `plan=degraded_ego_speed_unavailable` | `ego.source: none` | wire a real speed channel; see README |
| every frame `safety=min_risk_maneuver`, `ego_state_invalid` | same | same |
| `HEALTH_BIND_REFUSED` | port in use | change `health.port`, or `--no-health` |
| `/healthz` 503 with `"stale": true` | the frame loop stopped advancing | the watchdog should already have restarted it; check `journalctl` |
| events say `disk_full` | the log hit `min_free_mb` or ENOSPC | free space; the log records how long the hole was and how many records it swallowed |
| `adas_lane_is_mock == 1` on a vehicle | a mock backend is active | `allow_mock` is set somewhere, or an engine was missing at start |

## What deployment does not cover

There is no vehicle interface in this repository. `ControlCommand` is
`(throttle, brake, steering)` in normalised units; converting that to CAN frames,
arbitrating against the driver, and handling actuator faults are all outside it.
The ROS 2 bridge is the nearest thing and it has never been executed. Nothing
here has been through ISO 26262 work of any kind.

Four more things a deployment does not get, stated so nobody assumes otherwise:

* **No soak evidence.** The longest run ever performed against this codebase is a few
  hundred frames — about 25 seconds of clip. There is no 8 h result, no RSS curve, no
  file-descriptor audit. `MemoryMax=3G` in the unit is a guess against a process
  measured at 1.09 GiB RSS / 1.38 GiB high-water over 300 frames.
* **No camera has ever been attached** to the development board, so the CSI path in
  `deploy/` has never moved a frame.
* **The unit has never been started.** `sudo` needs a password in this environment.
  Everything under `deploy/` is validated by `systemd-analyze verify` and review only.
* **Several safety thresholds are not reachable from a config file.** The new
  `SafetyLimits` fields — including `min_dt_s`, `max_dt_s`, `max_frame_gap_s` and the
  whole `aeb_*` group, two of which can latch the terminal DISENGAGE state — are
  settable programmatically but have no key in `SafetyConfig` or
  `adas.cli.build_safety_limits`. A deployed config file therefore cannot pin them.
  So are `failsafe_hold_s`, `emergency_stop_time_s` and the log-gate periods.
