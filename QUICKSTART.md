# Quick start

Everything below has been run on the target board (Jetson Xavier NX, JetPack
5.1.6, Python 3.8.10, TensorRT 8.5.2.2) on 2026-09-13. Read
[docs/JETSON.md](docs/JETSON.md) for the board's quirks and
[README.md](README.md#what-is-not-production-ready) before drawing any
conclusion from the output.

## On this Jetson

`python3-venv` is not installed. Use the system interpreter with `PYTHONPATH`,
and hold the board's GPU mutex for anything that loads TensorRT — other agents
share this hardware.

```bash
cd ~/myspace/ADAS

# 1. Tests. 743 pass; the engine-backed ones skip when a file is missing.
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=src python3 -m pytest tests/ -q"

# 2. Real engines, blank frames. Proves the engines deserialise and the loop runs.
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=src python3 -m adas.cli --frames 10"

# 3. Real engines on the replay clip, closed loop against a simulated plant.
flock /tmp/jetson-gpu.lock -c "PYTHONPATH=src python3 -m adas.cli \
  --detector yolox --lane ufld \
  --source Ultra-Fast-Lane-Detection-v2/example.mp4 \
  --ego-source simulated --ego-speed 15 \
  --frames 200 --fps 0"
```

Actual output of (3):

```
frames=200 failures=0 dropped=0 reconnects=0 elapsed=13.09s
measured=15.28 FPS busy=15.44 FPS (64.8 ms/frame) reason=completed
  Detections:       136 (avg 0.68/frame)
  Tracks:           154 (avg 0.77/frame)
  Lane detected:    100.0% of frames
```

A representative per-frame line:

```
frame=198 det=1 trk=2 lane=yes lead=8.0m plan=follow_gap_8.0m|lane_center_err_0.11
          safety=nominal cmd=t0.00/b0.12/s+0.06
```

`cmd=` is the **arbitrated** command — what the actuators would receive — not the
controller's request. When they differ, the line is logged at WARNING with the
violations that caused it.

## Without engines, or on any other machine

Mock backends fabricate their output, so they must be enabled explicitly:

```bash
PYTHONPATH=src python3 -m adas.cli --detector mock --lane mock --allow-mock --frames 20
```

Without `--allow-mock` (or `ADAS_ALLOW_MOCK=1`, or `"allow_mock": true`) that
command exits 2 with:

```
Configuration error: detector.backend=mock and lane.backend=mock selected but
allow_mock is false. A mock backend FABRICATES geometry that nothing downstream
can distinguish from a measurement; ...
```

That is deliberate: the mock used to be the silent default.

## Installing elsewhere

```bash
git clone --recurse-submodules https://github.com/jakhon37/ADAS.git
cd ADAS
pip install -e ".[dev]"
adas-run --detector mock --lane mock --allow-mock --frames 20
```

`numpy` is the only runtime dependency pip will install. OpenCV and TensorRT are
deliberately **not** declared: on the Jetson both come from JetPack system
packages, and `pip install opencv-python` would shadow the board's
GStreamer-enabled build. `rclpy` cannot be installed with pip in any working
form — use apt and a ROS 2 underlay.

## The commands you will actually use

```bash
# What did my configuration resolve to?
PYTHONPATH=src python3 -m adas.cli --config my.json --print-config

# Run until stopped (SIGTERM / Ctrl-C), the way the systemd unit does
PYTHONPATH=src python3 -m adas.cli --config my.json --frames 0

# Health and metrics while it runs
curl -s http://127.0.0.1:8090/healthz | python3 -m json.tool
curl -s http://127.0.0.1:8090/metrics | grep adas_safety_state

# Quiet, structured logs for a log shipper
ADAS_LOG_FORMAT=json PYTHONPATH=src python3 -m adas.cli --log-level WARNING --frames 0

# Throughput, unpaced
PYTHONPATH=src python3 -m adas.cli --fps 0 --frames 300

# Replay a recorded ego speed channel alongside the clip
printf 'timestamp_s,speed_mps\n0.0,14.0\n0.5,13.5\n1.0,13.0\n' > /tmp/speed.csv
PYTHONPATH=src python3 -m adas.cli --detector yolox --lane ufld \
  --source Ultra-Fast-Lane-Detection-v2/example.mp4 --ego-file /tmp/speed.csv --frames 50
```

A `--ego-file` channel is the only ego source that reports `measured = True`.
Samples closer together than `ego.max_age_s` are linearly interpolated; a wider
gap is held for `max_age_s` and then reported invalid.

### Flags worth knowing

| flag | effect |
|---|---|
| `--frames 0` | run until stopped; anything > 0 is a bounded run |
| `--fps 0` | do not pace; measure real throughput |
| `--allow-mock` | permit fabricating backends. Bench only |
| `--ego-source {none,config,simulated,file}` | where ego speed comes from; `none` keeps the planner degraded |
| `--depth midas` | enable the independent range cross-check |
| `--lane yolop` | YOLOP; automatically drops to `every_n_frames=4` because it costs ~59 ms |
| `--print-config` | validate, print the resolved sections, exit |
| `--no-health` / `--no-events` | drop the ops surface for a throwaway run |

## Configuration

`config.example.json` is the **bench** profile — mock backends, synthetic source,
`allow_mock: true` — and doubles as the complete key reference, because every key
is validated and an unknown key is a hard error:

```
unknown key 'max_speed_mpsX' in config section 'safety'. Known keys: ...
```

The **vehicle** profile is in [README.md](README.md#the-vehicle-profile). The
three settings that matter most:

```json
"camera": { "calibrated": true, "mount_height_m": 1.30, "pitch_deg": -3.87 },
"ego":    { "source": "file", "file": "recordings/speed.csv", "max_age_s": 0.15 },
"allow_mock": false
```

Do not set `"calibrated": true` until you have measured the mount height and
pitch. It is the switch that tells the whole stack its metres are real, and the
shipped defaults are measurably wrong for the one clip we have — see
[README.md](README.md#calibration).

## Python API

```python
import time
from adas.cli import build_pipeline
from adas.core.config import default_config
from adas.core.models import EgoState, PerceptionFrame
from adas.runtime import synthetic_frame

config = default_config()
config.detector.backend = "mock"
config.lane.backend = "mock"
config.allow_mock = True
config.__post_init__()          # re-validate after any override

pipeline, config = build_pipeline(config=config)
try:
    payload = synthetic_frame()
    frame = PerceptionFrame(
        frame_id=0, timestamp_s=time.monotonic(), rgb=payload,
        width=payload["width"], height=payload["height"],
    )
    plan, command = pipeline.step(
        frame, ego=EgoState(speed_mps=15.0, valid=True), dt_s=0.05
    )
    print(plan.reason)
    print(command)                                  # the ARBITRATED command
    print(pipeline.last_arbitration.state.value, pipeline.last_arbitration.violations)
finally:
    pipeline.close()
```

Passing `ego=` (or `current_speed_mps=`) is not optional in practice: with
neither, `EgoState.valid` is `False`, the time-gap law is undefined and the
arbiter commands a minimum-risk manoeuvre. That is the honest degradation, not a
bug.

## Record and replay

```python
from adas.tools import DataRecorder, DataReplayer, RecordingConfig, RecordingPipeline, ReplayConfig
from adas.tools.replayer import replay_with_pipeline

recorder = DataRecorder(RecordingConfig(output_dir="recordings", recording_name="run1"))
recorder.start_recording()
wrapped = RecordingPipeline(pipeline, recorder)     # drop-in; records the arbitration too
...                                                  # run the loop against `wrapped`
recorder.stop_recording()

replayer = DataReplayer(ReplayConfig(recording_dir="recordings/run1", playback_speed=0.0))
print(replayer.safety_timeline())                    # every arbitration state change
for recorded, (plan, command) in replay_with_pipeline(replayer, other_pipeline):
    ...                                              # compare old vs new decisions
```

`replay_with_pipeline` swaps in perception backends that serve the *recorded*
detections and lanes, so tracking, planning, control and arbitration re-run on
any machine, GPU or not.

## Troubleshooting

**`Cuda Runtime (out of memory)` loading the UFLD engine.** It is 413 MB and
needs one contiguous allocation. Check `free -m`; the factory warns when
available memory is under 1.6× the engine size. The pipeline already loads the
largest engine first, which is what makes detector + UFLD fit at all.

**`plan=degraded_ego_speed_unavailable` on every frame.** `ego.source` is `none`.
That is the default and it is correct for a board with no vehicle bus; pass
`--ego-source simulated` for a bench run or point `ego.file` at a recording.

**`safety=min_risk_maneuver`, then `safety=disengage`, `violations=ego_state_invalid`.**
Same cause. A missing ego state is a persistent *fault*, so after
`safety.disengage_after_frames` (40) the arbiter latches DISENGAGE and stays
there until `pipeline.reset()`. A bare 200-frame run with the default
`ego.source: none` ends that way; it is the designed behaviour, not a crash.

**`lane_offset_unavailable` in the arbiter reason.** There is no metric lane
geometry, because the camera is uncalibrated or the lane is mock.
`safety.max_lateral_offset_m` cannot be enforced without it, and the arbiter says
so rather than passing the check silently.

**Health endpoint refuses to bind.** Another process holds 8090 (DMS uses 8088).
It fails soft and logs `HEALTH_BIND_REFUSED`; set `health.required: true` to make
it fatal, or `--no-health`.

**Tests import-error on `cv2` or `tensorrt`.** Those tests skip by design. The
core suite needs only numpy.
