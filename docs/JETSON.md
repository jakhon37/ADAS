# ADAS on this Jetson — pause / pickup notes

Paused: 2026-09-09. Branch: `jetson-native-py38` (v0.2.0).
Workspace: `/home/nvidia/myspace/ADAS`.

This is the document to read before continuing development.

## Hardware

| Item | Value |
|---|---|
| Board | NVIDIA Jetson Xavier NX Developer Kit |
| JetPack | 5.1.7 (L4T 35.6.5) |
| RAM | 6.7 GB (leave ≥2.5 GB free before `trtexec` or TensorRT) |
| Power | MODE_20W_6CORE |
| CUDA / TensorRT | 11.4.315 / 8.5.2.2 |
| Host Python | **3.8.10** (system). 3.9 is present; **no 3.10** |
| OpenCV | 4.5.4 (CPU, no CUDA) |
| ROS 2 | Not on the host. `ros:humble` and `auto_ros:humble` Docker images exist |
| Camera | No `/dev/video*`. Argus / `nvv4l2camerasrc` plugins are installed |

Neighbor projects on this board:

- `/home/nvidia/myspace/DMS` — TensorRT + GStreamer on Python 3.8. Uses ~0.8 GB + GPU when replaying. **Do not run `trtexec` or ADAS TensorRT while `dms.app` is up.**
- `/home/nvidia/myspace/autoJetsonBot` — ROS 2 Humble in Docker
- `/home/nvidia/myspace/cv-research` — Jetson FPS/power experiments

`python3-venv` is not installed (sudo needs a password). Run with `PYTHONPATH=src` and system site packages (`numpy`, `cv2`, `tensorrt`).

## What works now

- Planning (ACC + LKA), tracking, PID-like control, safety monitor
- Native Python 3.8 (no `dataclass(slots=True)`)
- Mock perception (default) — synthetic ACC demo is physically sane (~10.5 m lead car, not 0.3 m)
- **TensorRT YOLOv5n** vehicle detector when `models/yolov5n.engine` exists
- Video file / OpenCV camera / CSI GStreamer source (`--source`)
- Pinhole range: `distance_m = (object_height_m * focal_length_px) / box_height` (defaults 1.5 m, 910 px)
- Track range-rate velocity
- 37 unit tests on host Python 3.8

Measured on `Ultra-Fast-Lane-Detection-v2/example.mp4` around frame 150:

- 1× `car` tracked, range ~10–11 m, plan `follow_close_*`
- Full pipeline **~45 ms/frame (~22 FPS)** including decode
- Engine GPU compute ~7.3 ms (trtexec)

## What does not work yet

- Lane estimator is still **mock** (fixed 36%/64% geometry). `--lane ufld` is coded but there is **no UFLD engine**
- No overlay / saved debug video
- Ego-lane gating default is **off** (`ego_lane_half_width_frac=0`); `config.example.json` sets `0.2`
- ROS 2 bridge is source-only; Humble is Docker-only
- No live camera attached
- Safety steering-rate check is still a TODO
- Dockerfile is `python:3.11-slim` — unit tests only, not TensorRT/Argus

## How to run

From `/home/nvidia/myspace/ADAS`:

```bash
# tests (CPU, no TensorRT)
PYTHONPATH=src python3 -m pytest tests/ -v --tb=short

# mock synthetic (no GPU)
PYTHONPATH=src python3 -m adas.cli --frames 10

# TensorRT YOLO on the bundled lane video
PYTHONPATH=src python3 -m adas.cli \
  --detector tensorrt \
  --source Ultra-Fast-Lane-Detection-v2/example.mp4 \
  --frames 200
```

CLI flags: `--config`, `--source` (`synthetic` | video path | `camera` | `csi` | `camera:N`), `--detector mock|tensorrt`, `--lane mock|ufld`, `--frames`, `--log-level`.

Rebuild the YOLO engine only when the board is idle (≥2.5 GB available, no `dms.app`):

```bash
python3 scripts/build_yolo_engine.py
```

Downloads official YOLOv5n ONNX (YOLOv8n ONNX URLs 404’d) and runs `/usr/src/tensorrt/bin/trtexec --fp16`.

## Models (gitignored binaries)

| File | Role |
|---|---|
| `models/yolov5n.onnx` | Official Ultralytics YOLOv5n v7.0 (~3.8 MB) |
| `models/yolov5n.engine` | TensorRT 8.5 FP16, input `images` 1×3×640×640, output `output0` 1×25200×85 |
| `models/ufldv2_*.engine` | **Not built.** Needed for `--lane ufld` |

See `models/README.md`. Do not commit `.onnx` / `.engine`.

## Layout added in 0.2.0

```
src/adas/infer/          # TensorRT + libcudart (lazy import, no pycuda)
src/adas/perception/
  factory.py             # mock | tensorrt | ufld
  yolo.py                # YOLOv5/v8 decode + NMS
  ufld.py                # UFLDv2 decode (needs engine)
src/adas/runtime/capture.py
scripts/build_yolo_engine.py
```

Default detector/lane backends are **mock**, so CI and `pytest` never load CUDA.

## Next when development resumes

1. Overlay + save video (boxes, IDs, range, throttle/brake) — same idea as DMS replay
2. UFLD TensorRT engine + `--lane ufld` on `example.mp4`
3. Turn on ego-lane gating by default (`ego_lane_half_width_frac=0.2`)
4. Live CSI/USB camera when hardware is attached
5. ROS 2 Humble in `auto_ros:humble`, not native Foxy
6. Then vehicle/CAN — not before overlay + real lanes

## Courtesy

Xavier NX has 6.7 GB. DMS replay (`python3.8 -m dms.app`) takes the GPU. ADAS TensorRT and `trtexec` should wait until `MemAvailable` ≥ 2500 MB and `GR3D_FREQ` is ~0%.

Root-owned `.pytest_cache/` and some `__pycache__` dirs from an earlier Docker run cannot be deleted without sudo. They are gitignored.
