# ADAS Core (Production-Oriented Reference)

This repository now provides a **modular ADAS software architecture** rather than a single legacy script. It is designed as a production-style baseline for Jetson-class edge compute (Jetson Nano/Xavier/Orin), with clear interfaces for replacing mock components with optimized inference engines (TensorRT / ONNX Runtime / DeepStream).

## What was redesigned

- Moved from tutorial snippets to a package-based architecture (`src/adas/...`).
- Added structured domain models and typed pipeline contracts.
- Added explicit modules for:
  - perception (object detection + lane estimation)
  - tracking
  - behavior planning
  - control
  - runtime and CLI orchestration
- Added configuration support via JSON configuration files.
- Added automated tests for key planning/control behavior.

---

## Architecture

```text
Sensor Input (Camera)
   -> Perception (Detector + Lane Estimator)
   -> Tracking (ID association + distance proxy)
   -> Behavior Planning (lane centering + follow distance)
   -> Control (throttle/brake/steering command)
   -> Vehicle Interface / Actuation
```

### Package layout

```text
src/adas/
  config.py               # runtime + planner + detector configs
  types.py                # domain models (frames, tracks, plans, commands)
  pipeline.py             # end-to-end ADASPipeline.step(...)
  runtime.py              # runtime loop + synthetic frame generator
  cli.py                  # command-line entrypoint
  control.py              # longitudinal + steering command conversion
  planning.py             # behavior policy
  tracking.py             # object tracker
  perception/
    detection.py          # detector interface (replaceable backend)
    lane.py               # lane model estimation
tests/
  test_pipeline.py
```

---

## Quick start

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### 2) Synthetic smoke test

```bash
adas-run --synthetic
```

### 3) Multi-frame synthetic run

```bash
adas-run --frames 200
```

---

## Production integration notes

To take this into a real vehicle stack:

1. **Detector backend**
   - Replace `ObjectDetector.infer()` with TensorRT/DeepStream execution.
   - Keep output contract as `list[BoundingBox]`.

2. **State estimation**
   - Replace simple tracker with EKF/UKF + motion model and timestamped velocity estimates.

3. **Planning**
   - Upgrade `BehaviorPlanner` to scenario-based planner:
     - lane keeping
     - cut-in handling
     - emergency braking
     - stop-and-go logic

4. **Control**
   - Replace proportional controller with validated PID/MPC and actuator constraints.

5. **Safety and observability**
   - Add watchdogs, fail-operational state machine, health metrics, and structured telemetry.

---

## Testing

```bash
pytest
ruff check src tests
python -m adas.cli --synthetic
```

This repository is intentionally clean and modular so that each subsystem can be replaced by production-grade components without rewriting the rest of the stack.
