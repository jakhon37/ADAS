# ADAS Core - Quick Start Guide

On the Xavier NX in this workspace, skip PyPI/`pip install` and use `PYTHONPATH`. Full board notes: [docs/JETSON.md](docs/JETSON.md).

```bash
cd /home/nvidia/myspace/ADAS
PYTHONPATH=src python3 -m pytest tests/ -v --tb=short
PYTHONPATH=src python3 -m adas.cli --frames 10
# TensorRT YOLO (needs models/yolov5n.engine)
PYTHONPATH=src python3 -m adas.cli --detector tensorrt \
  --source Ultra-Fast-Lane-Detection-v2/example.mp4 --frames 60
```

Expected mock synthetic line (lead vehicle ~10.5 m, not emergency brake):

```
plan=follow_close_10.5m|lane_center_err_0.00, cmd=t0.47/b0.00/s0.00
```

## 🚀 Other machines

### 1. Install Package

```bash
# From source
git clone --recurse-submodules https://github.com/jakhon37/ADAS.git
cd ADAS
pip install -e ".[dev]"
```

### 2. Run Synthetic Test

```bash
adas-run --frames 60
```

### 3. Customize Configuration

```bash
# Copy example config
cp config.example.json my-config.json

# Edit as needed
vim my-config.json

# Run with custom config
adas-run --config my-config.json --frames 100
```

### 4. Run Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_safety.py -v
```

### 5. Build Docker Image

```bash
# Build
docker build -t adas-core:latest .

# Run
docker run --rm adas-core:latest --frames 10

# Or use docker-compose
docker-compose up
```

## 📋 Common Commands

```bash
# Run with debug logging
adas-run --log-level DEBUG --frames 10

# Run with custom FPS in config
adas-run --config config.json

# Run tests with coverage
pytest --cov=src/adas tests/

# Format code
make format

# Lint code
make lint

# Clean build artifacts
make clean
```

## 🎯 Key Configuration Parameters

### Speed Control
```json
"planner": {
  "cruise_speed_mps": 15.0,        // Target cruise speed (~54 km/h)
  "min_follow_distance_m": 12.0,   // Minimum safe following distance
  "time_gap_s": 2.0                // Desired time gap (2-second rule)
}
```

### Safety Limits
```json
"safety": {
  "max_speed_mps": 33.0,           // Maximum speed limit (~120 km/h)
  "max_deceleration_mps2": 8.0,    // Emergency braking limit
  "min_following_distance_m": 2.0  // Absolute minimum distance
}
```

### Controller Tuning
```json
"controller": {
  "kp_speed": 0.15,                // Speed control gain
  "steering_deadband_deg": 0.5     // Ignore small steering inputs
}
```

## 🔍 Troubleshooting

### Module Import Errors
```bash
# Reinstall package
pip install --upgrade adas-core

# Or from source
pip install -e .
```

### Tests Failing
```bash
# Check Python version (need 3.8+)
python3 --version

# Install test dependencies
pip install pytest
```

### Docker Build Issues
```bash
# Clean Docker cache
docker system prune -a

# Rebuild
docker build --no-cache -t adas-core:latest .
```

## 📚 Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Read [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
- Check [README.md](README.md) for full documentation

## 💡 Example Python Usage

```python
from adas.cli import build_pipeline
from adas.models import PerceptionFrame
from adas.runtime import synthetic_frame
import time

# Build pipeline with default config
pipeline, fps = build_pipeline()

# Create synthetic frame
frame_data = synthetic_frame()
frame = PerceptionFrame(
    frame_id=0,
    timestamp_s=time.time(),
    rgb=frame_data,
    width=frame_data["width"],
    height=frame_data["height"],
)

# Process frame (with current speed from vehicle)
plan, command = pipeline.step(frame, current_speed_mps=15.0)

# Use outputs
print(f"Target speed: {plan.target_speed_mps:.1f} m/s")
print(f"Steering: {plan.steering_angle_deg:.1f}°")
print(f"Throttle: {command.throttle:.2f}")
print(f"Brake: {command.brake:.2f}")
print(f"Reason: {plan.reason}")
```

## ⚠️ Important Notes

1. **This is a reference implementation** - Additional safety validation required for production
2. **Mock detectors** - Replace with real TensorRT models for actual deployment
3. **Safety critical** - Perform extensive testing before vehicle integration
4. **Calibration needed** - Adjust camera parameters for your specific setup

---

**Ready to go!** Run `adas-run --frames 10` to verify installation.
