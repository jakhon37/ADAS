# ADAS Core - Production-Grade Advanced Driver Assistance System

[![CI](https://github.com/jakhon37/ADAS/workflows/CI/badge.svg)](https://github.com/jakhon37/ADAS/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready Advanced Driver Assistance System (ADAS) implementation designed for edge deployment on platforms like NVIDIA Jetson. This system provides adaptive cruise control, lane keeping assistance, and object tracking with comprehensive safety monitoring.

## 🚗 Features

### Core Capabilities
- **Adaptive Cruise Control (ACC)**: Maintains safe following distance with dynamic speed control
- **Lane Keeping Assist (LKA)**: Automatic lane centering with proportional steering control
- **Multi-Object Tracking**: Persistent track IDs with data association
- **Safety Monitor**: Real-time safety constraint enforcement and command sanitization
- **ROS2 Integration**: Complete ROS2 bridge with standard topics and message types
- **Replay Tools**: Record and replay pipeline data for debugging and analysis

### Production Features
- ✅ **Comprehensive Error Handling**: Domain-specific exceptions with graceful degradation
- ✅ **Structured Logging**: Production-grade logging with performance metrics
- ✅ **Configuration Validation**: Type-checked configuration with range validation
- ✅ **Safety Enforcement**: Multi-layer safety checks and bounds enforcement
- ✅ **Stateless Design**: Thread-safe components for concurrent execution
- ✅ **Metrics & Observability**: Performance tracking and health monitoring
- ✅ **Docker Support**: Production-ready containerization
- ✅ **Comprehensive Tests**: Unit tests for all critical components
- ✅ **ROS2 Integration**: Full ROS2 bridge for robotics integration
- ✅ **Debug Tools**: Record/replay functionality for offline analysis

## 🏗️ Architecture

```
Camera → Perception → Tracking → Planning → Control → Safety Monitor → Actuators
```

**Key Components:**
- **Perception**: Object detection + lane estimation
- **Tracking**: Multi-object tracker with persistent IDs
- **Planning**: Behavior planner (ACC + lane keeping)
- **Control**: PID controller for speed and steering
- **Safety**: Multi-layer safety validation and enforcement

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## 📦 Installation

### Quick Start

```bash
# Clone repository
git clone https://github.com/jakhon37/ADAS.git
cd ADAS

# Install package
pip install -e .

# Run synthetic test
adas-run --frames 60
```

### Docker Deployment

```bash
# Build image
docker build -t adas-core:latest .

# Run container
docker-compose up -d

# View logs
docker-compose logs -f
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment guide.

## 🚀 Usage

### Command Line Interface

```bash
# Run with default configuration
adas-run --frames 100

# Run with custom configuration
adas-run --config config.json --frames 100

# Debug mode
adas-run --log-level DEBUG --frames 10
```

### Python API

```python
from adas.pipeline import ADASPipeline
from adas.models import PerceptionFrame
from adas.cli import build_pipeline
import time

# Build pipeline
pipeline, fps = build_pipeline()

# Process frame
frame = PerceptionFrame(
    frame_id=0,
    timestamp_s=time.time(),
    rgb=camera_data,
    width=1280,
    height=720,
)

plan, command = pipeline.step(frame, current_speed_mps=15.0)

print(f"Plan: {plan.target_speed_mps:.1f} m/s, {plan.steering_angle_deg:.1f}°")
print(f"Command: throttle={command.throttle:.2f}, brake={command.brake:.2f}")
```

## ⚙️ Configuration

Configuration is managed via JSON files. See `config.example.json` for template.

### Key Parameters

```json
{
  "planner": {
    "cruise_speed_mps": 15.0,       // ~54 km/h
    "min_follow_distance_m": 12.0,  // Minimum safe distance
    "time_gap_s": 2.0               // Desired time gap
  },
  "safety": {
    "max_speed_mps": 33.0,          // ~120 km/h limit
    "max_deceleration_mps2": 8.0,   // Emergency brake limit
    "min_following_distance_m": 2.0 // Absolute minimum
  }
}
```

All configuration parameters are validated on load with clear error messages.

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_safety.py -v

# Run with coverage
pytest --cov=src/adas tests/
```

## 📊 Performance

**Current (Mock Implementations):**
- Latency: <1ms per frame
- Throughput: 1000+ FPS
- Memory: <100MB

**Production (TensorRT on Jetson Xavier):**
- Latency: 30-50ms per frame
- Throughput: 20-30 FPS
- Memory: 500MB-2GB

## 🛡️ Safety

### Safety Features
- Speed and acceleration limits
- Steering rate and angle constraints
- Minimum following distance enforcement
- Command sanitization and clamping
- Graceful degradation on sensor failures

### Safety Certification Notes

⚠️ **IMPORTANT**: This is a reference implementation for development and testing.

For production automotive deployment:
1. Implement ISO 26262 functional safety requirements
2. Add redundancy and fail-safe mechanisms
3. Perform extensive validation (HIL/SIL testing)
4. Integrate with vehicle safety systems
5. Obtain necessary certifications

## 🔧 Development

### Setup Development Environment

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run linter
make lint

# Format code
make format

# Clean build artifacts
make clean
```

### Code Quality Standards
- **Type Hints**: All functions fully type-annotated
- **Docstrings**: Google-style docstrings
- **Error Handling**: Comprehensive exception handling
- **Logging**: Structured logging throughout
- **Testing**: >80% code coverage target

## 📚 Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and design
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment guide
- [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) - Package organization guide
- [ROS2_INTEGRATION.md](docs/ROS2_INTEGRATION.md) - ROS2 integration guide
- [TOOLS_GUIDE.md](docs/TOOLS_GUIDE.md) - Debugging and replay tools
- [config.example.json](config.example.json) - Configuration template

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

Built for edge deployment on:
- NVIDIA Jetson (Nano, Xavier, Orin)
- Similar ARM-based edge compute platforms

Designed to integrate with:
- TensorRT for GPU-accelerated inference
- DeepStream for video analytics
- ROS/ROS2 for robotics integration

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- See documentation in this repository
- Check [DEPLOYMENT.md](DEPLOYMENT.md) for troubleshooting

---

**Status**: Production-ready reference implementation  
**Version**: 0.1.0  
**Last Updated**: 2026-02-26
