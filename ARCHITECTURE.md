# ADAS Core - Architecture Documentation

## System Overview

ADAS Core is a production-grade Advanced Driver Assistance System implemented in Python, designed for edge deployment on platforms like NVIDIA Jetson.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Camera Input                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Perception Layer                            │
│  ┌──────────────┐              ┌─────────────────┐          │
│  │   Object     │              │  Lane           │          │
│  │   Detector   │              │  Estimator      │          │
│  └──────────────┘              └─────────────────┘          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Tracking Layer                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │   Multi-Object Tracker (Data Association)            │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Planning Layer                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │   Behavior Planner (ACC + Lane Keeping)              │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Control Layer                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │   PID Controller (Speed + Steering)                  │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Safety Monitor                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │   Safety Checks + Command Sanitization               │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 Vehicle Actuators                            │
│             (Throttle, Brake, Steering)                      │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Perception Layer (`src/adas/perception/`)

#### Object Detector (`detection.py`)
- **Purpose**: Detect vehicles and obstacles in camera frames
- **Current**: Mock detector with fixed geometry
- **Production**: Replace with TensorRT-optimized YOLO/SSD model
- **Output**: List of `BoundingBox` with confidence scores

#### Lane Estimator (`lane.py`)
- **Purpose**: Estimate lane boundaries and center
- **Current**: Fixed lane geometry
- **Production**: Replace with learned segmentation + polynomial fitting
- **Output**: `LaneModel` with lane coefficients and center

### 2. Tracking Layer (`src/adas/tracking.py`)

#### Multi-Object Tracker
- **Algorithm**: Nearest-neighbor data association with track management
- **Features**:
  - Persistent track IDs across frames
  - Track creation, association, and deletion
  - Distance estimation from bounding box height
- **Optimizations**:
  - Uses `math.hypot()` for efficient distance calculation
  - Configurable association threshold
  - Track pruning after N missed frames

### 3. Planning Layer (`src/adas/planning.py`)

#### Behavior Planner
- **Functions**:
  - **Longitudinal**: Adaptive cruise control with safe following distance
  - **Lateral**: Lane centering with proportional control
- **Safety Features**:
  - Time-gap based following distance
  - Smooth speed transitions
  - Steering angle limits
- **Output**: `MotionPlan` with target speed and steering

### 4. Control Layer (`src/adas/control.py`)

#### PID Controller
- **Type**: Proportional controller for speed and steering
- **Features**:
  - Stateless design (thread-safe)
  - Separate throttle/brake logic
  - Steering deadband to reduce jitter
  - Configurable gains and limits
- **Output**: `ControlCommand` with actuator values

### 5. Safety Monitor (`src/adas/safety.py`)

#### Safety Checks
- **Speed Limits**: Maximum speed enforcement
- **Acceleration Limits**: Max accel/decel bounds
- **Steering Limits**: Rate and angle constraints
- **Following Distance**: Minimum safe distance
- **Command Sanitization**: Clamp all outputs to safe ranges

### 6. Pipeline Orchestration (`src/adas/pipeline.py`)

#### ADASPipeline
- **Responsibilities**:
  - Coordinate all components
  - Handle errors gracefully
  - Integrate safety monitor
  - Log pipeline state
- **Error Handling**: Fail-safe on perception errors, continue with empty data

## Data Flow

### Input → Output Pipeline

```python
PerceptionFrame (camera data)
    ↓
List[BoundingBox] (detections)
    ↓
List[TrackedObject] (persistent tracks with IDs)
    ↓
MotionPlan (target speed + steering)
    ↓
ControlCommand (throttle/brake/steering)
    ↓
Vehicle Actuation
```

### Data Models (`src/adas/models.py`)

```python
BoundingBox:      # Detection output
  - x1, y1, x2, y2: Bounding box coordinates
  - confidence: Detection confidence [0, 1]
  - label: Object class

TrackedObject:    # Tracker output
  - track_id: Persistent ID
  - box: BoundingBox
  - distance_m: Estimated distance
  - velocity_mps: Relative velocity

MotionPlan:       # Planner output
  - target_speed_mps: Desired speed
  - steering_angle_deg: Desired steering
  - reason: Decision rationale (for debugging)

ControlCommand:   # Controller output
  - throttle: [0, 1]
  - brake: [0, 1]
  - steering: [-1, 1]
```

## Configuration System (`src/adas/config.py`)

### Hierarchical Configuration
- **DetectorConfig**: Model path, thresholds
- **TrackerConfig**: Association parameters, distance estimation
- **PlannerConfig**: Cruise speed, following distance, time gap
- **ControllerConfig**: Gains, limits, deadbands
- **SafetyConfig**: All safety limits
- **RuntimeConfig**: Top-level system config

### Validation
- All configs validated on construction
- Type checking and range validation
- Clear error messages for invalid values

## Error Handling Strategy

### Exception Hierarchy

```
ADASException (base)
├── ValidationError
├── ConfigurationError
├── SensorError
├── PerceptionError
├── TrackingError
├── PlanningError
├── ControlError
└── SafetyViolation
```

### Error Recovery
1. **Perception Failures**: Continue with empty detections
2. **Tracking Failures**: Skip frame, maintain existing tracks
3. **Planning Failures**: Raise exception, stop execution
4. **Control Failures**: Raise exception, trigger fail-safe
5. **Safety Violations**: Log warning, sanitize command

## Logging and Observability

### Structured Logging (`src/adas/logger.py`)
- Timestamp + module + level + message format
- Configurable log levels
- Performance logging helpers
- Safety event logging

### Metrics (`src/adas/metrics.py`)
- Frame processing rate (FPS)
- Component timing breakdown
- Detection/track statistics
- Safety event counters

## Performance Characteristics

### Current Performance (CPU)
- **Latency**: <1ms per frame (mock implementations)
- **Throughput**: 1000+ FPS
- **Memory**: <100MB

### Production Performance (with real models)
- **Latency**: 30-50ms per frame (TensorRT on Jetson)
- **Throughput**: 20-30 FPS
- **Memory**: 500MB-2GB (model dependent)

## Thread Safety

### Stateless Components
- **Detector**: Stateless (thread-safe)
- **LaneEstimator**: Stateless (thread-safe)
- **Planner**: Stateless (thread-safe)
- **Controller**: Stateless (thread-safe)

### Stateful Components
- **Tracker**: Maintains track state (not thread-safe)
- **Pipeline**: Maintains frame counter (not thread-safe)
- **SafetyMonitor**: Maintains last steering/speed (not thread-safe)

For multi-threaded deployment, use separate instances per thread.

## Extension Points

### Adding New Perception Modules
1. Implement interface matching `ObjectDetector` or `LaneEstimator`
2. Return appropriate data models
3. Handle errors with custom exceptions
4. Register in `cli.py` builder

### Custom Planners
1. Subclass or replace `BehaviorPlanner`
2. Implement `plan()` method returning `MotionPlan`
3. Add validation in `__post_init__`

### Integration with ROS/ROS2
- Wrap `ADASPipeline` in ROS node
- Subscribe to camera topics
- Publish control commands
- Use ROS logging and parameters

## Production Deployment Considerations

### Real-Time Requirements
- Pipeline must run at camera frame rate (20-30 Hz)
- Use real-time OS scheduling if needed
- Monitor for deadline misses

### Safety Certification
- Implement ASIL-D requirements for safety-critical paths
- Add redundancy (dual pipeline, watchdog)
- Extensive testing and validation
- Formal verification of safety monitor

### Hardware Integration
- CAN bus interface for vehicle signals
- Camera calibration and synchronization
- Sensor fusion (radar, lidar)
- Fail-safe on hardware faults

---

For implementation details, see source code documentation and `DEPLOYMENT.md`.
