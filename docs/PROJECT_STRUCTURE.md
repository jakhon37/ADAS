# ADAS Core - Project Structure Guide

## Overview

The ADAS Core project follows a **domain-driven organization** where code is grouped by functional area rather than being flat. This makes the codebase more maintainable, scalable, and easier to navigate.

## Directory Structure

```
src/adas/
├── __init__.py                    # Main package with common exports
├── cli.py                         # Command-line interface entry point
│
├── core/                          # 🔧 Core Infrastructure
│   ├── __init__.py               # Exports all core components
│   ├── config.py                 # Configuration management
│   ├── exceptions.py             # Exception hierarchy
│   ├── logger.py                 # Structured logging
│   ├── metrics.py                # Performance metrics
│   ├── models.py                 # Domain data models
│   └── validation.py             # Input validation utilities
│
├── infer/                         # TensorRT + CUDA runtime (Jetson)
│   ├── cudart.py
│   └── trt_engine.py
│
├── perception/                    # 👁️ Perception
│   ├── __init__.py
│   ├── factory.py                # mock | tensorrt | ufld
│   ├── detection.py              # Mock object detector
│   ├── yolo.py                   # YOLOv5/v8 TensorRT detector
│   ├── lane.py                   # Mock lane estimator
│   └── ufld.py                   # UFLDv2 TensorRT estimator
│
├── tracking/                      # 🎯 Tracking
│   ├── __init__.py
│   └── tracker.py                # Multi-object tracker
│
├── planning/                      # 🧠 Planning
│   ├── __init__.py
│   └── behavior_planner.py       # Motion planning (ACC + LKA)
│
├── control/                       # 🎮 Control
│   ├── __init__.py
│   ├── controller.py             # PID controller
│   └── safety.py                 # Safety monitor
│
└── runtime/                       # ⚙️ Runtime
    ├── __init__.py
    ├── pipeline.py               # Pipeline orchestration
    ├── runner.py                 # Runtime execution
    └── capture.py                # synthetic | video | camera | CSI
```

## Module Descriptions

### 🔧 Core (`adas.core`)

**Purpose**: Fundamental building blocks used across the entire system.

- **`config.py`**: Configuration management with validation
  - Loads JSON configuration files
  - Validates all parameters
  - Provides defaults
  
- **`exceptions.py`**: Domain-specific exception hierarchy
  - `ADASException` (base)
  - `ValidationError`, `SafetyViolation`, etc.
  
- **`logger.py`**: Structured logging utilities
  - Consistent timestamp formatting
  - Performance logging helpers
  - Safety event logging
  
- **`metrics.py`**: Performance metrics collection
  - Frame processing statistics
  - Component timing
  - Safety event counters
  
- **`models.py`**: Core data models
  - `BoundingBox`, `TrackedObject`
  - `MotionPlan`, `ControlCommand`
  - `PerceptionFrame`, `LaneModel`
  
- **`validation.py`**: Input validation functions
  - Validates all external inputs
  - Bounds checking
  - Type validation

### 👁️ Perception (`adas.perception`)

**Purpose**: Sensor data processing and feature extraction.

- **`factory.py`**: Selects mock / TensorRT YOLO / UFLD from config
- **`detection.py`**: Mock detector (default)
- **`yolo.py`**: TensorRT YOLOv5/v8 (lazy import)
- **`lane.py`**: Mock lane estimator (default)
- **`ufld.py`**: TensorRT UFLDv2 (lazy import; engine not built yet)

### 🎯 Tracking (`adas.tracking`)

**Purpose**: Multi-object tracking with persistent IDs.

- **`tracker.py`**: Multi-object tracker
  - Data association (nearest neighbor)
  - Track management (create/update/delete)
  - Distance estimation

### 🧠 Planning (`adas.planning`)

**Purpose**: Motion planning and decision making.

- **`behavior_planner.py`**: Behavior planner
  - Adaptive cruise control (ACC)
  - Lane keeping assist (LKA)
  - Returns `MotionPlan`

### 🎮 Control (`adas.control`)

**Purpose**: Low-level control and safety monitoring.

- **`controller.py`**: PID controller
  - Speed control (throttle/brake)
  - Steering control
  - Stateless design
  
- **`safety.py`**: Safety monitor
  - Validates motion plans
  - Enforces safety limits
  - Command sanitization

### ⚙️ Runtime (`adas.runtime`)

**Purpose**: Pipeline orchestration and execution.

- **`pipeline.py`**: ADAS pipeline
  - Coordinates all components
  - Error handling
  - Safety integration
  
- **`runner.py`**: Runtime execution
  - Synthetic testing
  - Performance logging
  - FPS control

## Import Patterns

### Common Imports

```python
# Import from top-level package (most common)
from adas import BoundingBox, MotionPlan, ADASPipeline

# Import from specific modules
from adas.core.models import TrackedObject
from adas.core.exceptions import SafetyViolation
from adas.tracking import MultiObjectTracker
from adas.planning import BehaviorPlanner
from adas.control import PIDLikeLongitudinalController, SafetyMonitor
from adas.runtime import PipelineRunner
```

### Internal Imports (within package)

```python
# Core modules importing from core
from adas.core.models import BoundingBox
from adas.core.exceptions import ValidationError
from adas.core.logger import setup_logger

# Other modules importing from core
from adas.core.models import MotionPlan
from adas.core.validation import validate_motion_plan
```

## Design Principles

### 1. **Separation of Concerns**
Each directory has a single, well-defined responsibility:
- `core/` - Infrastructure
- `perception/` - Sensing
- `tracking/` - Temporal consistency
- `planning/` - Decision making
- `control/` - Actuation
- `runtime/` - Orchestration

### 2. **Dependency Direction**
```
runtime → planning → tracking → perception
   ↓         ↓         ↓          ↓
control → core (everything depends on core)
```

Core modules have no dependencies on other ADAS modules.

### 3. **Package Exports**
Each `__init__.py` exports the public API of that module:
```python
# adas/tracking/__init__.py
from adas.tracking.tracker import MultiObjectTracker
__all__ = ["MultiObjectTracker"]
```

### 4. **Flat is Better Than Nested**
We use only 2 levels of nesting (package → module → class/function), which is optimal for most projects.

## Adding New Modules

### Adding to Existing Package

1. Create new file in appropriate directory:
   ```bash
   # Example: add path planning
   touch src/adas/planning/path_planner.py
   ```

2. Update the package `__init__.py`:
   ```python
   # adas/planning/__init__.py
   from adas.planning.behavior_planner import BehaviorPlanner
   from adas.planning.path_planner import PathPlanner  # NEW
   
   __all__ = ["BehaviorPlanner", "PathPlanner"]
   ```

### Creating New Package

1. Create directory and files:
   ```bash
   mkdir src/adas/localization
   touch src/adas/localization/__init__.py
   touch src/adas/localization/ekf.py
   ```

2. Implement module:
   ```python
   # adas/localization/ekf.py
   from adas.core.models import ...
   from adas.core.logger import setup_logger
   
   class ExtendedKalmanFilter:
       ...
   ```

3. Export in `__init__.py`:
   ```python
   # adas/localization/__init__.py
   from adas.localization.ekf import ExtendedKalmanFilter
   __all__ = ["ExtendedKalmanFilter"]
   ```

## Testing Structure

Tests mirror the source structure:

```
tests/
├── test_control.py        # Tests for adas.control
├── test_planning.py       # Tests for adas.planning
├── test_tracking.py       # Tests for adas.tracking
├── test_safety.py         # Tests for adas.control.safety
├── test_validation.py     # Tests for adas.core.validation
└── test_pipeline.py       # Integration tests
```

## Benefits of This Structure

✅ **Maintainability**: Easy to find and modify code  
✅ **Scalability**: Can add new modules without cluttering  
✅ **Testability**: Clear boundaries for unit testing  
✅ **Collaboration**: Multiple developers can work on different packages  
✅ **IDE Support**: Better autocomplete and navigation  
✅ **Industry Standard**: Follows Python best practices  

## Migration from Flat Structure

If you have old code using the flat imports:

```python
# Old (flat structure)
from adas.models import BoundingBox
from adas.exceptions import ValidationError
from adas.tracking import MultiObjectTracker

# New (organized structure)
from adas.core.models import BoundingBox
from adas.core.exceptions import ValidationError
from adas.tracking import MultiObjectTracker  # Unchanged

# Or use top-level imports (recommended)
from adas import BoundingBox, ValidationError
from adas.tracking import MultiObjectTracker
```

## File Count Summary

- **Total files**: 22 Python files
- **Core**: 7 files
- **Perception**: 3 files
- **Tracking**: 2 files
- **Planning**: 2 files
- **Control**: 3 files
- **Runtime**: 3 files
- **Root**: 2 files (cli.py, __init__.py)

## Further Reading

- [Python Packaging Guide](https://packaging.python.org/)
- [The Hitchhiker's Guide to Python - Structuring Your Project](https://docs.python-guide.org/writing/structure/)
- [Real Python - Python Application Layouts](https://realpython.com/python-application-layouts/)

---

**Last Updated**: 2026-02-26  
**Version**: 0.1.0
