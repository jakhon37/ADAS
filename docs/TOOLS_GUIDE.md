# Debugging & Replay Tools Guide

## Overview

ADAS Core provides comprehensive tools for debugging, replay, and analysis:

- **DataRecorder**: Record pipeline execution for offline analysis
- **DataReplayer**: Replay recorded data for debugging
- **Performance Profiling**: Analyze timing and performance

## Recording Data

### Quick Start

```python
from adas.cli import build_pipeline
from adas.tools import DataRecorder, RecordingConfig

# Create pipeline
pipeline, _ = build_pipeline()

# Configure recorder
config = RecordingConfig(
    output_dir="recordings",
    recording_name="test_run_001",
    record_images=True,  # Include image data
    record_detections=True,
    record_plans=True,
    record_commands=True,
)

# Create recorder
recorder = DataRecorder(config)

# Start recording
recorder.start_recording()

# Run pipeline (data is recorded automatically)
for frame in frames:
    plan, command = pipeline.step(frame, current_speed_mps=15.0)
    recorder.record_frame(frame, plan, command)

# Stop and save
recorder.stop_recording()
```

### Recording with Pipeline Wrapper

```python
from adas.tools.recorder import RecordingPipeline

# Wrap pipeline with recorder
recording_pipeline = RecordingPipeline(pipeline, recorder)

# Use normally - automatically records
recorder.start_recording()
for frame in frames:
    plan, command = recording_pipeline.step(frame, 15.0)
recorder.stop_recording()
```

### Configuration Options

```python
config = RecordingConfig(
    output_dir="recordings",        # Output directory
    recording_name=None,             # Auto-generate timestamp name
    record_images=False,             # Images can be large
    record_detections=True,          # Record detection outputs
    record_plans=True,               # Record motion plans
    record_commands=True,            # Record control commands
    compression=True,                # Compress data (future)
    format="json",                   # 'json' or 'pickle'
)
```

## Replaying Data

### Basic Replay

```python
from adas.tools import DataReplayer, ReplayConfig

# Configure replay
config = ReplayConfig(
    recording_dir="recordings/test_run_001",
    playback_speed=1.0,  # 1.0 = real-time, 0.0 = fast
    loop=False,          # Loop replay
    start_frame=0,       # Start from frame N
    end_frame=None,      # End at frame N (None = all)
)

# Create replayer
replayer = DataReplayer(config)

# Replay frames
for frame_data in replayer.replay_iterator():
    print(f"Frame {frame_data['frame_id']}")
    print(f"  Detections: {len(frame_data.get('detections', []))}")
    print(f"  Plan: {frame_data.get('plan', {}).get('reason', 'N/A')}")
```

### Replay Through Pipeline

```python
from adas.tools.replayer import replay_with_pipeline

# Run recorded data through pipeline for comparison
for recorded, (new_plan, new_cmd) in replay_with_pipeline(replayer, pipeline):
    # Compare recorded vs new results
    recorded_plan = recorded.get('plan', {})
    
    print(f"Frame {recorded['frame_id']}:")
    print(f"  Recorded speed: {recorded_plan.get('target_speed_mps')}")
    print(f"  New speed: {new_plan.target_speed_mps}")
    
    # Analyze differences
    if abs(recorded_plan.get('target_speed_mps', 0) - new_plan.target_speed_mps) > 0.1:
        print("  ⚠️ Speed mismatch!")
```

### Accessing Specific Frames

```python
# Get specific frame
frame_50 = replayer.get_frame(50)

# Reconstruct PerceptionFrame
perception_frame = replayer.get_perception_frame(50)

# Get frame summary
summary = replayer.get_frame_summary(50)
print(summary)
```

### Export Summary

```python
# Export replay summary to file
replayer.export_summary("analysis/summary.txt")
```

## Use Cases

### 1. Regression Testing

```python
# Record golden run
recorder = DataRecorder(RecordingConfig(recording_name="golden_v1.0"))
recorder.start_recording()
# ... run pipeline ...
recorder.stop_recording()

# Later: replay and compare
replayer = DataReplayer(ReplayConfig(recording_dir="recordings/golden_v1.0"))
for recorded, (plan, cmd) in replay_with_pipeline(replayer, new_pipeline):
    # Assert outputs match
    assert abs(recorded['plan']['target_speed_mps'] - plan.target_speed_mps) < 0.01
```

### 2. Performance Analysis

```python
# Record with timing
recorder = DataRecorder(RecordingConfig(recording_name="perf_test"))
recorder.start_recording()

for frame in frames:
    start = time.time()
    plan, cmd = pipeline.step(frame, 15.0)
    elapsed_ms = (time.time() - start) * 1000
    
    recorder.record_frame(frame, plan, cmd, timing_info={"total_ms": elapsed_ms})

recorder.stop_recording()

# Analyze timing
replayer = DataReplayer(ReplayConfig(recording_dir="recordings/perf_test"))
timings = [frame.get('timing', {}).get('total_ms', 0) for frame in replayer.frames]
print(f"Avg: {sum(timings)/len(timings):.2f}ms")
print(f"Max: {max(timings):.2f}ms")
print(f"Min: {min(timings):.2f}ms")
```

### 3. Debugging Specific Issues

```python
# Record problematic scenario
recorder = DataRecorder(RecordingConfig(recording_name="bug_case_123"))
# ... record issue ...

# Replay and inspect
replayer = DataReplayer(ReplayConfig(recording_dir="recordings/bug_case_123"))

# Find frame where issue occurs
for idx, frame in enumerate(replayer.frames):
    if frame.get('plan', {}).get('target_speed_mps', 0) < 0:
        print(f"Issue at frame {idx}:")
        print(replayer.get_frame_summary(idx))
```

### 4. A/B Testing

```python
# Record once
recorder = DataRecorder()
recorder.start_recording()
# ... record run ...
recorder.stop_recording()

# Test multiple pipeline versions
replayer = DataReplayer(ReplayConfig(recording_dir=recorder.recording_dir))

results_v1 = []
results_v2 = []

for recorded, (plan_v1, _) in replay_with_pipeline(replayer, pipeline_v1):
    results_v1.append(plan_v1.target_speed_mps)

replayer.current_frame_idx = 0  # Reset

for recorded, (plan_v2, _) in replay_with_pipeline(replayer, pipeline_v2):
    results_v2.append(plan_v2.target_speed_mps)

# Compare
differences = [abs(v1 - v2) for v1, v2 in zip(results_v1, results_v2)]
print(f"Avg difference: {sum(differences)/len(differences):.3f} m/s")
```

## Recording Directory Structure

```
recordings/
└── test_run_001/
    ├── metadata.json      # Recording metadata
    ├── frames.json        # Frame data (or frames.pkl)
    └── summary.txt        # Human-readable summary
```

### metadata.json
```json
{
  "recording_name": "test_run_001",
  "start_time": 1709020800.123,
  "end_time": 1709020900.456,
  "total_frames": 1500,
  "config": {
    "record_images": true,
    "format": "json"
  }
}
```

### frames.json
```json
[
  {
    "frame_id": 0,
    "timestamp": 1709020800.123,
    "width": 1280,
    "height": 720,
    "detections": [...],
    "lane": {...},
    "plan": {...},
    "command": {...},
    "timing": {"total_ms": 32.45}
  },
  ...
]
```

## CLI Tools

### Record from CLI

```bash
# Run with recording enabled
python -m adas.cli --frames 100 --record --record-name my_test
```

### Replay from CLI

```bash
# Replay recorded data
python -m adas.tools.replayer recordings/my_test
```

## Best Practices

1. **Storage**: Disable `record_images=True` for long recordings (images are large)
2. **Format**: Use JSON for human-readable, pickle for binary efficiency
3. **Naming**: Use descriptive recording names (e.g., `highway_merge_case_001`)
4. **Cleanup**: Regularly clean old recordings
5. **Versioning**: Include version info in recording names

## Troubleshooting

### Large File Sizes

```python
# Disable image recording
config = RecordingConfig(record_images=False)

# Or use pickle format (more compact)
config = RecordingConfig(format="pickle")
```

### Memory Issues

```python
# Replay in chunks instead of loading all
for frame in replayer.replay_iterator():
    # Process frame-by-frame
    process(frame)
    # Memory is freed after each iteration
```

## See Also

- [Architecture Guide](ARCHITECTURE.md)
- [Testing Guide](../tests/README.md)

---

**Status**: Production Ready  
**Version**: 0.1.0
