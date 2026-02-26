# ROS2 Integration Guide

## Overview

The ADAS Core system provides a complete ROS2 integration layer, allowing seamless integration with ROS2-based autonomous driving stacks.

## Features

- **ROS2 Bridge Node**: Wraps ADAS pipeline as a ROS2 node
- **Standard Topics**: Compatible with common ROS2 message types
- **Message Converters**: Bidirectional conversion between ADAS models and ROS2 messages
- **Diagnostics**: Built-in diagnostic publishing
- **QoS Profiles**: Optimized Quality of Service settings

## Installation

### Prerequisites

```bash
# Install ROS2 (Humble recommended)
# Follow: https://docs.ros.org/en/humble/Installation.html

# Install Python dependencies
pip install rclpy
pip install sensor-msgs vision-msgs diagnostic-msgs
```

### Optional Dependencies

```bash
# For advanced features
pip install derived-object-msgs
pip install autoware-auto-msgs
```

## Quick Start

### 1. Run ADAS Bridge Node

```bash
# Start the bridge node
python -m adas.ros2.bridge

# Or with ROS2 run
ros2 run adas adas_bridge
```

### 2. Publish Camera Data

```bash
# Publish test image
ros2 topic pub /camera/image_raw sensor_msgs/Image ...

# Publish vehicle speed
ros2 topic pub /vehicle/speed std_msgs/Float32 "data: 10.0"
```

### 3. Monitor Outputs

```bash
# View control commands
ros2 topic echo /adas/control/throttle_cmd
ros2 topic echo /adas/control/brake_cmd
ros2 topic echo /adas/control/steering_cmd

# View diagnostics
ros2 topic echo /adas/diagnostics
```

## Topic Interface

### Subscribed Topics

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Best Effort | Camera input |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | Best Effort | Camera calibration |
| `/vehicle/speed` | `std_msgs/Float32` | Best Effort | Vehicle speed (m/s) |
| `/vehicle/odometry` | `nav_msgs/Odometry` | Best Effort | Vehicle odometry |

### Published Topics

| Topic | Type | QoS | Description |
|-------|------|-----|-------------|
| `/adas/perception/detections` | `vision_msgs/Detection2DArray` | Default | Object detections |
| `/adas/perception/lane_markers` | `visualization_msgs/MarkerArray` | Default | Lane markers |
| `/adas/tracking/objects` | `derived_object_msgs/ObjectArray` | Default | Tracked objects |
| `/adas/planning/motion_plan` | Custom | Default | Motion plan |
| `/adas/control/throttle_cmd` | `std_msgs/Float32` | Reliable | Throttle command |
| `/adas/control/brake_cmd` | `std_msgs/Float32` | Reliable | Brake command |
| `/adas/control/steering_cmd` | `std_msgs/Float32` | Reliable | Steering command |
| `/adas/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Reliable | System diagnostics |
| `/adas/performance` | `std_msgs/String` | Default | Performance metrics |

## Python API

### Using the Bridge Programmatically

```python
import rclpy
from adas.cli import build_pipeline
from adas.ros2 import ADASBridgeNode

# Initialize ROS2
rclpy.init()

# Build ADAS pipeline
pipeline, _ = build_pipeline()

# Create bridge node
node = ADASBridgeNode(pipeline, node_name="my_adas")

# Spin
rclpy.spin(node)

# Cleanup
rclpy.shutdown()
```

### Message Conversion

```python
from adas.core.models import BoundingBox, ControlCommand
from adas.ros2.converters import (
    bbox_to_detection_msg,
    detection_msg_to_bbox,
    control_cmd_to_msg,
)

# Convert ADAS BoundingBox to ROS2
bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200, confidence=0.9, label="car")
detection_msg = bbox_to_detection_msg(bbox, frame_id="camera")

# Convert ROS2 Detection2D to ADAS BoundingBox
bbox_back = detection_msg_to_bbox(detection_msg)

# Convert control command
cmd = ControlCommand(throttle=0.5, brake=0.0, steering=0.1)
throttle_msg, brake_msg, steering_msg = control_cmd_to_msg(cmd)
```

## Launch Files

Create a launch file for easy startup:

```python
# launch/adas_bridge.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='adas',
            executable='adas_bridge',
            name='adas_bridge',
            output='screen',
            parameters=[{
                'config_file': '/path/to/config.json',
            }]
        ),
    ])
```

Run with:
```bash
ros2 launch adas adas_bridge.launch.py
```

## Configuration

The bridge node respects ADAS configuration files:

```bash
# Run with custom config
python -m adas.ros2.bridge --config /path/to/config.json
```

## Diagnostics

The bridge publishes diagnostic information:

```python
# View diagnostics
ros2 topic echo /adas/diagnostics

# Expected output:
# status:
#   - name: "ADAS Pipeline"
#     level: 0  # OK
#     message: "Operating normally"
#     values:
#       - key: "frame_count"
#         value: "150"
#       - key: "processing_time_ms"
#         value: "32.45"
```

## Performance Monitoring

```bash
# Monitor performance
ros2 topic echo /adas/performance

# Output: frame=150,time=32.45ms,fps=30.8
```

## Troubleshooting

### ROS2 Not Found

```bash
# Check ROS2 installation
ros2 --version

# Source ROS2 setup
source /opt/ros/humble/setup.bash
```

### Message Type Errors

```bash
# Install missing message packages
sudo apt install ros-humble-vision-msgs
sudo apt install ros-humble-diagnostic-msgs
```

### Import Errors

The ROS2 integration gracefully handles missing dependencies:

```python
from adas.ros2 import ROS2_AVAILABLE

if not ROS2_AVAILABLE:
    print("ROS2 not available - install with: pip install rclpy")
```

## Integration with Existing Stacks

### Autoware Integration

The ADAS bridge is compatible with Autoware topics:

```bash
# Remap topics to Autoware names
ros2 run adas adas_bridge \
  --ros-args \
  -r /camera/image_raw:=/sensing/camera/front/image_raw \
  -r /adas/control/throttle_cmd:=/control/command/throttle
```

### Apollo Integration

Similar topic remapping for Apollo:

```bash
ros2 run adas adas_bridge \
  --ros-args \
  -r /camera/image_raw:=/apollo/sensor/camera/front \
  -r /vehicle/speed:=/apollo/canbus/chassis
```

## Best Practices

1. **QoS Configuration**: Use appropriate QoS profiles for your use case
2. **Topic Remapping**: Use ROS2 remapping for integration
3. **Diagnostics**: Monitor `/adas/diagnostics` for system health
4. **Performance**: Check `/adas/performance` for latency
5. **Safety**: Always validate control commands before actuation

## See Also

- [ADAS Architecture](ARCHITECTURE.md)
- [Deployment Guide](../DEPLOYMENT.md)
- [ROS2 Documentation](https://docs.ros.org/)

---

**Status**: Production Ready  
**ROS2 Version**: Humble (recommended), Foxy, Galactic supported
