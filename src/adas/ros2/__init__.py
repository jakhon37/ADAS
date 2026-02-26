"""ROS2 integration layer for ADAS system.

This package provides ROS2 integration including:
- Bridge node for ADAS pipeline
- Topic definitions and message converters
- Launch files and configuration
"""

# Note: ROS2 is an optional dependency
try:
    import rclpy  # noqa: F401
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

__all__ = ["ROS2_AVAILABLE"]

if ROS2_AVAILABLE:
    from adas.ros2.bridge import ADASBridgeNode  # noqa: F401
    from adas.ros2.converters import (  # noqa: F401
        bbox_to_detection_msg,
        detection_msg_to_bbox,
        control_cmd_to_msg,
    )
    
    __all__.extend([
        "ADASBridgeNode",
        "bbox_to_detection_msg",
        "detection_msg_to_bbox",
        "control_cmd_to_msg",
    ])
