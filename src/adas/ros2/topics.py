"""ROS2 topic definitions for ADAS system.

This module defines the standard topic names and QoS profiles
used by the ADAS ROS2 bridge.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ADASTopics:
    """Standard ROS2 topic names for ADAS system."""
    
    # Input topics
    CAMERA_IMAGE = "/camera/image_raw"
    CAMERA_INFO = "/camera/camera_info"
    VEHICLE_ODOMETRY = "/vehicle/odometry"
    VEHICLE_SPEED = "/vehicle/speed"
    
    # Perception output topics
    DETECTIONS = "/adas/perception/detections"
    LANE_MARKERS = "/adas/perception/lane_markers"
    
    # Tracking output topics
    TRACKED_OBJECTS = "/adas/tracking/objects"
    
    # Planning output topics
    MOTION_PLAN = "/adas/planning/motion_plan"
    
    # Control output topics
    CONTROL_COMMAND = "/adas/control/command"
    STEERING_CMD = "/vehicle/steering_cmd"
    THROTTLE_CMD = "/vehicle/throttle_cmd"
    BRAKE_CMD = "/vehicle/brake_cmd"
    
    # Diagnostic topics
    DIAGNOSTICS = "/adas/diagnostics"
    PERFORMANCE = "/adas/performance"
    SAFETY_EVENTS = "/adas/safety/events"


@dataclass(frozen=True)
class QoSProfiles:
    """QoS profile configurations for different topic types."""
    
    # Sensor data: Best effort, volatile (high throughput, allow drops)
    SENSOR_DATA = {
        "reliability": "best_effort",
        "durability": "volatile",
        "history": "keep_last",
        "depth": 10,
    }
    
    # Control commands: Reliable, volatile (critical, no drops)
    CONTROL = {
        "reliability": "reliable",
        "durability": "volatile",
        "history": "keep_last",
        "depth": 5,
    }
    
    # Diagnostics: Reliable, transient local (persistent)
    DIAGNOSTICS = {
        "reliability": "reliable",
        "durability": "transient_local",
        "history": "keep_last",
        "depth": 50,
    }


# Message type hints (for documentation)
MESSAGE_TYPES = {
    ADASTopics.CAMERA_IMAGE: "sensor_msgs/Image",
    ADASTopics.CAMERA_INFO: "sensor_msgs/CameraInfo",
    ADASTopics.VEHICLE_ODOMETRY: "nav_msgs/Odometry",
    ADASTopics.VEHICLE_SPEED: "std_msgs/Float32",
    ADASTopics.DETECTIONS: "vision_msgs/Detection2DArray",
    ADASTopics.LANE_MARKERS: "visualization_msgs/MarkerArray",
    ADASTopics.TRACKED_OBJECTS: "derived_object_msgs/ObjectArray",
    ADASTopics.MOTION_PLAN: "autoware_auto_msgs/Trajectory",
    ADASTopics.CONTROL_COMMAND: "autoware_auto_msgs/VehicleControlCommand",
    ADASTopics.STEERING_CMD: "std_msgs/Float32",
    ADASTopics.THROTTLE_CMD: "std_msgs/Float32",
    ADASTopics.BRAKE_CMD: "std_msgs/Float32",
    ADASTopics.DIAGNOSTICS: "diagnostic_msgs/DiagnosticArray",
    ADASTopics.PERFORMANCE: "std_msgs/String",
    ADASTopics.SAFETY_EVENTS: "std_msgs/String",
}
