"""OPTIONAL ROS 2 integration.

**This package has never been executed.**  ``rclpy`` is not installed on the
Jetson this project targets (JetPack 5.1.6 ships no ROS 2, and the project's
package rules forbid pip-installing one), so the bridge is written against the
ROS 2 Humble API and reviewed, not tested.

Importing :mod:`adas.ros2` is always safe.  :data:`ROS2_AVAILABLE` says whether
``rclpy`` resolved; the bridge class and the converters import either way, and
:class:`~adas.ros2.bridge.ADASBridgeNode` raises :class:`ImportError` at
construction when ROS 2 is missing rather than degrading into a node that
silently publishes nothing.

Install ROS 2 Humble from apt (``ros-humble-rclpy``, ``ros-humble-vision-msgs``,
``ros-humble-diagnostic-msgs``) and source its setup script; ``pip install
rclpy`` does not produce a working installation.
"""

try:
    import rclpy  # noqa: F401

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

from adas.ros2.bridge import ADASBridgeNode
from adas.ros2.converters import (
    bbox_to_detection_msg,
    control_cmd_to_msg,
    detection_msg_to_bbox,
    ros_image_to_numpy,
)
from adas.ros2.topics import ADASTopics, MESSAGE_TYPES, QoSProfiles

__all__ = [
    "ADASBridgeNode",
    "ADASTopics",
    "MESSAGE_TYPES",
    "QoSProfiles",
    "ROS2_AVAILABLE",
    "bbox_to_detection_msg",
    "control_cmd_to_msg",
    "detection_msg_to_bbox",
    "ros_image_to_numpy",
]
