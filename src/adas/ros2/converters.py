"""Message converters between ADAS models and ROS2 messages.

This module provides conversion functions between ADAS internal
data structures and standard ROS2 message types.
"""

from __future__ import annotations

import numpy as np

from adas.core.models import BoundingBox, ControlCommand, LaneModel

# Type hints for ROS2 messages (avoid hard dependency)
try:
    from sensor_msgs.msg import Image
    from std_msgs.msg import Header, Float32
    from geometry_msgs.msg import Point
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
    from visualization_msgs.msg import Marker, MarkerArray
    ROS2_MSGS_AVAILABLE = True
except ImportError:
    ROS2_MSGS_AVAILABLE = False
    # Define dummy types for type hints
    Image = None
    Header = None
    Detection2D = None
    Detection2DArray = None
    Marker = None
    MarkerArray = None


def bbox_to_detection_msg(bbox: BoundingBox, frame_id: str = "camera", seq: int = 0):
    """Convert BoundingBox to ROS2 Detection2D message.
    
    Args:
        bbox: ADAS bounding box
        frame_id: Frame ID for the header
        seq: Sequence number
        
    Returns:
        Detection2D message
        
    Raises:
        ImportError: If ROS2 messages not available
    """
    if not ROS2_MSGS_AVAILABLE:
        raise ImportError("ROS2 vision_msgs not available. Install with: pip install vision-msgs")
    
    detection = Detection2D()
    
    # Header
    detection.header.frame_id = frame_id
    detection.header.stamp.sec = seq
    
    # Bounding box center and size
    center_x = (bbox.x1 + bbox.x2) / 2.0
    center_y = (bbox.y1 + bbox.y2) / 2.0
    size_x = bbox.x2 - bbox.x1
    size_y = bbox.y2 - bbox.y1
    
    detection.bbox.center.position.x = center_x
    detection.bbox.center.position.y = center_y
    detection.bbox.size_x = size_x
    detection.bbox.size_y = size_y
    
    # Classification
    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = bbox.label
    hypothesis.hypothesis.score = bbox.confidence
    detection.results.append(hypothesis)
    
    return detection


def detection_msg_to_bbox(detection: Detection2D) -> BoundingBox:
    """Convert ROS2 Detection2D message to BoundingBox.
    
    Args:
        detection: ROS2 Detection2D message
        
    Returns:
        ADAS BoundingBox
    """
    # Extract center and size
    center_x = detection.bbox.center.position.x
    center_y = detection.bbox.center.position.y
    size_x = detection.bbox.size_x
    size_y = detection.bbox.size_y
    
    # Convert to corner representation
    x1 = center_x - size_x / 2.0
    y1 = center_y - size_y / 2.0
    x2 = center_x + size_x / 2.0
    y2 = center_y + size_y / 2.0
    
    # Extract classification
    if detection.results:
        label = detection.results[0].hypothesis.class_id
        confidence = detection.results[0].hypothesis.score
    else:
        label = "unknown"
        confidence = 0.0
    
    return BoundingBox(
        x1=x1, y1=y1, x2=x2, y2=y2,
        confidence=confidence,
        label=label
    )


def control_cmd_to_msg(cmd: ControlCommand):
    """Convert ADAS ControlCommand to ROS2 messages.
    
    Args:
        cmd: ADAS control command
        
    Returns:
        Tuple of (throttle_msg, brake_msg, steering_msg)
    """
    if not ROS2_MSGS_AVAILABLE:
        raise ImportError("ROS2 std_msgs not available")
    
    throttle_msg = Float32()
    throttle_msg.data = float(cmd.throttle)
    
    brake_msg = Float32()
    brake_msg.data = float(cmd.brake)
    
    steering_msg = Float32()
    steering_msg.data = float(cmd.steering)
    
    return throttle_msg, brake_msg, steering_msg


def lane_to_marker_array(lane: LaneModel, frame_id: str = "camera") -> MarkerArray:
    """Convert LaneModel to ROS2 MarkerArray for visualization.
    
    Args:
        lane: ADAS lane model
        frame_id: Frame ID
        
    Returns:
        MarkerArray for visualization
    """
    if not ROS2_MSGS_AVAILABLE:
        raise ImportError("ROS2 visualization_msgs not available")
    
    marker_array = MarkerArray()
    
    # Create line marker for lane center
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = "lane_markers"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    
    # Set color (green for lane)
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    
    marker.scale.x = 0.1  # Line width
    
    # Create points along lane (simplified - just center line)
    for y in range(0, 100, 10):
        point = Point()
        point.x = lane.lane_center_px / 100.0  # Normalize
        point.y = float(y)
        point.z = 0.0
        marker.points.append(point)
    
    marker_array.markers.append(marker)
    return marker_array


def numpy_to_ros_image(np_image: np.ndarray, encoding: str = "rgb8") -> Image:
    """Convert NumPy image to ROS2 Image message.
    
    Args:
        np_image: NumPy array (H, W, C)
        encoding: Image encoding (rgb8, bgr8, mono8, etc.)
        
    Returns:
        ROS2 Image message
    """
    if not ROS2_MSGS_AVAILABLE:
        raise ImportError("ROS2 sensor_msgs not available")
    
    image_msg = Image()
    image_msg.height = np_image.shape[0]
    image_msg.width = np_image.shape[1]
    image_msg.encoding = encoding
    image_msg.is_bigendian = 0
    image_msg.step = np_image.shape[1] * np_image.shape[2]
    image_msg.data = np_image.tobytes()
    
    return image_msg


def ros_image_to_numpy(image_msg: Image) -> np.ndarray:
    """Convert ROS2 Image message to NumPy array.
    
    Args:
        image_msg: ROS2 Image message
        
    Returns:
        NumPy array (H, W, C)
    """
    dtype = np.uint8
    np_image = np.frombuffer(image_msg.data, dtype=dtype)
    
    if image_msg.encoding == "rgb8" or image_msg.encoding == "bgr8":
        np_image = np_image.reshape((image_msg.height, image_msg.width, 3))
    elif image_msg.encoding == "mono8":
        np_image = np_image.reshape((image_msg.height, image_msg.width))
    else:
        raise ValueError(f"Unsupported encoding: {image_msg.encoding}")
    
    return np_image
