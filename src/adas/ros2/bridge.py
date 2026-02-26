"""ROS2 bridge node for ADAS pipeline.

This module provides a ROS2 node that wraps the ADAS pipeline,
subscribing to camera and vehicle topics, and publishing
perception, planning, and control outputs.
"""

from __future__ import annotations

import time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32, String
    from vision_msgs.msg import Detection2DArray
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object  # Dummy base class

from adas.core.logger import setup_logger
from adas.core.models import PerceptionFrame
from adas.runtime import ADASPipeline
from adas.ros2.topics import ADASTopics
from adas.ros2.converters import (
    bbox_to_detection_msg,
    control_cmd_to_msg,
    ros_image_to_numpy,
)

logger = setup_logger(__name__)


class ADASBridgeNode(Node):
    """ROS2 node that bridges ADAS pipeline with ROS2 topics.
    
    This node:
    - Subscribes to camera and vehicle data
    - Runs ADAS pipeline on incoming frames
    - Publishes perception, planning, and control outputs
    - Publishes diagnostics and performance metrics
    """
    
    def __init__(self, pipeline: ADASPipeline, node_name: str = "adas_bridge"):
        """Initialize ADAS bridge node.
        
        Args:
            pipeline: ADAS pipeline instance
            node_name: ROS2 node name
        """
        if not ROS2_AVAILABLE:
            raise ImportError("ROS2 not available. Install with: pip install rclpy")
        
        super().__init__(node_name)
        
        self.pipeline = pipeline
        self.frame_count = 0
        self.current_speed_mps = 0.0
        
        # Create QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        control_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=5
        )
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            ADASTopics.CAMERA_IMAGE,
            self.image_callback,
            sensor_qos
        )
        
        self.speed_sub = self.create_subscription(
            Float32,
            ADASTopics.VEHICLE_SPEED,
            self.speed_callback,
            sensor_qos
        )
        
        # Publishers - Perception
        self.detections_pub = self.create_publisher(
            Detection2DArray,
            ADASTopics.DETECTIONS,
            10
        )
        
        # Publishers - Control
        self.throttle_pub = self.create_publisher(
            Float32,
            ADASTopics.THROTTLE_CMD,
            control_qos
        )
        
        self.brake_pub = self.create_publisher(
            Float32,
            ADASTopics.BRAKE_CMD,
            control_qos
        )
        
        self.steering_pub = self.create_publisher(
            Float32,
            ADASTopics.STEERING_CMD,
            control_qos
        )
        
        # Publishers - Diagnostics
        self.diagnostics_pub = self.create_publisher(
            DiagnosticArray,
            ADASTopics.DIAGNOSTICS,
            10
        )
        
        self.performance_pub = self.create_publisher(
            String,
            ADASTopics.PERFORMANCE,
            10
        )
        
        self.safety_pub = self.create_publisher(
            String,
            ADASTopics.SAFETY_EVENTS,
            10
        )
        
        logger.info(f"ADAS Bridge Node '{node_name}' initialized")
        logger.info(f"Subscribed to: {ADASTopics.CAMERA_IMAGE}, {ADASTopics.VEHICLE_SPEED}")
        logger.info(f"Publishing to: {ADASTopics.THROTTLE_CMD}, {ADASTopics.BRAKE_CMD}, {ADASTopics.STEERING_CMD}")
    
    def speed_callback(self, msg: Float32) -> None:
        """Handle vehicle speed updates.
        
        Args:
            msg: Speed message in m/s
        """
        self.current_speed_mps = msg.data
    
    def image_callback(self, msg: Image) -> None:
        """Handle incoming camera images.
        
        Args:
            msg: ROS2 Image message
        """
        start_time = time.time()
        
        try:
            # Convert ROS image to numpy
            np_image = ros_image_to_numpy(msg)
            
            # Create perception frame
            frame = PerceptionFrame(
                frame_id=self.frame_count,
                timestamp_s=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                rgb={"width": msg.width, "height": msg.height, "data": np_image},
                width=msg.width,
                height=msg.height,
            )
            
            # Run ADAS pipeline
            plan, command = self.pipeline.step(frame, current_speed_mps=self.current_speed_mps)
            
            # Publish detections
            if frame.detections:
                self._publish_detections(frame.detections, msg.header.frame_id)
            
            # Publish control commands
            self._publish_control(command)
            
            # Publish diagnostics
            elapsed_ms = (time.time() - start_time) * 1000.0
            self._publish_diagnostics(elapsed_ms, frame, plan)
            
            self.frame_count += 1
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            self._publish_error(str(e))
    
    def _publish_detections(self, detections, frame_id: str) -> None:
        """Publish detection results.
        
        Args:
            detections: List of BoundingBox
            frame_id: Frame ID for header
        """
        detection_array = Detection2DArray()
        detection_array.header.frame_id = frame_id
        detection_array.header.stamp = self.get_clock().now().to_msg()
        
        for bbox in detections:
            detection_msg = bbox_to_detection_msg(bbox, frame_id, self.frame_count)
            detection_array.detections.append(detection_msg)
        
        self.detections_pub.publish(detection_array)
    
    def _publish_control(self, command) -> None:
        """Publish control commands.
        
        Args:
            command: ControlCommand instance
        """
        throttle_msg, brake_msg, steering_msg = control_cmd_to_msg(command)
        
        self.throttle_pub.publish(throttle_msg)
        self.brake_pub.publish(brake_msg)
        self.steering_pub.publish(steering_msg)
    
    def _publish_diagnostics(self, elapsed_ms: float, frame, plan) -> None:
        """Publish diagnostics information.
        
        Args:
            elapsed_ms: Processing time in milliseconds
            frame: PerceptionFrame
            plan: MotionPlan
        """
        # Diagnostics array
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()
        
        # Pipeline status
        status = DiagnosticStatus()
        status.name = "ADAS Pipeline"
        status.level = DiagnosticStatus.OK
        status.message = "Operating normally"
        
        status.values.append(KeyValue(key="frame_count", value=str(self.frame_count)))
        status.values.append(KeyValue(key="processing_time_ms", value=f"{elapsed_ms:.2f}"))
        status.values.append(KeyValue(key="detections", value=str(len(frame.detections))))
        status.values.append(KeyValue(key="has_lane", value=str(frame.lane is not None)))
        status.values.append(KeyValue(key="target_speed_mps", value=f"{plan.target_speed_mps:.2f}"))
        status.values.append(KeyValue(key="plan_reason", value=plan.reason))
        
        diag_array.status.append(status)
        self.diagnostics_pub.publish(diag_array)
        
        # Performance metrics
        perf_msg = String()
        perf_msg.data = f"frame={self.frame_count},time={elapsed_ms:.2f}ms,fps={1000.0/elapsed_ms:.1f}"
        self.performance_pub.publish(perf_msg)
    
    def _publish_error(self, error_msg: str) -> None:
        """Publish error diagnostics.
        
        Args:
            error_msg: Error message
        """
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()
        
        status = DiagnosticStatus()
        status.name = "ADAS Pipeline"
        status.level = DiagnosticStatus.ERROR
        status.message = error_msg
        
        diag_array.status.append(status)
        self.diagnostics_pub.publish(diag_array)


def main(args=None):
    """Main entry point for ROS2 node."""
    if not ROS2_AVAILABLE:
        print("ERROR: ROS2 not available. Install with: pip install rclpy")
        return
    
    rclpy.init(args=args)
    
    try:
        # Build ADAS pipeline
        from adas.cli import build_pipeline
        pipeline, _ = build_pipeline()
        
        # Create and spin node
        node = ADASBridgeNode(pipeline)
        
        logger.info("ADAS Bridge Node started. Spinning...")
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
