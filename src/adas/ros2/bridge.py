"""ROS 2 bridge node for the ADAS pipeline.

**Status on this board: NEVER EXECUTED.**  ``rclpy`` is not installed on this
Jetson (JetPack 5.1.6 ships no ROS 2, and installing one is outside the project's
package rules), so nothing in this module has run.  It is written against the
ROS 2 Humble API, it imports and unit-tests without ``rclpy``, and its
construction raises a clear error rather than half-working.  Treat the first real
run as a bring-up, not a regression.

Four defects in the previous version were fixed here, all of which fail *open*
in a vehicle:

1. **Fail-open ego speed.**  ``speed_callback`` assigned ``msg.data`` with no
   finiteness, range or staleness check, and ``current_speed_mps`` started at
   ``0.0``.  A dead ``/vehicle/speed`` therefore made the planner see a stopped
   vehicle, target cruise and command full throttle forever.  Speed is now
   wrapped in an :class:`~adas.core.models.EgoState` that is ``valid=False``
   until a plausible message arrives and invalid again after
   ``speed_timeout_s``; the arbiter already forces a minimum-risk manoeuvre on an
   invalid ego state.
2. **A latched command on the exception path.**  A failed frame published
   nothing, leaving whatever throttle the actuators last received applied.  The
   bridge now publishes
   :meth:`~adas.runtime.pipeline.ADASPipeline.failsafe_command` instead.
3. **Hard-coded healthy diagnostics.**  ``DiagnosticStatus.OK`` /
   ``"Operating normally"`` was published on every frame regardless of what the
   arbiter had just decided.  The level and message now come from
   ``pipeline.last_arbitration``.
4. **A dict where the detector expects an image.**  ``rgb`` was set to
   ``{"width", "height", "data"}``, which a TensorRT backend cannot consume
   (ADAS-OPS-06).  The decoded array is passed directly.

Topics are defined in :mod:`adas.ros2.topics`.
"""

from __future__ import annotations

import time
from typing import Any, Optional

try:  # pragma: no cover - exercised only where ROS 2 is installed
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32, String
    from vision_msgs.msg import Detection2DArray

    ROS2_AVAILABLE = True
except ImportError:  # pragma: no cover - the path taken on this board
    rclpy = None  # type: ignore[assignment]
    ROS2_AVAILABLE = False

    class Node:  # type: ignore[no-redef]
        """Stand-in base class so this module imports without ROS 2.

        Only :class:`ADASBridgeNode` inherits from it, and that class refuses to
        construct when ``ROS2_AVAILABLE`` is False, so no stub method is ever
        reached.
        """

    DiagnosticArray = DiagnosticStatus = KeyValue = None  # type: ignore[assignment]
    QoSProfile = ReliabilityPolicy = DurabilityPolicy = None  # type: ignore[assignment]
    Image = Float32 = String = Detection2DArray = None  # type: ignore[assignment]

from adas.core.logger import setup_logger
from adas.core.models import ControlCommand, EgoState, PerceptionFrame, SafetyState
from adas.ros2.converters import (
    bbox_to_detection_msg,
    control_cmd_to_msg,
    ros_image_to_numpy,
)
from adas.ros2.topics import ADASTopics
from adas.runtime import ADASPipeline

logger = setup_logger(__name__)

#: A speed message older than this makes the ego state invalid.  Three frame
#: periods at 20 Hz: long enough to ride out one dropped message, short enough
#: that a dead bus is noticed within 150 ms.
DEFAULT_SPEED_TIMEOUT_S = 0.15

#: Speeds outside this band are rejected as implausible rather than used.
MAX_PLAUSIBLE_SPEED_MPS = 90.0

_STATE_TO_DIAGNOSTIC = {
    SafetyState.NOMINAL: ("OK", "Operating normally"),
    SafetyState.LIMITED: ("WARN", "Degraded: the arbiter is limiting authority"),
    SafetyState.MIN_RISK_MANEUVER: ("ERROR", "Minimum-risk manoeuvre in progress"),
    SafetyState.DISENGAGE: ("STALE", "Disengaged and latched; reset required"),
}


class ADASBridgeNode(Node):
    """Wraps an :class:`ADASPipeline` behind ROS 2 topics.

    Args:
        pipeline: the pipeline to drive.
        node_name: ROS 2 node name.
        speed_timeout_s: age above which ``/vehicle/speed`` is treated as dead.

    Raises:
        ImportError: ``rclpy`` is not installed.  The bridge does not degrade to
            a no-op node: a vehicle integration that silently does nothing is
            worse than one that refuses to start.
    """

    def __init__(
        self,
        pipeline: ADASPipeline,
        node_name: str = "adas_bridge",
        speed_timeout_s: float = DEFAULT_SPEED_TIMEOUT_S,
    ) -> None:
        if not ROS2_AVAILABLE:
            raise ImportError(
                "ROS 2 (rclpy) is not available. It is not installed on this Jetson and "
                "cannot be pip-installed here; source a ROS 2 Humble underlay first. "
                "The bridge has never been executed on this board -- see the module "
                "docstring and docs/JETSON.md."
            )

        super().__init__(node_name)

        self.pipeline = pipeline
        self.frame_count = 0
        self.speed_timeout_s = float(speed_timeout_s)

        self._speed_mps: float = 0.0
        self._speed_stamp_s: Optional[float] = None
        self._speed_rejects = 0
        self._last_frame_mono: Optional[float] = None
        self._logged_speed_gap = False

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        control_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=5,
        )

        self.image_sub = self.create_subscription(
            Image, ADASTopics.CAMERA_IMAGE, self.image_callback, sensor_qos
        )
        self.speed_sub = self.create_subscription(
            Float32, ADASTopics.VEHICLE_SPEED, self.speed_callback, sensor_qos
        )

        self.detections_pub = self.create_publisher(Detection2DArray, ADASTopics.DETECTIONS, 10)
        self.throttle_pub = self.create_publisher(Float32, ADASTopics.THROTTLE_CMD, control_qos)
        self.brake_pub = self.create_publisher(Float32, ADASTopics.BRAKE_CMD, control_qos)
        self.steering_pub = self.create_publisher(Float32, ADASTopics.STEERING_CMD, control_qos)
        self.diagnostics_pub = self.create_publisher(DiagnosticArray, ADASTopics.DIAGNOSTICS, 10)
        self.performance_pub = self.create_publisher(String, ADASTopics.PERFORMANCE, 10)
        self.safety_pub = self.create_publisher(String, ADASTopics.SAFETY_EVENTS, 10)

        logger.info("ADAS bridge node %r initialised", node_name)
        logger.warning(
            "The ROS 2 bridge has never been executed on this hardware. Verify the "
            "actuator topics and the speed source before enabling any output."
        )

    # ---------------------------------------------------------------- callbacks

    def speed_callback(self, msg: Any) -> None:
        """Accept a plausible ``/vehicle/speed`` sample and stamp it.

        An absent, non-finite, negative or implausibly large value is REJECTED,
        not clamped: a clamped bad speed is indistinguishable from a good one.
        The rejection leaves the previous stamp untouched, so the state ages out
        and becomes invalid on its own.
        """
        value = getattr(msg, "data", None)
        try:
            speed = float(value)
        except (TypeError, ValueError):
            self._speed_rejects += 1
            return
        if not (speed == speed) or speed in (float("inf"), float("-inf")):  # NaN / inf
            self._speed_rejects += 1
            return
        if speed < 0.0 or speed > MAX_PLAUSIBLE_SPEED_MPS:
            self._speed_rejects += 1
            logger.warning("Rejecting implausible vehicle speed %.2f m/s", speed)
            return
        self._speed_mps = speed
        self._speed_stamp_s = time.monotonic()

    def ego_state(self, timestamp_s: float) -> EgoState:
        """Current ego state, invalid until a fresh plausible speed exists."""
        stamp = self._speed_stamp_s
        if stamp is None:
            return EgoState(speed_mps=0.0, valid=False, timestamp_s=timestamp_s)
        age = time.monotonic() - stamp
        if age > self.speed_timeout_s:
            if not self._logged_speed_gap:
                logger.error(
                    "No vehicle speed for %.3f s (> %.3f s): ego state is now INVALID and "
                    "the arbiter will command a minimum-risk manoeuvre.",
                    age,
                    self.speed_timeout_s,
                )
                self._logged_speed_gap = True
            return EgoState(speed_mps=0.0, valid=False, timestamp_s=timestamp_s)
        self._logged_speed_gap = False
        return EgoState(speed_mps=self._speed_mps, valid=True, timestamp_s=timestamp_s)

    def image_callback(self, msg: Any) -> None:
        """Run one pipeline step and publish the ARBITRATED command."""
        started = time.monotonic()
        dt_s = 0.05 if self._last_frame_mono is None else max(1e-3, started - self._last_frame_mono)
        self._last_frame_mono = started

        frame: Optional[PerceptionFrame] = None
        plan = None
        try:
            image = ros_image_to_numpy(msg)
            frame = PerceptionFrame(
                frame_id=self.frame_count,
                timestamp_s=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                rgb=image,
                width=int(msg.width),
                height=int(msg.height),
            )
            plan, command = self.pipeline.step(
                frame, dt_s=dt_s, ego=self.ego_state(started)
            )
            if frame.detections:
                self._publish_detections(frame.detections, msg.header.frame_id)
        except Exception as exc:  # noqa: BLE001 - a failed frame must still actuate
            command = self.pipeline.failsafe_command(dt_s=dt_s)
            logger.error("Frame %s failed: %s; publishing fail-safe", self.frame_count, exc,
                         exc_info=True)
            self._publish_error(str(exc))

        self._publish_control(command)
        self._publish_diagnostics((time.monotonic() - started) * 1000.0, frame, plan)
        self.frame_count += 1

    # --------------------------------------------------------------- publishers

    def _publish_detections(self, detections: Any, frame_id: str) -> None:
        array = Detection2DArray()
        array.header.frame_id = frame_id
        array.header.stamp = self.get_clock().now().to_msg()
        for bbox in detections:
            array.detections.append(bbox_to_detection_msg(bbox, frame_id, self.frame_count))
        self.detections_pub.publish(array)

    def _publish_control(self, command: ControlCommand) -> None:
        throttle_msg, brake_msg, steering_msg = control_cmd_to_msg(command)
        self.throttle_pub.publish(throttle_msg)
        self.brake_pub.publish(brake_msg)
        self.steering_pub.publish(steering_msg)

    def _publish_diagnostics(self, elapsed_ms: float, frame: Any, plan: Any) -> None:
        """Publish the arbiter's ACTUAL state, not a hard-coded OK."""
        arbitration = self.pipeline.last_arbitration
        state = arbitration.state if arbitration is not None else SafetyState.NOMINAL
        level_name, message = _STATE_TO_DIAGNOSTIC.get(state, ("ERROR", "Unknown safety state"))
        if arbitration is not None and arbitration.violations:
            message = "%s: %s" % (message, ", ".join(arbitration.violations))

        status = DiagnosticStatus()
        status.name = "ADAS Pipeline"
        status.level = getattr(DiagnosticStatus, level_name, DiagnosticStatus.ERROR)
        status.message = message
        status.values.append(KeyValue(key="frame_count", value=str(self.frame_count)))
        status.values.append(KeyValue(key="processing_time_ms", value="%.2f" % elapsed_ms))
        status.values.append(KeyValue(key="safety_state", value=state.value))
        status.values.append(KeyValue(key="speed_rejects", value=str(self._speed_rejects)))
        ego = self.pipeline.last_ego
        status.values.append(
            KeyValue(key="ego_speed_valid", value=str(bool(ego is not None and ego.valid)))
        )
        if frame is not None:
            status.values.append(KeyValue(key="detections", value=str(len(frame.detections))))
            status.values.append(KeyValue(key="has_lane", value=str(frame.lane is not None)))
            if frame.status is not None:
                status.values.append(KeyValue(key="perception_ok", value=str(frame.status.ok)))
        if plan is not None:
            status.values.append(
                KeyValue(key="target_speed_mps", value="%.2f" % plan.target_speed_mps)
            )
            status.values.append(KeyValue(key="plan_reason", value=plan.reason))

        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.status.append(status)
        self.diagnostics_pub.publish(array)

        perf = String()
        perf.data = "frame=%d,time=%.2fms,fps=%.1f" % (
            self.frame_count,
            elapsed_ms,
            (1000.0 / elapsed_ms) if elapsed_ms > 0 else 0.0,
        )
        self.performance_pub.publish(perf)

        if arbitration is not None and arbitration.state is not SafetyState.NOMINAL:
            event = String()
            event.data = "state=%s violations=%s reason=%s" % (
                arbitration.state.value,
                "|".join(arbitration.violations),
                arbitration.reason,
            )
            self.safety_pub.publish(event)

    def _publish_error(self, error_msg: str) -> None:
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.name = "ADAS Pipeline"
        status.level = DiagnosticStatus.ERROR
        status.message = error_msg
        array.status.append(status)
        self.diagnostics_pub.publish(array)


def main(args: Optional[list] = None) -> int:
    """ROS 2 entry point.  Returns a process exit code."""
    if not ROS2_AVAILABLE:
        print(
            "ERROR: ROS 2 (rclpy) is not available. It is not installed on this Jetson; "
            "source a ROS 2 Humble underlay and re-run. The bridge is UNTESTED on this "
            "hardware."
        )
        return 1

    rclpy.init(args=args)
    node = None
    pipeline = None
    try:
        from adas.cli import build_pipeline

        pipeline, _config = build_pipeline()
        node = ADASBridgeNode(pipeline)
        logger.info("ADAS bridge spinning")
        rclpy.spin(node)
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down")
        return 0
    except Exception as exc:  # noqa: BLE001 - top level
        logger.error("Fatal error: %s", exc, exc_info=True)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if pipeline is not None:
            pipeline.close()
        if rclpy.ok():
            rclpy.shutdown()


__all__ = ["ADASBridgeNode", "DEFAULT_SPEED_TIMEOUT_S", "ROS2_AVAILABLE", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
