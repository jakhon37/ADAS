"""Record pipeline inputs and outputs for offline analysis.

What changed and why: the previous version read ``LaneModel.coefficients``,
``LaneModel.lateral_offset_m`` and ``LaneModel.heading_error_rad``, none of which
exist, so :meth:`DataRecorder.record_frame` raised ``AttributeError`` the first
time a lane was present (ADAS-OPS-03).  It also recorded only the plan and the
command, which is the least useful pair: what an incident investigation needs is
the *arbitration* -- what the planner asked for, what the arbiter allowed, and
why they differed.

The record is JSON by default so it can be read without this package.  Images are
off by default because a 1280x720 clip at 20 Hz is 55 MB/s; when
``record_images`` is on the frames are written as individual ``.npy`` files
alongside the JSON and referenced by name, never inlined.

Failure behaviour: recording never raises into the pipeline.  A per-frame error
is counted in :attr:`DataRecorder.errors`, logged once, and the run continues --
a debugging tool that can stop the vehicle is worse than no debugging tool.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from adas.core.logger import setup_logger
from adas.core.models import (
    ArbitrationResult,
    ControlCommand,
    EgoState,
    LaneModel,
    MotionPlan,
    PerceptionFrame,
)

logger = setup_logger(__name__)

#: Bumped when the record layout changes incompatibly. The replayer refuses a
#: newer version rather than silently misreading fields.
RECORD_VERSION = 2


@dataclass
class RecordingConfig:
    """Configuration for one recording.

    Attributes:
        output_dir: parent directory; one subdirectory per recording.
        recording_name: subdirectory name, defaulted to a UTC timestamp.
        record_images: write each frame as ``frames/<id>.npy``.  Large.
        record_detections, record_plans, record_commands: field selection.
        record_tracks: include the tracker's output, which is what makes a
            recording useful for a range or association post-mortem.
        format: ``json`` (portable, inspectable) or ``pickle`` (faster, and only
            loadable by a matching Python).
        flush_every: write the JSON to disk every N frames so a crash does not
            lose the whole run.  ``0`` writes only at :meth:`stop_recording`.
    """

    output_dir: str = "recordings"
    recording_name: Optional[str] = None
    record_images: bool = False
    record_detections: bool = True
    record_plans: bool = True
    record_commands: bool = True
    record_tracks: bool = True
    format: str = "json"
    flush_every: int = 200

    def __post_init__(self) -> None:
        if self.recording_name is None:
            self.recording_name = "adas_recording_%s" % time.strftime("%Y%m%d_%H%M%S")
        if self.format not in ("json", "pickle"):
            raise ValueError("RecordingConfig.format must be 'json' or 'pickle'")


def lane_to_dict(lane: LaneModel) -> Dict[str, Any]:
    """Serialise a lane model using the fields it actually has.

    ``lateral_offset_m`` / ``heading_error_rad`` / ``curvature_radius_m`` are
    recomputed from the METRIC ground-plane fit stored on ``lane.lines`` (indices
    1 and 2) via :func:`adas.perception.lane.lane_geometry_from_model`.  They are
    ``None`` -- not 0.0 -- for a mock lane or one fitted without a camera, because
    "no metric geometry exists" and "the vehicle is exactly centred" are different
    claims.
    """
    record: Dict[str, Any] = {
        "lane_center_px": lane.lane_center_px,
        "left_coeffs": list(lane.left_coeffs),
        "right_coeffs": list(lane.right_coeffs),
        "curvature_m": lane.curvature_m,
        "confidence": lane.confidence,
        "is_mock": lane.is_mock,
        "lines": [
            {
                "index": line.index,
                "confidence": line.confidence,
                "coeffs": None if line.coeffs is None else [float(c) for c in line.coeffs],
                "n_points": len(line.points_px),
            }
            for line in lane.lines
        ],
        "lateral_offset_m": None,
        "heading_error_rad": None,
        "curvature_radius_m": None,
        "lane_width_m": None,
        "plausible": None,
    }
    try:
        from adas.perception.lane import lane_geometry_from_model

        geometry = lane_geometry_from_model(lane)
    except Exception as exc:  # noqa: BLE001 - a recording helper, never fatal
        logger.debug("lane_geometry_from_model failed while recording: %s", exc)
        geometry = None
    if geometry is not None:
        record.update(
            lateral_offset_m=geometry.lateral_offset_m,
            heading_error_rad=geometry.heading_error_rad,
            curvature_radius_m=geometry.curvature_radius_m,
            lane_width_m=geometry.lane_width_m,
            plausible=geometry.plausible,
        )
    return record


def arbitration_to_dict(result: ArbitrationResult) -> Dict[str, Any]:
    """Serialise what the safety arbiter decided and what it actuated."""
    return {
        "state": result.state.value,
        "violations": list(result.violations),
        "reason": result.reason,
        "command": {
            "throttle": result.command.throttle,
            "brake": result.command.brake,
            "steering": result.command.steering,
        },
    }


class DataRecorder:
    """Append-only recorder for pipeline frames.

    Not thread-safe: it belongs to the pipeline thread, like the metrics object.
    """

    def __init__(self, config: Optional[RecordingConfig] = None) -> None:
        self.config = config or RecordingConfig()
        self.recording_dir = Path(self.config.output_dir) / str(self.config.recording_name)
        self.recording_dir.mkdir(parents=True, exist_ok=True)
        if self.config.record_images:
            (self.recording_dir / "frames").mkdir(exist_ok=True)

        self.frame_data: List[Dict[str, Any]] = []
        self.errors = 0
        self.is_recording = False
        self.metadata: Dict[str, Any] = {
            "record_version": RECORD_VERSION,
            "recording_name": self.config.recording_name,
            "start_time": time.time(),
            "config": asdict(self.config),
        }
        self._since_flush = 0
        logger.info("DataRecorder initialised: %s", self.recording_dir)

    # ------------------------------------------------------------------ control

    def start_recording(self) -> None:
        self.is_recording = True
        self.metadata["start_time"] = time.time()
        logger.info("Recording started: %s", self.recording_dir)

    def stop_recording(self) -> None:
        self.is_recording = False
        self.metadata["end_time"] = time.time()
        self.metadata["total_frames"] = len(self.frame_data)
        self.metadata["errors"] = self.errors
        self._save()
        logger.info("Recording stopped: %d frames, %d errors", len(self.frame_data), self.errors)

    # ------------------------------------------------------------------- record

    def record_frame(
        self,
        frame: PerceptionFrame,
        plan: Optional[MotionPlan] = None,
        command: Optional[ControlCommand] = None,
        timing_info: Optional[dict] = None,
        arbitration: Optional[ArbitrationResult] = None,
        tracks: Optional[List[Any]] = None,
        ego: Optional[EgoState] = None,
    ) -> None:
        """Record one frame.  Never raises; a failure is counted and logged."""
        if not self.is_recording:
            return
        try:
            self.frame_data.append(
                self._build_record(frame, plan, command, timing_info, arbitration, tracks, ego)
            )
        except Exception as exc:  # noqa: BLE001 - a debugging tool must not stop the loop
            self.errors += 1
            if self.errors == 1:
                logger.error("Recording frame %s failed: %s", frame.frame_id, exc, exc_info=True)
            return

        self._since_flush += 1
        if self.config.flush_every and self._since_flush >= self.config.flush_every:
            self._since_flush = 0
            self._save()

    def _build_record(
        self,
        frame: PerceptionFrame,
        plan: Optional[MotionPlan],
        command: Optional[ControlCommand],
        timing_info: Optional[dict],
        arbitration: Optional[ArbitrationResult],
        tracks: Optional[List[Any]],
        ego: Optional[EgoState],
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp_s,
            "width": frame.width,
            "height": frame.height,
        }

        if self.config.record_images and frame.rgb is not None:
            record["image"] = self._write_image(frame)

        if frame.status is not None:
            record["perception"] = {
                "ok": frame.status.ok,
                "consecutive_failures": frame.status.consecutive_failures,
                "detector_ok": frame.status.detector_ok,
                "lane_ok": frame.status.lane_ok,
                "reason": frame.status.reason,
            }

        source_ego = ego if ego is not None else frame.ego
        if source_ego is not None:
            record["ego"] = {
                "speed_mps": source_ego.speed_mps,
                "accel_mps2": source_ego.accel_mps2,
                "yaw_rate_dps": source_ego.yaw_rate_dps,
                "valid": source_ego.valid,
            }

        if self.config.record_detections and frame.detections:
            record["detections"] = [
                {
                    "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                    "confidence": d.confidence, "label": d.label,
                }
                for d in frame.detections
            ]

        if self.config.record_tracks and tracks:
            record["tracks"] = [
                {
                    "track_id": t.track_id,
                    "distance_m": t.distance_m,
                    "velocity_mps": t.velocity_mps,
                    "lateral_offset_m": t.lateral_offset_m,
                    "ttc_s": None if t.ttc_s == float("inf") else t.ttc_s,
                    "in_ego_lane": t.in_ego_lane,
                    "age_frames": t.age_frames,
                    "hits": t.hits,
                    "time_since_update": t.time_since_update,
                    "label": t.box.label,
                    "range_source": (
                        None if t.range_estimate is None else t.range_estimate.source.value
                    ),
                    "range_confidence": (
                        None if t.range_estimate is None else t.range_estimate.confidence
                    ),
                }
                for t in tracks
            ]

        if frame.lane is not None:
            record["lane"] = lane_to_dict(frame.lane)

        if frame.drivable is not None:
            record["drivable"] = {
                "width": frame.drivable.width,
                "height": frame.drivable.height,
                "confidence": frame.drivable.confidence,
                "has_mask": frame.drivable.mask is not None,
            }

        if self.config.record_plans and plan is not None:
            record["plan"] = {
                "target_speed_mps": plan.target_speed_mps,
                "steering_angle_deg": plan.steering_angle_deg,
                "reason": plan.reason,
            }

        if self.config.record_commands and command is not None:
            record["command"] = {
                "throttle": command.throttle,
                "brake": command.brake,
                "steering": command.steering,
            }

        if arbitration is not None:
            record["arbitration"] = arbitration_to_dict(arbitration)

        if timing_info:
            record["timing"] = dict(timing_info)

        return record

    def _write_image(self, frame: PerceptionFrame) -> Any:
        """Write the frame buffer next to the JSON and return a reference."""
        if isinstance(frame.rgb, dict):
            return dict(frame.rgb)
        try:
            import numpy as np

            name = "frames/%08d.npy" % frame.frame_id
            np.save(str(self.recording_dir / name), frame.rgb)
            return {"file": name}
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not write frame %s: %s", frame.frame_id, exc)
            return {"file": None, "error": str(exc)}

    # --------------------------------------------------------------------- save

    def _save(self) -> None:
        try:
            (self.recording_dir / "metadata.json").write_text(json.dumps(self.metadata, indent=2))
            if self.config.format == "json":
                (self.recording_dir / "frames.json").write_text(
                    json.dumps(self.frame_data, indent=2, default=str)
                )
            else:
                with open(self.recording_dir / "frames.pkl", "wb") as handle:
                    pickle.dump(self.frame_data, handle)
            self._write_summary()
        except OSError as exc:
            self.errors += 1
            logger.error("Could not save recording to %s: %s", self.recording_dir, exc)

    def _write_summary(self) -> None:
        started = self.metadata["start_time"]
        ended = self.metadata.get("end_time", time.time())
        states: Dict[str, int] = {}
        for record in self.frame_data:
            state = record.get("arbitration", {}).get("state")
            if state:
                states[state] = states.get(state, 0) + 1
        lines = [
            "ADAS Recording Summary",
            "=" * 50,
            "",
            "Recording:    %s" % self.config.recording_name,
            "Record ver:   %d" % RECORD_VERSION,
            "Total frames: %d" % len(self.frame_data),
            "Duration:     %.2fs" % (ended - started),
            "Format:       %s" % self.config.format,
            "Images:       %s" % self.config.record_images,
            "Errors:       %d" % self.errors,
            "Safety states: %s" % (", ".join("%s=%d" % kv for kv in sorted(states.items())) or "-"),
            "",
        ]
        (self.recording_dir / "summary.txt").write_text("\n".join(lines))

    def get_stats(self) -> dict:
        return {
            "total_frames": len(self.frame_data),
            "recording_time": time.time() - self.metadata["start_time"],
            "is_recording": self.is_recording,
            "output_dir": str(self.recording_dir),
            "errors": self.errors,
        }


class RecordingPipeline:
    """Pipeline wrapper that records every step.

    Delegates everything it does not override, so it can be dropped in front of
    an :class:`~adas.runtime.pipeline.ADASPipeline` with no other change.  It
    records the ARBITRATED command and the arbitration result, which is what
    makes the recording an account of what the vehicle was told to do rather than
    what the planner wanted.
    """

    def __init__(self, pipeline: Any, recorder: DataRecorder) -> None:
        self.pipeline = pipeline
        self.recorder = recorder

    def step(self, frame: PerceptionFrame, current_speed_mps: Optional[float] = None,
             dt_s: float = 0.05, ego: Optional[EgoState] = None):
        started = time.monotonic()
        plan, command = self.pipeline.step(
            frame, current_speed_mps=current_speed_mps, dt_s=dt_s, ego=ego
        )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        self.recorder.record_frame(
            frame=frame,
            plan=plan,
            command=command,
            timing_info={"total_ms": elapsed_ms, "dt_s": dt_s},
            arbitration=getattr(self.pipeline, "last_arbitration", None),
            tracks=getattr(self.pipeline, "last_tracks", None),
            ego=ego if ego is not None else getattr(self.pipeline, "last_ego", None),
        )
        return plan, command

    def __getattr__(self, name: str) -> Any:
        return getattr(self.pipeline, name)


__all__ = [
    "DataRecorder",
    "RECORD_VERSION",
    "RecordingConfig",
    "RecordingPipeline",
    "arbitration_to_dict",
    "lane_to_dict",
]
