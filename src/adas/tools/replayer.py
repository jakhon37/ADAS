"""Replay a recording made by :mod:`adas.tools.recorder`.

Two uses:

* **Inspection.**  Iterate the frames, print a summary, find the frame where the
  arbiter escalated.
* **Regression.**  Push the recorded perception back through a pipeline and
  compare the new decision with the recorded one.  That is what
  :func:`replay_with_pipeline` does, and it is the only cheap way to tell whether
  a change to the planner or the arbiter alters behaviour on real data.

What replay reproduces, and what it does not.  A regression harness that
silently reproduces nothing is worse than no harness, so the envelope is stated
here and enforced by :func:`replay_with_pipeline`:

* **Reproduced exactly** -- the detections, the lane model, the ego state
  (validity included), the frame geometry, the frame timestamps and the measured
  ``dt`` of every frame.  Tracking, planning, control and arbitration are then
  RE-RUN on that input, which is the point: a change to the planner or the
  arbiter shows up as a difference against the recorded decision.
* **Not reproduced** -- anything the recorder does not store.  The drivable-area
  mask is stored as a flag, not a mask, and the depth channel is not stored at
  all.  A recording that used either is replayed WITHOUT it and
  :func:`replay_with_pipeline` says so, loudly, once per replay; the arbitration
  can then legitimately differ and the caller has been told why.

How the recorded perception reaches the pipeline.  ``ADASPipeline._run_perception``
calls ``detector.infer(frame.rgb, ...)`` -- it hands the backend the IMAGE, not
the :class:`PerceptionFrame` -- so a replay backend keyed on ``frame_id`` cannot
find its frame from that argument alone.  It used to return ``[]`` instead, which
was indistinguishable from "the recorded detector found nothing": a recorded
full-authority AEB replayed as brake 0.0 on every frame with no error anywhere
(ADAS-OPS-05).  The replay driver now sets the frame id on both backends before
each step (:meth:`RecordedDetector.set_frame`), the ``rgb`` placeholder carries
``frame_id`` as a second route, and a backend that STILL cannot tell which frame
it is on raises instead of inventing an empty road.

What changed and why: the previous version reconstructed a ``LaneModel`` with
``coefficients=``, ``lateral_offset_m=`` and ``heading_error_rad=`` keyword
arguments, none of which are fields of ``LaneModel``, so
``get_perception_frame`` raised ``TypeError`` on any recording that contained a
lane (ADAS-OPS-03).  It also replayed with ``current_speed_mps=0.0``, which now
means "assert 0 m/s" and would silently change every planning decision; the
recorded :class:`~adas.core.models.EgoState` is replayed instead, including its
``valid`` flag.
"""

from __future__ import annotations

import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from adas.core.exceptions import ConfigurationError
from adas.core.logger import setup_logger
from adas.core.models import (
    BoundingBox,
    EgoState,
    LaneLine,
    LaneModel,
    PerceptionFrame,
)
from adas.tools.recorder import RECORD_VERSION

logger = setup_logger(__name__)


@dataclass
class ReplayConfig:
    """Replay parameters.

    Attributes:
        recording_dir: directory written by :class:`DataRecorder`.
        playback_speed: ``1.0`` real time, ``0.0`` as fast as possible.
        loop: restart at the end.
        start_frame, end_frame: half-open index range into the recording.
        load_images: read the ``.npy`` frame buffers back, when present.  Needed
            to re-run perception; not needed to re-run planning and arbitration.
    """

    recording_dir: str
    playback_speed: float = 1.0
    loop: bool = False
    start_frame: int = 0
    end_frame: Optional[int] = None
    load_images: bool = False


class DataReplayer:
    """Reads a recording and reconstructs pipeline inputs from it."""

    def __init__(self, config: ReplayConfig) -> None:
        self.config = config
        self.recording_dir = Path(config.recording_dir)
        if not self.recording_dir.exists():
            raise FileNotFoundError("Recording directory not found: %s" % self.recording_dir)

        self.metadata = self._load_metadata()
        version = int(self.metadata.get("record_version", 1))
        if version > RECORD_VERSION:
            raise ConfigurationError(
                "recording %s is record_version %d; this build understands up to %d"
                % (self.recording_dir, version, RECORD_VERSION)
            )
        self.record_version = version
        self.frames = self._load_frames()
        if config.start_frame < 0 or config.start_frame > len(self.frames):
            raise ConfigurationError(
                "start_frame %d is outside the recording (%d frames)"
                % (config.start_frame, len(self.frames))
            )
        self.current_frame_idx = config.start_frame
        logger.info(
            "DataReplayer: %s, %d frames, record_version %d",
            self.recording_dir,
            len(self.frames),
            version,
        )

    # --------------------------------------------------------------------- load

    def _load_metadata(self) -> dict:
        path = self.recording_dir / "metadata.json"
        if not path.exists():
            logger.warning("No metadata.json in %s; assuming record_version 1", self.recording_dir)
            return {}
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ConfigurationError("metadata.json in %s is invalid: %s" % (self.recording_dir, exc)) from exc

    def _load_frames(self) -> List[dict]:
        json_path = self.recording_dir / "frames.json"
        if json_path.exists():
            try:
                frames = json.loads(json_path.read_text())
            except json.JSONDecodeError as exc:
                raise ConfigurationError("frames.json in %s is invalid: %s" % (self.recording_dir, exc)) from exc
            logger.info("Loaded %d frames from JSON", len(frames))
            return frames

        pkl_path = self.recording_dir / "frames.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as handle:
                frames = pickle.load(handle)
            logger.info("Loaded %d frames from pickle", len(frames))
            return frames

        raise FileNotFoundError("No frame data in %s (frames.json or frames.pkl)" % self.recording_dir)

    # ---------------------------------------------------------------- accessors

    def __len__(self) -> int:
        return len(self.frames)

    def get_frame(self, frame_idx: int) -> Optional[dict]:
        if 0 <= frame_idx < len(self.frames):
            return self.frames[frame_idx]
        return None

    def get_ego(self, frame_idx: int) -> Optional[EgoState]:
        """The recorded ego state, including whether it was valid at the time."""
        data = self.get_frame(frame_idx)
        if data is None or "ego" not in data:
            return None
        ego = data["ego"]
        return EgoState(
            speed_mps=float(ego.get("speed_mps", 0.0)),
            accel_mps2=float(ego.get("accel_mps2", 0.0)),
            yaw_rate_dps=float(ego.get("yaw_rate_dps", 0.0)),
            timestamp_s=float(data.get("timestamp", 0.0)),
            valid=bool(ego.get("valid", False)),
        )

    def get_perception_frame(self, frame_idx: int) -> Optional[PerceptionFrame]:
        """Reconstruct a :class:`PerceptionFrame` from the recording.

        The detections and the lane are the ones that were recorded, so replaying
        through a pipeline exercises tracking, planning, control and arbitration
        against real perception output without needing a GPU.  ``rgb`` is the
        stored ``.npy`` buffer when ``load_images`` is set and one was recorded,
        otherwise a ``{"width", "height"}`` placeholder -- which is enough for
        everything downstream of perception.
        """
        data = self.get_frame(frame_idx)
        if data is None:
            return None

        frame = PerceptionFrame(
            frame_id=int(data["frame_id"]),
            timestamp_s=float(data.get("timestamp", 0.0)),
            rgb=self._load_image(data),
            width=int(data["width"]),
            height=int(data["height"]),
        )

        for det in data.get("detections", []):
            frame.detections.append(
                BoundingBox(
                    x1=float(det["x1"]), y1=float(det["y1"]),
                    x2=float(det["x2"]), y2=float(det["y2"]),
                    confidence=float(det["confidence"]), label=str(det["label"]),
                )
            )

        lane = data.get("lane")
        if lane:
            frame.lane = _lane_from_dict(lane)

        frame.ego = self.get_ego(frame_idx)
        return frame

    def _load_image(self, data: dict) -> Any:
        # The placeholder carries ``frame_id`` because it is the ONLY thing the
        # pipeline hands to the perception backends: ``_run_perception`` calls
        # ``detector.infer(frame.rgb, ...)``.  Without it a replay backend cannot
        # tell which frame it is on.  See the module docstring (ADAS-OPS-05).
        placeholder = {
            "width": int(data["width"]),
            "height": int(data["height"]),
            "frame_id": int(data["frame_id"]),
        }
        image = data.get("image")
        if not self.config.load_images or not isinstance(image, dict):
            return placeholder
        name = image.get("file")
        if not name:
            return placeholder
        try:
            import numpy as np

            return np.load(str(self.recording_dir / name))
        except Exception as exc:  # noqa: BLE001 - a missing buffer is not fatal
            logger.warning("Could not load frame buffer %s: %s", name, exc)
            return placeholder

    # ---------------------------------------------------------------- iteration

    def replay_iterator(self) -> Iterator[dict]:
        """Yield frame records, optionally paced against the recorded timestamps."""
        end_frame = self.config.end_frame if self.config.end_frame is not None else len(self.frames)
        end_frame = max(self.config.start_frame, min(end_frame, len(self.frames)))

        while True:
            for idx in range(self.config.start_frame, end_frame):
                if self.config.playback_speed > 0 and idx > self.config.start_frame:
                    dt = float(self.frames[idx].get("timestamp", 0.0)) - float(
                        self.frames[idx - 1].get("timestamp", 0.0)
                    )
                    sleep_s = dt / self.config.playback_speed
                    # A recording made with wall-clock stamps can contain a
                    # backwards or absurd step; never sleep on one.
                    if math.isfinite(sleep_s) and 0.0 < sleep_s < 5.0:
                        time.sleep(sleep_s)
                self.current_frame_idx = idx
                yield self.frames[idx]

            if not self.config.loop:
                return
            logger.info("Looping replay")

    # ------------------------------------------------------------------ reports

    def get_stats(self) -> dict:
        return {
            "total_frames": len(self.frames),
            "current_frame": self.current_frame_idx,
            "recording_name": self.metadata.get("recording_name", "unknown"),
            "record_version": self.record_version,
            "playback_speed": self.config.playback_speed,
            "loop": self.config.loop,
        }

    def safety_timeline(self) -> List[Tuple[int, str, List[str]]]:
        """Every arbitration state CHANGE in the recording.

        Returns ``(frame_id, state, violations)`` for each transition -- the
        first thing to look at after an event, and far shorter than the frames.
        """
        timeline: List[Tuple[int, str, List[str]]] = []
        previous = None
        for record in self.frames:
            arbitration = record.get("arbitration")
            if not arbitration:
                continue
            state = arbitration.get("state")
            if state != previous:
                timeline.append(
                    (int(record.get("frame_id", -1)), str(state), list(arbitration.get("violations", [])))
                )
                previous = state
        return timeline

    def get_frame_summary(self, frame_idx: int) -> str:
        data = self.get_frame(frame_idx)
        if data is None:
            return "Frame not found"

        lines = [
            "Frame %s:" % data.get("frame_id"),
            "  Timestamp:  %.3fs" % float(data.get("timestamp", 0.0)),
            "  Resolution: %sx%s" % (data.get("width"), data.get("height")),
        ]
        if "ego" in data:
            ego = data["ego"]
            lines.append(
                "  Ego:        %.2f m/s (valid=%s)"
                % (float(ego.get("speed_mps", 0.0)), ego.get("valid"))
            )
        if "perception" in data and not data["perception"].get("ok", True):
            lines.append(
                "  Perception: FAILED (%s, %d consecutive)"
                % (data["perception"].get("reason", "?"),
                   data["perception"].get("consecutive_failures", 0))
            )
        if "detections" in data:
            lines.append("  Detections: %d" % len(data["detections"]))
        if "tracks" in data:
            lines.append(
                "  Tracks:     %d%s"
                % (
                    len(data["tracks"]),
                    "".join(
                        " [#%s %.1fm%s]" % (t["track_id"], t["distance_m"],
                                            " in-lane" if t.get("in_ego_lane") else "")
                        for t in data["tracks"][:3]
                    ),
                )
            )
        if "lane" in data:
            lane = data["lane"]
            offset = lane.get("lateral_offset_m")
            lines.append(
                "  Lane:       center=%.1fpx conf=%.2f mock=%s offset=%s"
                % (
                    float(lane.get("lane_center_px", 0.0)),
                    float(lane.get("confidence", 0.0)),
                    lane.get("is_mock"),
                    "n/a" if offset is None else "%.2fm" % offset,
                )
            )
        if "plan" in data:
            plan = data["plan"]
            lines.append(
                "  Plan:       %.1f m/s, %.1f deg, %s"
                % (plan["target_speed_mps"], plan["steering_angle_deg"], plan["reason"])
            )
        if "arbitration" in data:
            arb = data["arbitration"]
            lines.append(
                "  Arbiter:    %s%s"
                % (arb.get("state"), (" " + ",".join(arb.get("violations", []))).rstrip())
            )
        if "command" in data:
            cmd = data["command"]
            lines.append(
                "  Command:    t=%.2f b=%.2f s=%+.2f"
                % (cmd["throttle"], cmd["brake"], cmd["steering"])
            )
        if "timing" in data:
            lines.append("  Timing:     %.2f ms" % float(data["timing"].get("total_ms", 0.0)))
        return "\n".join(lines)

    def export_summary(self, output_path: Optional[str] = None) -> Path:
        """Write a human-readable summary and return the path it was written to."""
        path = Path(output_path) if output_path else (self.recording_dir / "replay_summary.txt")
        lines = [
            "ADAS Replay Summary",
            "=" * 50,
            "",
            "Recording:    %s" % self.metadata.get("recording_name", "unknown"),
            "Total frames: %d" % len(self.frames),
            "",
            "Safety timeline:",
        ]
        timeline = self.safety_timeline()
        lines.extend(
            ["  frame %-8d %-18s %s" % (fid, state, ",".join(v) or "-") for fid, state, v in timeline]
            or ["  (no arbitration recorded)"]
        )
        lines.append("")
        if self.frames:
            for idx in sorted({0, len(self.frames) // 2, len(self.frames) - 1}):
                lines.append(self.get_frame_summary(idx))
                lines.append("")
        path.write_text("\n".join(lines))
        logger.info("Summary exported to %s", path)
        return path


def _lane_from_dict(data: Dict[str, Any]) -> LaneModel:
    """Rebuild a :class:`LaneModel` from a recorded lane, using its real fields."""
    left = tuple(float(c) for c in data.get("left_coeffs", (0.0, 0.0, 0.0)))
    right = tuple(float(c) for c in data.get("right_coeffs", (0.0, 0.0, 0.0)))
    lines = [
        LaneLine(
            points_px=[],
            coeffs=None if line.get("coeffs") is None else tuple(float(c) for c in line["coeffs"]),
            confidence=float(line.get("confidence", 0.0)),
            index=int(line.get("index", -1)),
        )
        for line in data.get("lines", [])
    ]
    return LaneModel(
        left_coeffs=left if len(left) == 3 else (0.0, 0.0, 0.0),
        right_coeffs=right if len(right) == 3 else (0.0, 0.0, 0.0),
        lane_center_px=float(data.get("lane_center_px", 0.0)),
        curvature_m=float(data.get("curvature_m", 0.0)),
        confidence=float(data.get("confidence", 0.0)),
        lines=lines,
        is_mock=bool(data.get("is_mock", True)),
    )


class _FrameKeyedBackend:
    """Common frame-identification for the replay backends.

    ``ADASPipeline._run_perception`` passes ``frame.rgb`` -- the image -- to the
    backends, so a backend keyed on ``frame_id`` has three possible routes to it,
    tried in this order:

    1. the cursor the replay driver set with :meth:`set_frame` before the step.
       This is the authoritative one and always available under
       :func:`replay_with_pipeline`; it works even when ``load_images`` is on and
       ``rgb`` is a bare numpy array carrying no identity at all.
    2. ``frame.frame_id`` when the caller passed the whole
       :class:`~adas.core.models.PerceptionFrame`.
    3. ``rgb["frame_id"]`` from the replayer's placeholder dict.

    If none of them resolves, the backend RAISES.  It must not return an empty
    result: "I could not tell which frame this is" and "there was nothing on this
    frame" are different claims, and conflating them is what let a recorded AEB
    replay as a clear road (ADAS-OPS-05).  The raise surfaces through
    ``_run_perception`` as a perception FAILURE, which the arbiter degrades on --
    visible, rather than silent.
    """

    def __init__(self) -> None:
        self._cursor: Optional[int] = None

    def set_frame(self, frame_id: Optional[int]) -> None:
        """Tell the backend which recorded frame the pipeline is about to step."""
        self._cursor = None if frame_id is None else int(frame_id)

    def _frame_id(self, frame: Any) -> int:
        if self._cursor is not None:
            return self._cursor
        candidate = getattr(frame, "frame_id", None)
        if candidate is None and isinstance(frame, dict):
            candidate = frame.get("frame_id")
        if candidate is None:
            raise ConfigurationError(
                "%s could not identify the frame it was asked about (%r). A replay "
                "backend must never answer 'nothing here' when it means 'I do not "
                "know which frame this is'; call set_frame() before stepping the "
                "pipeline, or use replay_with_pipeline()."
                % (type(self).__name__, type(frame).__name__)
            )
        return int(candidate)


class RecordedDetector(_FrameKeyedBackend):
    """Detector backend that replays a recording's detections.

    This is what makes offline regression possible without a GPU: the pipeline
    calls ``infer`` exactly as it would call YOLOX, and gets the boxes that were
    actually produced on that frame.  A frame id that IS resolved but is absent
    from the index yields an empty list -- that correctly means "the recorded
    detector found nothing", because a recording only omits detections when there
    were none.  An UNRESOLVABLE frame id raises; see :class:`_FrameKeyedBackend`.
    """

    is_mock = False

    def __init__(self, replayer: "DataReplayer") -> None:
        super().__init__()
        self.by_frame: Dict[int, List[BoundingBox]] = {}
        for idx in range(len(replayer)):
            frame = replayer.get_perception_frame(idx)
            if frame is not None:
                self.by_frame[frame.frame_id] = list(frame.detections)

    def infer(self, frame: Any, width: int, height: int) -> List[BoundingBox]:
        return list(self.by_frame.get(self._frame_id(frame), []))

    def close(self) -> None:
        self.by_frame.clear()


class RecordedLaneEstimator(_FrameKeyedBackend):
    """Lane backend that replays a recording's lane models.

    ``drivable_area`` always returns ``None``: the recorder stores whether a
    drivable area existed, not its mask.  :func:`replay_with_pipeline` warns when
    a recording it is replaying contained one, because its absence can change
    ``in_ego_lane`` and therefore the arbitration.
    """

    name = "recorded"
    is_mock = False

    def __init__(self, replayer: "DataReplayer") -> None:
        super().__init__()
        self.by_frame: Dict[int, Optional[LaneModel]] = {}
        for record in replayer.frames:
            lane = record.get("lane")
            self.by_frame[int(record["frame_id"])] = _lane_from_dict(lane) if lane else None

    def estimate(self, frame: Any, width: int, height: int) -> Optional[LaneModel]:
        return self.by_frame.get(self._frame_id(frame))

    def drivable_area(self):
        return None

    def close(self) -> None:
        self.by_frame.clear()


def replay_fidelity_gaps(replayer: "DataReplayer") -> List[str]:
    """What this recording contains that a replay cannot reproduce.

    Empty means the replay is a faithful reproduction of the decision path for
    this recording.  Non-empty is not an error -- it is the list of reasons the
    replayed arbitration may legitimately differ from the recorded one, and it is
    what a caller comparing the two must be told.
    """
    gaps: List[str] = []
    if any(record.get("drivable") for record in replayer.frames):
        gaps.append(
            "the recording used a drivable-area mask; the recorder stores only a "
            "flag, so tracking replays without it and in_ego_lane may differ"
        )

    environment = replayer.metadata.get("environment")
    used_depth = None
    if isinstance(environment, dict) and "depth_channel" in environment:
        used_depth = bool(environment["depth_channel"])
    elif any(
        track.get("range_source") == "depth_model"
        for record in replayer.frames
        for track in record.get("tracks", [])
    ):
        # A pre-``environment`` recording. Note that ``fused`` is NOT evidence of
        # a depth channel: the tracker reports it for its own pinhole/width
        # fusion, with no depth model in the run at all.
        used_depth = True
    if used_depth:
        gaps.append(
            "the recording used the independent depth range channel; it is not "
            "recorded, so replayed ranges come from the pinhole model alone"
        )
    elif used_depth is None:
        gaps.append(
            "the recording predates environment capture, so whether the depth "
            "range channel was in use is unknown and cannot be reproduced either way"
        )
    if not any("arbitration" in record for record in replayer.frames):
        gaps.append(
            "the recording predates arbitration; there is no arbitrated command "
            "to compare a replay against"
        )
    return gaps


def build_replay_backends(replayer: "DataReplayer"):
    """Return ``(detector, lane_estimator)`` that replay *replayer*'s perception.

    Wire these into an :class:`~adas.runtime.pipeline.ADASPipeline` before calling
    :func:`replay_with_pipeline`; otherwise the pipeline re-runs the *live*
    perception backends against the recording's frame buffers, which needs a GPU
    and, without ``load_images``, has no buffers to run on.
    """
    return RecordedDetector(replayer), RecordedLaneEstimator(replayer)


def replay_with_pipeline(replayer: DataReplayer, pipeline: Any, dt_s: float = 0.05):
    """Push a recording back through a pipeline and yield old vs new.

    The pipeline's perception backends are replaced with
    :func:`build_replay_backends` for the duration of the replay and restored
    afterwards, so tracking, planning, control and arbitration are re-run against
    the recorded perception on any machine, GPU or not.

    Yields:
        ``(recorded_frame_dict, (plan, command))``.  ``command`` is the newly
        ARBITRATED command, so comparing it with
        ``recorded["arbitration"]["command"]`` is a like-for-like regression
        check.  Compare against ``recorded["command"]`` only for a recording made
        before arbitration existed.

    The recorded :class:`EgoState` -- validity included -- is replayed, so a
    recording made with no ego speed replays as degraded rather than as 0 m/s.

    Fidelity is reported, not assumed: :func:`replay_fidelity_gaps` is evaluated
    up front and every gap is logged as a WARNING before the first frame, so a
    caller that then finds a difference knows whether the recording could ever
    have reproduced.  The depth channel is also detached for the duration --
    replaying recorded boxes through a LIVE depth model would mix a recorded
    range with a freshly computed one and produce a decision that never happened.
    """
    for gap in replay_fidelity_gaps(replayer):
        logger.warning("Replay is not a faithful reproduction: %s", gap)
    saved = (pipeline.detector, pipeline.lane_estimator)
    saved_depth = getattr(pipeline, "depth_channel", None)
    pipeline.detector, pipeline.lane_estimator = build_replay_backends(replayer)
    if saved_depth is not None:
        pipeline.depth_channel = None
    try:
        yield from _replay_frames(replayer, pipeline, dt_s)
    finally:
        pipeline.detector, pipeline.lane_estimator = saved
        if saved_depth is not None:
            pipeline.depth_channel = saved_depth


def _replay_frames(replayer: DataReplayer, pipeline: Any, dt_s: float):
    pipeline.reset()
    for frame_data in replayer.replay_iterator():
        perception = replayer.get_perception_frame(replayer.current_frame_idx)
        if perception is None:
            continue
        # Tell the replay backends which frame this is BEFORE stepping. The
        # pipeline hands them ``frame.rgb``, not the frame, so this is the only
        # route that works for every recording -- images or not (ADAS-OPS-05).
        for backend in (pipeline.detector, pipeline.lane_estimator):
            setter = getattr(backend, "set_frame", None)
            if setter is not None:
                setter(perception.frame_id)
        recorded_dt = frame_data.get("timing", {}).get("dt_s", dt_s)
        plan, command = pipeline.step(perception, dt_s=float(recorded_dt), ego=perception.ego)
        yield frame_data, (plan, command)


__all__ = [
    "DataReplayer",
    "RecordedDetector",
    "RecordedLaneEstimator",
    "ReplayConfig",
    "build_replay_backends",
    "replay_fidelity_gaps",
    "replay_with_pipeline",
]
