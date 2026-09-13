"""Frame sources and ego-speed sources.

Frame sources
-------------
``synthetic`` (a blank canvas, for wiring tests), ``video`` (an mp4 replayed
through OpenCV) and ``camera`` (a V4L2 index or a Jetson CSI GStreamer pipeline).
There is no camera on this board -- ``/dev/video*`` is absent -- so everything is
validated by replaying local clips, and the CSI path is written but untested
here.

A read that returns nothing is ambiguous, and the ambiguity matters: end of file
on a replay clip is a normal termination, while a mid-stream failure on a live
camera is a fault that should be retried and counted.  :class:`FrameSource`
therefore exposes :attr:`eof` alongside :meth:`read`, and
:class:`ReconnectingSource` reopens only on the fault case.

Ego-speed sources
-----------------
There is no vehicle bus on this board.  Rather than let a fabricated speed leak
into the constant-time-gap law, every ego-speed source states what it is:

===============  ================================================  ========
source           what it is                                        measured
===============  ================================================  ========
``none``         no ego speed at all; ``EgoState.valid`` is False   no
``config``       a constant the operator declared                   no
``simulated``    a point-mass plant driven by the actuated command  no
``file``         a recorded ``timestamp_s,speed_mps`` channel       yes
===============  ================================================  ========

``valid`` says whether the planner may use the number; ``measured`` says whether
it came from a sensor.  Only ``file`` sets both, and only for as long as its
samples are fresher than ``ego.max_age_s``.  Everything that reports a speed to
an operator -- the log banner, ``/healthz``, the event log -- carries
``measured`` so a bench run can never be mistaken for a vehicle run.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

from adas.core.config import EgoConfig, SourceConfig
from adas.core.exceptions import ConfigurationError, SensorError
from adas.core.logger import setup_logger
from adas.core.models import ControlCommand, EgoState

logger = setup_logger(__name__)

JETSON_CSI_PIPELINE = (
    "nvarguscamerasrc sensor-id={sensor} ! "
    "video/x-raw(memory:NVMM), width={width}, height={height}, "
    "framerate=30/1, format=NV12 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
)


@dataclass
class CapturedFrame:
    """One frame plus the size it actually arrived at (not the requested size)."""

    image: object
    width: int
    height: int


class FrameSource:
    """Base frame source.

    Attributes:
        eof: set by :meth:`read` when ``None`` means "the stream ended normally".
            A ``None`` with ``eof`` still False is a read FAILURE and a caller may
            retry or reconnect.
    """

    eof: bool = False

    def read(self) -> Optional[CapturedFrame]:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def reopen(self) -> bool:
        """Reopen the underlying device. Returns True on success."""
        return False

    def frames(self) -> Iterator[CapturedFrame]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class SyntheticSource(FrameSource):
    """A blank frame generator.

    ``as_image=False`` yields a ``{"width", "height"}`` dict, which exercises the
    orchestration without importing numpy; ``as_image=True`` yields a real black
    ``uint8`` array so a TensorRT backend can run.  A black frame is a legitimate
    input, not a fault: the detector should return zero detections and the lane
    estimator ``None``, and anything else is a bug worth seeing.
    """

    def __init__(self, width: int = 1280, height: int = 720, as_image: bool = False) -> None:
        self.width = int(width)
        self.height = int(height)
        self.as_image = bool(as_image)

    def read(self) -> Optional[CapturedFrame]:
        if self.as_image:
            import numpy as np

            image: Any = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            image = {"width": self.width, "height": self.height}
        return CapturedFrame(image=image, width=self.width, height=self.height)


class OpenCVSource(FrameSource):
    """OpenCV ``VideoCapture`` over a file, a device index or a GStreamer pipeline.

    Args:
        uri: file path, device index, or GStreamer pipeline string.
        width, height: requested capture size (device sources only; a file's own
            size always wins).
        loop: restart a *file* at end of stream instead of reporting EOF.  For
            soak testing only -- it makes a finite clip look like a live feed.
    """

    def __init__(
        self,
        uri: Any,
        width: Optional[int] = None,
        height: Optional[int] = None,
        loop: bool = False,
    ) -> None:
        import cv2

        self._cv2 = cv2
        self.uri = uri
        self.width = width
        self.height = height
        self.loop = bool(loop)
        self.cap = None
        self._open()

    def _open(self) -> None:
        cv2 = self._cv2
        cap = cv2.VideoCapture(self.uri)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            cap.release()
            raise SensorError("failed to open video source: %s" % (self.uri,))
        self.cap = cap
        self.eof = False
        logger.info("Opened video source %s", self.uri)

    def read(self) -> Optional[CapturedFrame]:
        if self.cap is None:
            return None
        ok, image = self.cap.read()
        if not ok or image is None:
            if self.loop and self._rewind():
                ok, image = self.cap.read()
            if not ok or image is None:
                # A file that has run out is EOF; a device that stops delivering
                # is a fault. `CAP_PROP_FRAME_COUNT > 0` is the only signal
                # OpenCV gives us to tell them apart.
                self.eof = self._is_file()
                return None
        h, w = image.shape[:2]
        return CapturedFrame(image=image, width=int(w), height=int(h))

    def _is_file(self) -> bool:
        try:
            return float(self.cap.get(self._cv2.CAP_PROP_FRAME_COUNT)) > 0.0
        except Exception:  # noqa: BLE001 - a probe, never fatal
            return isinstance(self.uri, str) and Path(self.uri).exists()

    def _rewind(self) -> bool:
        try:
            self.cap.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
            logger.info("Looping video source %s", self.uri)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not rewind %s: %s", self.uri, exc)
            return False

    def reopen(self) -> bool:
        self.close()
        try:
            self._open()
            return True
        except SensorError as exc:
            logger.error("Reopen failed for %s: %s", self.uri, exc)
            return False

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class ReconnectingSource(FrameSource):
    """Wrap a source so a mid-stream read FAILURE is retried (ADAS-OPS-05).

    End of file is passed straight through -- a clip that ended has not failed
    and reopening it would silently turn a 400-frame replay into an infinite one.
    Each successful reopen increments :attr:`reconnects` so the metric and the
    event log can show how flaky the link is.
    """

    def __init__(self, inner: FrameSource, attempts: int = 3, delay_s: float = 1.0) -> None:
        self.inner = inner
        self.attempts = max(0, int(attempts))
        self.delay_s = max(0.0, float(delay_s))
        self.reconnects = 0
        self.failures = 0

    def read(self) -> Optional[CapturedFrame]:
        frame = self.inner.read()
        if frame is not None:
            return frame
        if self.inner.eof:
            self.eof = True
            return None
        self.failures += 1
        for attempt in range(1, self.attempts + 1):
            logger.warning(
                "Source read failed; reconnect attempt %d/%d in %.1fs",
                attempt,
                self.attempts,
                self.delay_s,
            )
            if self.delay_s > 0:
                time.sleep(self.delay_s)
            if not self.inner.reopen():
                continue
            self.reconnects += 1
            frame = self.inner.read()
            if frame is not None:
                logger.info("Source reconnected after %d attempt(s)", attempt)
                return frame
            if self.inner.eof:
                self.eof = True
                return None
        logger.error("Source is unrecoverable after %d attempt(s)", self.attempts)
        self.eof = False
        return None

    def reopen(self) -> bool:
        return self.inner.reopen()

    def close(self) -> None:
        self.inner.close()


def open_source(
    source_type: str,
    uri: str = "",
    width: int = 1280,
    height: int = 720,
    as_image: bool = False,
    loop: bool = False,
    reconnect_attempts: int = 0,
    reconnect_delay_s: float = 1.0,
) -> FrameSource:
    """Open a frame source from config or CLI arguments.

    Raises:
        SensorError: unknown type, missing URI, or the device would not open.
    """
    kind = (source_type or "synthetic").lower()
    if kind == "synthetic":
        return SyntheticSource(width=width, height=height, as_image=as_image)

    if kind == "video":
        if not uri:
            raise SensorError("video source requires a file path")
        path = Path(uri)
        if not path.exists():
            raise SensorError("video file not found: %s" % uri)
        inner: FrameSource = OpenCVSource(str(path), loop=loop)
    elif kind == "camera":
        inner = _open_camera(uri, width, height)
    else:
        raise SensorError("unknown source type: %s" % source_type)

    if reconnect_attempts > 0:
        return ReconnectingSource(inner, attempts=reconnect_attempts, delay_s=reconnect_delay_s)
    return inner


def _open_camera(uri: str, width: int, height: int) -> FrameSource:
    if uri.startswith("nvargus") or uri.startswith("nvv4l2") or "!" in uri:
        return OpenCVSource(uri)
    if uri in ("", "csi", "csi:0"):
        pipeline = JETSON_CSI_PIPELINE.format(sensor=0, width=width, height=height)
        logger.info("Using Jetson CSI pipeline (UNTESTED on this board: no camera present)")
        return OpenCVSource(pipeline)
    if uri.startswith("csi:"):
        try:
            sensor = int(uri.split(":", 1)[1] or "0")
        except ValueError as exc:
            raise SensorError("invalid CSI sensor id in %r" % uri) from exc
        pipeline = JETSON_CSI_PIPELINE.format(sensor=sensor, width=width, height=height)
        return OpenCVSource(pipeline)
    try:
        index: Any = int(uri)
    except ValueError:
        index = uri
    return OpenCVSource(index, width=width, height=height)


def parse_source_arg(value: Optional[str]) -> Tuple[str, str]:
    """Parse the CLI ``--source`` argument into ``(type, uri)``."""
    if not value or value == "synthetic":
        return "synthetic", ""
    if value in ("camera", "csi"):
        return "camera", "csi:0"
    if value.startswith("camera:"):
        return "camera", value.split(":", 1)[1]
    return "video", value


# --------------------------------------------------------------------------- #
# Ego speed
# --------------------------------------------------------------------------- #


class EgoSpeedSource:
    """Base ego-speed source.  See the module docstring for the honesty contract.

    Attributes:
        measured: ``True`` only when the value originates from a sensor recording.
            A declared constant and a simulated plant are both ``False``, and the
            health endpoint publishes this as ``adas_ego_speed_valid``'s companion
            so a fleet query can separate bench units from vehicles.
        kind: the configuration name of this source.
    """

    kind = "none"
    measured = False

    def state(self, timestamp_s: float) -> EgoState:
        """The ego state for a frame captured at *timestamp_s* (monotonic)."""
        raise NotImplementedError

    def apply_command(self, command: ControlCommand, dt_s: float) -> None:
        """Feed back the ACTUATED command.  Only a simulated plant uses it."""
        return None

    def describe(self) -> str:
        return self.kind

    def close(self) -> None:
        return None


class NoEgoSpeed(EgoSpeedSource):
    """No ego speed at all.  The honest default for a board with no vehicle bus.

    Every state is ``valid=False``.  The longitudinal planner then holds its last
    target and ramps it down at ``max_decel_mps2`` (and at ``mrm_decel_mps2``
    after three consecutive dropouts), and the arbiter forces a minimum-risk
    manoeuvre.  The vehicle will not drive -- which is correct, because nothing
    knows how fast it is going.
    """

    kind = "none"
    measured = False

    def state(self, timestamp_s: float) -> EgoState:
        return EgoState(speed_mps=0.0, valid=False, timestamp_s=timestamp_s)

    def describe(self) -> str:
        return "none (no ego speed; the planner will stay degraded)"


class ConstantEgoSpeed(EgoSpeedSource):
    """A fixed speed the operator declared in the configuration.

    ``valid=True`` because the operator asserted it, ``measured=False`` because
    nothing measured it.  Useful for replaying a clip that was recorded at a
    roughly known speed; never correct in a vehicle.
    """

    kind = "config"
    measured = False

    def __init__(self, speed_mps: float) -> None:
        if not math.isfinite(speed_mps) or speed_mps < 0.0:
            raise ConfigurationError("ego.speed_mps must be finite and >= 0, got %r" % (speed_mps,))
        self.speed_mps = float(speed_mps)
        logger.warning(
            "Ego speed is a DECLARED CONSTANT of %.2f m/s, not a measurement. "
            "The time-gap law will be exactly as wrong as this number is.",
            self.speed_mps,
        )

    def state(self, timestamp_s: float) -> EgoState:
        return EgoState(speed_mps=self.speed_mps, valid=True, timestamp_s=timestamp_s)

    def describe(self) -> str:
        return "config (declared constant %.2f m/s)" % self.speed_mps


class SimulatedEgoSpeed(EgoSpeedSource):
    """Closed-loop point-mass plant driven by the ACTUATED command.

    Demanded acceleration is ``a_throttle*throttle - a_brake*brake - drag*v``.
    The ACHIEVED acceleration follows it through a first-order lag with time
    constant ``actuator_tau_s``, and the speed integrates that::

        a += (a_demand - a) * (1 - exp(-dt/tau))
        v  = max(0, v + a*dt)

    The lag is not cosmetic.  Without it a pedal command that steps between
    frames produces an instantaneous acceleration step, i.e. infinite jerk, and
    the arbiter -- which differences the MEASURED ego speed to check jerk --
    correctly reports 40-60 m/s^3 violations and holds the system in LIMITED
    forever.  That is the plant being unphysical, not the arbiter being wrong;
    0.15 s is a plausible brake-system rise time and keeps the bench inside the
    envelope a real actuator would.

    Still no grade, no aerodynamic drag model, no dead time, no tyre limit.  It
    exists so a replay run exercises the full closed loop instead of running the
    planner permanently degraded; it is not a vehicle model and no claim about
    vehicle behaviour may be drawn from it.

    ``valid=True`` (the loop needs a number) and ``measured=False``, and the
    construction logs a NOT-FOR-VEHICLE-USE warning.
    """

    kind = "simulated"
    measured = False

    def __init__(
        self,
        initial_speed_mps: float = 0.0,
        accel_authority_mps2: float = 2.5,
        brake_authority_mps2: float = 8.0,
        drag_per_s: float = 0.02,
        actuator_tau_s: float = 0.15,
        max_speed_mps: float = 60.0,
    ) -> None:
        self.speed_mps = max(0.0, float(initial_speed_mps))
        self.accel_authority_mps2 = float(accel_authority_mps2)
        self.brake_authority_mps2 = float(brake_authority_mps2)
        self.drag_per_s = float(drag_per_s)
        self.actuator_tau_s = max(0.0, float(actuator_tau_s))
        self.max_speed_mps = float(max_speed_mps)
        self.last_accel_mps2 = 0.0
        logger.warning(
            "Ego speed is SIMULATED by a point-mass plant (initial %.2f m/s, "
            "+%.1f/-%.1f m/s^2 authority). NOT FOR VEHICLE USE: the loop is closed "
            "against a model, not against a vehicle.",
            self.speed_mps,
            self.accel_authority_mps2,
            self.brake_authority_mps2,
        )

    def state(self, timestamp_s: float) -> EgoState:
        return EgoState(
            speed_mps=self.speed_mps,
            accel_mps2=self.last_accel_mps2,
            valid=True,
            timestamp_s=timestamp_s,
        )

    def apply_command(self, command: ControlCommand, dt_s: float) -> None:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            return
        throttle = command.throttle if math.isfinite(command.throttle) else 0.0
        brake = command.brake if math.isfinite(command.brake) else 1.0
        demand = (
            self.accel_authority_mps2 * max(0.0, min(1.0, throttle))
            - self.brake_authority_mps2 * max(0.0, min(1.0, brake))
            - self.drag_per_s * self.speed_mps
        )
        if self.actuator_tau_s > 0.0:
            alpha = 1.0 - math.exp(-dt_s / self.actuator_tau_s)
            self.last_accel_mps2 += (demand - self.last_accel_mps2) * alpha
        else:
            self.last_accel_mps2 = demand
        self.speed_mps = max(
            0.0, min(self.max_speed_mps, self.speed_mps + self.last_accel_mps2 * dt_s)
        )

    def describe(self) -> str:
        return "simulated (point-mass plant, currently %.2f m/s)" % self.speed_mps


class FileEgoSpeed(EgoSpeedSource):
    """A recorded speed channel replayed against the frame clock.

    Accepted formats:

    * CSV with a header containing ``timestamp_s`` and ``speed_mps`` columns, or
      a headerless two-column CSV of ``t,v``.
    * JSON: a list of ``{"timestamp_s": t, "speed_mps": v}`` objects, a list of
      ``[t, v]`` pairs, or an object with a ``"samples"`` key holding either.

    Timestamps are seconds and are interpreted RELATIVE to the first frame, so a
    recording made with wall-clock stamps and a run started later still line up.

    Between two samples whose gap is at most ``max_age_s`` the speed is linearly
    INTERPOLATED -- a channel recorded at 10-50 Hz is dense enough for that to be
    a fair reconstruction, and holding the earlier sample instead would inject a
    staircase that the arbiter reads as jerk.  Across a larger gap, and past the
    last sample, the value is HELD for at most ``max_age_s`` and then reported
    ``valid=False``.  A frozen speed held indefinitely is the failure mode that
    makes a dead bus look healthy -- the same bug as the ROS 2 bridge's old
    ``current_speed_mps = 0.0`` default.
    """

    kind = "file"
    measured = True

    def __init__(self, path: str, max_age_s: float = 0.5) -> None:
        self.path = str(path)
        self.max_age_s = float(max_age_s)
        self.samples = _load_speed_samples(self.path)
        if not self.samples:
            raise ConfigurationError("ego speed file %s contains no usable samples" % self.path)
        self._t0: Optional[float] = None
        self._index = 0
        self._stale_logged = False
        logger.info(
            "Ego speed from %s: %d samples spanning %.1f s (%.2f..%.2f m/s)",
            self.path,
            len(self.samples),
            self.samples[-1][0] - self.samples[0][0],
            min(v for _, v in self.samples),
            max(v for _, v in self.samples),
        )

    def state(self, timestamp_s: float) -> EgoState:
        if self._t0 is None:
            self._t0 = timestamp_s
        elapsed = timestamp_s - self._t0 + self.samples[0][0]

        while self._index + 1 < len(self.samples) and self.samples[self._index + 1][0] <= elapsed:
            self._index += 1
        sample_t, sample_v = self.samples[self._index]

        nxt = self.samples[self._index + 1] if self._index + 1 < len(self.samples) else None
        if nxt is not None and (nxt[0] - sample_t) <= self.max_age_s:
            span = nxt[0] - sample_t
            frac = 0.0 if span <= 0.0 else (elapsed - sample_t) / span
            speed = sample_v + (nxt[1] - sample_v) * max(0.0, min(1.0, frac))
            self._stale_logged = False
            return EgoState(speed_mps=speed, valid=True, timestamp_s=timestamp_s)

        age = elapsed - sample_t
        if age > self.max_age_s:
            if not self._stale_logged:
                logger.warning(
                    "Ego speed channel %s is stale (%.2f s > %.2f s at t=%.2f); "
                    "reporting invalid",
                    self.path,
                    age,
                    self.max_age_s,
                    elapsed,
                )
                self._stale_logged = True
            return EgoState(speed_mps=0.0, valid=False, timestamp_s=timestamp_s)
        self._stale_logged = False
        return EgoState(speed_mps=sample_v, valid=True, timestamp_s=timestamp_s)

    def describe(self) -> str:
        return "file (%s, %d samples)" % (self.path, len(self.samples))


def _load_speed_samples(path: str) -> List[Tuple[float, float]]:
    """Parse a CSV or JSON ego-speed sidecar into sorted ``(t, v)`` pairs.

    Raises:
        ConfigurationError: the file is missing, unreadable, or in no recognised
            shape.  Rows with a non-finite or negative speed are dropped with a
            warning rather than failing the load.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ConfigurationError("ego speed file not found: %s" % path)

    raw: List[Sequence[Any]] = []
    text = file_path.read_text()
    if file_path.suffix.lower() == ".json" or text.lstrip()[:1] in ("[", "{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigurationError("ego speed file %s is not valid JSON: %s" % (path, exc)) from exc
        if isinstance(payload, dict):
            payload = payload.get("samples", [])
        if not isinstance(payload, list):
            raise ConfigurationError("ego speed JSON %s must be a list or {'samples': [...]}" % path)
        for item in payload:
            if isinstance(item, dict):
                raw.append((item.get("timestamp_s", item.get("t")), item.get("speed_mps", item.get("v"))))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                raw.append((item[0], item[1]))
    else:
        rows = list(csv.reader(text.splitlines()))
        if not rows:
            raise ConfigurationError("ego speed CSV %s is empty" % path)
        header = [c.strip().lower() for c in rows[0]]
        if "speed_mps" in header:
            t_col = header.index("timestamp_s") if "timestamp_s" in header else 0
            v_col = header.index("speed_mps")
            body = rows[1:]
        else:
            t_col, v_col = 0, 1
            body = rows
        for row in body:
            if len(row) <= max(t_col, v_col):
                continue
            raw.append((row[t_col], row[v_col]))

    samples: List[Tuple[float, float]] = []
    dropped = 0
    for t, v in raw:
        try:
            tt, vv = float(t), float(v)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if not (math.isfinite(tt) and math.isfinite(vv)) or vv < 0.0:
            dropped += 1
            continue
        samples.append((tt, vv))
    if dropped:
        logger.warning("Dropped %d unusable row(s) from ego speed file %s", dropped, path)
    samples.sort(key=lambda pair: pair[0])
    return samples


def open_ego_source(config: EgoConfig) -> EgoSpeedSource:
    """Build the ego-speed source named by ``ego.source``.

    Raises:
        ConfigurationError: unknown source, or a ``file`` source whose sidecar is
            missing or empty.
    """
    kind = config.source
    if kind == "none":
        return NoEgoSpeed()
    if kind == "config":
        return ConstantEgoSpeed(config.speed_mps)
    if kind == "simulated":
        return SimulatedEgoSpeed(
            initial_speed_mps=config.speed_mps,
            accel_authority_mps2=config.accel_authority_mps2,
            brake_authority_mps2=config.brake_authority_mps2,
            drag_per_s=config.drag_per_s,
            actuator_tau_s=config.actuator_tau_s,
        )
    if kind == "file":
        return FileEgoSpeed(config.file, max_age_s=config.max_age_s)
    raise ConfigurationError("Unknown ego source: %s" % kind)


def open_source_from_config(config: SourceConfig, as_image: bool = False) -> FrameSource:
    """Open the frame source described by a :class:`SourceConfig`."""
    return open_source(
        config.type,
        uri=config.uri,
        width=config.width,
        height=config.height,
        as_image=as_image,
        loop=config.loop,
        reconnect_attempts=config.reconnect_attempts,
        reconnect_delay_s=config.reconnect_delay_s,
    )


__all__ = [
    "CapturedFrame",
    "ConstantEgoSpeed",
    "EgoSpeedSource",
    "FileEgoSpeed",
    "FrameSource",
    "JETSON_CSI_PIPELINE",
    "NoEgoSpeed",
    "OpenCVSource",
    "ReconnectingSource",
    "SimulatedEgoSpeed",
    "SyntheticSource",
    "open_ego_source",
    "open_source",
    "open_source_from_config",
    "parse_source_arg",
]
