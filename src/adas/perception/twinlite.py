"""TwinLiteNet lane + drivable-area backend.

STATUS ON THIS BOARD: **the engine does not exist.**
--------------------------------------------------
The engines workstream built UFLD-v2, YOLOP, YOLOX-nano/tiny, MiDaS and YOLOv5n.
TwinLiteNet was not among them, and ``models/MANIFEST.json`` has no entry for
it. Everything below is the production interface and the real decode, written
against the published TwinLiteNet ONNX contract so the engine can drop in with
no call-site change -- but nothing here has ever been run against real weights.

Two entry points, and the difference matters:

* :class:`TwinLiteNetLaneEstimator` -- the real backend. Its constructor
  **raises** :class:`~adas.core.exceptions.PerceptionError` when the engine file
  is missing. It never degrades into a mock.
* :class:`TwinLiteNetUnavailable` -- the honest stub. ``estimate`` returns
  ``None`` (no lane measurement exists) and ``drivable_area`` returns a
  :class:`~adas.core.models.DrivableArea` with ``mask=None`` and
  ``confidence=0.0``. It never fabricates geometry and never reports a lane.

Expected engine contract (TwinLiteNet, chequeredhat/TwinLiteNet, BSD-3-Clause,
trained on BDD100K)::

    input            1x3x360x640  float32  NCHW RGB
    <drivable head>  1x2x360x640  float32  {not-drivable, drivable} logits
    <lane head>      1x2x360x640  float32  {not-lane, lane} logits

Preprocessing: BGR -> RGB, **stretch** the whole frame to 640x360 (aspect NOT
preserved -- TwinLiteNet has no letterbox), ``/255``, no mean/std.

The two segmentation heads are distinguished by binding name (``da``/``drive``
vs ``ll``/``lane``); if the names are uninformative the constructor raises
rather than guessing, because swapping the drivable mask for the lane mask
produces a plausible-looking and completely wrong free-space signal.

Units and failure behaviour
---------------------------
Identical to :mod:`adas.perception.yolop`: the decode stays at NETWORK
resolution and only the decoded polyline points are scaled back to source
pixels, the lane fit goes through
:func:`~adas.perception.lane.lane_model_from_boundaries`, ``LaneModel`` points
and coefficients are in source pixels, no lane means ``None``, and malformed
input raises :class:`~adas.core.exceptions.PerceptionError`.

To enable it: add a manifest entry with the ONNX URL and sha256, run
``scripts/fetch_models.sh`` then ``scripts/build_engines.py --only twinlitenet``
(both under ``flock /tmp/jetson-gpu.lock``), and point the lane backend at
``models/twinlitenet_360x640.engine``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from adas.core.exceptions import ConfigurationError, PerceptionError
from adas.core.logger import setup_logger
from adas.core.models import DrivableArea, LaneModel
from adas.perception.geometry import CameraConfig, LaneGeometry
from adas.perception.lane import LaneBackend, lane_model_from_boundaries
from adas.perception.yolop import (
    class1_margin,
    drivable_area_from_margin,
    lane_points_from_mask,
    sample_margin_confidence,
)

logger = setup_logger(__name__)

#: Canonical TwinLiteNet input, width x height.
DEFAULT_INPUT_SIZE = (640, 360)

#: Substrings that identify each segmentation head in a binding name.
DRIVABLE_NAME_HINTS = ("da", "drive", "drivable", "seg_da")
LANE_NAME_HINTS = ("ll", "lane", "line", "seg_ll")


def preprocess(image: np.ndarray, size: Tuple[int, int] = DEFAULT_INPUT_SIZE) -> np.ndarray:
    """Build the ``1x3x360x640`` float32 NCHW blob.

    ``image`` must be uint8 HxWx3 BGR. The frame is stretched, not letterboxed,
    so there is no padding to undo: source pixel ``(x, y)`` maps to
    ``(x * W/w, y * H/h)``.
    """
    import cv2

    if not isinstance(image, np.ndarray):
        raise PerceptionError("twinlite preprocess needs a numpy image, got %s" % type(image))
    if image.ndim != 3 or image.shape[2] != 3:
        raise PerceptionError("twinlite preprocess needs HxWx3 BGR, got shape %s" % (image.shape,))
    if image.dtype != np.uint8:
        raise PerceptionError(
            "twinlite preprocess needs a uint8 BGR frame, got dtype %s" % image.dtype
        )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    stretched = cv2.resize(rgb, (int(size[0]), int(size[1])), interpolation=cv2.INTER_LINEAR)
    blob = stretched.astype(np.float32) / 255.0
    return np.ascontiguousarray(blob.transpose(2, 0, 1)[None, ...])


def resize_mask_to_frame(
    prob: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Stretch a network-resolution probability map back to the source frame.

    TwinLiteNet does not letterbox, so this is a plain anisotropic resize -- the
    inverse of :func:`preprocess`.
    """
    import cv2

    return cv2.resize(
        prob,
        (int(frame_width), int(frame_height)),
        interpolation=cv2.INTER_LINEAR,
    )


def _classify_heads(output_names: List[str]) -> Tuple[str, str]:
    """Map binding names onto ``(drivable_name, lane_name)``.

    Raises :class:`~adas.core.exceptions.ConfigurationError` when the names do
    not identify both heads unambiguously. Guessing here would silently swap
    free space for lane markings.
    """
    drivable = [n for n in output_names if any(h in n.lower() for h in DRIVABLE_NAME_HINTS)]
    lane = [n for n in output_names if any(h in n.lower() for h in LANE_NAME_HINTS)]
    # "lane" also matches nothing in DRIVABLE_NAME_HINTS except via "da" inside
    # words; resolve overlaps by preferring the more specific match.
    drivable = [n for n in drivable if n not in lane] or drivable
    lane = [n for n in lane if n not in drivable] or lane
    if len(drivable) != 1 or len(lane) != 1 or drivable[0] == lane[0]:
        raise ConfigurationError(
            "cannot tell the TwinLiteNet drivable head from the lane head. "
            "engine outputs: %s (drivable candidates %s, lane candidates %s). "
            "Re-export with explicit output names." % (output_names, drivable, lane)
        )
    return drivable[0], lane[0]


class TwinLiteNetLaneEstimator(LaneBackend):
    """TwinLiteNet-backed lane estimator with an independent free-space mask.

    Not thread-safe. Raises :class:`~adas.core.exceptions.PerceptionError` at
    construction when the engine is absent -- see the module docstring; it does
    not currently exist on this board.
    """

    name = "twinlitenet"

    def __init__(
        self,
        engine_path: str,
        camera: Optional[CameraConfig] = None,
        min_confidence: float = 0.0,
    ) -> None:
        from adas.infer.trt_engine import TrtEngine

        path = Path(engine_path)
        if not path.exists():
            raise PerceptionError(
                "TwinLiteNet engine not found: %s. No TwinLiteNet engine has been built "
                "on this board; use the 'ufld' or 'yolop' lane backend, or build one "
                "(see adas.perception.twinlite module docstring)." % engine_path
            )
        self.engine = TrtEngine(str(path))
        shape = tuple(int(s) for s in self.engine.input_shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
            self.engine.close()
            raise ConfigurationError(
                "TwinLiteNet engine %s has input shape %s; expected 1x3xHxW" % (path, list(shape))
            )
        self.input_height = shape[2]
        self.input_width = shape[3]
        try:
            self.drivable_name, self.lane_name = _classify_heads(list(self.engine.output_names))
        except ConfigurationError:
            self.engine.close()
            raise
        self.camera = camera
        self.min_confidence = float(min_confidence)
        self._drivable: Optional[DrivableArea] = None
        self.last_geometry: Optional[LaneGeometry] = None
        self.last_lane_mask: Optional[np.ndarray] = None
        logger.info(
            "TwinLiteNetLaneEstimator loaded %s (%dx%d, drivable=%s lane=%s)",
            path,
            self.input_width,
            self.input_height,
            self.drivable_name,
            self.lane_name,
        )
        if camera is None:
            logger.warning(
                "TwinLiteNetLaneEstimator has NO camera model: curvature and metric "
                "lateral offset are unavailable; lane output is pixel-only."
            )

    def estimate(self, frame: object, width: int, height: int) -> Optional[LaneModel]:
        blob = preprocess(frame, (self.input_width, self.input_height))
        outputs = self.engine.infer({self.engine.input_name: blob})
        frame_h, frame_w = frame.shape[:2]
        return self.decode(outputs, frame_w, frame_h)

    def decode(
        self,
        outputs: Dict[str, np.ndarray],
        frame_width: int,
        frame_height: int,
    ) -> Optional[LaneModel]:
        """Decode raw engine outputs. Split out so fixtures can drive it with no GPU.

        Works at NETWORK resolution (360x640) and scales the handful of decoded
        polyline points back to source pixels; TwinLiteNet stretches rather than
        letterboxes, so the two axes have different scales.
        """
        drive_margin = class1_margin(outputs[self.drivable_name])
        lane_margin = class1_margin(outputs[self.lane_name])
        self._drivable = drivable_area_from_margin(drive_margin)
        lane_mask = lane_margin > 0.0
        self.last_lane_mask = lane_mask

        scale_x = float(self.input_width) / float(frame_width)
        scale_y = float(self.input_height) / float(frame_height)
        ego_x_src = self.camera.cx if self.camera is not None else frame_width / 2.0
        left_net, right_net = lane_points_from_mask(lane_mask, ego_x_px=ego_x_src * scale_x)
        if not left_net and not right_net:
            self.last_geometry = None
            return None

        left_conf = sample_margin_confidence(lane_margin, left_net)
        right_conf = sample_margin_confidence(lane_margin, right_net)
        left_pts = [(x / scale_x, y / scale_y) for x, y in left_net]
        right_pts = [(x / scale_x, y / scale_y) for x, y in right_net]

        built = lane_model_from_boundaries(
            left_points_px=left_pts or None,
            right_points_px=right_pts or None,
            frame_width=frame_width,
            frame_height=frame_height,
            camera=self.camera,
            left_confidence=left_conf,
            right_confidence=right_conf,
        )
        if built is None:
            self.last_geometry = None
            return None
        model, geometry = built
        self.last_geometry = geometry
        if model.confidence < self.min_confidence:
            return None
        return model

    def drivable_area(self) -> Optional[DrivableArea]:
        return self._drivable

    def close(self) -> None:
        engine = getattr(self, "engine", None)
        if engine is not None:
            engine.close()
            self.engine = None


class TwinLiteNetUnavailable(LaneBackend):
    """Honest stand-in for the TwinLiteNet backend while no engine exists.

    ``estimate`` returns ``None`` -- there is no lane measurement, and this
    class will not invent one. ``drivable_area`` returns a
    :class:`~adas.core.models.DrivableArea` with ``mask=None`` and
    ``confidence=0.0``; note that ``DrivableArea.is_free`` answers ``True`` for
    a ``None`` mask (its documented "unknown, do not constrain" behaviour), so
    consumers MUST gate on ``confidence > 0`` before using it as evidence of
    free space.

    ``is_mock`` is ``True`` so the flag propagates the same way the mock lane
    estimator's does.
    """

    name = "twinlitenet-unavailable"

    def __init__(self, reason: str = "no TwinLiteNet engine has been built on this board") -> None:
        self.reason = str(reason)
        logger.warning(
            "TwinLiteNet backend selected but unavailable (%s). Lane output will be "
            "None and free space will report confidence 0.0. NOT FOR VEHICLE USE.",
            self.reason,
        )

    @property
    def is_mock(self) -> bool:
        return True

    def estimate(self, frame: object, width: int, height: int) -> Optional[LaneModel]:
        return None

    def drivable_area(self) -> Optional[DrivableArea]:
        return DrivableArea(mask=None, width=0, height=0, confidence=0.0)


def build_twinlite_backend(
    engine_path: str,
    camera: Optional[CameraConfig] = None,
    allow_unavailable: bool = False,
) -> LaneBackend:
    """Return the real backend, or the honest stub when the engine is missing.

    ``allow_unavailable`` must be set explicitly: by default a missing engine is
    a hard error, so a vehicle configuration cannot quietly fall back to "no
    lane, forever".
    """
    if Path(engine_path).exists():
        return TwinLiteNetLaneEstimator(engine_path, camera=camera)
    if allow_unavailable:
        return TwinLiteNetUnavailable("engine file %s not present" % engine_path)
    raise PerceptionError(
        "TwinLiteNet engine not found: %s. Pass allow_unavailable=True only for "
        "non-vehicle use." % engine_path
    )
