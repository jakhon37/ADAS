"""YOLOP panoptic driving perception: lane lines + drivable area + vehicles.

Model
-----
``models/yolop_640.engine`` -- hustvl/YOLOP, MIT code licence, trained on
BDD100K (whose dataset terms are non-commercial; see ``models/MANIFEST.json``).
Static FP16 engine:

===================  ==========================  =====================================
binding              shape                       meaning
===================  ==========================  =====================================
``images``           ``1x3x640x640`` float32     NCHW RGB
``det_out``          ``1x25200x6``   float32     ``[cx, cy, w, h, obj, car]``, already
                                                 decoded to 640-frame pixels, sigmoid
                                                 already applied, single class
``drive_area_seg``   ``1x2x640x640`` float32     {not-drivable, drivable} logits
``lane_line_seg``    ``1x2x640x640`` float32     {not-lane, lane} logits
===================  ==========================  =====================================

Preprocessing: BGR -> RGB, **centred** letterbox to 640x640 with pad value 114,
``/255``, then ImageNet mean/std (unlike YOLOv5, which stops at ``/255``).

Cost
----
26.98 ms GPU median on this Xavier NX. That does NOT fit a 20 Hz per-frame
budget alongside a detector; schedule it at 2-5 Hz from the runtime. This module
deliberately does no internal frame skipping -- silently re-serving a stale lane
model as if it were current is exactly the kind of dishonesty the rest of this
stack is being cleaned of.

Units and failure behaviour
---------------------------
* Masks are returned in SOURCE-frame resolution (the letterbox is undone).
* :class:`~adas.core.models.DrivableArea` carries a downsampled boolean grid
  (:data:`DRIVABLE_GRID_W` x :data:`DRIVABLE_GRID_H`) plus the mean softmax
  probability over the cells it marks free, as its confidence.
* Lane boundaries are extracted from ``lane_line_seg`` by a bottom-up sliding
  window and then go through the shared
  :func:`~adas.perception.lane.lane_model_from_boundaries` fit, so YOLOP and
  UFLD produce identical ``LaneModel`` semantics.
* No lane found -> ``estimate`` returns ``None``. Malformed input raises
  :class:`~adas.core.exceptions.PerceptionError`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from adas.core.exceptions import ConfigurationError, PerceptionError
from adas.core.logger import setup_logger
from adas.core.models import BoundingBox, DrivableArea, LaneModel
from adas.perception.geometry import CameraConfig, LaneGeometry
from adas.perception.lane import LaneBackend, lane_model_from_boundaries

logger = setup_logger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ``(px/255 - mean) / std`` refactored to ``px * SCALE - SHIFT`` so the whole
# normalisation, the BGR->RGB swap and the HWC->CHW transpose happen in one
# fused pass per channel over the uint8 canvas.
_PIXEL_SCALE = (1.0 / (255.0 * IMAGENET_STD)).astype(np.float32)
_PIXEL_SHIFT = (IMAGENET_MEAN / IMAGENET_STD).astype(np.float32)

LETTERBOX_PAD_VALUE = 114

#: Downsampled free-space grid published on ``DrivableArea``. 64x36 keeps the
#: 16:9 aspect and costs 2.3 kB, cheap enough to query per planning tick.
DRIVABLE_GRID_W = 64
DRIVABLE_GRID_H = 36

#: A grid cell counts as free when at least this fraction of its pixels are
#: classified drivable. Deliberately above 0.5 so a cell straddling the road
#: edge is treated as blocked.
DRIVABLE_CELL_THRESHOLD = 0.6


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def letterbox_centred(
    image: np.ndarray,
    size: int,
    pad_value: int = LETTERBOX_PAD_VALUE,
) -> Tuple[np.ndarray, float, int, int]:
    """Aspect-preserving resize into a centred ``size x size`` canvas.

    Returns ``(canvas, scale, pad_x, pad_y)`` where a source pixel ``(x, y)``
    maps to ``(x * scale + pad_x, y * scale + pad_y)``. The canvas keeps the
    input dtype, so a float frame is not truncated into a uint8 buffer.
    """
    import cv2

    h, w = image.shape[:2]
    scale = min(float(size) / float(h), float(size) / float(w))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, image.shape[2]), pad_value, dtype=image.dtype)
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def preprocess(image: np.ndarray, size: int = 640) -> Tuple[np.ndarray, float, int, int]:
    """Build the YOLOP input blob. ``image`` must be uint8 HxWx3 BGR.

    Returns ``(blob, scale, pad_x, pad_y)``; the letterbox parameters are needed
    to undo the padding on the segmentation masks and the boxes.
    """
    import cv2

    if not isinstance(image, np.ndarray):
        raise PerceptionError("yolop preprocess needs a numpy image, got %s" % type(image))
    if image.ndim != 3 or image.shape[2] != 3:
        raise PerceptionError("yolop preprocess needs HxWx3 BGR, got shape %s" % (image.shape,))
    if image.dtype != np.uint8:
        raise PerceptionError(
            "yolop preprocess needs a uint8 BGR frame, got dtype %s" % image.dtype
        )
    canvas, scale, pad_x, pad_y = letterbox_centred(image, size)
    blob = np.empty((1, 3, size, size), dtype=np.float32)
    for c in range(3):
        # ``2 - c`` performs the BGR -> RGB swap inside the same gather that
        # writes the NCHW plane, so no separate cvtColor pass is needed.
        np.multiply(canvas[:, :, 2 - c], _PIXEL_SCALE[c], out=blob[0, c], dtype=np.float32)
        blob[0, c] -= _PIXEL_SHIFT[c]
    return blob, scale, pad_x, pad_y


# --------------------------------------------------------------------------- #
# Segmentation decode
# --------------------------------------------------------------------------- #


def _softmax2(logits: np.ndarray) -> np.ndarray:
    """Softmax over axis 0 of a ``2xHxW`` logit volume; returns the class-1 map."""
    z = logits - logits.max(axis=0, keepdims=True)
    e = np.exp(z)
    return (e[1] / (e.sum(axis=0) + 1e-12)).astype(np.float32)


def class1_margin(logits: np.ndarray) -> np.ndarray:
    """``logit[1] - logit[0]`` for a 2-class head.

    The sign is the argmax (``> 0`` means class 1) and ``sigmoid`` of it is the
    softmax probability of class 1, so one subtraction replaces a full softmax.
    Used because the two YOLOP heads are 640x640 each and an exp over 819k
    elements twice per frame is pure waste when only a handful of sampled
    probabilities are ever read.
    """
    arr = np.asarray(logits, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[0] != 2:
        raise PerceptionError(
            "expected a 2xHxW segmentation head, got shape %s" % (arr.shape,)
        )
    return arr[1] - arr[0]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float32)))).astype(np.float32)


def crop_letterbox(
    plane: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Cut the padding off a network-resolution plane, leaving only the frame.

    The result still lives in NETWORK pixels: a source pixel ``(x, y)`` sits at
    ``(x * scale, y * scale)`` in it. Nothing is upsampled -- resizing a
    640x640 float plane to 1280x720 costs more than everything else in the
    decode put together, and the lane fit does not need it.
    """
    h, w = plane.shape[:2]
    nw = min(w - int(pad_x), max(1, int(round(frame_width * scale))))
    nh = min(h - int(pad_y), max(1, int(round(frame_height * scale))))
    cropped = plane[int(pad_y) : int(pad_y) + nh, int(pad_x) : int(pad_x) + nw]
    if cropped.size == 0:
        raise PerceptionError(
            "letterbox crop is empty (scale=%.4f pad=%d,%d plane=%s)"
            % (scale, pad_x, pad_y, plane.shape)
        )
    return cropped


def unletterbox_mask(
    mask: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    frame_width: int,
    frame_height: int,
    interpolation: Optional[int] = None,
) -> np.ndarray:
    """Undo the centred letterbox and resize a plane to SOURCE resolution.

    Only needed for visualisation and for callers that want a per-source-pixel
    mask; the decode path itself stays in network resolution.
    """
    import cv2

    cropped = crop_letterbox(mask, scale, pad_x, pad_y, frame_width, frame_height)
    if interpolation is None:
        interpolation = cv2.INTER_NEAREST if cropped.dtype == np.uint8 else cv2.INTER_LINEAR
    return cv2.resize(cropped, (int(frame_width), int(frame_height)), interpolation=interpolation)


def drivable_area_from_margin(
    margin: np.ndarray,
    grid_w: int = DRIVABLE_GRID_W,
    grid_h: int = DRIVABLE_GRID_H,
    cell_threshold: float = DRIVABLE_CELL_THRESHOLD,
) -> DrivableArea:
    """Downsample a drivable-class margin map to the boolean query grid.

    ``margin`` is ``logit[drivable] - logit[not drivable]`` over the region of
    the network input that corresponds to the source frame (i.e. after
    :func:`crop_letterbox`). A cell is free when at least ``cell_threshold`` of
    its pixels are drivable; confidence is the mean softmax probability over the
    free cells. An empty free set yields an all-blocked grid with confidence
    0.0, which is the fail-safe direction.
    """
    import cv2

    binary = (margin > 0.0).astype(np.float32)
    coarse = cv2.resize(binary, (int(grid_w), int(grid_h)), interpolation=cv2.INTER_AREA)
    free = coarse >= float(cell_threshold)
    if free.any():
        coarse_margin = cv2.resize(
            margin.astype(np.float32), (int(grid_w), int(grid_h)), interpolation=cv2.INTER_AREA
        )
        confidence = float(np.clip(float(_sigmoid(coarse_margin[free]).mean()), 0.0, 1.0))
    else:
        confidence = 0.0
    return DrivableArea(
        mask=free,
        width=int(grid_w),
        height=int(grid_h),
        confidence=confidence,
    )


def drivable_area_from_probability(
    prob: np.ndarray,
    grid_w: int = DRIVABLE_GRID_W,
    grid_h: int = DRIVABLE_GRID_H,
    cell_threshold: float = DRIVABLE_CELL_THRESHOLD,
) -> DrivableArea:
    """As :func:`drivable_area_from_margin` but taking a probability map in [0, 1]."""
    eps = 1e-6
    clipped = np.clip(np.asarray(prob, dtype=np.float32), eps, 1.0 - eps)
    return drivable_area_from_margin(
        np.log(clipped / (1.0 - clipped)), grid_w, grid_h, cell_threshold
    )


class _TrackedSide:
    """Sliding-window state for one lane boundary, with slope momentum.

    Following the previous x alone loses the line as soon as it slants, which on
    a 640-wide image at the top of the frame it always does. Predicting the next
    anchor from the running slope keeps the gate tight (so a neighbouring line
    cannot capture it) while still following a curve.
    """

    __slots__ = ("x", "y", "slope", "misses", "done", "points")

    def __init__(self) -> None:
        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self.slope = 0.0
        self.misses = 0
        self.done = False
        self.points: List[Tuple[float, float]] = []

    def seed(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.slope = 0.0
        self.misses = 0
        self.points.append((float(x), float(y)))

    def predict(self, y: float) -> float:
        return self.x + self.slope * (float(y) - self.y)

    def accept(self, x: float, y: float) -> None:
        dy = float(y) - self.y
        if abs(dy) > 1e-6:
            instant = (float(x) - self.x) / dy
            self.slope = 0.5 * self.slope + 0.5 * instant
        self.x = float(x)
        self.y = float(y)
        self.misses = 0
        self.points.append((float(x), float(y)))

    def miss(self, max_misses: int) -> None:
        self.misses += 1
        if self.misses >= int(max_misses):
            self.done = True


def lane_points_from_mask(
    mask: np.ndarray,
    ego_x_px: Optional[float] = None,
    num_bands: int = 24,
    top_frac: float = 0.45,
    min_run_px: int = 2,
    min_band_pixels: int = 4,
    search_px: Optional[float] = None,
    max_consecutive_misses: int = 3,
    min_points: int = 4,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Extract the left and right ego-lane boundaries from a binary lane mask.

    Bottom-up sliding window. The frame below ``top_frac`` of its height is cut
    into ``num_bands`` horizontal bands; in each band the mask columns are
    summed, runs of at least ``min_run_px`` lit columns become candidate
    centroids, and each side follows the candidate nearest its *predicted*
    position (previous anchor advanced by the running slope) within
    ``search_px``. A side that misses ``max_consecutive_misses`` bands in a row
    stops -- following a gap further just invents geometry.

    Seeding prefers the lowest band that offers a candidate on BOTH sides of
    ``ego_x_px``, so the two boundaries are picked as a pair from the same row
    rather than independently from different rows. Failing that, each side seeds
    from the lowest band that offers it a candidate.

    ``ego_x_px`` should be the camera principal point when one is known; the
    image centre is only correct for a perfectly centred camera. Coordinates are
    in the same pixel space as ``mask``.

    Returns ``(left_points, right_points)``; a side with fewer than
    ``min_points`` accepted points is returned empty, because three points on a
    dashed line fit a quadratic that means nothing.
    """
    if mask.ndim != 2:
        raise PerceptionError("lane mask must be 2-D, got shape %s" % (mask.shape,))
    height, width = mask.shape[:2]
    binary = mask.astype(bool)
    if ego_x_px is None:
        ego_x_px = width / 2.0
    if search_px is None:
        search_px = max(12.0, 0.05 * width)

    y_top = int(max(0, min(height - 2, round(height * float(top_frac)))))
    band_edges = np.linspace(height, y_top, int(num_bands) + 1).astype(int)

    bands: List[Tuple[float, List[float]]] = []
    for i in range(int(num_bands)):
        y1 = int(band_edges[i])
        y0 = int(band_edges[i + 1])
        if y1 - y0 < 1:
            continue
        counts = binary[y0:y1, :].sum(axis=0)
        y_mid = (y0 + y1) / 2.0
        if int(counts.sum()) < int(min_band_pixels):
            bands.append((y_mid, []))
            continue
        bands.append((y_mid, _run_centroids(counts >= 1, counts, int(min_run_px))))

    left = _TrackedSide()
    right = _TrackedSide()

    # Paired seeding: the lowest band with candidates either side of the ego x.
    for y_mid, centroids in bands:
        below = [c for c in centroids if c < ego_x_px]
        above = [c for c in centroids if c > ego_x_px]
        if below and above:
            left.seed(max(below), y_mid)
            right.seed(min(above), y_mid)
            break

    for y_mid, centroids in bands:
        for side, want_left in ((left, True), (right, False)):
            if side.done:
                continue
            if side.y is not None and y_mid >= side.y:
                continue  # already consumed by the paired seed
            if not centroids:
                if side.x is not None:
                    side.miss(max_consecutive_misses)
                continue
            if side.x is None:
                pool = [c for c in centroids if (c < ego_x_px) == want_left]
                if not pool:
                    continue
                side.seed(max(pool) if want_left else min(pool), y_mid)
                continue
            predicted = side.predict(y_mid)
            pick = min(centroids, key=lambda c: abs(c - predicted))
            gate = float(search_px) * (1.0 + 0.5 * side.misses)
            if abs(pick - predicted) <= gate:
                side.accept(pick, y_mid)
            else:
                side.miss(max_consecutive_misses)
        if left.done and right.done:
            break

    left_points = left.points if len(left.points) >= int(min_points) else []
    right_points = right.points if len(right.points) >= int(min_points) else []
    return left_points, right_points


def _run_centroids(lit: np.ndarray, counts: np.ndarray, min_run_px: int) -> List[float]:
    """Intensity-weighted centroids of the contiguous runs of lit columns."""
    centroids: List[float] = []
    lit = np.asarray(lit, dtype=bool)
    if not lit.any():
        return centroids
    padded = np.concatenate(([False], lit, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    weights = counts.astype(np.float64)
    for start, stop in zip(edges[0::2], edges[1::2]):
        if stop - start < int(min_run_px):
            continue
        w = weights[start:stop]
        total = float(w.sum())
        if total <= 0.0:
            continue
        idx = np.arange(start, stop, dtype=np.float64)
        centroids.append(float((idx * w).sum() / total))
    return centroids


# --------------------------------------------------------------------------- #
# Detection decode
# --------------------------------------------------------------------------- #


def nms_single_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    max_detections: int = 100,
) -> List[int]:
    """Greedy NMS over xyxy boxes. YOLOP has one class, so class-agnostic is correct."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1][: max(1, int(max_detections) * 10)]
    keep: List[int] = []
    while order.size > 0 and len(keep) < int(max_detections):
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0.0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= float(iou_threshold)]
    return keep


def decode_detections(
    det_out: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    frame_width: int,
    frame_height: int,
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
    label: str = "car",
) -> List[BoundingBox]:
    """Decode ``det_out`` (``1x25200x6``) to source-frame boxes.

    The rows are ``[cx, cy, w, h, obj, car]`` already in 640-letterbox pixels
    with sigmoid applied, so the only work is score gating, un-letterboxing and
    NMS. Boxes are clipped to the frame; callers that need truncation flags
    should use :func:`adas.perception.geometry.truncation_flags` on the result.
    """
    arr = np.asarray(det_out, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise PerceptionError(
            "YOLOP det_out must be (N, 6) after dropping the batch dim, got %s" % (arr.shape,)
        )
    obj = arr[:, 4]
    cls = arr[:, 5]
    scores = obj * cls
    keep_mask = scores >= float(conf_threshold)
    if not keep_mask.any():
        return []
    sel = arr[keep_mask]
    sel_scores = scores[keep_mask]
    cx, cy, bw, bh = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
    x1 = (cx - bw / 2.0 - pad_x) / scale
    y1 = (cy - bh / 2.0 - pad_y) / scale
    x2 = (cx + bw / 2.0 - pad_x) / scale
    y2 = (cy + bh / 2.0 - pad_y) / scale
    boxes = np.stack(
        [
            np.clip(x1, 0.0, float(frame_width - 1)),
            np.clip(y1, 0.0, float(frame_height - 1)),
            np.clip(x2, 0.0, float(frame_width - 1)),
            np.clip(y2, 0.0, float(frame_height - 1)),
        ],
        axis=1,
    )
    keep = nms_single_class(boxes, sel_scores, iou_threshold, max_detections)
    out: List[BoundingBox] = []
    for i in keep:
        bx = boxes[i]
        if bx[2] <= bx[0] or bx[3] <= bx[1]:
            continue
        out.append(
            BoundingBox(
                x1=float(bx[0]),
                y1=float(bx[1]),
                x2=float(bx[2]),
                y2=float(bx[3]),
                confidence=float(min(1.0, max(0.0, sel_scores[i]))),
                label=label,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #


class YolopLaneEstimator(LaneBackend):
    """YOLOP-backed lane estimator that also publishes free space and vehicles.

    Not thread-safe (one TensorRT execution context, shared host buffers).

    A single :meth:`estimate` call runs the whole network; the drivable-area
    mask and the vehicle boxes from that same call are then available through
    :meth:`drivable_area` and :meth:`detections` with no extra inference.
    """

    name = "yolop"

    def __init__(
        self,
        engine_path: str,
        camera: Optional[CameraConfig] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        min_confidence: float = 0.0,
    ) -> None:
        from adas.infer.trt_engine import TrtEngine

        path = Path(engine_path)
        if not path.exists():
            raise PerceptionError("YOLOP engine not found: %s" % engine_path)
        self.engine = TrtEngine(str(path))
        shape = tuple(int(s) for s in self.engine.input_shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3 or shape[2] != shape[3]:
            self.engine.close()
            raise ConfigurationError(
                "YOLOP engine %s has input shape %s; expected 1x3xSxS" % (path, list(shape))
            )
        self.input_size = int(shape[2])
        self.camera = camera
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.min_confidence = float(min_confidence)
        self._drivable: Optional[DrivableArea] = None
        self._detections: List[BoundingBox] = []
        self.last_geometry: Optional[LaneGeometry] = None
        self.last_lane_mask: Optional[np.ndarray] = None
        #: Source -> network scale factor for ``last_lane_mask`` (network pixels
        #: per source pixel). The mask is NOT at source resolution.
        self.last_lane_mask_scale: float = 1.0

        names = set(self.engine.output_names)
        required = {"det_out", "drive_area_seg", "lane_line_seg"}
        if not required.issubset(names):
            self.engine.close()
            raise ConfigurationError(
                "YOLOP engine %s outputs %s; expected %s"
                % (path, sorted(names), sorted(required))
            )
        logger.info(
            "YolopLaneEstimator loaded %s (input %dx%d, ~27 ms/frame - schedule at 2-5 Hz)",
            path,
            self.input_size,
            self.input_size,
        )
        if camera is None:
            logger.warning(
                "YolopLaneEstimator has NO camera model: curvature and metric lateral "
                "offset are unavailable; lane output is pixel-only."
            )

    def estimate(self, frame: object, width: int, height: int) -> Optional[LaneModel]:
        blob, scale, pad_x, pad_y = preprocess(frame, self.input_size)
        outputs = self.engine.infer({self.engine.input_name: blob})
        frame_h, frame_w = frame.shape[:2]
        return self.decode(outputs, scale, pad_x, pad_y, frame_w, frame_h)

    def decode(
        self,
        outputs: Dict[str, np.ndarray],
        scale: float,
        pad_x: int,
        pad_y: int,
        frame_width: int,
        frame_height: int,
    ) -> Optional[LaneModel]:
        """Decode raw engine outputs. Split out so fixtures can drive it with no GPU.

        Everything happens at NETWORK resolution: the two 640x640 heads are
        cropped to the letterbox interior and the lane polylines are extracted
        there, then the handful of resulting points is divided by ``scale`` to
        return to source pixels. Upsampling either mask to 1280x720 first cost
        more than the rest of the decode combined and bought nothing.
        """
        drive_margin = class1_margin(outputs["drive_area_seg"])
        lane_margin = class1_margin(outputs["lane_line_seg"])
        drive_crop = crop_letterbox(drive_margin, scale, pad_x, pad_y, frame_width, frame_height)
        lane_crop = crop_letterbox(lane_margin, scale, pad_x, pad_y, frame_width, frame_height)

        self._drivable = drivable_area_from_margin(drive_crop)
        lane_mask = lane_crop > 0.0
        self.last_lane_mask = lane_mask
        self.last_lane_mask_scale = float(scale)

        if "det_out" in outputs:
            self._detections = decode_detections(
                outputs["det_out"],
                scale,
                pad_x,
                pad_y,
                frame_width,
                frame_height,
                self.conf_threshold,
                self.iou_threshold,
                self.max_detections,
            )
        else:
            self._detections = []

        ego_x_src = self.camera.cx if self.camera is not None else frame_width / 2.0
        left_net, right_net = lane_points_from_mask(lane_mask, ego_x_px=ego_x_src * scale)
        if not left_net and not right_net:
            self.last_geometry = None
            return None

        left_conf = sample_margin_confidence(lane_crop, left_net)
        right_conf = sample_margin_confidence(lane_crop, right_net)
        left_pts = [(x / scale, y / scale) for x, y in left_net]
        right_pts = [(x / scale, y / scale) for x, y in right_net]

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

    def detections(self) -> List[BoundingBox]:
        """Vehicle boxes from the most recent :meth:`estimate`, in source pixels."""
        return list(self._detections)

    def close(self) -> None:
        engine = getattr(self, "engine", None)
        if engine is not None:
            engine.close()
            self.engine = None


def _sample_confidence(prob: np.ndarray, points: Sequence[Tuple[float, float]]) -> float:
    """Mean class probability sampled at the polyline points, in ``[0, 1]``."""
    if not points:
        return 0.0
    h, w = prob.shape[:2]
    total = 0.0
    for x, y in points:
        xi = int(min(w - 1, max(0, round(x))))
        yi = int(min(h - 1, max(0, round(y))))
        total += float(prob[yi, xi])
    return float(min(1.0, max(0.0, total / len(points))))


def sample_margin_confidence(
    margin: np.ndarray,
    points: Sequence[Tuple[float, float]],
) -> float:
    """Mean ``sigmoid(margin)`` at the polyline points, in ``[0, 1]``.

    ``points`` must be in the same pixel space as ``margin``. Returns 0.0 for an
    empty polyline -- no points is no evidence, not full confidence.
    """
    if not points:
        return 0.0
    h, w = margin.shape[:2]
    xs = np.clip(np.round([p[0] for p in points]).astype(int), 0, w - 1)
    ys = np.clip(np.round([p[1] for p in points]).astype(int), 0, h - 1)
    values = _sigmoid(margin[ys, xs])
    return float(min(1.0, max(0.0, float(values.mean()))))
