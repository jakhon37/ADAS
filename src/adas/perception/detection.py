"""Detector abstraction plus the primitives every detector backend shares.

What lives here
---------------
* The COCO class tables and the named class sets an integrator selects from.
  A detector that cannot see a pedestrian is not an ADAS, so ``person`` is in
  the default set.
* :class:`PreprocSpec` -- an explicit, per-model declaration of colour order,
  scaling, mean/std and padding. Every backend states its contract instead of
  inheriting one module's hardcoded convention. Feeding YOLOX ``/255`` RGB, or
  YOLOv5 raw BGR, produces zero detections and no error; this dataclass is the
  guard against that.
* :class:`LetterboxTransform` -- the forward and, crucially, the *inverse*
  mapping between source-frame pixels and network-input pixels.
* NMS, including the class-aware variant, and the shared "rows of
  ``(cx, cy, w, h, score, class)`` to :class:`~adas.core.models.BoundingBox`"
  tail used by every backend.
* :func:`analyse_detection_head` -- the engine output contract check. The
  layout is derived from the *declared* shapes and validated, never guessed
  from "whichever output tensor is biggest".
* :class:`ObjectDetector` -- the deterministic mock, which is honest about
  being a mock.

Units and conventions
---------------------
Every coordinate is pixels. Source-frame coordinates use ``u`` right / ``v``
down with the origin at the top-left of the **capture** frame, which is what
:mod:`adas.perception.geometry` and :mod:`adas.tracking` expect. Network-input
coordinates are pixels of the square letterboxed blob. Confidences are in
``[0, 1]``.

Failure behaviour
-----------------
Anything that cannot be decoded raises
:class:`~adas.core.exceptions.PerceptionError` with both the observed and the
expected shape. Nothing in this module returns a plausible constant, and
nothing silently reinterprets a tensor it does not recognise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from adas.core.exceptions import PerceptionError
from adas.core.logger import setup_logger
from adas.core.models import BoundingBox

logger = setup_logger(__name__)


# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #

#: The 80 COCO class names in training order. Index == class id.
COCO_CLASSES: Tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

#: Road users an ACC / AEB stack must see. ``person`` (0) is a pedestrian and is
#: the reason this table exists: the previous ``COCO_VEHICLE`` omitted it while
#: its own comment claimed VRU coverage.
COCO_ROAD_USERS: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck",
}

#: Static traffic-control objects. **Not** in the default set: the tracker
#: range-estimates everything it is given and a planner that treats a traffic
#: light as a lead vehicle brakes for nothing. Opt in explicitly when a
#: consumer exists.
COCO_TRAFFIC_CONTROL: Dict[int, str] = {
    9: "traffic light",
    11: "stop sign",
}

#: Backwards-compatible alias for the old vehicle-only table. Deprecated: it
#: excludes pedestrians. Kept so existing imports keep working.
COCO_VEHICLE: Dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

_ROAD_USERS_AND_SIGNS = dict(COCO_ROAD_USERS)
_ROAD_USERS_AND_SIGNS.update(COCO_TRAFFIC_CONTROL)

#: Named class sets an integrator may select by string.
CLASS_SETS: Dict[str, Tuple[int, ...]] = {
    "road_users": tuple(sorted(COCO_ROAD_USERS)),
    "vehicles": tuple(sorted(COCO_VEHICLE)),
    "traffic_control": tuple(sorted(COCO_TRAFFIC_CONTROL)),
    "road_users_and_signs": tuple(sorted(_ROAD_USERS_AND_SIGNS)),
    "all": tuple(range(len(COCO_CLASSES))),
}

#: What a detector filters to when nothing else is configured.
DEFAULT_CLASS_IDS: Tuple[int, ...] = CLASS_SETS["road_users"]


def resolve_class_ids(spec: object, num_classes: int = len(COCO_CLASSES)) -> Tuple[int, ...]:
    """Normalise a class-set specification to a sorted tuple of valid ids.

    ``spec`` may be ``None`` (the default road-user set), the name of an entry
    in :data:`CLASS_SETS`, or any iterable of integer ids. Raises
    :class:`~adas.core.exceptions.PerceptionError` for an unknown name, an
    empty selection, or an id outside ``[0, num_classes)`` -- a silently
    dropped out-of-range id would look exactly like a model that never fires
    for that class.
    """
    if spec is None:
        ids: Iterable[int] = DEFAULT_CLASS_IDS
    elif isinstance(spec, str):
        key = spec.strip().lower()
        if key not in CLASS_SETS:
            raise PerceptionError(
                "unknown class set %r; known sets are %s" % (spec, sorted(CLASS_SETS))
            )
        ids = CLASS_SETS[key]
    else:
        ids = spec
    try:
        resolved = sorted({int(i) for i in ids})
    except (TypeError, ValueError) as exc:
        raise PerceptionError("class ids must be integers, got %r" % (spec,)) from exc
    if not resolved:
        raise PerceptionError("class id selection is empty")
    bad = [i for i in resolved if i < 0 or i >= int(num_classes)]
    if bad:
        raise PerceptionError(
            "class ids %s are outside [0, %d) for this model" % (bad, int(num_classes))
        )
    return tuple(resolved)


def class_name(class_id: int) -> str:
    """COCO name for ``class_id``, or the numeric id as a string if unknown."""
    index = int(class_id)
    if 0 <= index < len(COCO_CLASSES):
        return COCO_CLASSES[index]
    return str(index)


# --------------------------------------------------------------------------- #
# Preprocessing contract
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PreprocSpec:
    """How one model wants its input built, stated explicitly.

    Attributes
    ----------
    colour:
        ``"rgb"`` or ``"bgr"``. Source frames are always BGR (OpenCV), so
        ``"rgb"`` means a channel swap is applied.
    divisor:
        Value every pixel is divided by. ``255.0`` for the ``[0,1]`` models,
        ``1.0`` for YOLOX and SFace which want raw ``0..255``.
    mean, std:
        Per-channel, applied *after* the division, in the model's own channel
        order. ``None`` means no shift / no scaling.
    pad_value:
        Letterbox fill, in 0..255 source units.
    pad_mode:
        ``"center"`` (YOLOv5/YOLOP), ``"topleft"`` (YOLOX), or ``"stretch"``
        (MiDaS, UFLD) which ignores aspect ratio entirely and does not pad.
    interpolation:
        ``"linear"``, ``"area"`` or ``"cubic"``. This is a *contract*, not a
        taste: the reference implementations these engines were validated
        against use bilinear, and INTER_AREA is also 9x slower on this board at
        a 0.33 downscale (11.9 ms vs 1.4 ms for 1280x720 -> 416).
    """

    colour: str = "rgb"
    divisor: float = 255.0
    mean: Optional[Tuple[float, float, float]] = None
    std: Optional[Tuple[float, float, float]] = None
    pad_value: int = 114
    pad_mode: str = "center"
    interpolation: str = "linear"

    def __post_init__(self) -> None:
        if self.colour not in ("rgb", "bgr"):
            raise PerceptionError("PreprocSpec.colour must be 'rgb' or 'bgr', got %r" % (self.colour,))
        if self.pad_mode not in ("center", "topleft", "stretch"):
            raise PerceptionError(
                "PreprocSpec.pad_mode must be 'center', 'topleft' or 'stretch', got %r"
                % (self.pad_mode,)
            )
        if not (self.divisor > 0.0):
            raise PerceptionError("PreprocSpec.divisor must be > 0, got %r" % (self.divisor,))
        if self.interpolation not in ("linear", "area", "cubic", "nearest"):
            raise PerceptionError(
                "PreprocSpec.interpolation must be linear|area|cubic|nearest, got %r"
                % (self.interpolation,)
            )

    @property
    def swap_rb(self) -> bool:
        """True when a BGR source frame must have its channels swapped."""
        return self.colour == "rgb"


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: YOLOv5 / YOLOv8 (Ultralytics): RGB, ``/255``, centred letterbox, pad 114.
PREPROC_YOLOV5 = PreprocSpec(colour="rgb", divisor=255.0, pad_value=114, pad_mode="center")
#: YOLOX: **BGR**, **raw 0..255**, top-left letterbox, pad 114. Dividing by 255
#: silences the model completely -- verified on this board: 10 detections
#: became 0 with no error raised.
PREPROC_YOLOX = PreprocSpec(colour="bgr", divisor=1.0, pad_value=114, pad_mode="topleft")
#: MiDaS v2.1 small: RGB, ``/255``, ImageNet mean/std, anisotropic stretch.
PREPROC_MIDAS = PreprocSpec(
    colour="rgb",
    divisor=255.0,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    pad_mode="stretch",
    interpolation="cubic",
)


def cv_interpolation(name: str) -> int:
    """Map a :class:`PreprocSpec` interpolation name to an OpenCV flag."""
    import cv2

    flags = {
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
    }
    try:
        return flags[name]
    except KeyError:
        raise PerceptionError("unknown interpolation %r" % (name,)) from None


# --------------------------------------------------------------------------- #
# Letterbox
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LetterboxTransform:
    """Forward and inverse mapping between source pixels and network pixels.

    ``u_net = u_src * scale + pad_x`` and ``v_net = v_src * scale + pad_y``.
    The inverse is what turns a decoded box back into a source-frame box, and
    getting the pad term wrong shifts every detection by up to half the padding
    -- silently, because the boxes still look like boxes.
    """

    scale: float
    pad_x: float
    pad_y: float
    net_size: int
    src_width: int
    src_height: int

    def to_net(self, u_src: float, v_src: float) -> Tuple[float, float]:
        """Map one source pixel into network-input coordinates."""
        return (u_src * self.scale + self.pad_x, v_src * self.scale + self.pad_y)

    def to_source(self, u_net: float, v_net: float) -> Tuple[float, float]:
        """Map one network-input pixel back to source coordinates."""
        return ((u_net - self.pad_x) / self.scale, (v_net - self.pad_y) / self.scale)

    def boxes_to_source(self, xyxy_net: np.ndarray) -> np.ndarray:
        """Vectorised inverse for an ``(N, 4)`` array of network-space xyxy boxes.

        Returns a new float32 array; the caller is responsible for clipping.
        """
        arr = np.asarray(xyxy_net, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise PerceptionError("boxes_to_source expects (N,4), got %s" % (arr.shape,))
        out = np.empty_like(arr)
        inv = np.float32(1.0 / self.scale)
        out[:, 0] = (arr[:, 0] - np.float32(self.pad_x)) * inv
        out[:, 1] = (arr[:, 1] - np.float32(self.pad_y)) * inv
        out[:, 2] = (arr[:, 2] - np.float32(self.pad_x)) * inv
        out[:, 3] = (arr[:, 3] - np.float32(self.pad_y)) * inv
        return out


def letterbox_transform(
    src_width: int,
    src_height: int,
    net_size: int,
    pad_mode: str = "center",
) -> LetterboxTransform:
    """Compute the letterbox parameters without touching pixels."""
    if src_width <= 0 or src_height <= 0:
        raise PerceptionError("invalid source size %dx%d" % (src_width, src_height))
    if net_size <= 0:
        raise PerceptionError("invalid network size %r" % (net_size,))
    scale = min(net_size / float(src_height), net_size / float(src_width))
    new_h = int(round(src_height * scale))
    new_w = int(round(src_width * scale))
    if pad_mode == "topleft":
        pad_x = 0.0
        pad_y = 0.0
    elif pad_mode == "center":
        pad_x = float((net_size - new_w) // 2)
        pad_y = float((net_size - new_h) // 2)
    else:
        raise PerceptionError("letterbox pad_mode must be 'center' or 'topleft', got %r" % (pad_mode,))
    return LetterboxTransform(
        scale=float(scale),
        pad_x=pad_x,
        pad_y=pad_y,
        net_size=int(net_size),
        src_width=int(src_width),
        src_height=int(src_height),
    )


def require_bgr_uint8(image: object) -> np.ndarray:
    """Validate a capture frame: uint8, HxWx3, BGR.

    Every source in :mod:`adas.runtime.capture` yields uint8 BGR, and the
    preprocessing here assumes it. Silently accepting a float frame used to
    truncate it into a uint8 canvas -- a normalised ``[0,1]`` frame collapsed to
    all zeros and the detector returned nothing, with no error anywhere.
    """
    if not hasattr(image, "shape"):
        raise PerceptionError("expected an image array, got %s" % type(image).__name__)
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise PerceptionError("expected an HxWx3 BGR image, got shape %s" % (array.shape,))
    if array.dtype != np.uint8:
        raise PerceptionError(
            "expected a uint8 BGR image, got dtype %s. Convert at the capture "
            "boundary; this module will not guess a scale." % (array.dtype,)
        )
    return array


def letterbox_canvas(
    image: np.ndarray,
    net_size: int,
    pad_value: int = 114,
    pad_mode: str = "center",
    interpolation: str = "linear",
) -> Tuple[np.ndarray, LetterboxTransform]:
    """Resize preserving aspect ratio and pad to ``net_size`` square.

    Returns the uint8 canvas (still in the input's channel order) and the
    transform needed to invert the mapping.
    """
    import cv2

    array = require_bgr_uint8(image)
    src_h, src_w = array.shape[:2]
    transform = letterbox_transform(src_w, src_h, net_size, pad_mode=pad_mode)
    new_w = int(round(src_w * transform.scale))
    new_h = int(round(src_h * transform.scale))
    new_w = max(1, min(net_size, new_w))
    new_h = max(1, min(net_size, new_h))
    resized = cv2.resize(array, (new_w, new_h), interpolation=cv_interpolation(interpolation))
    if new_w == net_size and new_h == net_size:
        return resized, transform
    canvas = np.full((net_size, net_size, 3), int(pad_value), dtype=np.uint8)
    top = int(transform.pad_y)
    left = int(transform.pad_x)
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, transform


def blob_from_canvas(
    canvas: np.ndarray,
    spec: PreprocSpec,
    out_dtype: object = np.float32,
) -> np.ndarray:
    """Turn a uint8 BGR canvas into a ``(1, 3, H, W)`` NCHW blob.

    Uses ``cv2.dnn.blobFromImage`` (a single C++ pass) when available, and an
    equivalent numpy path otherwise. ``spec.mean``/``spec.std`` are applied
    after the division, in the model's own channel order.

    ``out_dtype`` should be the engine's declared input dtype. For a float16
    engine the conversion is done here with ``cv2.convertFp16`` (0.55 ms for a
    1x3x640x640 blob) rather than left to a numpy cast inside the engine
    wrapper (5.5 ms for the same blob) -- measured on this Xavier NX.
    """
    import cv2

    scale = 1.0 / float(spec.divisor)
    blob = None
    dnn = getattr(cv2, "dnn", None)
    if dnn is not None:
        blob = dnn.blobFromImage(
            canvas,
            scalefactor=scale,
            size=(canvas.shape[1], canvas.shape[0]),
            mean=(0.0, 0.0, 0.0),
            swapRB=spec.swap_rb,
            crop=False,
        )
    else:  # pragma: no cover - OpenCV on this board always ships dnn
        source = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB) if spec.swap_rb else canvas
        blob = (source.astype(np.float32) * np.float32(scale)).transpose(2, 0, 1)[None, ...]
        blob = np.ascontiguousarray(blob)
    if spec.mean is not None:
        blob -= np.asarray(spec.mean, dtype=np.float32).reshape(1, 3, 1, 1)
    if spec.std is not None:
        blob /= np.asarray(spec.std, dtype=np.float32).reshape(1, 3, 1, 1)
    target = np.dtype(out_dtype)
    if target == np.float32:
        return blob
    if target == np.float16 and hasattr(cv2, "convertFp16"):
        shape = blob.shape
        # cv2.convertFp16 produces CV_16F, which the Python bindings hand back
        # as **int16** carrying the raw half-precision bit pattern. Reading it
        # without the view() is not an error, it is 15000-valued garbage that
        # the network happily consumes and returns nothing for.
        half = cv2.convertFp16(blob.reshape(-1, 1))
        if half.dtype == np.float16:
            return half.reshape(shape)
        if half.dtype == np.int16:
            return half.view(np.float16).reshape(shape)
        logger.warning(
            "cv2.convertFp16 returned dtype %s; falling back to a numpy cast", half.dtype
        )
    return blob.astype(target, copy=False)


def preprocess_letterbox(
    image: np.ndarray,
    net_size: int,
    spec: PreprocSpec,
    out_dtype: object = np.float32,
) -> Tuple[np.ndarray, LetterboxTransform]:
    """Full source-frame -> NCHW blob path for a letterboxing detector."""
    canvas, transform = letterbox_canvas(
        image,
        net_size,
        pad_value=spec.pad_value,
        pad_mode=spec.pad_mode,
        interpolation=spec.interpolation,
    )
    return blob_from_canvas(canvas, spec, out_dtype=out_dtype), transform


# --------------------------------------------------------------------------- #
# NMS
# --------------------------------------------------------------------------- #


def nms_indices(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    top_k: int = 300,
) -> np.ndarray:
    """Greedy NMS over ``(N,4)`` xyxy boxes; returns kept indices, best first.

    ``top_k`` bounds the candidate set *before* the loop. Without it, a low
    confidence threshold on a 25200-row head can push thousands of boxes into
    an O(n^2) Python loop and blow a 50 ms frame budget; the cap makes the
    worst case deterministic.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.size == 0:
        return np.empty(0, dtype=np.int64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise PerceptionError("nms expects (N,4) boxes, got %s" % (boxes.shape,))
    if scores.shape[0] != boxes.shape[0]:
        raise PerceptionError(
            "nms got %d boxes but %d scores" % (boxes.shape[0], scores.shape[0])
        )
    order = np.argsort(-scores, kind="stable")
    if top_k > 0 and order.size > top_k:
        order = order[:top_k]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    keep: List[int] = []
    while order.size > 0:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[best], x1[rest])
        yy1 = np.maximum(y1[best], y1[rest])
        xx2 = np.minimum(x2[best], x2[rest])
        yy2 = np.minimum(y2[best], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[best] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    top_k: int = 300,
) -> np.ndarray:
    """Per-class NMS via the coordinate-offset trick.

    Boxes of different classes are shifted into disjoint coordinate strips so
    they can never overlap, then a single NMS pass runs. Class-agnostic NMS
    makes a car and a truck at the same location suppress each other, so the
    surviving label -- and therefore the per-class height the range model uses
    -- flickers frame to frame and the reported distance jumps by 2x.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.empty(0, dtype=np.int64)
    class_ids = np.asarray(class_ids)
    span = float(np.max(boxes[:, 2] - boxes[:, 0]).item()) if boxes.shape[0] else 0.0
    extent = float(max(np.max(boxes).item(), span, 1.0)) + 1.0
    offset = class_ids.astype(np.float32) * np.float32(extent)
    shifted = boxes.copy()
    shifted[:, 0] += offset
    shifted[:, 2] += offset
    return nms_indices(shifted, scores, iou_threshold, top_k=top_k)


def nms_xyxy(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    iou_threshold: float,
) -> List[int]:
    """Greedy class-agnostic NMS over a list of ``[x1, y1, x2, y2]`` boxes.

    Thin list-oriented wrapper around :func:`nms_indices`, kept for callers and
    tests that predate the array API.
    """
    if boxes is None or len(boxes) == 0:
        return []
    return [int(i) for i in nms_indices(np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32), iou_threshold, top_k=0)]


# --------------------------------------------------------------------------- #
# Shared decode tail
# --------------------------------------------------------------------------- #


def finalise_detections(
    xywh_net: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    transform: LetterboxTransform,
    iou_threshold: float,
    max_detections: int,
    pre_nms_topk: int = 300,
) -> List[BoundingBox]:
    """Map network-space ``(cx, cy, w, h)`` rows to source-frame boxes.

    Applies, in order: centre-form to corner-form, the inverse letterbox, a
    clip to the source frame, a degenerate-box drop, class-aware NMS, and the
    ``max_detections`` cap. Boxes are clipped to ``[0, w-1] x [0, h-1]`` so that
    :func:`adas.perception.geometry.truncation_flags` can recognise a box that
    ran off the frame edge -- a truncated box under-reports its height and so
    over-reports its range, always in the unsafe direction.
    """
    xywh_net = np.asarray(xywh_net, dtype=np.float32)
    if xywh_net.size == 0:
        return []
    if xywh_net.ndim != 2 or xywh_net.shape[1] != 4:
        raise PerceptionError("expected (N,4) xywh rows, got %s" % (xywh_net.shape,))
    half_w = xywh_net[:, 2] * np.float32(0.5)
    half_h = xywh_net[:, 3] * np.float32(0.5)
    corners = np.empty_like(xywh_net)
    corners[:, 0] = xywh_net[:, 0] - half_w
    corners[:, 1] = xywh_net[:, 1] - half_h
    corners[:, 2] = xywh_net[:, 0] + half_w
    corners[:, 3] = xywh_net[:, 1] + half_h

    boxes = transform.boxes_to_source(corners)
    max_x = np.float32(transform.src_width - 1)
    max_y = np.float32(transform.src_height - 1)
    np.clip(boxes[:, 0::2], 0.0, max_x, out=boxes[:, 0::2])
    np.clip(boxes[:, 1::2], 0.0, max_y, out=boxes[:, 1::2])

    scores = np.asarray(scores, dtype=np.float32)
    class_ids = np.asarray(class_ids, dtype=np.int32)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & np.isfinite(scores)
    if not bool(np.any(valid)):
        return []
    boxes = boxes[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    keep = class_aware_nms(boxes, scores, class_ids, iou_threshold, top_k=pre_nms_topk)
    if keep.size > max_detections:
        keep = keep[:max_detections]
    return [
        BoundingBox(
            x1=float(boxes[i, 0]),
            y1=float(boxes[i, 1]),
            x2=float(boxes[i, 2]),
            y2=float(boxes[i, 3]),
            confidence=float(min(1.0, max(0.0, scores[i]))),
            label=class_name(int(class_ids[i])),
        )
        for i in keep
    ]


# --------------------------------------------------------------------------- #
# Engine head contract
# --------------------------------------------------------------------------- #

#: Feature-map strides shared by every YOLO family this project runs.
FPN_STRIDES: Tuple[int, int, int] = (8, 16, 32)


def fpn_cell_count(net_size: int, strides: Sequence[int] = FPN_STRIDES) -> int:
    """Total number of feature-map cells across the FPN levels."""
    return int(sum((int(net_size) // int(s)) ** 2 for s in strides))


@dataclass(frozen=True)
class HeadContract:
    """The validated interpretation of a detection engine's output tensor."""

    layout: str
    rows: int
    width: int
    num_classes: int
    has_objectness: bool
    grid_decode: bool
    transposed: bool
    tensor_name: str = ""

    @property
    def class_offset(self) -> int:
        """Column index of class score 0."""
        return 5 if self.has_objectness else 4


#: Output tensor names, in preference order, that a detection head is known by.
DETECTION_OUTPUT_NAMES: Tuple[str, ...] = ("output0", "output", "outputs", "det_out", "detections")


def select_detection_output(
    output_shapes: Dict[str, Sequence[int]],
    net_size: int,
) -> str:
    """Pick the detection tensor by name, or by matching a known head shape.

    Never "whichever tensor is biggest": for a four-output EfficientNMS engine
    the biggest tensor is ``boxes``, and decoding it as a raw head produces
    confident nonsense. Raises when the choice is ambiguous.
    """
    if not output_shapes:
        raise PerceptionError("engine exposes no outputs")
    if len(output_shapes) == 1:
        return next(iter(output_shapes))
    for candidate in DETECTION_OUTPUT_NAMES:
        if candidate in output_shapes:
            return candidate
    matches = []
    for name, shape in output_shapes.items():
        try:
            analyse_detection_head(shape, net_size)
        except PerceptionError:
            continue
        matches.append(name)
    if len(matches) == 1:
        return matches[0]
    raise PerceptionError(
        "cannot identify the detection output among %s (net_size=%d); %s"
        % (
            {k: tuple(v) for k, v in output_shapes.items()},
            net_size,
            "several tensors match a known head" if matches else "none matches a known head",
        )
    )


def analyse_detection_head(
    shape: Sequence[int],
    net_size: int,
    layout_hint: Optional[str] = None,
    strides: Sequence[int] = FPN_STRIDES,
) -> HeadContract:
    """Validate a detection output shape against the known head layouts.

    ``shape`` is the engine's declared output shape, with or without a leading
    batch dimension. Returns the :class:`HeadContract` describing how to read
    it, or raises :class:`~adas.core.exceptions.PerceptionError` naming the
    observed shape and what each layout would have required.

    The three layouts are distinguished by row count and row width, both of
    which follow from ``net_size``:

    =========  =====================  ==============  =============
    layout     rows                   width           box encoding
    =========  =====================  ==============  =============
    yolov5     ``3 * fpn_cells``      ``5 + nc``      pixels
    yolov8     ``fpn_cells``          ``4 + nc``      pixels
    yolox      ``fpn_cells``          ``5 + nc``      grid units
    =========  =====================  ==============  =============

    ``yolov8`` and ``yolox`` share a row count and are separated by width; for
    the COCO-80 models this project ships that is 84 vs 85 and unambiguous. A
    non-COCO class count can collide, which is what ``layout_hint`` is for.
    """
    dims = [int(d) for d in shape]
    if len(dims) == 3:
        if dims[0] != 1:
            raise PerceptionError("only batch 1 detection heads are supported, got %s" % (shape,))
        dims = dims[1:]
    if len(dims) != 2:
        raise PerceptionError("detection head must be rank 2 or 3, got %s" % (shape,))

    rows, width = dims
    transposed = False
    # A channels-first export ((1, 84, 8400)) has far fewer rows than columns.
    if rows < width:
        rows, width = width, rows
        transposed = True

    cells = fpn_cell_count(net_size, strides)
    hint = (layout_hint or "auto").strip().lower()
    if hint not in ("auto", "yolov5", "yolov8", "yolox"):
        raise PerceptionError("unknown detector layout %r" % (layout_hint,))

    candidates: List[HeadContract] = []
    if rows == 3 * cells and width >= 6:
        candidates.append(
            HeadContract("yolov5", rows, width, width - 5, True, False, transposed)
        )
    if rows == cells:
        if width >= 5:
            candidates.append(
                HeadContract("yolov8", rows, width, width - 4, False, False, transposed)
            )
        if width >= 6:
            candidates.append(
                HeadContract("yolox", rows, width, width - 5, True, True, transposed)
            )

    if hint != "auto":
        for contract in candidates:
            if contract.layout == hint:
                return contract
        raise PerceptionError(
            "output %s is not a %s head at net_size=%d: that layout needs rows=%d and width=%s"
            % (
                shape,
                hint,
                net_size,
                3 * cells if hint == "yolov5" else cells,
                ">=6" if hint != "yolov8" else ">=5",
            )
        )

    if not candidates:
        raise PerceptionError(
            "unrecognised detection head %s at net_size=%d: expected rows=%d (yolov5) "
            "or rows=%d (yolov8/yolox)" % (shape, net_size, 3 * cells, cells)
        )
    if len(candidates) == 1:
        return candidates[0]
    # yolov8 (4+nc) vs yolox (5+nc) at the same row count. Resolve on the COCO-80
    # width; anything else needs an explicit hint rather than a coin flip.
    for contract in candidates:
        if contract.num_classes == len(COCO_CLASSES):
            return contract
    raise PerceptionError(
        "detection head %s at net_size=%d is ambiguous between %s; set an explicit "
        "layout" % (shape, net_size, [c.layout for c in candidates])
    )


def gather_class_scores(
    rows: np.ndarray,
    contract: HeadContract,
    allowed: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Best allowed class score and id per row.

    Slices to the allowed class columns *before* the argmax. Taking the argmax
    over all 80 classes first and filtering afterwards throws away a valid
    detection whenever some disallowed class happens to score higher -- a car
    behind a pedestrian commonly scores person 0.55 / car 0.48 and the car
    simply vanishes.
    """
    start = contract.class_offset
    sub = rows[:, start + allowed] if allowed.size else rows[:, start:start]
    if sub.size == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty.astype(np.int32)
    local = sub.argmax(axis=1)
    best = sub[np.arange(sub.shape[0]), local]
    return best.astype(np.float32, copy=False), allowed[local].astype(np.int32, copy=False)


def infer_net_size(
    shape,
    layout_hint=None,
    candidates=(224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640,
                672, 704, 736, 768, 800, 832, 864, 896, 960, 1024, 1088, 1280),
):
    """Recover the square network input size implied by a detection head shape.

    Only needed when the caller cannot state it (the legacy ``decode_yolo``
    signature carries the letterbox scale but not the input size). Every
    candidate is a multiple of 32; the search raises rather than guessing when
    no candidate produces a consistent head.
    """
    for size in candidates:
        try:
            contract = analyse_detection_head(shape, size, layout_hint=layout_hint)
        except PerceptionError:
            continue
        return int(size), contract
    raise PerceptionError(
        "cannot infer the network input size from detection head %s; pass net_size "
        "explicitly" % (tuple(int(d) for d in shape),)
    )


def decode_head(
    output,
    contract: HeadContract,
    transform: LetterboxTransform,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    class_ids=None,
    pre_nms_topk: int = 300,
    max_candidates: int = 3000,
) -> List[BoundingBox]:
    """Decode one raw YOLO-family head into source-frame bounding boxes.

    The single decode implementation behind :func:`adas.perception.yolo.decode_yolo`
    and :func:`adas.perception.yolox.decode_yolox`; ``contract`` says which
    layout is being read.

    Cost
    ----
    The expensive part of a naive decode is materialising ``obj * cls`` for
    every row: 25200x80 float32 is 8.1 MB of arithmetic per frame before a
    single box has been thresholded. Here, for a head with objectness, the
    first pass touches only column 4 (25200 values) and gates on it. That gate
    is *exact*, not an approximation: the final score is ``obj * cls`` with
    ``cls`` in ``[0, 1]``, so no row with ``obj < conf_threshold`` can survive.
    Only the surviving rows are converted to float32 and scored.

    ``max_candidates`` bounds the work between the gate and NMS. When more rows
    pass the gate than that, the highest-gate rows are kept -- so a pathological
    frame costs a bounded amount of time instead of an unbounded one.
    """
    arr = np.asarray(output)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise PerceptionError("only batch 1 is supported, got %s" % (arr.shape,))
        arr = arr[0]
    if arr.ndim != 2:
        raise PerceptionError("detection head must be rank 2 or 3, got %s" % (arr.shape,))
    if contract.transposed:
        arr = arr.T
    if arr.shape != (contract.rows, contract.width):
        raise PerceptionError(
            "detection head is %s but the contract says %s"
            % (arr.shape, (contract.rows, contract.width))
        )

    allowed = np.asarray(resolve_class_ids(class_ids, contract.num_classes), dtype=np.int32)
    threshold = float(conf_threshold)

    if contract.has_objectness:
        gate = arr[:, 4]
    else:
        gate = arr[:, 4 + allowed].max(axis=1)
    candidate_idx = np.flatnonzero(np.asarray(gate) >= threshold)
    if candidate_idx.size == 0:
        return []
    if max_candidates > 0 and candidate_idx.size > max_candidates:
        gate_values = np.asarray(gate, dtype=np.float32)[candidate_idx]
        top = np.argpartition(-gate_values, max_candidates - 1)[:max_candidates]
        candidate_idx = np.sort(candidate_idx[top])

    rows = np.asarray(arr[candidate_idx], dtype=np.float32)
    scores, class_out = gather_class_scores(rows, contract, allowed)
    if contract.has_objectness:
        scores = scores * rows[:, 4]
    keep = scores >= threshold
    if not bool(np.any(keep)):
        return []
    rows = rows[keep]
    scores = scores[keep]
    class_out = class_out[keep]
    candidate_idx = candidate_idx[keep]

    xywh = rows[:, :4].copy()
    if contract.grid_decode:
        from adas.perception.yolox import decode_yolox_boxes

        xywh = decode_yolox_boxes(xywh, transform.net_size, row_index=candidate_idx)

    return finalise_detections(
        xywh,
        scores,
        class_out,
        transform,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        pre_nms_topk=pre_nms_topk,
    )


# --------------------------------------------------------------------------- #
# Mock detector
# --------------------------------------------------------------------------- #

MOCK_LABEL = "vehicle"
_MOCK_BANNER = (
    "MOCK OBJECT DETECTOR ACTIVE - NOT FOR VEHICLE USE. It fabricates one "
    "lead-vehicle box at a fixed image position on every frame (~10.5 m at the "
    "default 1280x720 / f=910 calibration). Nothing downstream can tell this "
    "apart from real perception. Set detector.backend='tensorrt'."
)


@dataclass
class ObjectDetector:
    """Deterministic mock detector for pipeline smoke tests.

    Emits exactly one box, 12% of frame width by 18% of frame height, with its
    bottom edge at 70% of frame height. It is not perception: it does not look
    at the frame at all.

    Honesty
    -------
    ``is_mock`` is ``True``, the class name is announced at WARNING on
    construction and re-announced every ``warn_every_n_frames`` calls, and
    :attr:`last_was_mock` is set on every call so a caller can propagate the
    flag. The label is :data:`MOCK_LABEL` (``"vehicle"``), which is *not* a
    COCO class name -- a real detector never emits it.
    """

    confidence_threshold: float = 0.35
    warn_every_n_frames: int = 200
    is_mock: bool = True
    last_was_mock: bool = field(default=True, init=False)
    _calls: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        logger.warning(_MOCK_BANNER)

    def infer(self, frame: object, width: int, height: int) -> List[BoundingBox]:
        """Return the single synthetic box. ``frame`` is ignored by design."""
        self._calls += 1
        self.last_was_mock = True
        if self.warn_every_n_frames > 0 and self._calls % self.warn_every_n_frames == 1 and self._calls > 1:
            logger.warning("%s (frame %d)", _MOCK_BANNER, self._calls)
        conf = 0.8
        if conf < self.confidence_threshold:
            return []
        box_w, box_h = width * 0.12, height * 0.18
        center_x, bottom_y = width / 2.0, height * 0.7
        return [
            BoundingBox(
                x1=center_x - box_w / 2.0,
                y1=bottom_y - box_h,
                x2=center_x + box_w / 2.0,
                y2=bottom_y,
                confidence=conf,
                label=MOCK_LABEL,
            )
        ]

    def close(self) -> None:
        """No resources to release; present so callers can close uniformly."""
        return None
