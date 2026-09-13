"""TensorRT YOLO object detector (YOLOv5 / YOLOv8 / YOLOX).

This module owns the engine-backed detector; the layout-independent
primitives -- class tables, preprocessing specs, letterbox, NMS and the decode
core -- live in :mod:`adas.perception.detection`, and the YOLOX-specific grid
decode lives in :mod:`adas.perception.yolox`.

Engine contract
---------------
The detector reads its contract from the engine at construction and refuses to
start if it does not recognise it:

* exactly one input, rank 4, ``(1, 3, S, S)``. ``S`` comes from the **engine**,
  never from configuration, so a 416-input YOLOX engine cannot be fed a 640
  blob because a config file said 640.
* one detection output, chosen by name (``output0``/``output``/...) or by
  matching a known head shape -- never "whichever tensor is biggest", which
  picks ``boxes`` out of a four-output EfficientNMS engine and decodes it as
  confident nonsense.
* the head shape must match one of the three known layouts for that ``S``
  (see :func:`adas.perception.detection.analyse_detection_head`).

Preprocessing
-------------
Selected from the resolved layout, not hardcoded: YOLOv5/v8 get RGB, ``/255``
and a centred letterbox; YOLOX gets BGR, raw ``0..255`` and a top-left
letterbox. Source frames must be uint8 HxWx3 BGR.

Units
-----
Boxes are returned in **source-frame pixels** (the letterbox is inverted and
the result clipped to the frame), confidences in ``[0, 1]``, labels are COCO
names. Timings are wall-clock milliseconds.

Thread safety
-------------
None. One instance owns one TensorRT execution context and one set of staging
buffers. Do not share an instance between threads.

Failure behaviour
-----------------
A missing engine file, an unrecognised contract or a non-uint8 frame raises
:class:`~adas.core.exceptions.PerceptionError`. A frame that simply contains
nothing returns an empty list -- callers must not conflate the two, which is
what :class:`~adas.core.models.PerceptionStatus` exists for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from adas.core.exceptions import PerceptionError
from adas.core.logger import setup_logger
from adas.core.models import BoundingBox
from adas.perception.detection import (
    COCO_CLASSES,
    COCO_ROAD_USERS,
    COCO_TRAFFIC_CONTROL,
    COCO_VEHICLE,
    DEFAULT_CLASS_IDS,
    PREPROC_YOLOV5,
    PREPROC_YOLOX,
    HeadContract,
    LetterboxTransform,
    PreprocSpec,
    analyse_detection_head,
    class_aware_nms,
    class_name,
    decode_head,
    infer_net_size,
    letterbox_canvas,
    letterbox_transform,
    nms_indices,
    nms_xyxy,
    preprocess_letterbox,
    require_bgr_uint8,
    resolve_class_ids,
    select_detection_output,
)

logger = setup_logger(__name__)

__all__ = [
    "class_aware_nms",
    "letterbox_canvas",
    "letterbox_transform",
    "nms_indices",
    "preprocess_letterbox",
    "COCO_CLASSES",
    "COCO_ROAD_USERS",
    "COCO_TRAFFIC_CONTROL",
    "COCO_VEHICLE",
    "DEFAULT_CLASS_IDS",
    "YoloTensorRTDetector",
    "decode_yolo",
    "letterbox",
    "nms_xyxy",
]


def letterbox(image, size: int = 640, pad_value: int = 114) -> Tuple[object, float, int, int]:
    """Centred aspect-preserving resize and pad to ``size`` square.

    Legacy 4-tuple form ``(canvas, scale, pad_x, pad_y)`` kept for existing
    callers. New code should use
    :func:`adas.perception.detection.letterbox_canvas`, which returns a
    :class:`~adas.perception.detection.LetterboxTransform` carrying the inverse
    mapping and the source size.

    Requires uint8 HxWx3 input. The previous implementation accepted float
    frames and then assigned them into a uint8 canvas, truncating every value --
    a ``[0,1]`` normalised frame became all zeros and the detector silently
    returned nothing.
    """
    canvas, transform = letterbox_canvas(image, int(size), pad_value=int(pad_value), pad_mode="center")
    return canvas, transform.scale, int(transform.pad_x), int(transform.pad_y)


def decode_yolo(
    output,
    scale: float,
    pad_x: float,
    pad_y: float,
    orig_w: int,
    orig_h: int,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    class_ids: Optional[Iterable[int]] = None,
    layout: str = "auto",
    net_size: Optional[int] = None,
    pre_nms_topk: int = 300,
    contract: Optional[HeadContract] = None,
) -> List[BoundingBox]:
    """Decode a YOLOv5 / YOLOv8 / YOLOX head to source-frame boxes.

    ``scale``/``pad_x``/``pad_y`` are the letterbox parameters that produced the
    network input; ``orig_w``/``orig_h`` are the source frame size. ``net_size``
    is the square network input size -- when omitted it is recovered from the
    head shape, which only works for the standard multiple-of-32 sizes, so pass
    it when you have it.

    ``class_ids`` defaults to :data:`adas.perception.detection.DEFAULT_CLASS_IDS`
    (road users **including pedestrians**). It may also be the name of a set in
    :data:`adas.perception.detection.CLASS_SETS`.
    """
    arr = np.asarray(output)
    head = contract
    if net_size is None:
        hint = head.layout if head is not None else layout
        net_size, inferred = infer_net_size(arr.shape, layout_hint=hint)
        if head is None:
            head = inferred
    elif head is None:
        head = analyse_detection_head(arr.shape, int(net_size), layout_hint=layout)
    transform = LetterboxTransform(
        scale=float(scale),
        pad_x=float(pad_x),
        pad_y=float(pad_y),
        net_size=int(net_size),
        src_width=int(orig_w),
        src_height=int(orig_h),
    )
    return decode_head(
        arr,
        head,
        transform,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        class_ids=class_ids,
        pre_nms_topk=pre_nms_topk,
    )


def preproc_for_layout(layout: str) -> PreprocSpec:
    """The preprocessing contract a decoded layout requires."""
    return PREPROC_YOLOX if layout == "yolox" else PREPROC_YOLOV5


class YoloTensorRTDetector:
    """Road-user detector backed by a TensorRT YOLO engine.

    Parameters
    ----------
    engine_path:
        Path to a serialized, fully static TensorRT engine.
    confidence_threshold:
        Minimum ``obj * class`` score, in ``[0, 1]``.
    iou_threshold:
        IoU above which a lower-scoring box of the *same class* is suppressed.
    max_detections:
        Cap on returned boxes, applied after NMS.
    input_size:
        Advisory only. The engine's own input size always wins; a mismatch is
        logged at WARNING so a stale config is visible instead of silent.
    class_ids:
        ``None`` for the default road-user set, a name from
        :data:`adas.perception.detection.CLASS_SETS`, or explicit COCO ids.
    layout:
        ``"auto"`` (default), ``"yolov5"``, ``"yolov8"`` or ``"yolox"``.
    pre_nms_topk:
        Candidate cap entering NMS. Bounds worst-case decode time.
    engine:
        An already-constructed engine (or
        :class:`~adas.infer.trt_engine.FakeTrtEngine`) to use instead of
        loading ``engine_path``. This is how the decode path is tested without
        a GPU.
    """

    def __init__(
        self,
        engine_path: str = "",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        max_detections: int = 100,
        input_size: int = 640,
        class_ids: Optional[Iterable[int]] = None,
        layout: str = "auto",
        pre_nms_topk: int = 300,
        engine: object = None,
    ) -> None:
        if engine is None:
            from adas.infer.trt_engine import EngineError, TrtEngine

            path = Path(engine_path)
            if not path.exists():
                raise PerceptionError("YOLO engine not found: %s" % engine_path)
            try:
                engine = TrtEngine(str(path))
            except EngineError as exc:
                raise PerceptionError("could not load %s: %s" % (path, exc)) from exc
            self.engine_path = str(path)
        else:
            self.engine_path = str(engine_path or getattr(engine, "engine_path", "<engine>"))
        self.engine = engine

        # Anything that fails from here on must release the engine: on a board
        # with ~2.5 GiB free, a leaked execution context and its device buffers
        # are not a rounding error.
        try:
            self.net_size = self._resolve_input_size(engine, int(input_size))
            self.input_dtype = np.dtype(getattr(engine, "input_dtype", np.float32))
            self.output_name = select_detection_output(engine.output_shapes, self.net_size)
            contract = analyse_detection_head(
                engine.output_shapes[self.output_name], self.net_size, layout_hint=layout
            )
            self.contract = HeadContract(
                layout=contract.layout,
                rows=contract.rows,
                width=contract.width,
                num_classes=contract.num_classes,
                has_objectness=contract.has_objectness,
                grid_decode=contract.grid_decode,
                transposed=contract.transposed,
                tensor_name=self.output_name,
            )
            self.preproc = preproc_for_layout(self.contract.layout)
            self.class_ids = resolve_class_ids(class_ids, self.contract.num_classes)
        except PerceptionError as exc:
            self.close()
            raise PerceptionError(
                "%s: %s (engine contract: %s)" % (self.engine_path, exc, _describe(engine))
            ) from exc
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.pre_nms_topk = int(pre_nms_topk)

        # Rolling instrumentation. The pipeline has no per-stage timing of its
        # own, and "38 ms of the 45 ms is numpy" is not something you can act on
        # unless you can measure it.
        self.frames = 0
        self.last_preproc_ms = 0.0
        self.last_infer_ms = 0.0
        self.last_decode_ms = 0.0
        self.total_preproc_ms = 0.0
        self.total_infer_ms = 0.0
        self.total_decode_ms = 0.0

        logger.info(
            "%s loaded: layout=%s input=%dx%d head=%s(%d rows x %d cols, %d classes) "
            "classes=%s preproc=%s/%s pad=%s",
            type(self).__name__,
            self.contract.layout,
            self.net_size,
            self.net_size,
            self.output_name,
            self.contract.rows,
            self.contract.width,
            self.contract.num_classes,
            _summarise_classes(self.class_ids),
            self.preproc.colour,
            "raw" if self.preproc.divisor == 1.0 else "1/%g" % self.preproc.divisor,
            self.preproc.pad_mode,
        )

    # -- construction helpers --------------------------------------------- #

    @staticmethod
    def _resolve_input_size(engine: object, configured: int) -> int:
        shape = tuple(int(d) for d in engine.input_shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
            raise PerceptionError(
                "detector engine input must be (1, 3, S, S), got %s" % (shape,)
            )
        if shape[2] != shape[3]:
            raise PerceptionError(
                "detector engine input must be square, got %dx%d" % (shape[2], shape[3])
            )
        net_size = int(shape[2])
        if configured and configured != net_size:
            logger.warning(
                "configured detector input_size=%d ignored; the engine declares %d",
                configured,
                net_size,
            )
        return net_size

    # -- inference --------------------------------------------------------- #

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, LetterboxTransform]:
        """Source frame -> ``(blob, transform)`` per this engine's contract.

        The blob is built directly in the engine's declared input dtype, so no
        cast happens inside the engine wrapper.
        """
        return preprocess_letterbox(
            image, self.net_size, self.preproc, out_dtype=self.input_dtype
        )

    def infer(self, frame: object, width: int, height: int) -> List[BoundingBox]:
        """Detect road users in one uint8 BGR frame.

        ``width``/``height`` are the declared source size and are only used to
        flag a mismatch; the actual frame's shape governs the geometry.
        """
        import time

        image = require_bgr_uint8(frame)
        src_h, src_w = image.shape[:2]
        if (width and int(width) != src_w) or (height and int(height) != src_h):
            logger.debug(
                "frame is %dx%d but the pipeline declared %sx%s; using the frame",
                src_w,
                src_h,
                width,
                height,
            )

        t0 = time.perf_counter()
        blob, transform = self.preprocess(image)
        t1 = time.perf_counter()
        # Views, not copies: the head is up to 8.6 MB and is consumed here.
        outputs = self.engine.infer_views({self.engine.input_name: blob})
        t2 = time.perf_counter()
        detections = decode_head(
            outputs[self.output_name],
            self.contract,
            transform,
            conf_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
            class_ids=self.class_ids,
            pre_nms_topk=self.pre_nms_topk,
        )
        t3 = time.perf_counter()

        self.frames += 1
        self.last_preproc_ms = (t1 - t0) * 1000.0
        self.last_infer_ms = (t2 - t1) * 1000.0
        self.last_decode_ms = (t3 - t2) * 1000.0
        self.total_preproc_ms += self.last_preproc_ms
        self.total_infer_ms += self.last_infer_ms
        self.total_decode_ms += self.last_decode_ms
        return detections

    # -- introspection ----------------------------------------------------- #

    def timing_summary(self) -> dict:
        """Mean per-stage milliseconds since construction (``{}`` before use)."""
        if self.frames == 0:
            return {}
        n = float(self.frames)
        return {
            "frames": self.frames,
            "preproc_ms": self.total_preproc_ms / n,
            "infer_ms": self.total_infer_ms / n,
            "decode_ms": self.total_decode_ms / n,
            "total_ms": (self.total_preproc_ms + self.total_infer_ms + self.total_decode_ms) / n,
        }

    @property
    def class_names(self) -> List[str]:
        """Human-readable names of the classes this detector reports."""
        return [class_name(i) for i in self.class_ids]

    @property
    def is_mock(self) -> bool:
        """Always ``False``. Present so callers can test any detector uniformly."""
        return False

    def close(self) -> None:
        """Release the engine. Idempotent."""
        engine = getattr(self, "engine", None)
        if engine is not None:
            close = getattr(engine, "close", None)
            if callable(close):
                close()
            self.engine = None

    def __enter__(self) -> "YoloTensorRTDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _summarise_classes(class_ids: Sequence[int], limit: int = 12) -> str:
    """Readable class list for the startup log, truncated for the ``all`` set."""
    names = [class_name(i) for i in class_ids]
    if len(names) <= limit:
        return ", ".join(names)
    return "%s, ... (%d classes)" % (", ".join(names[:limit]), len(names))


def _describe(engine: object) -> str:
    describe = getattr(engine, "describe", None)
    if callable(describe):
        return describe()
    return "in %s -> out %s" % (
        getattr(engine, "input_shape", "?"),
        getattr(engine, "output_shapes", "?"),
    )
