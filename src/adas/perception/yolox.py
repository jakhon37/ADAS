"""YOLOX TensorRT backend -- the licence-clean (Apache-2.0) object detector.

Why this module exists
----------------------
The project's original detector is Ultralytics YOLOv5n, whose weights are
AGPL-3.0-only while ``pyproject.toml`` declares MIT. YOLOX-Nano and YOLOX-Tiny
(Megvii, Apache-2.0) are drop-in replacements that are also *faster* on this
board -- 4.69 ms vs 7.23 ms GPU compute for nano at 416 vs 640.

They are not, however, drop-in at the tensor level. YOLOX differs from YOLOv5
in three ways, each of which fails silently if ignored:

1. **Preprocessing.** BGR, raw ``0..255``, top-left letterbox. Feeding it the
   YOLOv5 recipe (RGB, ``/255``, centred letterbox) does not raise: measured on
   frame 150 of ``Ultra-Fast-Lane-Detection-v2/example.mp4``, BGR+raw yields 10
   detections above 0.30 and ``/255`` yields **zero**.
2. **Anchor-free head.** One row per feature-map cell, not three, so a 416
   input gives 3549 rows (``52^2 + 26^2 + 13^2``) rather than YOLOv5's 10647.
3. **Undecoded boxes.** Columns 0..3 are ``(dx, dy, log_w, log_h)`` in *grid
   units*, not pixels. They must be combined with the cell origin and the level
   stride. Objectness and class scores, by contrast, are already sigmoid'd in
   the exported graph -- applying a second sigmoid halves every score.

Contract (verified by reading the built engine's bindings)
----------------------------------------------------------
``models/yolox_nano.engine`` and ``models/yolox_tiny.engine``::

    in  "images" float32 (1, 3, 416, 416)
    out "output" float32 (1, 3549, 85)      rows = [dx, dy, log_w, log_h, obj, cls_0..cls_79]

Units
-----
Grid coordinates are feature-map cells. Decoded boxes are pixels of the
416x416 letterboxed input until :func:`~adas.perception.detection.finalise_detections`
maps them back to source-frame pixels. Scores are probabilities in ``[0, 1]``.

Failure behaviour
-----------------
A shape that is not a YOLOX head at the engine's own input size raises
:class:`~adas.core.exceptions.PerceptionError` naming both shapes. Nothing here
falls back to a different layout.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from adas.core.exceptions import PerceptionError
from adas.core.logger import setup_logger
from adas.core.models import BoundingBox
from adas.perception.yolo import YoloTensorRTDetector
from adas.perception.detection import (
    PREPROC_YOLOX,
    HeadContract,
    LetterboxTransform,
    analyse_detection_head,
    decode_head,
    preprocess_letterbox,
)

logger = setup_logger(__name__)

#: FPN strides, in the order the head concatenates its levels.
YOLOX_STRIDES: Tuple[int, int, int] = (8, 16, 32)

# net_size -> (grid_xy (N,2) float32, stride (N,1) float32). Built once per size.
_GRID_CACHE: Dict[Tuple[int, Tuple[int, ...]], Tuple[np.ndarray, np.ndarray]] = {}


def yolox_grids(
    net_size: int,
    strides: Sequence[int] = YOLOX_STRIDES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cell origins and strides for every head row, in head order.

    Returns ``(grid_xy, stride)`` where ``grid_xy`` is ``(N, 2)`` of integer
    cell coordinates as float32 and ``stride`` is ``(N, 1)``. ``N`` equals the
    head's row count. Rows are level-major (stride 8 first) and, within a
    level, row-major with ``x`` varying fastest -- exactly the order produced by
    YOLOX's own ``demo_postprocess``.
    """
    key = (int(net_size), tuple(int(s) for s in strides))
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached
    grids: List[np.ndarray] = []
    expanded: List[np.ndarray] = []
    for stride in key[1]:
        if stride <= 0 or int(net_size) % stride != 0:
            raise PerceptionError(
                "YOLOX net_size %d is not divisible by stride %d" % (net_size, stride)
            )
        size = int(net_size) // int(stride)
        xv, yv = np.meshgrid(np.arange(size), np.arange(size))
        grids.append(np.stack((xv, yv), axis=2).reshape(-1, 2).astype(np.float32))
        expanded.append(np.full((size * size, 1), float(stride), dtype=np.float32))
    result = (np.concatenate(grids, axis=0), np.concatenate(expanded, axis=0))
    _GRID_CACHE[key] = result
    return result


def decode_yolox_boxes(
    raw_xywh: np.ndarray,
    net_size: int,
    row_index: Optional[np.ndarray] = None,
    strides: Sequence[int] = YOLOX_STRIDES,
) -> np.ndarray:
    """Grid-decode YOLOX box columns into ``(cx, cy, w, h)`` input pixels.

    ``raw_xywh`` is ``(M, 4)`` of ``(dx, dy, log_w, log_h)``. ``row_index`` maps
    each of those ``M`` rows back to its position in the full head; pass it
    whenever the rows have already been thresholded, which is the normal case.
    Without it, ``raw_xywh`` must be the complete head.

    The transform is YOLOX's own::

        cx = (dx + grid_x) * stride
        cy = (dy + grid_y) * stride
        w  = exp(log_w) * stride
        h  = exp(log_h) * stride
    """
    boxes = np.asarray(raw_xywh, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise PerceptionError("decode_yolox_boxes expects (M,4), got %s" % (boxes.shape,))
    grid, stride = yolox_grids(net_size, strides)
    if row_index is None:
        if boxes.shape[0] != grid.shape[0]:
            raise PerceptionError(
                "decode_yolox_boxes got %d rows but net_size=%d has %d cells; pass "
                "row_index for a pre-thresholded subset"
                % (boxes.shape[0], net_size, grid.shape[0])
            )
        sel_grid, sel_stride = grid, stride
    else:
        index = np.asarray(row_index, dtype=np.int64)
        if index.shape[0] != boxes.shape[0]:
            raise PerceptionError(
                "row_index has %d entries for %d rows" % (index.shape[0], boxes.shape[0])
            )
        if index.size and (int(index.min()) < 0 or int(index.max()) >= grid.shape[0]):
            raise PerceptionError(
                "row_index out of range for net_size=%d (%d cells)" % (net_size, grid.shape[0])
            )
        sel_grid = grid[index]
        sel_stride = stride[index]
    out = np.empty_like(boxes)
    out[:, 0] = (boxes[:, 0] + sel_grid[:, 0]) * sel_stride[:, 0]
    out[:, 1] = (boxes[:, 1] + sel_grid[:, 1]) * sel_stride[:, 0]
    np.exp(boxes[:, 2:4], out=out[:, 2:4])
    out[:, 2] *= sel_stride[:, 0]
    out[:, 3] *= sel_stride[:, 0]
    return out


def yolox_contract(output_shape: Sequence[int], net_size: int) -> HeadContract:
    """Validate ``output_shape`` as a YOLOX head for ``net_size``."""
    return analyse_detection_head(output_shape, net_size, layout_hint="yolox")


def preprocess_yolox(image: np.ndarray, net_size: int) -> Tuple[np.ndarray, LetterboxTransform]:
    """Build the YOLOX input blob from a uint8 BGR source frame.

    BGR is preserved, pixels stay in ``0..255``, and the padding goes on the
    bottom and right only (YOLOX does not centre its letterbox), so the inverse
    mapping is a pure division by the scale with no pad subtraction.
    """
    return preprocess_letterbox(image, net_size, PREPROC_YOLOX)


def decode_yolox(
    output,
    transform: LetterboxTransform,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    class_ids=None,
    pre_nms_topk: int = 300,
    contract: Optional[HeadContract] = None,
) -> List[BoundingBox]:
    """Decode a YOLOX head into source-frame boxes.

    Equivalent to YOLOX's ``demo_postprocess`` followed by ``multiclass_nms``,
    with two differences that matter for a real-time stack: the score gate runs
    on objectness *before* the class scores are materialised (the gate is exact
    because ``score = obj * cls`` and ``cls <= 1``), and NMS is class-aware.
    """
    arr = np.asarray(output)
    head = contract or yolox_contract(arr.shape, transform.net_size)
    if head.layout != "yolox":
        raise PerceptionError("decode_yolox given a %s contract" % head.layout)
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


class YoloXTensorRTDetector(YoloTensorRTDetector):
    """Object detector backed by an Apache-2.0 YOLOX TensorRT engine.

    Specialisation of :class:`adas.perception.yolo.YoloTensorRTDetector` that
    pins the layout to ``yolox`` instead of inferring it, so pointing it at a
    YOLOv5 engine fails at construction with a shape mismatch rather than
    quietly decoding grid-unit boxes as if they were pixels.

    Everything else -- preprocessing, decode, NMS, timing, lifecycle -- is the
    shared implementation; ``PREPROC_YOLOX`` is selected automatically from the
    layout.
    """

    def __init__(self, engine_path: str, **kwargs) -> None:
        kwargs["layout"] = "yolox"
        super().__init__(engine_path, **kwargs)
