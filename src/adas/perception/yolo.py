"""YOLOv5/v8 TensorRT detector for Jetson."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from adas.core.exceptions import PerceptionError
from adas.core.logger import setup_logger
from adas.core.models import BoundingBox

logger = setup_logger(__name__)

# COCO ids used for ACC (lead vehicle / VRU in path).
COCO_VEHICLE = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def letterbox(
    image,
    size: int = 640,
    pad_value: int = 114,
) -> Tuple[object, float, int, int]:
    """Resize with unchanged aspect ratio and pad to a square."""
    import cv2
    import numpy as np

    h, w = image.shape[:2]
    scale = min(size / float(h), size / float(w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, left, top


def nms_xyxy(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    iou_threshold: float,
) -> List[int]:
    """Greedy NMS. boxes are [x1, y1, x2, y2]."""
    if not boxes:
        return []
    import numpy as np

    b = np.asarray(boxes, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    order = s.argsort()[::-1]
    keep: List[int] = []
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(min=0) * (yy2 - yy1).clip(min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_threshold]
    return keep


def decode_yolo(
    output,
    scale: float,
    pad_x: int,
    pad_y: int,
    orig_w: int,
    orig_h: int,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    class_ids: Iterable[int] = COCO_VEHICLE.keys(),
) -> List[BoundingBox]:
    """Decode YOLOv5 (N,85) or YOLOv8 (84,N) output to image-space boxes."""
    import numpy as np

    arr = np.asarray(output)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise PerceptionError("unexpected YOLO output rank: %s" % (arr.shape,))

    # YOLOv8: (84, N) = 4 box + 80 classes. YOLOv5: (N, 85) = box + obj + 80.
    if arr.shape[0] < arr.shape[1] and arr.shape[0] in (84, 85):
        arr = arr.T

    arr = arr.astype(np.float32, copy=False)
    has_obj = arr.shape[1] == 85
    xywh = arr[:, :4]
    if has_obj:
        confs = arr[:, 5:] * arr[:, 4:5]
    else:
        confs = arr[:, 4:]
    cls_id = confs.argmax(axis=1)
    conf = confs.max(axis=1)
    allowed = np.fromiter(class_ids, dtype=np.int32)
    mask = (conf >= conf_threshold) & np.isin(cls_id, allowed)
    if not np.any(mask):
        return []

    xywh = xywh[mask]
    conf = conf[mask]
    cls_id = cls_id[mask]
    cx, cy, bw, bh = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = np.clip((cx - bw / 2.0 - pad_x) / scale, 0.0, float(orig_w - 1))
    y1 = np.clip((cy - bh / 2.0 - pad_y) / scale, 0.0, float(orig_h - 1))
    x2 = np.clip((cx + bw / 2.0 - pad_x) / scale, 0.0, float(orig_w - 1))
    y2 = np.clip((cy + bh / 2.0 - pad_y) / scale, 0.0, float(orig_h - 1))
    valid = (x2 > x1) & (y2 > y1)
    boxes = np.stack([x1, y1, x2, y2], axis=1)[valid].tolist()
    scores = conf[valid].tolist()
    ids = cls_id[valid].tolist()
    labels = [COCO_VEHICLE.get(int(i), str(int(i))) for i in ids]

    keep = nms_xyxy(boxes, scores, iou_threshold)[:max_detections]
    return [
        BoundingBox(
            x1=boxes[i][0],
            y1=boxes[i][1],
            x2=boxes[i][2],
            y2=boxes[i][3],
            confidence=scores[i],
            label=labels[i],
        )
        for i in keep
    ]


class YoloTensorRTDetector:
    """Vehicle detector backed by a TensorRT YOLO engine."""

    def __init__(
        self,
        engine_path: str,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        max_detections: int = 100,
        input_size: int = 640,
    ) -> None:
        from adas.infer.trt_engine import TrtEngine

        path = Path(engine_path)
        if not path.exists():
            raise PerceptionError("YOLO engine not found: %s" % engine_path)
        self.engine = TrtEngine(str(path))
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.input_size = int(self.engine.input_shape[-1]) if self.engine.input_shape else input_size
        logger.info("YoloTensorRTDetector loaded %s input=%s", path, self.engine.input_shape)

    def infer(self, frame: object, width: int, height: int) -> list[BoundingBox]:
        import cv2
        import numpy as np

        if not hasattr(frame, "shape"):
            raise PerceptionError("tensorrt detector needs an image array, got %s" % type(frame))
        image = frame
        if image.ndim != 3 or image.shape[2] != 3:
            raise PerceptionError("expected HxWx3 image, got %s" % (image.shape,))
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.dtype == np.uint8 else image
        canvas, scale, pad_x, pad_y = letterbox(rgb, size=self.input_size)
        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        outputs: Dict[str, object] = self.engine.infer({self.engine.input_name: blob})
        # Prefer the largest output tensor (YOLO detections).
        det = max(outputs.values(), key=lambda a: int(np.asarray(a).size))
        return decode_yolo(
            det,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            orig_w=w,
            orig_h=h,
            conf_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
        )
