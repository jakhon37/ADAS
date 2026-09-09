"""Ultra-Fast-Lane-Detection-v2 TensorRT lane estimator."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from adas.core.exceptions import PerceptionError
from adas.core.logger import setup_logger
from adas.core.models import LaneModel

logger = setup_logger(__name__)


def _softmax(x, axis: int = 0):
    import numpy as np

    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


def pred_to_coords(
    loc_row,
    exist_row,
    loc_col,
    exist_col,
    ori_w: int,
    ori_h: int,
    num_row: int,
    num_col: int,
) -> List[List[Tuple[int, int]]]:
    """Decode UFLDv2 heads to polylines in the network crop frame."""
    import numpy as np

    loc_row = np.asarray(loc_row)
    exist_row = np.asarray(exist_row)
    loc_col = np.asarray(loc_col)
    exist_col = np.asarray(exist_col)
    if loc_row.ndim == 4:
        loc_row = loc_row[0]
        exist_row = exist_row[0]
        loc_col = loc_col[0]
        exist_col = exist_col[0]

    # loc_row: (grid, cls, lanes)
    max_indices_row = loc_row.argmax(0)
    valid_row = exist_row.argmax(0)
    max_indices_col = loc_col.argmax(0)
    valid_col = exist_col.argmax(0)
    num_grid_row = loc_row.shape[0]
    num_grid_col = loc_col.shape[0]
    num_cls_row = loc_row.shape[1]
    num_cls_col = loc_col.shape[1]

    row_anchor = np.linspace(0.42, 1.0, num_row)
    col_anchor = np.linspace(0.0, 1.0, num_col)
    coords: List[List[Tuple[int, int]]] = []

    for lane_i in (1, 2):
        if lane_i >= valid_row.shape[1]:
            continue
        if valid_row[:, lane_i].sum() <= num_cls_row / 2:
            continue
        pts = []
        for k in range(valid_row.shape[0]):
            if not valid_row[k, lane_i]:
                continue
            idx = int(max_indices_row[k, lane_i])
            lo = max(0, idx - ori_w)
            hi = min(num_grid_row - 1, idx + ori_w)
            window = loc_row[lo : hi + 1, k, lane_i]
            weights = _softmax(window, axis=0)
            out = (weights * np.arange(lo, hi + 1)).sum() + 0.5
            x = out / (num_grid_row - 1) * ori_w
            y = row_anchor[k] * ori_h
            pts.append((int(x), int(y)))
        if pts:
            coords.append(pts)

    for lane_i in (0, 3):
        if lane_i >= valid_col.shape[1]:
            continue
        if valid_col[:, lane_i].sum() <= num_cls_col / 4:
            continue
        pts = []
        for k in range(valid_col.shape[0]):
            if not valid_col[k, lane_i]:
                continue
            idx = int(max_indices_col[k, lane_i])
            lo = max(0, idx - ori_w)
            hi = min(num_grid_col - 1, idx + ori_w)
            window = loc_col[lo : hi + 1, k, lane_i]
            weights = _softmax(window, axis=0)
            out = (weights * np.arange(lo, hi + 1)).sum() + 0.5
            y = out / (num_grid_col - 1) * ori_h
            x = col_anchor[k] * ori_w
            pts.append((int(x), int(y)))
        if pts:
            coords.append(pts)
    return coords


def _fit_quadratic(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, float]:
    import numpy as np

    if len(xs) < 3:
        if not xs:
            return (0.0, 0.0, 0.0)
        return (0.0, 0.0, float(xs[-1]))
    coeff = np.polyfit(ys, xs, 2)
    return (float(coeff[0]), float(coeff[1]), float(coeff[2]))


def coords_to_lane_model(
    lanes: Sequence[Sequence[Tuple[int, int]]],
    frame_width: int,
    frame_height: int,
    crop_offset_y: int,
    scale_x: float,
    scale_y: float,
) -> Optional[LaneModel]:
    """Map crop-space polylines back to the original frame and fit a lane model."""
    if not lanes:
        return None
    mapped = []
    for lane in lanes:
        pts = []
        for x, y in lane:
            ox = x / scale_x
            oy = y / scale_y + crop_offset_y
            pts.append((ox, oy))
        if pts:
            mapped.append(pts)
    if not mapped:
        return None

    bottom_y = frame_height * 0.9

    def x_at_bottom(pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        a, b, c = _fit_quadratic(xs, ys)
        return a * bottom_y * bottom_y + b * bottom_y + c

    mapped.sort(key=x_at_bottom)
    mid = frame_width / 2.0
    left_pts = None
    right_pts = None
    for pts in mapped:
        xb = x_at_bottom(pts)
        if xb <= mid and (left_pts is None or abs(xb - mid) < abs(x_at_bottom(left_pts) - mid)):
            left_pts = pts
        if xb >= mid and (right_pts is None or abs(xb - mid) < abs(x_at_bottom(right_pts) - mid)):
            right_pts = pts
    if left_pts is None and right_pts is None:
        return None
    if left_pts is None:
        right_x = x_at_bottom(right_pts)
        left_x = right_x - frame_width * 0.28
        left_coeffs = (0.0, 0.0, left_x)
        right_coeffs = _fit_quadratic([p[0] for p in right_pts], [p[1] for p in right_pts])
        center = (left_x + right_x) / 2.0
    elif right_pts is None:
        left_x = x_at_bottom(left_pts)
        right_x = left_x + frame_width * 0.28
        left_coeffs = _fit_quadratic([p[0] for p in left_pts], [p[1] for p in left_pts])
        right_coeffs = (0.0, 0.0, right_x)
        center = (left_x + right_x) / 2.0
    else:
        left_coeffs = _fit_quadratic([p[0] for p in left_pts], [p[1] for p in left_pts])
        right_coeffs = _fit_quadratic([p[0] for p in right_pts], [p[1] for p in right_pts])
        center = (x_at_bottom(left_pts) + x_at_bottom(right_pts)) / 2.0
    return LaneModel(
        left_coeffs=left_coeffs,
        right_coeffs=right_coeffs,
        lane_center_px=float(center),
        curvature_m=220.0,
    )


class UFLDLaneEstimator:
    """Lane estimator backed by a UFLDv2 TensorRT engine."""

    def __init__(
        self,
        engine_path: str,
        input_width: int = 1600,
        input_height: int = 320,
        crop_ratio: float = 0.6,
        num_row: int = 72,
        num_col: int = 81,
    ) -> None:
        from adas.infer.trt_engine import TrtEngine

        path = Path(engine_path)
        if not path.exists():
            raise PerceptionError("UFLD engine not found: %s" % engine_path)
        self.engine = TrtEngine(str(path))
        self.input_width = input_width
        self.input_height = input_height
        self.crop_ratio = crop_ratio
        self.num_row = num_row
        self.num_col = num_col
        logger.info("UFLDLaneEstimator loaded %s", path)

    def estimate(self, frame: object, width: int, height: int) -> LaneModel | None:
        import cv2
        import numpy as np

        if not hasattr(frame, "shape"):
            raise PerceptionError("ufld estimator needs an image array, got %s" % type(frame))
        image = frame
        h, w = image.shape[:2]
        scale = self.input_width / float(w)
        resized = cv2.resize(image, (self.input_width, max(1, int(round(h * scale)))))
        if resized.shape[0] > self.input_height:
            crop_offset_y = resized.shape[0] - self.input_height
            crop = resized[crop_offset_y:, :, :]
        else:
            crop_offset_y = 0
            crop = cv2.resize(resized, (self.input_width, self.input_height))
        # Map crop y back to original: orig_y = (crop_y + crop_offset_y) / scale
        orig_crop_offset = crop_offset_y / scale
        blob = crop.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        outputs = self.engine.infer({self.engine.input_name: blob})
        loc_row = exist_row = loc_col = exist_col = None
        for name, arr in outputs.items():
            key = name.lower()
            if "exist_row" in key:
                exist_row = arr
            elif "exist_col" in key:
                exist_col = arr
            elif "loc_row" in key or key.endswith("row") and "exist" not in key:
                loc_row = arr
            elif "loc_col" in key or key.endswith("col") and "exist" not in key:
                loc_col = arr
        if loc_row is None or exist_row is None or loc_col is None or exist_col is None:
            # Fall back to binding order if names are generic.
            vals = list(outputs.values())
            if len(vals) >= 4:
                loc_row, exist_row, loc_col, exist_col = vals[:4]
            else:
                logger.warning("UFLD output tensors not recognized: %s", list(outputs))
                return None
        lanes = pred_to_coords(
            loc_row,
            exist_row,
            loc_col,
            exist_col,
            ori_w=self.input_width,
            ori_h=self.input_height,
            num_row=self.num_row,
            num_col=self.num_col,
        )
        return coords_to_lane_model(
            lanes,
            frame_width=w,
            frame_height=h,
            crop_offset_y=int(orig_crop_offset),
            scale_x=scale,
            scale_y=scale,
        )
