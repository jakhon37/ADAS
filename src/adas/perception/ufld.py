"""Ultra-Fast-Lane-Detection-v2 TensorRT lane backend.

Model
-----
``models/ufldv2_culane_res18.engine`` -- UFLD-v2 ResNet-18 trained on CULane,
MIT code licence (CULane dataset terms are research-only; see
``models/MANIFEST.json``). Static FP16 engine:

===============  ==========================  ============
binding          shape                       meaning
===============  ==========================  ============
``input``        ``1x3x320x1600`` float32    NCHW RGB
``loc_row``      ``1x200x72x4``   float32    row-head location logits
``loc_col``      ``1x100x81x4``   float32    column-head location logits
``exist_row``    ``1x2x72x4``     float32    row-head {absent, present} logits
``exist_col``    ``1x2x81x4``     float32    column-head {absent, present} logits
===============  ==========================  ============

Preprocessing (canonical, from ``data/dataloader.py`` + ``data/dataset.py`` of
the vendored reference, which is the transform the published F1 numbers come
from -- **not** the vendored ``deploy/trt_infer.py``, which omits the
normalisation and is an upstream bug):

1. BGR -> RGB.
2. Resize the WHOLE frame to ``train_width x int(train_height / crop_ratio)``
   = 1600x533. Aspect ratio is deliberately NOT preserved.
3. ``/255``, then ImageNet mean ``(0.485, 0.456, 0.406)`` / std
   ``(0.229, 0.224, 0.225)``.
4. Keep the bottom ``train_height`` rows (``img[-320:, :, :]``). The 213 dropped
   rows are sky.

Coordinate convention (this is the part that is easy to get wrong)
------------------------------------------------------------------
UFLD-v2's decoder emits coordinates in the **original full frame**, not in the
network crop. ``row_anchor = linspace(0.42, 1.0, num_row)`` is a fraction of the
ORIGINAL image height, and the location expectation is scaled by the ORIGINAL
image width. That is precisely why the crop keeps the bottom 60%: rows 0.42..1.0
of the source all survive it. Consequently there is no inverse-crop mapping to
apply -- :func:`pred_to_coords` is handed the source frame size and its output
is already in source pixels. (The reference ``demo.py`` draws ``pred2coords``
output straight onto the un-cropped image.)

Units and failure behaviour
---------------------------
* Polyline points are source-frame pixels, floats (the softmax expectation
  computes sub-pixel positions; rounding them away throws that precision out).
* Confidences are the existence head's softmax probability, in ``[0, 1]``.
* Lane indices follow UFLD-v2: ``1``/``2`` are the ego-lane left/right
  boundaries and come from the ROW head; ``0``/``3`` are the adjacent lanes and
  come from the COLUMN head. Adjacent lanes are reported as informational
  ``LaneLine`` entries and are NEVER allowed to become an ego boundary -- they
  are near-horizontal in image space, so an ``x = f(y)`` fit through them is
  ill-conditioned and its bottom-of-frame extrapolation is arbitrary.
* Engine/dataset mismatches raise
  :class:`~adas.core.exceptions.ConfigurationError` at construction, not a
  per-frame ``ValueError`` that the pipeline would swallow into "no lane".
* A frame with no lane returns ``None``. Malformed input raises
  :class:`~adas.core.exceptions.PerceptionError`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from adas.core.exceptions import ConfigurationError, PerceptionError
from adas.core.logger import setup_logger
from adas.core.models import LaneLine, LaneModel
from adas.perception.geometry import CameraConfig, LaneGeometry
from adas.perception.lane import LaneBackend, lane_model_from_boundaries

logger = setup_logger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ``(px/255 - mean) / std`` refactored to ``px * SCALE - SHIFT`` so the whole
# normalisation is one fused pass per channel over the cropped uint8 image
# instead of four passes over the full stretched float image. Measured on this
# Xavier NX with a 1280x720 frame: 30.5 ms -> 8.1 ms, max abs difference
# 4.8e-07 against the naive form.
_PIXEL_SCALE = (1.0 / (255.0 * IMAGENET_STD)).astype(np.float32)
_PIXEL_SHIFT = (IMAGENET_MEAN / IMAGENET_STD).astype(np.float32)

#: Reference decoder window: the expectation is taken over the argmax cell plus
#: this many cells either side (``demo.py`` default ``local_width = 1``).
#: Widening it to the whole grid -- as ``deploy/trt_infer.py`` does -- drags
#: every recovered x toward the grid mean, i.e. toward the image centre, which
#: makes a lane-centering controller silently under-steer.
DEFAULT_LOCAL_WIDTH = 1


@dataclass(frozen=True)
class UFLDDatasetSpec:
    """Dataset-dependent constants. The anchors are NOT the same across datasets.

    ``row_anchor`` / ``col_anchor`` are fractions of the ORIGINAL frame height /
    width respectively (see the module docstring).
    """

    name: str
    train_width: int
    train_height: int
    crop_ratio: float
    num_row: int
    num_col: int
    num_cell_row: int
    num_cell_col: int
    num_lanes: int = 4
    row_lane_idx: Tuple[int, ...] = (1, 2)
    col_lane_idx: Tuple[int, ...] = (0, 3)
    row_anchor_start: float = 0.42
    row_anchor_end: float = 1.0
    col_anchor_start: float = 0.0
    col_anchor_end: float = 1.0

    @property
    def resize_height(self) -> int:
        """Height the full frame is stretched to before the bottom crop.

        ``int(train_height / crop_ratio)`` -- truncated, matching
        ``torchvision.transforms.Resize((int(h/r), w))`` in the reference.
        """
        return int(self.train_height / self.crop_ratio)

    @property
    def row_anchor(self) -> np.ndarray:
        return np.linspace(self.row_anchor_start, self.row_anchor_end, self.num_row)

    @property
    def col_anchor(self) -> np.ndarray:
        return np.linspace(self.col_anchor_start, self.col_anchor_end, self.num_col)

    def expected_shapes(self) -> Dict[str, Tuple[int, int, int]]:
        """Output shapes with the batch dimension already dropped."""
        return {
            "loc_row": (self.num_cell_row, self.num_row, self.num_lanes),
            "loc_col": (self.num_cell_col, self.num_col, self.num_lanes),
            "exist_row": (2, self.num_row, self.num_lanes),
            "exist_col": (2, self.num_col, self.num_lanes),
        }


CULANE = UFLDDatasetSpec(
    name="culane",
    train_width=1600,
    train_height=320,
    crop_ratio=0.6,
    num_row=72,
    num_col=81,
    num_cell_row=200,
    num_cell_col=100,
)

# Tusimple's row anchors are absolute pixel rows of a 720-high image, normalised:
# linspace(160, 710, num_row) / 720 -> 0.2222 .. 0.9861.
TUSIMPLE = UFLDDatasetSpec(
    name="tusimple",
    train_width=800,
    train_height=320,
    crop_ratio=0.8,
    num_row=56,
    num_col=41,
    num_cell_row=100,
    num_cell_col=100,
    row_anchor_start=160.0 / 720.0,
    row_anchor_end=710.0 / 720.0,
)

CURVELANES = UFLDDatasetSpec(
    name="curvelanes",
    train_width=1600,
    train_height=800,
    crop_ratio=0.8,
    num_row=72,
    num_col=81,
    num_cell_row=200,
    num_cell_col=100,
    row_anchor_start=0.4,
    row_anchor_end=1.0,
)

DATASET_SPECS: Dict[str, UFLDDatasetSpec] = {
    CULANE.name: CULANE,
    TUSIMPLE.name: TUSIMPLE,
    CURVELANES.name: CURVELANES,
}


@dataclass
class LanePolyline:
    """One decoded lane boundary in source-frame pixels."""

    index: int
    points: List[Tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.0
    head: str = "row"


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    # -inf entries (masked-out grid cells) become 0 after exp; guard the all-inf row.
    z = np.where(np.isfinite(z), z, -np.inf)
    e = np.exp(z)
    total = e.sum(axis=axis, keepdims=True)
    return e / np.where(total > 0.0, total, 1.0)


def _drop_batch(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise PerceptionError("UFLD %s: batch size must be 1, got %d" % (name, arr.shape[0]))
        arr = arr[0]
    if arr.ndim != 3:
        raise PerceptionError("UFLD %s: expected 3 or 4 dims, got shape %s" % (name, arr.shape))
    return arr


def _decode_head(
    loc: np.ndarray,
    exist: np.ndarray,
    lane_indices: Sequence[int],
    anchor: np.ndarray,
    presence_ratio: float,
    local_width: int,
    scale_along: float,
    scale_across: float,
    axis_is_row: bool,
) -> Dict[int, LanePolyline]:
    """Shared row/column decode.

    ``scale_along`` scales the location expectation (original width for the row
    head, original height for the column head); ``scale_across`` scales the
    anchor (original height for the row head, original width for the column
    head). ``presence_ratio`` is the reference gate: a row lane needs more than
    ``num_cls / 2`` present anchors, a column lane more than ``num_cls / 4``.
    """
    num_grid, num_cls, num_lanes = loc.shape
    if anchor.shape[0] != num_cls:
        raise PerceptionError(
            "UFLD anchor length %d does not match tensor num_cls %d -- the engine "
            "and the configured dataset disagree" % (anchor.shape[0], num_cls)
        )
    present = exist.argmax(0).astype(bool)  # (num_cls, num_lanes)
    p_present = _softmax(exist, axis=0)[1]  # (num_cls, num_lanes)
    max_idx = loc.argmax(0)  # (num_cls, num_lanes)
    offsets = np.arange(-int(local_width), int(local_width) + 1, dtype=np.int64)

    out: Dict[int, LanePolyline] = {}
    for lane_i in lane_indices:
        if lane_i >= num_lanes:
            continue
        column = present[:, lane_i]
        if int(column.sum()) <= num_cls * presence_ratio:
            continue
        ks = np.nonzero(column)[0]
        idx = max_idx[ks, lane_i].astype(np.int64)
        ind = idx[:, None] + offsets[None, :]
        in_range = (ind >= 0) & (ind <= num_grid - 1)
        ind_clipped = np.clip(ind, 0, num_grid - 1)
        logits = loc[ind_clipped, ks[:, None], lane_i]
        logits = np.where(in_range, logits, -np.inf)
        weights = _softmax(logits, axis=1)
        expectation = (weights * ind_clipped).sum(axis=1) + 0.5
        along = expectation / float(num_grid - 1) * float(scale_along)
        across = anchor[ks] * float(scale_across)
        if axis_is_row:
            points = [(float(x), float(y)) for x, y in zip(along, across)]
        else:
            points = [(float(x), float(y)) for x, y in zip(across, along)]
        probs = p_present[ks, lane_i]
        coverage = float(len(ks)) / float(num_cls)
        confidence = float(np.clip(float(probs.mean()) * coverage, 0.0, 1.0))
        out[int(lane_i)] = LanePolyline(
            index=int(lane_i),
            points=points,
            confidence=confidence,
            head="row" if axis_is_row else "col",
        )
    return out


def pred_to_coords(
    loc_row,
    exist_row,
    loc_col,
    exist_col,
    spec: UFLDDatasetSpec = CULANE,
    original_width: int = 1640,
    original_height: int = 590,
    local_width: int = DEFAULT_LOCAL_WIDTH,
) -> Dict[int, LanePolyline]:
    """Decode the four UFLD-v2 heads into per-lane polylines in SOURCE pixels.

    Faithful to ``demo.py::pred2coords`` with two deliberate improvements: the
    lane index is preserved (the reference throws it away into a flat list) and
    the coordinates stay floats.

    Returns a dict keyed by UFLD lane index. Lanes that fail the existence gate
    are absent from the dict; an empty dict means no lane was detected, which is
    a valid result, not an error.
    """
    loc_row = _drop_batch(loc_row, "loc_row")
    exist_row = _drop_batch(exist_row, "exist_row")
    loc_col = _drop_batch(loc_col, "loc_col")
    exist_col = _drop_batch(exist_col, "exist_col")
    if exist_row.shape[0] != 2 or exist_col.shape[0] != 2:
        raise PerceptionError(
            "UFLD existence heads must have 2 channels, got exist_row%s exist_col%s"
            % (exist_row.shape, exist_col.shape)
        )

    lanes = _decode_head(
        loc_row,
        exist_row,
        spec.row_lane_idx,
        spec.row_anchor,
        presence_ratio=0.5,
        local_width=local_width,
        scale_along=float(original_width),
        scale_across=float(original_height),
        axis_is_row=True,
    )
    lanes.update(
        _decode_head(
            loc_col,
            exist_col,
            spec.col_lane_idx,
            spec.col_anchor,
            presence_ratio=0.25,
            local_width=local_width,
            scale_along=float(original_height),
            scale_across=float(original_width),
            axis_is_row=False,
        )
    )
    return lanes


def coords_to_lane_model(
    lanes: Dict[int, LanePolyline],
    frame_width: int,
    frame_height: int,
    camera: Optional[CameraConfig] = None,
    spec: UFLDDatasetSpec = CULANE,
) -> Optional[Tuple[LaneModel, Optional[LaneGeometry]]]:
    """Assemble a :class:`LaneModel` from decoded UFLD lanes.

    The ego boundaries are taken by INDEX (``spec.row_lane_idx``, i.e. 1 and 2
    for CULane), never re-guessed from image geometry. Column-head lanes (the
    adjacent lanes) are attached as informational ``LaneLine`` entries with
    ``coeffs=None`` -- they are near-horizontal, so no ``x = f(y)`` fit is
    attempted for them.

    Returns ``None`` when neither ego boundary was detected.
    """
    left_idx, right_idx = (spec.row_lane_idx + (1, 2))[:2]
    left = lanes.get(left_idx)
    right = lanes.get(right_idx)
    if left is None and right is None:
        return None

    extra: List[LaneLine] = []
    for idx in spec.col_lane_idx:
        poly = lanes.get(idx)
        if poly is None:
            continue
        extra.append(
            LaneLine(
                points_px=[(float(x), float(y)) for x, y in poly.points],
                coeffs=None,
                confidence=float(poly.confidence),
                index=int(idx),
            )
        )

    return lane_model_from_boundaries(
        left_points_px=left.points if left is not None else None,
        right_points_px=right.points if right is not None else None,
        frame_width=frame_width,
        frame_height=frame_height,
        camera=camera,
        left_confidence=left.confidence if left is not None else 0.0,
        right_confidence=right.confidence if right is not None else 0.0,
        extra_lines=extra,
    )


def resolve_output_tensors(
    outputs: Dict[str, np.ndarray],
    spec: UFLDDatasetSpec,
) -> Dict[str, np.ndarray]:
    """Bind the four UFLD tensors by name, verified against the expected shapes.

    Name matching is tried first (the reference exporter writes
    ``loc_row, loc_col, exist_row, exist_col``). If any name is missing or its
    shape disagrees with ``spec``, the tensors are re-resolved purely by shape.
    Positional order is never guessed: the historical fallback unpacked
    ``loc_row, exist_row, loc_col, exist_col`` while the exporter emits
    ``loc_row, loc_col, exist_row, exist_col``, silently binding ``exist_row``
    to ``loc_col``.

    Raises :class:`~adas.core.exceptions.PerceptionError` listing every observed
    shape when the four tensors cannot be identified unambiguously.
    """
    expected = spec.expected_shapes()
    observed = {name: np.asarray(arr) for name, arr in outputs.items()}

    def squeezed(arr: np.ndarray) -> Tuple[int, ...]:
        shape = tuple(int(s) for s in arr.shape)
        if len(shape) == 4 and shape[0] == 1:
            shape = shape[1:]
        return shape

    by_name: Dict[str, np.ndarray] = {}
    for key in expected:
        for name, arr in observed.items():
            if key in name.lower():
                by_name[key] = arr
                break
    if len(by_name) == 4 and all(squeezed(by_name[k]) == expected[k] for k in expected):
        return by_name

    by_shape: Dict[str, np.ndarray] = {}
    used = set()
    for key, want in expected.items():
        for name, arr in observed.items():
            if name in used:
                continue
            if squeezed(arr) == want:
                by_shape[key] = arr
                used.add(name)
                break
    if len(by_shape) == 4:
        logger.warning(
            "UFLD outputs resolved by shape, not by name (engine names: %s)",
            sorted(observed),
        )
        return by_shape

    raise PerceptionError(
        "cannot identify the four UFLD tensors for dataset %r. expected %s; "
        "engine produced %s"
        % (
            spec.name,
            {k: list(v) for k, v in expected.items()},
            {name: list(squeezed(arr)) for name, arr in observed.items()},
        )
    )


def preprocess(
    image: np.ndarray,
    spec: UFLDDatasetSpec = CULANE,
    interpolation: Optional[int] = None,
) -> np.ndarray:
    """Build the ``1x3xHxW`` float32 NCHW blob the UFLD engine expects.

    ``image`` must be an ``HxWx3`` uint8 **BGR** frame, as produced by every
    source in ``adas.runtime.capture``. Anything else raises
    :class:`~adas.core.exceptions.PerceptionError` rather than being silently
    truncated into a uint8 canvas.
    """
    import cv2

    if not isinstance(image, np.ndarray):
        raise PerceptionError("ufld preprocess needs a numpy image, got %s" % type(image))
    if image.ndim != 3 or image.shape[2] != 3:
        raise PerceptionError("ufld preprocess needs HxWx3 BGR, got shape %s" % (image.shape,))
    if image.dtype != np.uint8:
        raise PerceptionError(
            "ufld preprocess needs a uint8 BGR frame, got dtype %s. Convert at the "
            "capture boundary; do not hand float frames to the perception stack." % image.dtype
        )
    if interpolation is None:
        interpolation = cv2.INTER_LINEAR

    stretched = cv2.resize(
        image,
        (spec.train_width, spec.resize_height),
        interpolation=interpolation,
    )
    crop = stretched[-spec.train_height :, :, :]
    blob = np.empty((1, 3, spec.train_height, spec.train_width), dtype=np.float32)
    for c in range(3):
        # ``2 - c`` performs the BGR -> RGB swap during the same gather that
        # writes the NCHW plane, so no separate cvtColor pass is needed.
        np.multiply(crop[:, :, 2 - c], _PIXEL_SCALE[c], out=blob[0, c], dtype=np.float32)
        blob[0, c] -= _PIXEL_SHIFT[c]
    return blob


class UFLDLaneEstimator(LaneBackend):
    """Lane estimator backed by a UFLD-v2 TensorRT engine.

    Not thread-safe: it owns one execution context and one set of host buffers.

    Raises :class:`~adas.core.exceptions.ConfigurationError` at construction if
    the engine's input shape or output shapes disagree with the configured
    dataset, so a wrong engine fails loudly at startup instead of raising every
    frame into the pipeline's exception handler and reporting "no lane" forever.
    """

    name = "ufld"

    def __init__(
        self,
        engine_path: str,
        dataset: str = "culane",
        camera: Optional[CameraConfig] = None,
        local_width: int = DEFAULT_LOCAL_WIDTH,
        interpolation: Optional[int] = None,
        min_confidence: float = 0.0,
        input_width: Optional[int] = None,
        input_height: Optional[int] = None,
        crop_ratio: Optional[float] = None,
        num_row: Optional[int] = None,
        num_col: Optional[int] = None,
    ) -> None:
        """``input_width``/``input_height``/``crop_ratio``/``num_row``/``num_col``
        are the legacy ``LaneConfig`` fields that ``adas.perception.factory``
        still passes. They are no longer the source of truth -- ``dataset``
        selects a whole coherent :class:`UFLDDatasetSpec`, because these five
        numbers plus the anchor formula have to agree or the decode is silently
        wrong. Any supplied value that disagrees with the selected dataset
        raises :class:`~adas.core.exceptions.ConfigurationError` instead of
        being quietly honoured."""
        from adas.infer.trt_engine import TrtEngine

        key = str(dataset).lower()
        if key not in DATASET_SPECS:
            raise ConfigurationError(
                "unknown UFLD dataset %r; known: %s" % (dataset, sorted(DATASET_SPECS))
            )
        self.spec = DATASET_SPECS[key]
        for field_name, supplied in (
            ("input_width", input_width),
            ("input_height", input_height),
            ("crop_ratio", crop_ratio),
            ("num_row", num_row),
            ("num_col", num_col),
        ):
            if supplied is None:
                continue
            attr = {
                "input_width": "train_width",
                "input_height": "train_height",
                "crop_ratio": "crop_ratio",
                "num_row": "num_row",
                "num_col": "num_col",
            }[field_name]
            expected = getattr(self.spec, attr)
            if abs(float(supplied) - float(expected)) > 1e-6:
                raise ConfigurationError(
                    "LaneConfig.%s=%r contradicts UFLD dataset %r (%s=%r). These are "
                    "not independent knobs: change `dataset`, not the individual "
                    "dimensions." % (field_name, supplied, self.spec.name, attr, expected)
                )
        self.local_width = int(local_width)
        if self.local_width < 0:
            raise ConfigurationError("local_width must be >= 0, got %r" % (local_width,))
        self.min_confidence = float(min_confidence)
        self.interpolation = interpolation
        self.camera = camera
        self.last_geometry: Optional[LaneGeometry] = None
        self.last_lanes: Dict[int, LanePolyline] = {}

        path = Path(engine_path)
        if not path.exists():
            raise PerceptionError("UFLD engine not found: %s" % engine_path)
        self.engine = TrtEngine(str(path))

        shape = tuple(int(s) for s in self.engine.input_shape)
        want = (1, 3, self.spec.train_height, self.spec.train_width)
        if shape != want:
            self.engine.close()
            raise ConfigurationError(
                "UFLD engine %s has input shape %s but dataset %r expects %s. "
                "Either the engine or the configured dataset is wrong."
                % (path, list(shape), self.spec.name, list(want))
            )

        # One zero-input inference at construction: it proves the engine runs and
        # gives the authoritative output shapes to validate against the dataset
        # spec. Costs ~17 ms once, and turns a per-frame IndexError deep in the
        # decoder into a clear startup error.
        try:
            probe = self.engine.infer(
                {self.engine.input_name: np.zeros(want, dtype=np.float32)}
            )
            resolve_output_tensors(probe, self.spec)
        except PerceptionError as exc:
            self.engine.close()
            raise ConfigurationError(str(exc)) from exc
        except Exception as exc:
            self.engine.close()
            raise ConfigurationError(
                "UFLD engine %s failed its startup probe inference: %s" % (path, exc)
            ) from exc

        if self.camera is not None:
            logger.info(
                "UFLDLaneEstimator loaded %s (dataset=%s, camera=%s, calibrated=%s)",
                path,
                self.spec.name,
                self.camera.label,
                self.camera.calibrated,
            )
        else:
            logger.warning(
                "UFLDLaneEstimator loaded %s (dataset=%s) with NO camera model: "
                "lane geometry will be pixel-only, curvature_m will report "
                "'straight' and lateral offset in metres is unavailable.",
                path,
                self.spec.name,
            )

    def estimate(self, frame: object, width: int, height: int) -> Optional[LaneModel]:
        """Run one frame. Returns ``None`` when no ego lane boundary is detected."""
        blob = preprocess(frame, self.spec, self.interpolation)
        outputs = self.engine.infer({self.engine.input_name: blob})
        image = frame  # validated by preprocess
        frame_h, frame_w = image.shape[:2]
        return self.decode(outputs, frame_w, frame_h)

    def decode(
        self,
        outputs: Dict[str, np.ndarray],
        frame_width: int,
        frame_height: int,
    ) -> Optional[LaneModel]:
        """Decode raw engine outputs. Separated from :meth:`estimate` so the
        decoder can be unit-tested against saved fixture tensors with no GPU."""
        tensors = resolve_output_tensors(outputs, self.spec)
        lanes = pred_to_coords(
            tensors["loc_row"],
            tensors["exist_row"],
            tensors["loc_col"],
            tensors["exist_col"],
            spec=self.spec,
            original_width=int(frame_width),
            original_height=int(frame_height),
            local_width=self.local_width,
        )
        self.last_lanes = lanes
        built = coords_to_lane_model(
            lanes,
            frame_width=int(frame_width),
            frame_height=int(frame_height),
            camera=self.camera,
            spec=self.spec,
        )
        if built is None:
            self.last_geometry = None
            return None
        model, geometry = built
        self.last_geometry = geometry
        if model.confidence < self.min_confidence:
            logger.debug(
                "UFLD lane confidence %.3f below min_confidence %.3f; reporting no lane",
                model.confidence,
                self.min_confidence,
            )
            return None
        return model

    def close(self) -> None:
        engine = getattr(self, "engine", None)
        if engine is not None:
            engine.close()
            self.engine = None


def euclidean_lane_width_px(model: LaneModel, y_px: float) -> float:
    """Pixel separation of the two boundaries at row ``y_px``. Diagnostic helper."""
    la, lb, lc = model.left_coeffs
    ra, rb, rc = model.right_coeffs
    left = la * y_px * y_px + lb * y_px + lc
    right = ra * y_px * y_px + rb * y_px + rc
    width = right - left
    return float(width) if math.isfinite(width) else 0.0
