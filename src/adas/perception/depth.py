"""Monocular relative depth (MiDaS v2.1 small) as a *relative* depth channel.

Status: NOT a metric range channel
----------------------------------
This channel does **not** publish metres by default, and on the hardware and
footage this project ships with it cannot be made to. That is a measured
conclusion, not a caution; the numbers are below.

The original design goal was an independent metric cross-check on the
box-geometry range. Both :func:`adas.perception.geometry.pinhole_range` and
:func:`adas.perception.geometry.ground_plane_range` read the *same* box from the
*same* detector, so a detector that shrinks a box corrupts both estimates in the
same direction at the same instant. A per-pixel depth network shares none of
that, so its disagreement would be real information.

The idea is sound. The model is not good enough to carry it.

What was measured (Jetson Xavier NX, 2026-09-13)
------------------------------------------------
Engine ``models/midas_v21_small_256.engine`` (FP16, 256x256), detector
``yolox_nano.engine``, clip ``Ultra-Fast-Lane-Detection-v2/example.mp4``,
default uncalibrated :class:`~adas.perception.geometry.CameraConfig`.

Per-object rank correlation against the box-height pinhole range, over 199
(detection, frame) pairs sampled every 10th frame. Rank correlation is invariant
under any monotone rescaling, so these numbers do not depend on how the affine
fit is anchored or on whether the camera is calibrated -- they measure only
whether the network can tell a near vehicle from a far one::

    sampling of the object's disparity        Spearman   far(>30m) vs near mean disparity
    median, lower-central 55-95% (shipped)      +0.139        399  vs  392
    75th percentile, lower-central              +0.177        413  vs  415
    25th percentile, lower-central              +0.142        381  vs  374
    median, wheel band 80-100%                  +0.212        416  vs  427
    median, upper body 5-45%                    +0.179        333  vs  323
    10-90% trimmed mean, whole box              +0.180        364  vs  358
    median over a 3x-zoomed second inference    -0.275        653  vs  562

Objects beyond 30 m carry the *same* mean disparity as objects inside 20 m. The
network is not compressing the range scale -- it is not resolving range at all
at the object level. Eight sampling statistics and a second, higher-resolution
inference pass over the horizon band were tried; the best of them reaches 0.21
against a bar of 0.80, and the extra-resolution pass is actively inverted.

End to end, with the road-plane affine fit applied, the published metres were::

    pinhole  min 5.3  p50 16.9  max 61.5 m      (n=101, every 20th frame)
    midas    min 3.7  p50  8.7  max 13.4 m
    pearson 0.214  spearman 0.252
    far objects (pinhole > 30 m): n=17, midas median 8.9 m, max 11.9 m

A true 5-61 m spread arrived as 3.7-13.4 m. Consumed by
``adas.control.arbiter._fuse_range``, which took ``min(pinhole, depth)``, that
dragged every range toward ~9 m and produced phantom braking: 400 frames at
``--ego-speed 25`` gave 98 ``range_channel_disagreement`` violations and 8 AEB
lines with ``--depth midas`` against 0 and 5 with ``--depth off``.

Why it fails, and what still works
----------------------------------
The failure is specific and worth recording, because it is *not* the affine fit.
Probing the road surface at known forward ranges through the camera homography
recovers a clean ``disparity = a / Z + b`` law with a relative fit residual of
0.04-0.08 on this clip. The network resolves the *road* depth gradient well.

What it cannot do is place an *object* on that gradient. MiDaS v2.1 small at
256x256 assigns a vehicle a disparity driven by its appearance and saliency, not
by its position in the scene, and a car 25 px tall at 720p is ~9 cells on the
256x256 map. Concretely, on frame 300 the near car (pinhole 13.4 m) sampled 348
and the far car (pinhole 54.3 m) sampled 410 -- the far car read *nearer*.

One variant does better: sampling the road strip just below the box bottom edge
reaches Spearman 0.686. It is excluded anyway, for two reasons. It still
collapses the far field (median 7.8 m, max 11.1 m for objects beyond 30 m), and
it consumes ``box.y2`` -- the same pixel row ``ground_plane_range`` consumes -- so
it forfeits the independence that was the entire justification for the channel.
It would be a slower, noisier copy of a measurement the stack already has.

A second, independent problem that this channel is *not* the right place to fix:
on the default uncalibrated camera, ``ground_plane_range`` itself spans only
3.5-10.1 m over this clip while ``pinhole_range`` spans 5.3-61.5 m, a median
relative disagreement of 132%. The road-plane anchors are projected through that
same uncalibrated homography, so even a perfect depth model anchored on them
would inherit its scale error. Metric depth here needs a calibrated camera
*and* a better model; neither is available.

What this module publishes now
------------------------------
``update()`` returns one :class:`~adas.core.models.RangeEstimate` per box, and
by default every one of them is ``RangeSource.UNAVAILABLE`` with confidence 0,
which the pipeline drops before the arbiter ever sees it. No metres, no
0.70-confidence second opinion, no ``min(pinhole, depth)``.

The relative information the network *does* carry is published separately, as
:class:`OrdinalDepth` via :meth:`DepthRangeChannel.ordinal_readings`. That type
has no metric field and is not a ``RangeEstimate``, so it cannot be mistaken for
metres by a consumer that was not written for it. Given the correlations above,
even the ordering should be treated as weak evidence.

Re-enabling metres, if the model or the calibration improves
------------------------------------------------------------
``publish_metric=True`` asks for metres. It is not sufficient. The channel
continuously audits itself: every frame it records the (reference range, sampled
disparity) pairs it was given and computes a rolling Spearman correlation over
the last ``ordering_window`` of them. Metres are published only while that
measured correlation is at or above ``min_ordering_spearman`` (0.80) over at
least ``min_ordering_pairs`` samples. On this clip the audit settles around 0.2
and the gate never opens. It fails closed: no reference ranges means no
evidence, which means no metres.

So the way to turn this channel back on is to make it earn the correlation on
real footage. Setting the flag alone does nothing, which is the point.

The metric machinery -- :class:`DepthScale`, :func:`fit_depth_scale`,
:func:`fit_depth_scale_robust`, :func:`road_plane_anchors` -- is kept and still
runs, because the road-plane fit is genuinely good and its diagnostics are
reported in :meth:`DepthRangeChannel.stats`. Only publication is gated.

Cadence
-------
The engine runs once every ``cadence_frames`` frames (6.3 ms of GPU each time)
and the disparity map is reused for the rest of the window with a linearly
decayed confidence. Past ``cadence_frames - 1`` frames of staleness the map is
discarded outright.

Units
-----
``distance_m``, when it is ever published, is metres forward along the optical
axis, matching :mod:`adas.perception.geometry`. Disparity is unitless network
output, larger = nearer. :class:`OrdinalDepth` is unitless throughout. Cadence
is frames. Timings are wall-clock milliseconds.

Engine contract (read back from ``models/midas_v21_small_256.engine``)
----------------------------------------------------------------------
::

    in  "0"   float32 (1, 3, 256, 256)
    out "797" float32 (1, 256, 256)

The binding names really are the strings ``"0"`` and ``"797"`` -- torch tensor
ids kept by the v2.1 export -- so they are read from the engine, never
hardcoded. Preprocessing: BGR->RGB, **stretch** to 256x256 (aspect ratio is not
preserved), ``/255``, ImageNet mean/std.

Failure behaviour
-----------------
If the engine file is absent or fails to load, the channel constructs anyway
with ``available = False`` and ``is_mock = True``, logs one WARNING, and returns
``RangeSource.UNAVAILABLE`` estimates forever. It never fabricates a range.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from adas.core.exceptions import PerceptionError
from adas.core.logger import setup_logger
from adas.core.models import BoundingBox, RangeEstimate, RangeSource
from adas.perception.detection import PREPROC_MIDAS, blob_from_canvas

logger = setup_logger(__name__)

#: Ranges outside this band are refused as scale anchors: closer than this the
#: ground-plane contact point is usually truncated, further than this a pixel of
#: contact-point error is worth many metres.
MIN_ANCHOR_RANGE_M = 4.0
MAX_ANCHOR_RANGE_M = 120.0

#: An anchor whose reference range is less confident than this is not used.
MIN_ANCHOR_CONFIDENCE = 0.15

#: Ceiling on the confidence of a metric estimate, whatever the fit quality --
#: the channel inherits the camera geometry's scale error through its anchors.
#: Only reachable when the metric gate is open; the default path publishes
#: confidence 0.
MAX_DEPTH_CONFIDENCE = 0.75

#: Reported range is clamped here; beyond it the inverse-depth inversion is
#: numerically meaningless.
MAX_DEPTH_RANGE_M = 200.0

#: Forward ranges, in metres, at which the road surface is probed for anchors.
#: Geometric spacing because the pixel-to-range sensitivity is ~1/Z^2: a linear
#: ladder would put almost every sample in the far field where it is worthless.
ROAD_ANCHOR_RANGES_M = (6.0, 8.0, 11.0, 15.0, 20.0, 27.0, 36.0, 48.0, 64.0, 85.0)

#: Lateral offsets, in metres from the optical axis, probed at each range. Kept
#: inside a single lane so the samples stay on tarmac rather than on a verge.
ROAD_ANCHOR_LATERAL_M = (-1.4, 0.0, 1.4)

#: A road probe this close to a detection box (in box-width fractions) is
#: dropped: the vehicle, not the road, owns those pixels.
ROAD_ANCHOR_BOX_MARGIN = 0.08

#: Box anchoring is a weaker, partly circular reference than the road plane.
BOX_ANCHOR_CONFIDENCE_FACTOR = 0.55

#: Rank correlation the channel must demonstrate, against the reference ranges
#: it is handed, before it is allowed to publish metres. 0.80 is the bar this
#: project set for calling a range channel informative; MiDaS v2.1 small at
#: 256x256 measured 0.14-0.25 on the shipped clip (see the module docstring), so
#: on that footage this gate stays shut.
MIN_ORDERING_SPEARMAN = 0.80

#: How many (reference range, sampled disparity) pairs the rolling audit keeps.
#: ~12 s of a 2-vehicle scene at 20 Hz with cadence 5.
ORDERING_WINDOW_PAIRS = 240

#: Pairs required before the audit's correlation is believed at all. Below this
#: the estimate is dominated by sampling noise, and an unbelieved audit means no
#: metres: the gate fails closed.
MIN_ORDERING_PAIRS = 24

#: Measured per-object rank correlation of this model, at this resolution, on
#: ``Ultra-Fast-Lane-Detection-v2/example.mp4``, over 199 (detection, frame)
#: pairs. Recorded so the shortfall against MIN_ORDERING_SPEARMAN is a documented
#: number rather than folklore. Best of eight sampling statistics plus a
#: 3x-zoomed second inference pass; see the module docstring for the full table.
MEASURED_OBJECT_SPEARMAN_MIDAS_SMALL_256 = 0.212


def spearman_rho(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation of two equal-length samples.

    Average ranks for ties, so a run of identical disparities -- which is what a
    saturated depth map produces -- neither inflates nor breaks the statistic.
    Returns NaN for fewer than 3 usable pairs or when either side is constant
    (rank correlation is undefined there, and the callers treat NaN as "no
    evidence", which is the honest reading).

    Kept local rather than pulled from scipy: scipy is not installed on the
    target board and this is twenty lines of numpy.
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size != y.size:
        raise PerceptionError("spearman_rho got %d and %d samples" % (x.size, y.size))
    usable = np.isfinite(x) & np.isfinite(y)
    x = x[usable]
    y = y[usable]
    if x.size < 3:
        return float("nan")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    if float(rx.std()) <= 1e-12 or float(ry.std()) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks of ``values``, ties sharing their mean rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    ordered = values[order]
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or ordered[index] != ordered[start]:
            if index - start > 1:
                ranks[order[start:index]] = float(np.mean(ranks[order[start:index]]))
            start = index
    return ranks


@dataclass(frozen=True)
class OrdinalDepth:
    """One box's *relative* depth reading. Unitless. Never metres.

    This is deliberately not a :class:`~adas.core.models.RangeEstimate` and has
    no ``distance_m``: the arbiter's range fusion cannot consume it by accident,
    which is the whole reason it exists as a separate type. See the module
    docstring for the measured correlation -- on the shipped model this ordering
    is weak evidence, not a measurement.

    Attributes
    ----------
    disparity:
        Raw network output sampled over the box's lower-central region. Larger
        means nearer. The scale is arbitrary and changes every frame.
    rank:
        0 for the nearest box in this frame, ``of - 1`` for the farthest, by
        sampled disparity. Ties broken arbitrarily but stably.
    of:
        How many boxes in this frame produced a usable sample.
    normalized:
        ``rank / (of - 1)``: 0.0 nearest, 1.0 farthest. 0.0 when ``of`` is 1.
    frame_id:
        Frame the *disparity map* came from, which with a cadence above 1 is not
        necessarily the frame this reading was requested for.
    stale_frames:
        How many frames old the map was when sampled. 0 is fresh.
    """

    disparity: float
    rank: int
    of: int
    normalized: float
    frame_id: int
    stale_frames: int


# --------------------------------------------------------------------------- #
# Scale alignment
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DepthScale:
    """The per-frame affine map from network disparity to metric inverse range.

    ``disparity = a * (1 / Z_metres) + b``. ``a`` must be positive: disparity
    grows as range shrinks. ``rmse`` is the fit residual in disparity units,
    ``rel_rmse`` the same normalised by the disparity spread of the anchors,
    which is what the confidence is derived from.
    """

    a: float = 0.0
    b: float = 0.0
    anchors: int = 0
    rmse: float = 0.0
    rel_rmse: float = 1.0
    valid: bool = False

    def range_m(self, disparity: float) -> Optional[float]:
        """Invert one disparity sample to metres, or ``None`` if unusable."""
        if not self.valid:
            return None
        denominator = float(disparity) - self.b
        if not math.isfinite(denominator) or denominator <= 1e-6:
            # At or beyond the fitted horizon: the model says "very far", which
            # is not a measurement.
            return None
        distance = self.a / denominator
        if not math.isfinite(distance) or distance <= 0.0:
            return None
        return float(min(distance, MAX_DEPTH_RANGE_M))


def fit_depth_scale(
    disparities: Sequence[float],
    ranges_m: Sequence[float],
    min_anchors: int = 2,
    previous: Optional[DepthScale] = None,
) -> DepthScale:
    """Least-squares fit of ``disparity = a / Z + b`` over the anchor set.

    With ``min_anchors`` or more anchors that have a usable spread in ``1/Z``,
    both terms are fitted. With exactly one anchor, or when the anchors are all
    at effectively the same range (the design matrix is singular), only ``a`` is
    fitted and ``b`` is held at the previous frame's value -- or 0 if there is
    no previous fit. That case is flagged by a ``rel_rmse`` of 1.0, which drives
    the confidence down; it is a degraded fit, not a good one.

    Returns an invalid :class:`DepthScale` (``valid=False``) whenever the fit
    would be meaningless, including a non-positive ``a``, which means the
    disparity map disagrees with the geometry about which object is nearer.
    """
    disp = np.asarray(disparities, dtype=np.float64).ravel()
    rng = np.asarray(ranges_m, dtype=np.float64).ravel()
    if disp.size != rng.size:
        raise PerceptionError(
            "fit_depth_scale got %d disparities and %d ranges" % (disp.size, rng.size)
        )
    usable = np.isfinite(disp) & np.isfinite(rng) & (rng > 0.0)
    disp = disp[usable]
    rng = rng[usable]
    if disp.size == 0:
        return DepthScale()

    inv = 1.0 / rng
    spread = float(inv.max() - inv.min())
    fitted_b = float(previous.b) if (previous is not None and previous.valid) else 0.0

    if disp.size >= max(2, int(min_anchors)) and spread > 1e-4:
        design = np.stack([inv, np.ones_like(inv)], axis=1)
        try:
            solution, _res, rank, _sv = np.linalg.lstsq(design, disp, rcond=None)
        except np.linalg.LinAlgError:
            return DepthScale()
        if rank < 2:
            return DepthScale()
        a = float(solution[0])
        b = float(solution[1])
        residual = disp - (a * inv + b)
        degraded = False
    else:
        # Scale-only fit around a held shift.
        denominator = float(np.dot(inv, inv))
        if denominator <= 1e-12:
            return DepthScale()
        a = float(np.dot(inv, disp - fitted_b) / denominator)
        b = fitted_b
        residual = disp - (a * inv + b)
        degraded = True

    if not math.isfinite(a) or a <= 0.0 or not math.isfinite(b):
        return DepthScale(anchors=int(disp.size), valid=False)

    rmse = float(np.sqrt(float(np.mean(residual ** 2)))) if residual.size else 0.0
    disparity_spread = float(disp.max() - disp.min())
    if degraded or disparity_spread <= 1e-6:
        rel = 1.0
    else:
        rel = float(min(1.0, rmse / disparity_spread))
    return DepthScale(
        a=a,
        b=b,
        anchors=int(disp.size),
        rmse=rmse,
        rel_rmse=rel,
        valid=True,
    )


def fit_depth_scale_robust(
    disparities: Sequence[float],
    ranges_m: Sequence[float],
    min_anchors: int = 3,
    previous: Optional[DepthScale] = None,
    trim_sigma: float = 3.0,
) -> DepthScale:
    """:func:`fit_depth_scale` with one median-absolute-deviation trim pass.

    A road probe that lands on a vehicle, a shadow or a bridge is an outlier
    whose disparity has nothing to do with its projected range, and least
    squares has no defence against it. One MAD-based rejection pass at
    ``trim_sigma`` removes those without the cost or the nondeterminism of
    RANSAC. The trim only applies if enough anchors survive it.
    """
    disp = np.asarray(disparities, dtype=np.float64).ravel()
    rng = np.asarray(ranges_m, dtype=np.float64).ravel()
    first = fit_depth_scale(disp, rng, min_anchors=min_anchors, previous=previous)
    if not first.valid or disp.size <= min_anchors:
        return first
    residual = disp - (first.a / rng + first.b)
    centre = float(np.median(residual))
    mad = float(np.median(np.abs(residual - centre)))
    if mad <= 1e-9:
        return first
    keep = np.abs(residual - centre) <= trim_sigma * 1.4826 * mad
    if int(keep.sum()) < max(min_anchors, 3) or int(keep.sum()) == disp.size:
        return first
    return fit_depth_scale(disp[keep], rng[keep], min_anchors=min_anchors, previous=previous)


def road_plane_anchors(
    field: "DepthField",
    camera: object,
    frame_width: int,
    frame_height: int,
    boxes: Sequence[BoundingBox] = (),
    ranges_m: Sequence[float] = ROAD_ANCHOR_RANGES_M,
    lateral_m: Sequence[float] = ROAD_ANCHOR_LATERAL_M,
) -> Tuple[List[float], List[float]]:
    """Probe the road surface for ``(disparity, range_m)`` anchor pairs.

    Each ``(lateral, forward)`` point on the road plane is projected into the
    image with the camera's ground-plane homography; points that fall outside
    the frame, at or above the horizon, or inside (a slightly grown) detection
    box are discarded, because a vehicle standing there owns those pixels and
    its disparity is not the road's.

    Returns two parallel lists. They are the *only* metric information the
    depth channel consumes in road-plane mode, and they come from the camera
    extrinsics rather than from any detection.
    """
    if not field.valid or camera is None:
        return [], []
    grown = []
    for box in boxes:
        width = float(box.x2) - float(box.x1)
        height = float(box.y2) - float(box.y1)
        margin_x = width * ROAD_ANCHOR_BOX_MARGIN
        margin_y = height * ROAD_ANCHOR_BOX_MARGIN
        grown.append(
            (
                float(box.x1) - margin_x,
                float(box.y1) - margin_y,
                float(box.x2) + margin_x,
                float(box.y2) + margin_y,
            )
        )
    us: List[float] = []
    vs: List[float] = []
    zs: List[float] = []
    for z_m in ranges_m:
        for x_m in lateral_m:
            projected = camera.ground_to_image(float(x_m), float(z_m))
            if projected is None:
                continue
            u_px, v_px = projected
            if any(x1 <= u_px <= x2 and y1 <= v_px <= y2 for x1, y1, x2, y2 in grown):
                continue
            us.append(float(u_px))
            vs.append(float(v_px))
            zs.append(float(z_m))
    if not us:
        return [], []
    samples = field.sample_points(us, vs, frame_width, frame_height)
    finite = np.isfinite(samples)
    disparities = [float(x) for x in samples[finite]]
    distances = [z for z, ok in zip(zs, finite) if ok]
    return disparities, distances


# --------------------------------------------------------------------------- #
# Disparity field
# --------------------------------------------------------------------------- #


@dataclass
class DepthField:
    """One inference's disparity map plus the provenance needed to age it.

    ``disparity`` is the raw network output at its own resolution (256x256 for
    MiDaS v2.1 small). It is deliberately **not** upsampled to the source frame:
    every consumer here samples a median over a region, and sampling the small
    map in normalised coordinates gives the same answer for a fraction of the
    cost.
    """

    disparity: Optional[np.ndarray] = None
    frame_id: int = -1
    timestamp_s: float = 0.0
    infer_ms: float = 0.0
    is_mock: bool = False

    @property
    def valid(self) -> bool:
        return self.disparity is not None and self.disparity.size > 0

    def sample_points(
        self,
        u_px: np.ndarray,
        v_px: np.ndarray,
        frame_width: int,
        frame_height: int,
        patch: int = 3,
    ) -> np.ndarray:
        """Median disparity in a ``patch`` x ``patch`` window around each pixel.

        Vectorised: one gather and one :func:`numpy.median` call for all points.
        The scalar form costs ~0.34 ms per point on this board (numpy's median
        has a large fixed overhead), which at 30 road probes per frame is 10 ms
        -- more than the network itself.

        ``patch`` is in *disparity-map* cells, so on the 256x256 MiDaS map a
        patch of 3 covers roughly 15x8 source pixels at 1280x720: enough to
        reject a speckle, small enough not to smear across a lane marking.
        Points outside the frame yield NaN.
        """
        u = np.asarray(u_px, dtype=np.float64).ravel()
        v = np.asarray(v_px, dtype=np.float64).ravel()
        if u.size != v.size:
            raise PerceptionError("sample_points got %d u and %d v" % (u.size, v.size))
        out = np.full(u.size, np.nan, dtype=np.float64)
        if not self.valid or u.size == 0 or frame_width <= 0 or frame_height <= 0:
            return out
        inside = (u >= 0.0) & (u < frame_width) & (v >= 0.0) & (v < frame_height)
        if not bool(np.any(inside)):
            return out
        height, width = self.disparity.shape[-2:]
        plane = np.asarray(self.disparity).reshape(height, width)
        xi = np.clip((u[inside] / float(frame_width) * width).astype(np.int64), 0, width - 1)
        yi = np.clip((v[inside] / float(frame_height) * height).astype(np.int64), 0, height - 1)
        half = max(0, int(patch) // 2)
        offsets = np.arange(-half, half + 1, dtype=np.int64)
        xs = np.clip(xi[:, None] + offsets[None, :], 0, width - 1)
        ys = np.clip(yi[:, None] + offsets[None, :], 0, height - 1)
        window = plane[ys[:, :, None], xs[:, None, :]]
        out[inside] = np.median(window.reshape(window.shape[0], -1), axis=1)
        return out

    def sample_point(
        self,
        u_px: float,
        v_px: float,
        frame_width: int,
        frame_height: int,
        patch: int = 3,
    ) -> Optional[float]:
        """Scalar form of :meth:`sample_points`. ``None`` when unusable."""
        value = float(
            self.sample_points([u_px], [v_px], frame_width, frame_height, patch=patch)[0]
        )
        return value if math.isfinite(value) else None

    def sample_box(
        self,
        box: BoundingBox,
        frame_width: int,
        frame_height: int,
        lower_frac: float = 0.55,
        upper_frac: float = 0.95,
        centre_frac: float = 0.5,
    ) -> Optional[float]:
        """Median disparity over the lower-central part of ``box``.

        The lower-central region is the part of a vehicle that faces the camera
        and stands on the road: the top of the box bleeds into sky or into the
        vehicle behind, and the last few per cent of the bottom is road surface.
        A median (not a mean) so one bright background pixel inside the box
        cannot move the estimate.

        Returns ``None`` when the map is absent or the region is empty.
        """
        if not self.valid or frame_width <= 0 or frame_height <= 0:
            return None
        height, width = self.disparity.shape[-2:]
        cx = (float(box.x1) + float(box.x2)) * 0.5
        half = (float(box.x2) - float(box.x1)) * 0.5 * float(centre_frac)
        box_h = float(box.y2) - float(box.y1)
        u0 = (cx - half) / float(frame_width)
        u1 = (cx + half) / float(frame_width)
        v0 = (float(box.y1) + box_h * float(lower_frac)) / float(frame_height)
        v1 = (float(box.y1) + box_h * float(upper_frac)) / float(frame_height)

        x0 = int(np.clip(math.floor(u0 * width), 0, width - 1))
        x1 = int(np.clip(math.ceil(u1 * width), 1, width))
        y0 = int(np.clip(math.floor(v0 * height), 0, height - 1))
        y1 = int(np.clip(math.ceil(v1 * height), 1, height))
        if x1 <= x0:
            x1 = min(width, x0 + 1)
        if y1 <= y0:
            y1 = min(height, y0 + 1)
        patch = np.asarray(self.disparity).reshape(height, width)[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        value = float(np.median(patch))
        return value if math.isfinite(value) else None


# --------------------------------------------------------------------------- #
# Engine wrapper
# --------------------------------------------------------------------------- #


class MiDaSDepthEstimator:
    """MiDaS v2.1 small TensorRT wrapper. Produces raw relative inverse depth.

    Reads its binding names and input size from the engine, so the numeric
    ``"0"`` / ``"797"`` names of the v2.1 export are never hardcoded.
    """

    def __init__(self, engine_path: str = "", engine: object = None) -> None:
        if engine is None:
            from adas.infer.trt_engine import EngineError, TrtEngine

            path = Path(engine_path)
            if not path.exists():
                raise PerceptionError("MiDaS engine not found: %s" % engine_path)
            try:
                engine = TrtEngine(str(path))
            except EngineError as exc:
                raise PerceptionError("could not load %s: %s" % (path, exc)) from exc
            self.engine_path = str(path)
        else:
            self.engine_path = str(engine_path or getattr(engine, "engine_path", "<engine>"))
        self.engine = engine

        shape = tuple(int(d) for d in engine.input_shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
            self.close()
            raise PerceptionError("MiDaS engine input must be (1, 3, H, W), got %s" % (shape,))
        self.input_height = shape[2]
        self.input_width = shape[3]
        self.input_dtype = np.dtype(getattr(engine, "input_dtype", np.float32))
        if len(engine.output_names) != 1:
            self.close()
            raise PerceptionError(
                "MiDaS engine must have exactly one output, got %s" % (engine.output_names,)
            )
        self.output_name = engine.output_names[0]
        out_shape = tuple(int(d) for d in engine.output_shapes[self.output_name])
        squeezed = tuple(d for d in out_shape if d != 1)
        if len(squeezed) != 2:
            self.close()
            raise PerceptionError(
                "MiDaS output must squeeze to a 2-D map, got %s" % (out_shape,)
            )
        self.output_height, self.output_width = squeezed
        self.frames = 0
        self.last_infer_ms = 0.0
        logger.info(
            "MiDaSDepthEstimator loaded %s: in %s %s -> out %s %s",
            self.engine_path,
            engine.input_name,
            shape,
            self.output_name,
            out_shape,
        )

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """uint8 BGR frame -> ``(1, 3, H, W)`` float32 blob.

        MiDaS stretches the frame to a square; the aspect ratio is deliberately
        not preserved, matching the reference transform.
        """
        import cv2

        from adas.perception.detection import cv_interpolation, require_bgr_uint8

        array = require_bgr_uint8(image)
        resized = cv2.resize(
            array,
            (self.input_width, self.input_height),
            interpolation=cv_interpolation(PREPROC_MIDAS.interpolation),
        )
        return blob_from_canvas(resized, PREPROC_MIDAS, out_dtype=self.input_dtype)

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Run one inference. Returns the ``(H, W)`` float32 disparity map.

        The values are relative inverse depth: unitless, larger = closer, with
        an unknown per-frame affine scale and shift. Do not treat them as
        metres.
        """
        blob = self.preprocess(image)
        start = time.perf_counter()
        outputs = self.engine.infer_views({self.engine.input_name: blob})
        raw = np.asarray(outputs[self.output_name], dtype=np.float32)
        # Copy: infer_views hands back the pinned staging buffer, which the next
        # inference overwrites, and this map is deliberately kept across frames.
        disparity = raw.reshape(self.output_height, self.output_width).copy()
        self.last_infer_ms = (time.perf_counter() - start) * 1000.0
        self.frames += 1
        return disparity

    def close(self) -> None:
        engine = getattr(self, "engine", None)
        if engine is not None:
            close = getattr(engine, "close", None)
            if callable(close):
                close()
            self.engine = None

    def __enter__(self) -> "MiDaSDepthEstimator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Range channel
# --------------------------------------------------------------------------- #


class DepthRangeChannel:
    """Reduced-cadence relative-depth channel. Publishes metres only if audited.

    By default this channel publishes **no metric range at all**: every
    :class:`~adas.core.models.RangeEstimate` it returns is
    ``RangeSource.UNAVAILABLE`` with confidence 0, which the pipeline drops. The
    relative signal is available separately via :meth:`ordinal_readings`. The
    module docstring records the measurements behind that decision.

    Parameters
    ----------
    engine_path:
        Path to ``midas_v21_small_256.engine``. If it does not exist the channel
        degrades to an honest stub (see the module docstring).
    cadence_frames:
        Run the network every N frames. 5 at 20 Hz is 4 Hz and 6.3 ms of GPU per
        run, i.e. ~1.3 ms/frame amortised.
    min_road_anchors:
        Road-surface probes that must survive projection, box exclusion and
        outlier trimming before the road-plane fit is accepted. The fit has two
        free parameters, so anything below ~4 is interpolation, not estimation.
    min_box_anchors:
        Detections with a usable reference range needed for the leave-one-out
        fallback fit. Must be at least 3: the fit for object *i* excludes
        object *i* and still needs a degree of freedom.
    publish_metric:
        Request metric ``RangeSource.DEPTH_MODEL`` estimates. Default ``False``.
        This is a *request*, not a switch: metres are published only while the
        rolling self-audit below also passes. Setting it on footage where the
        model has no ordering skill changes nothing.
    min_ordering_spearman:
        Rank correlation between the reference ranges the channel is handed and
        the disparities it samples that must hold before metres are published.
        Defaults to :data:`MIN_ORDERING_SPEARMAN`.
    min_ordering_pairs:
        Pairs the audit needs before its correlation counts as evidence.
        Defaults to :data:`MIN_ORDERING_PAIRS`. Fewer pairs means no evidence,
        which means no metres.
    ordering_window:
        How many recent pairs the audit keeps. Defaults to
        :data:`ORDERING_WINDOW_PAIRS`.
    require_engine:
        ``True`` turns a missing or broken engine into a
        :class:`~adas.core.exceptions.PerceptionError` at construction instead
        of a stub. Use it for a production profile that must not run blind.
    engine:
        Pre-built engine (or :class:`~adas.infer.trt_engine.FakeTrtEngine`) for
        tests.
    """

    def __init__(
        self,
        engine_path: str = "models/midas_v21_small_256.engine",
        cadence_frames: int = 5,
        min_road_anchors: int = 6,
        min_box_anchors: int = 3,
        publish_metric: bool = False,
        min_ordering_spearman: float = MIN_ORDERING_SPEARMAN,
        min_ordering_pairs: int = MIN_ORDERING_PAIRS,
        ordering_window: int = ORDERING_WINDOW_PAIRS,
        require_engine: bool = False,
        engine: object = None,
    ) -> None:
        if cadence_frames < 1:
            raise PerceptionError("cadence_frames must be >= 1, got %r" % (cadence_frames,))
        self.engine_path = str(engine_path)
        self.cadence_frames = int(cadence_frames)
        self.min_road_anchors = max(4, int(min_road_anchors))
        self.min_box_anchors = max(3, int(min_box_anchors))
        self.publish_metric = bool(publish_metric)
        self.min_ordering_spearman = float(min_ordering_spearman)
        self.min_ordering_pairs = max(3, int(min_ordering_pairs))
        self.ordering_window = max(self.min_ordering_pairs, int(ordering_window))
        #: Rolling audit of the channel's own ordering skill. NaN until there is
        #: enough evidence to say anything.
        self.ordering_spearman = float("nan")
        self._audit_reference: Deque[float] = deque(maxlen=self.ordering_window)
        self._audit_disparity: Deque[float] = deque(maxlen=self.ordering_window)
        #: True only while metres are actually being published.
        self.metric_gate_open = False
        self._gate_logged = False
        #: Last frame's unitless relative readings, index-aligned with the boxes.
        self.last_ordinal: List[Optional[OrdinalDepth]] = []
        #: Which anchoring mode produced the current scale: ``"road_plane"``,
        #: ``"box_loo"`` or ``""`` when there is no valid fit.
        self.scale_mode = ""
        self.max_stale_frames = max(0, self.cadence_frames - 1)
        self.estimator: Optional[MiDaSDepthEstimator] = None
        self.unavailable_reason = ""
        self.field = DepthField()
        self.scale = DepthScale()
        self.frames_seen = 0
        self.inferences = 0
        self.last_total_ms = 0.0

        if engine is not None or Path(self.engine_path).exists():
            try:
                self.estimator = MiDaSDepthEstimator(
                    "" if engine is not None else self.engine_path, engine=engine
                )
            except PerceptionError as exc:
                if require_engine:
                    raise
                self.unavailable_reason = str(exc)
                logger.warning(
                    "Depth range channel DISABLED: %s. Range estimates will report "
                    "source=UNAVAILABLE and there are no ordinal readings either. "
                    "Note the channel publishes no metric range even when it loads "
                    "(see adas.perception.depth), so this costs the relative signal "
                    "only.",
                    exc,
                )
        else:
            self.unavailable_reason = "engine not found: %s" % self.engine_path
            if require_engine:
                raise PerceptionError(self.unavailable_reason)
            logger.warning(
                "Depth range channel DISABLED: %s. Range estimates will report "
                "source=UNAVAILABLE (honest stub); the interface is live and the "
                "engine drops in with no call-site change.",
                self.unavailable_reason,
            )

    # -- state ------------------------------------------------------------- #

    @property
    def available(self) -> bool:
        """True when a real engine is loaded."""
        return self.estimator is not None

    @property
    def is_mock(self) -> bool:
        """True when this channel is the honest stub, i.e. measures nothing."""
        return self.estimator is None

    def due(self, frame_id: int) -> bool:
        """Whether the network should run on ``frame_id``."""
        if not self.available:
            return False
        if not self.field.valid:
            return True
        return (int(frame_id) - self.field.frame_id) >= self.cadence_frames

    # -- main entry point --------------------------------------------------- #

    def update(
        self,
        frame: object,
        frame_id: int,
        boxes: Sequence[BoundingBox],
        frame_width: int,
        frame_height: int,
        reference: Optional[Sequence[Optional[RangeEstimate]]] = None,
        camera: object = None,
        force: bool = False,
    ) -> List[RangeEstimate]:
        """Return one :class:`~adas.core.models.RangeEstimate` per box.

        **By default every one of them is ``RangeSource.UNAVAILABLE`` with
        confidence 0.** Metres are published only when ``publish_metric`` was
        requested *and* the rolling self-audit shows this channel ordering
        objects correctly on the current footage; see the class and module
        docstrings. The unitless relative readings for this frame are always
        available from :meth:`ordinal_readings`.

        ``camera`` is a :class:`~adas.perception.geometry.CameraConfig`. When it
        is supplied the scale is anchored on the road plane rather than on the
        detection boxes. The fit still runs, and is reported in :meth:`stats`,
        whether or not metres are published.

        ``reference`` supplies one metric range per box (normally the estimate
        the tracker already computed). It feeds the ordering audit, and is the
        anchor for the leave-one-out fallback fit when no camera is available.
        With no reference and no camera there is neither an identifiable fit nor
        any audit evidence, so every result is ``UNAVAILABLE``.

        ``force`` runs the network regardless of cadence.

        Never raises for an ordinary frame: an inference failure disables the
        channel for that frame (logged at ERROR) and yields ``UNAVAILABLE``
        results, because a monitor that throws is worse than a monitor that says
        "I do not know".
        """
        started = time.perf_counter()
        self.frames_seen += 1
        fresh_map = False
        count = len(boxes)
        self.last_ordinal = [None] * count
        if not self.available:
            self.last_total_ms = (time.perf_counter() - started) * 1000.0
            return [_unavailable() for _ in range(count)]

        if force or self.due(frame_id):
            try:
                disparity = self.estimator.infer(frame)
            except Exception as exc:  # noqa: BLE001 - a monitor must not kill the frame
                logger.error("MiDaS inference failed on frame %s: %s", frame_id, exc)
                self.field = DepthField()
                self.scale = DepthScale()
                self.scale_mode = ""
                self.last_total_ms = (time.perf_counter() - started) * 1000.0
                return [_unavailable() for _ in range(count)]
            self.inferences += 1
            fresh_map = True
            self.field = DepthField(
                disparity=disparity,
                frame_id=int(frame_id),
                timestamp_s=time.time(),
                infer_ms=self.estimator.last_infer_ms,
                is_mock=False,
            )

        stale = int(frame_id) - self.field.frame_id
        if not self.field.valid or stale < 0 or stale > self.max_stale_frames:
            self.scale = DepthScale()
            self.scale_mode = ""
            self.last_total_ms = (time.perf_counter() - started) * 1000.0
            return [_unavailable() for _ in range(count)]

        samples = [self.field.sample_box(box, frame_width, frame_height) for box in boxes]
        staleness = 1.0 - (stale / float(self.max_stale_frames + 1))

        # The audit and the ordinal readings do not depend on any fit, so they
        # are produced before the scale branches and regardless of which one
        # wins -- including when neither does.
        references = self._reference_ranges(boxes, reference, camera, frame_width, frame_height)
        self._record_ordering(samples, references)
        self._refresh_metric_gate()
        self.last_ordinal = _ordinal_readings(samples, self.field.frame_id, stale)

        # ---- preferred: road-plane anchoring, which reads no detection box ----
        # "Independent of the boxes" is true of the *anchors* -- they are road
        # pixels projected through the camera extrinsics. It was never true of
        # the whole channel: the per-object disparity still comes from pixels
        # inside a box the detector drew, and on the shipped model that sample
        # is the part that carries no range information. Publication is gated on
        # the self-audit for exactly that reason.
        # The fit belongs to the disparity map, so it is recomputed only when a
        # new map arrives; on the reuse frames the stored scale still applies
        # and re-probing the road would cost ~2 ms for the same answer.
        if camera is not None:
            if not fresh_map and self.scale_mode == "road_plane" and self.scale.valid:
                base = self._confidence(self.scale, staleness, 1.0)
                results = self._publish(
                    [self._estimate(self.scale, sample, base) for sample in samples]
                )
                self.last_total_ms = (time.perf_counter() - started) * 1000.0
                return results
            disparities, distances = road_plane_anchors(
                self.field, camera, frame_width, frame_height, boxes
            )
            if len(disparities) >= self.min_road_anchors:
                scale = fit_depth_scale_robust(
                    disparities,
                    distances,
                    min_anchors=self.min_road_anchors,
                    previous=self.scale if self.scale_mode == "road_plane" else None,
                )
                if scale.valid and scale.anchors >= self.min_road_anchors:
                    self.scale = scale
                    self.scale_mode = "road_plane"
                    base = self._confidence(scale, staleness, 1.0)
                    results = self._publish(
                        [self._estimate(scale, sample, base) for sample in samples]
                    )
                    self.last_total_ms = (time.perf_counter() - started) * 1000.0
                    return results
            self.scale = DepthScale()
            self.scale_mode = ""

        # ---- fallback: leave-one-out over the other detections' ranges ----
        usable = [
            index
            for index, (sample, ref) in enumerate(zip(samples, references))
            if sample is not None and _usable_anchor(ref)
        ]
        if len(usable) < self.min_box_anchors + 1:
            # Every anchor object would have to validate itself. Refuse.
            self.scale = DepthScale()
            self.scale_mode = ""
            self.last_total_ms = (time.perf_counter() - started) * 1000.0
            return [_unavailable() for _ in range(count)]

        full = fit_depth_scale(
            [samples[i] for i in usable],
            [references[i].distance_m for i in usable],
            min_anchors=self.min_box_anchors,
        )
        self.scale = full
        self.scale_mode = "box_loo" if full.valid else ""
        results = []
        for index, sample in enumerate(samples):
            if index in usable:
                others = [i for i in usable if i != index]
                scale = fit_depth_scale(
                    [samples[i] for i in others],
                    [references[i].distance_m for i in others],
                    min_anchors=self.min_box_anchors,
                )
            else:
                scale = full
            base = self._confidence(scale, staleness, BOX_ANCHOR_CONFIDENCE_FACTOR)
            results.append(self._estimate(scale, sample, base))
        results = self._publish(results)
        self.last_total_ms = (time.perf_counter() - started) * 1000.0
        return results

    # -- self-audit and the metric gate -------------------------------------- #

    def _record_ordering(
        self,
        samples: Sequence[Optional[float]],
        references: Sequence[Optional[RangeEstimate]],
    ) -> None:
        """File this frame's (reference range, sampled disparity) pairs.

        Only pairs where both sides are usable are kept, so a box the depth map
        could not sample, or one whose reference range is itself untrustworthy,
        contributes nothing rather than contributing noise.
        """
        for sample, ref in zip(samples, references):
            if sample is None or not math.isfinite(float(sample)):
                continue
            if not _usable_anchor(ref):
                continue
            self._audit_reference.append(float(ref.distance_m))
            self._audit_disparity.append(float(sample))

    def _refresh_metric_gate(self) -> None:
        """Recompute the audit correlation and decide whether metres may flow.

        The correlation is taken against **negated** disparity so that a channel
        which orders objects correctly scores +1: disparity grows as range
        shrinks. Because rank correlation is invariant under any monotone
        rescaling, this measures the network's ordering skill alone and is
        unaffected by how the affine fit is anchored or whether the camera is
        calibrated -- which is exactly the property that makes it a fair gate.

        Fails closed. Too few pairs, or a NaN correlation (constant disparity,
        no evidence), leaves the gate shut.

        Caveat worth stating: the reference is whatever the caller supplied, and
        that is box geometry. When no ``reference`` is passed the fallback is
        ``ground_plane_range``, which shares the camera homography with the road
        anchors, so the audit could in principle flatter the channel. It does
        not rescue it here -- measured against ``ground_plane_range`` the
        correlation is 0.455, against ``pinhole_range`` 0.252, and the gate's
        floor is 0.80. A gate that can only be generous and still refuses is
        refusing for a real reason.
        """
        pairs = len(self._audit_reference)
        if pairs < self.min_ordering_pairs:
            self.ordering_spearman = float("nan")
            self._set_gate(False, "only %d/%d audit pairs" % (pairs, self.min_ordering_pairs))
            return
        rho = spearman_rho(
            list(self._audit_reference), [-d for d in self._audit_disparity]
        )
        self.ordering_spearman = rho
        if not math.isfinite(rho):
            self._set_gate(False, "audit correlation undefined over %d pairs" % pairs)
            return
        self._set_gate(
            rho >= self.min_ordering_spearman,
            "audit spearman %.3f vs floor %.2f over %d pairs"
            % (rho, self.min_ordering_spearman, pairs),
        )

    def _set_gate(self, ordering_ok: bool, reason: str) -> None:
        """Open or close metric publication, logging the first refusal once."""
        opened = bool(self.publish_metric and ordering_ok)
        if self.publish_metric and not opened and not self._gate_logged:
            self._gate_logged = True
            logger.warning(
                "Depth channel is NOT publishing metric range: %s. Estimates stay "
                "RangeSource.UNAVAILABLE; the relative signal is on "
                "ordinal_readings(). This model measured spearman ~%.2f per object "
                "on the reference clip.",
                reason,
                MEASURED_OBJECT_SPEARMAN_MIDAS_SMALL_256,
            )
        if opened and self._gate_logged:
            self._gate_logged = False
            logger.info("Depth channel metric range ENABLED: %s", reason)
        self.metric_gate_open = opened

    def _publish(self, results: List[RangeEstimate]) -> List[RangeEstimate]:
        """Let metric estimates through, or replace them all with UNAVAILABLE.

        Replacing rather than suppressing keeps ``update`` index-aligned with the
        boxes, which the pipeline relies on to match results to tracks.
        """
        if self.metric_gate_open:
            return results
        return [_unavailable() for _ in results]

    def ordinal_readings(self) -> List[Optional[OrdinalDepth]]:
        """The last frame's unitless relative depths, aligned with its boxes.

        ``None`` where the map could not be sampled. These are **not metres**;
        see :class:`OrdinalDepth`. Given the correlations in the module
        docstring, treat even the ordering as weak evidence on the shipped
        model.
        """
        return list(self.last_ordinal)

    # -- helpers ------------------------------------------------------------ #

    def _confidence(self, scale: DepthScale, staleness: float, mode_factor: float) -> float:
        """Confidence for one frame's fit: quality x anchor count x staleness.

        Capped at :data:`MAX_DEPTH_CONFIDENCE` whatever the fit says: a
        relative-depth network with a borrowed scale must never out-vote a direct
        geometric measurement.

        This value only reaches a caller when the metric gate is open, which on
        the shipped model it is not. It measures the quality of the *affine fit*,
        not the channel's ability to rank objects -- those are different things,
        and it was the latter that failed. ``ordering_spearman`` in
        :meth:`stats` is the one that speaks to trustworthiness.
        """
        if not scale.valid:
            return 0.0
        fit_quality = max(0.0, 1.0 - scale.rel_rmse)
        anchor_bonus = min(1.0, scale.anchors / float(self.min_road_anchors + 2))
        value = MAX_DEPTH_CONFIDENCE * fit_quality * anchor_bonus * max(0.0, staleness)
        return float(max(0.0, min(MAX_DEPTH_CONFIDENCE, value * float(mode_factor))))

    @staticmethod
    def _estimate(scale: DepthScale, sample: Optional[float], confidence: float) -> RangeEstimate:
        """One box's metric result, or ``UNAVAILABLE`` if it cannot be formed.

        Forming a value here does not publish it: :meth:`_publish` still has to
        pass it through the metric gate.
        """
        if sample is None or not scale.valid or confidence <= 0.0:
            return _unavailable()
        distance = scale.range_m(sample)
        if distance is None or distance < MIN_ANCHOR_RANGE_M * 0.25:
            return _unavailable()
        return RangeEstimate(
            distance_m=float(distance),
            confidence=float(confidence),
            source=RangeSource.DEPTH_MODEL,
            truncated=False,
        )

    @staticmethod
    def _reference_ranges(
        boxes: Sequence[BoundingBox],
        reference: Optional[Sequence[Optional[RangeEstimate]]],
        camera: object,
        frame_width: int,
        frame_height: int,
    ) -> List[Optional[RangeEstimate]]:
        if reference is not None:
            if len(reference) != len(boxes):
                raise PerceptionError(
                    "reference has %d entries for %d boxes" % (len(reference), len(boxes))
                )
            return list(reference)
        if camera is None:
            return [None] * len(boxes)
        from adas.perception.geometry import ground_plane_range

        return [
            ground_plane_range(box, camera, int(frame_width), int(frame_height))
            for box in boxes
        ]

    def stats(self) -> Dict[str, float]:
        """Cadence, fit and self-audit diagnostics, for the metrics subsystem.

        ``ordering_spearman`` is the channel's own measured skill on the footage
        it is running on, and ``metric_gate_open`` says whether that was enough
        to publish metres. ``scale_*`` describe the road-plane affine fit, which
        runs and is reported even while publication is gated shut -- on the
        reference clip that fit is good (relative residual 0.04-0.08) and it is
        the *object sampling*, not the fit, that fails.
        """
        return {
            "available": 1.0 if self.available else 0.0,
            "frames_seen": float(self.frames_seen),
            "inferences": float(self.inferences),
            "duty_cycle": (self.inferences / float(self.frames_seen)) if self.frames_seen else 0.0,
            "scale_mode_road": 1.0 if self.scale_mode == "road_plane" else 0.0,
            "scale_a": self.scale.a,
            "scale_b": self.scale.b,
            "scale_anchors": float(self.scale.anchors),
            "scale_rel_rmse": self.scale.rel_rmse,
            "last_infer_ms": self.field.infer_ms,
            "last_total_ms": self.last_total_ms,
            "publish_metric_requested": 1.0 if self.publish_metric else 0.0,
            "metric_gate_open": 1.0 if self.metric_gate_open else 0.0,
            "ordering_spearman": float(self.ordering_spearman),
            "ordering_pairs": float(len(self._audit_reference)),
        }

    def close(self) -> None:
        """Release the engine. Idempotent."""
        if self.estimator is not None:
            self.estimator.close()
            self.estimator = None
        self.field = DepthField()
        self.scale = DepthScale()
        self.scale_mode = ""
        self.last_ordinal = []
        self._audit_reference.clear()
        self._audit_disparity.clear()
        self.ordering_spearman = float("nan")
        self.metric_gate_open = False

    def __enter__(self) -> "DepthRangeChannel":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _ordinal_readings(
    samples: Sequence[Optional[float]], frame_id: int, stale_frames: int
) -> List[Optional[OrdinalDepth]]:
    """Rank this frame's sampled disparities into unitless relative depths.

    Rank 0 is the largest disparity, i.e. the nearest box. Boxes the map could
    not sample get ``None`` and take no rank, so a missing sample does not shift
    the ranks of the boxes around it.
    """
    usable = [
        index
        for index, value in enumerate(samples)
        if value is not None and math.isfinite(float(value))
    ]
    out: List[Optional[OrdinalDepth]] = [None] * len(samples)
    if not usable:
        return out
    order = sorted(usable, key=lambda i: -float(samples[i]))
    total = len(order)
    for rank, index in enumerate(order):
        out[index] = OrdinalDepth(
            disparity=float(samples[index]),
            rank=rank,
            of=total,
            normalized=(rank / float(total - 1)) if total > 1 else 0.0,
            frame_id=int(frame_id),
            stale_frames=int(stale_frames),
        )
    return out


def _usable_anchor(ref: Optional[RangeEstimate]) -> bool:
    """Whether a reference range is trustworthy enough to anchor the scale."""
    if ref is None or ref.source is RangeSource.UNAVAILABLE or ref.truncated:
        return False
    if ref.confidence < MIN_ANCHOR_CONFIDENCE:
        return False
    return MIN_ANCHOR_RANGE_M <= ref.distance_m <= MAX_ANCHOR_RANGE_M


def _unavailable() -> RangeEstimate:
    """The honest "no measurement" result. Never a plausible constant."""
    return RangeEstimate(
        distance_m=0.0,
        confidence=0.0,
        source=RangeSource.UNAVAILABLE,
        truncated=False,
    )
