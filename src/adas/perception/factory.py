"""Build perception backends from configuration.

This is the single place where a configuration string becomes a loaded TensorRT
engine.  Three rules apply to every builder here:

* **A missing engine is an error, not a downgrade.**  Falling back to a
  fabricating stub because a file was absent is how a vehicle ends up driving on
  invented geometry.  That substitution needs ``allow_fallback``, which is off by
  default and which the CLI only sets when the operator passed ``--allow-mock``
  (or ``ADAS_ALLOW_MOCK=1``, or ``"allow_mock": true``); it is then logged at
  ERROR with the reason.  ``allow_mock`` is the separate, weaker permission to
  honour an EXPLICIT ``backend: "mock"`` request; the configuration layer refuses
  such a request unless the operator opted in
  (:meth:`adas.core.config.RuntimeConfig.cross_validation_errors`), so a
  config-driven or CLI-driven run cannot reach a stub by accident.
* **Every stub is self-identifying.**  Anything returned from here answers
  ``is_mock``; the pipeline copies that onto ``LaneModel.is_mock``, the health
  endpoint publishes it as ``adas_lane_is_mock``, and the event log records it at
  startup.
* **Relative model paths are resolved against the repository root as well as the
  working directory**, so ``models/yolox_nano.engine`` works whether the process
  was started from the repo or from ``/``.
* **Every engine is checked against ``models/MANIFEST.json`` before it is
  loaded.**  The digest is verified here rather than only inside
  :class:`~adas.infer.trt_engine.TrtEngine` so the refusal happens before four
  seconds of engine deserialisation and reaches the operator as one line.  A
  mismatch raises :class:`~adas.core.exceptions.PerceptionError`, which
  ``allow_fallback`` turns into the same loud mock downgrade as a missing file;
  an engine the manifest does not list is logged as unverifiable and loaded.

Backends
--------
detector
    ``mock`` | ``tensorrt`` (layout auto-detected from the engine) | ``yolov5``
    | ``yolox``
lane
    ``mock`` | ``ufld`` | ``yolop`` | ``twinlite``
depth
    ``off`` | ``midas``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

from adas.core.config import CameraConfig, DepthConfig, DetectorConfig, LaneConfig
from adas.core.exceptions import ConfigurationError, PerceptionError
from adas.core.logger import setup_logger
from adas.infer.trt_engine import EngineError, verify_engine_file
from adas.perception.detection import ObjectDetector
from adas.perception.lane import MockLaneEstimator

logger = setup_logger(__name__)

#: Repository root inferred from this file: src/adas/perception/factory.py -> repo.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_model_path(path: str) -> Path:
    """Resolve a model path against the CWD and then the repository root.

    An absolute path is returned unchanged.  A relative path is tried against the
    current working directory first (so an operator's ``--config`` with a local
    override wins) and then against the repository root, which is what makes
    ``models/yolox_nano.engine`` work from a systemd unit whose ``WorkingDirectory``
    is not the checkout.

    Returns:
        The first candidate that exists, or the CWD-relative candidate when
        neither does (so the caller's "not found" message names the path the
        operator most likely meant).
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    rooted = _REPO_ROOT / candidate
    if rooted.exists():
        return rooted
    return candidate


#: An engine needs roughly its own file size again in device memory for weights,
#: plus execution-context scratch. Below this multiple of the file size we warn:
#: the Xavier NX has ~2.5 GiB free on a good day and UFLD-v2 alone is 413 MB.
_MEMORY_HEADROOM = 1.6


def available_mb() -> Optional[float]:
    """MemAvailable from /proc/meminfo, or ``None`` where it cannot be read.

    The Xavier NX has unified memory, so host availability is what bounds a
    cudaMalloc; there is no separate VRAM pool to query.
    """
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def search_paths(path: str) -> List[Path]:
    """The absolute candidates :func:`resolve_model_path` actually tries, deduplicated.

    An absolute path has exactly one candidate.  A relative one is tried against
    the working directory and then the repository root -- which are the same
    directory whenever the process was started from the checkout, so the pair is
    collapsed rather than printed twice as if the search were broken.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return [candidate]
    out: List[Path] = []
    for base in (Path.cwd(), _REPO_ROOT):
        full = (Path(base) / candidate).resolve()
        if full not in out:
            out.append(full)
    return out


def verify_engine(resolved: Path, what: str) -> str:
    """Check *resolved* against ``models/MANIFEST.json``; return its sha256.

    Returns ``""`` when the manifest does not list the file (the warning comes
    from :func:`adas.infer.trt_engine.verify_engine_file`).  Raises
    :class:`PerceptionError` -- not the underlying
    :class:`~adas.infer.trt_engine.EngineIntegrityError` -- so a mismatch takes
    the same operator-facing path as a missing engine, including the
    ``allow_fallback`` downgrade.
    """
    try:
        digest = verify_engine_file(str(resolved))
    except EngineError as exc:
        raise PerceptionError("%s engine rejected: %s" % (what, exc)) from exc
    if digest:
        logger.info("%s engine %s verified: sha256 %s", what, resolved.name, digest)
    return digest


def _require_engine(path: str, what: str, verify: bool = True) -> Path:
    resolved = resolve_model_path(path)
    if not resolved.exists():
        tried = ", ".join(str(p) for p in search_paths(path))
        raise PerceptionError(
            "%s engine not found: %s (tried %s). Build it with "
            "`python3 scripts/build_engines.py` -- see models/README.md."
            % (what, path, tried)
        )
    size_mb = resolved.stat().st_size / (1024.0 * 1024.0)
    free_mb = available_mb()
    if free_mb is not None and free_mb < size_mb * _MEMORY_HEADROOM:
        # Deserialisation will probably fail with a TensorRT OutOfMemory whose
        # message says nothing about what to do; say it here instead.
        logger.warning(
            "%s engine %s is %.0f MB but only %.0f MB is available. TensorRT needs "
            "roughly %.0f MB to deserialise it and will fail with 'Cuda Runtime (out "
            "of memory)'. Close other GPU processes, or pick a smaller backend.",
            what,
            resolved.name,
            size_mb,
            free_mb,
            size_mb * _MEMORY_HEADROOM,
        )
    else:
        logger.debug("%s engine %s: %.0f MB", what, resolved.name, size_mb)
    if verify:
        verify_engine(resolved, what)
    return resolved


def allow_mock_from_env(default: bool = False) -> bool:
    """Read the ``ADAS_ALLOW_MOCK`` escape hatch.

    Set it to ``1``/``true``/``yes`` to permit fabricating backends without
    editing the configuration file.  Anything else (including unset) leaves
    *default* untouched.
    """
    raw = os.environ.get("ADAS_ALLOW_MOCK", "")
    if raw.strip().lower() in ("1", "true", "yes", "on"):
        return True
    return default


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #


def build_camera(
    config: CameraConfig,
    frame_width: int = 0,
    frame_height: int = 0,
) -> Optional[Any]:
    """Build the geometry camera model, rescaled to the live frame size.

    Args:
        config: the ``camera`` configuration section.
        frame_width, frame_height: the resolution frames will actually arrive at.
            When they differ from the resolution the intrinsics were measured at,
            the intrinsics are rescaled -- an ``fx`` measured at 1280 wide is 1.5x
            too small at 1920 wide, and nothing else in the stack notices
            (ADAS-PERC-26).

    Returns:
        An :class:`adas.perception.geometry.CameraConfig`, or ``None`` when
        ``camera.enabled`` is false, in which case every metric quantity in the
        stack is unavailable by design rather than assumed.
    """
    if not config.enabled:
        logger.warning(
            "camera.enabled is false: no road-plane geometry. Ranges fall back to the "
            "pinhole box-height model, lane output is pixel-only, the metric steering "
            "law stays disengaged and the depth channel cannot anchor its scale."
        )
        return None

    from adas.perception.geometry import CameraConfig as GeometryCamera

    camera = GeometryCamera(
        image_width=config.image_width,
        image_height=config.image_height,
        fx=config.fx,
        fy=config.fy,
        cx=config.cx,
        cy=config.cy,
        mount_height_m=config.mount_height_m,
        pitch_deg=config.pitch_deg,
        calibrated=config.calibrated,
        label=config.label,
    )
    if (
        frame_width > 0
        and frame_height > 0
        and (frame_width != camera.image_width or frame_height != camera.image_height)
    ):
        logger.info(
            "Rescaling camera intrinsics from %dx%d to the live %dx%d",
            camera.image_width,
            camera.image_height,
            frame_width,
            frame_height,
        )
        camera = camera.scaled_to(frame_width, frame_height)
    return camera


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #


def build_detector(
    config: DetectorConfig,
    allow_mock: bool = True,
    allow_fallback: bool = False,
) -> Any:
    """Return a detector implementing ``infer(frame, width, height)``.

    Args:
        config: the ``detector`` configuration section.
        allow_mock: honour an EXPLICIT ``backend: "mock"``.  The configuration
            layer gates this for config-driven runs; a direct programmatic call
            asking for the mock by name gets it, with the usual warning banner.
        allow_fallback: substitute the mock when a real engine is missing or
            will not load.  Off by default -- a downgrade must be a decision.

    Raises:
        ConfigurationError: unknown backend, or ``mock`` with *allow_mock* false.
        PerceptionError: the engine file is missing or will not load and
            *allow_fallback* is false.
    """
    backend = config.backend

    if backend == "mock":
        if not allow_mock:
            raise ConfigurationError(
                "detector.backend is 'mock' but mock backends are not allowed. "
                "Pass --allow-mock, set ADAS_ALLOW_MOCK=1, or set \"allow_mock\": true "
                "in the configuration to accept a FABRICATED detection stream."
            )
        logger.warning("Using the MOCK object detector (fabricated detections)")
        return ObjectDetector(confidence_threshold=config.confidence_threshold)

    if backend not in ("tensorrt", "yolov5", "yolox"):
        raise ConfigurationError("Unknown detector backend: %s" % backend)

    layout = config.layout
    if backend == "yolox" and layout == "auto":
        layout = "yolox"
    elif backend == "yolov5" and layout == "auto":
        layout = "yolov5"

    try:
        engine_path = _require_engine(config.model_path, "detector")
        from adas.perception.yolo import YoloTensorRTDetector

        detector = YoloTensorRTDetector(
            engine_path=str(engine_path),
            confidence_threshold=config.confidence_threshold,
            iou_threshold=config.iou_threshold,
            max_detections=config.max_detections,
            input_size=config.input_size,
            class_ids=config.class_ids,
            layout=layout,
            pre_nms_topk=config.pre_nms_topk,
        )
    except PerceptionError as exc:
        if not allow_fallback:
            raise
        logger.error(
            "DETECTOR FALLBACK TO MOCK: %s. Detections are now FABRICATED and this "
            "process must not be used for anything but bench work.",
            exc,
        )
        return ObjectDetector(confidence_threshold=config.confidence_threshold)

    logger.info("Detector ready: %s (%s, layout=%s)", backend, engine_path.name, layout)
    return detector


# --------------------------------------------------------------------------- #
# Lane
# --------------------------------------------------------------------------- #


def build_lane_estimator(
    config: LaneConfig,
    camera: Optional[Any] = None,
    allow_mock: bool = True,
    allow_fallback: bool = False,
) -> Any:
    """Return a lane estimator implementing ``estimate(frame, width, height)``.

    Args:
        config: the ``lane`` configuration section.
        camera: an :class:`adas.perception.geometry.CameraConfig`, or ``None``.
            Without it every backend still produces a pixel lane centre (which is
            what the non-metric steering law consumes) but no metric geometry.
        allow_mock: honour an EXPLICIT ``backend: "mock"``.
        allow_fallback: substitute the mock when the engine is missing.

    Raises:
        ConfigurationError: unknown backend, or ``mock`` without *allow_mock*.
        PerceptionError: the engine is missing or will not load and
            *allow_fallback* is false.

    Note:
        ``twinlite`` has no engine on this board.  With
        ``lane.allow_unavailable`` it returns :class:`TwinLiteNetUnavailable`,
        which reports ``is_mock=True``, ``estimate() -> None`` and a
        zero-confidence drivable area -- it never invents a lane.
    """
    backend = config.backend

    if backend == "mock":
        if not allow_mock:
            raise ConfigurationError(
                "lane.backend is 'mock' but mock backends are not allowed. "
                "Pass --allow-mock, set ADAS_ALLOW_MOCK=1, or set \"allow_mock\": true "
                "in the configuration to accept FABRICATED lane geometry."
            )
        logger.warning("Using the MOCK lane estimator (fabricated geometry)")
        return MockLaneEstimator()

    if backend not in ("ufld", "yolop", "twinlite"):
        raise ConfigurationError("Unknown lane backend: %s" % backend)

    try:
        if backend == "twinlite":
            from adas.perception.twinlite import build_twinlite_backend

            resolved = resolve_model_path(config.model_path)
            if resolved.exists():
                # An absent twinlite engine is a documented state (the backend
                # reports is_mock and never invents a lane); a present one that
                # is not the validated file is not.
                verify_engine(resolved, "lane")
            estimator = build_twinlite_backend(
                str(resolved),
                camera=camera,
                allow_unavailable=config.allow_unavailable,
            )
        else:
            engine_path = _require_engine(config.model_path, "lane")
            if backend == "ufld":
                from adas.perception.ufld import UFLDLaneEstimator

                estimator = UFLDLaneEstimator(
                    engine_path=str(engine_path),
                    dataset=config.dataset,
                    camera=camera,
                    min_confidence=config.min_confidence,
                )
            else:
                from adas.perception.yolop import YolopLaneEstimator

                estimator = YolopLaneEstimator(
                    engine_path=str(engine_path),
                    camera=camera,
                    min_confidence=config.min_confidence,
                )
    except PerceptionError as exc:
        if not allow_fallback:
            raise
        logger.error(
            "LANE FALLBACK TO MOCK: %s. Lane geometry is now FABRICATED and this "
            "process must not be used for anything but bench work.",
            exc,
        )
        return MockLaneEstimator()

    logger.info("Lane estimator ready: %s (%s)", backend, config.model_path)
    return estimator


# --------------------------------------------------------------------------- #
# Depth (independent range channel)
# --------------------------------------------------------------------------- #


def build_depth_channel(config: DepthConfig) -> Optional[Any]:
    """Return the independent range channel, or ``None`` when it is switched off.

    Args:
        config: the ``depth`` configuration section.

    Returns:
        A :class:`adas.perception.depth.DepthRangeChannel`, or ``None`` for
        ``backend: off``.  A channel whose engine is missing is returned in its
        honest stub state (``is_mock=True``, every estimate
        ``RangeSource.UNAVAILABLE``) unless ``require_engine`` is set, in which
        case construction raises.

    Raises:
        ConfigurationError: unknown backend.
        PerceptionError: ``require_engine`` is set and the engine is unusable,
            or the engine file on disk is not the one ``models/MANIFEST.json``
            records.  A missing engine still degrades to the stub; a *wrong*
            engine is refused, because "no range channel" is a safe state and
            "an unknown model's range channel" is not.
    """
    if config.backend == "off":
        return None
    if config.backend != "midas":
        raise ConfigurationError("Unknown depth backend: %s" % config.backend)

    from adas.perception.depth import DepthRangeChannel

    resolved = resolve_model_path(config.model_path)
    if resolved.exists():
        verify_engine(resolved, "depth")
    channel = DepthRangeChannel(
        engine_path=str(resolved),
        cadence_frames=config.cadence_frames,
        require_engine=config.require_engine,
    )
    logger.info(
        "Depth range channel: %s (cadence %d frames, available=%s)",
        resolved.name,
        config.cadence_frames,
        channel.available,
    )
    return channel


__all__ = [
    "allow_mock_from_env",
    "build_camera",
    "build_depth_channel",
    "build_detector",
    "build_lane_estimator",
    "resolve_model_path",
    "search_paths",
    "verify_engine",
]
