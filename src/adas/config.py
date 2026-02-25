"""Configuration objects and JSON loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DetectorConfig:
    model_path: str = "models/yolov8n.onnx"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.5


@dataclass(slots=True)
class PlannerConfig:
    cruise_speed_mps: float = 15.0
    min_follow_distance_m: float = 12.0
    max_steering_deg: float = 22.0


@dataclass(slots=True)
class RuntimeConfig:
    detector: DetectorConfig
    planner: PlannerConfig
    fps: int = 20


DEFAULT_CONFIG = RuntimeConfig(detector=DetectorConfig(), planner=PlannerConfig())


def load_config(path: str | Path | None = None) -> RuntimeConfig:
    if path is None:
        return DEFAULT_CONFIG

    payload = json.loads(Path(path).read_text())
    detector_payload = payload.get("detector", {})
    planner_payload = payload.get("planner", {})

    detector = DetectorConfig(
        model_path=detector_payload.get("model_path", DEFAULT_CONFIG.detector.model_path),
        confidence_threshold=float(
            detector_payload.get("confidence_threshold", DEFAULT_CONFIG.detector.confidence_threshold)
        ),
        iou_threshold=float(detector_payload.get("iou_threshold", DEFAULT_CONFIG.detector.iou_threshold)),
    )
    planner = PlannerConfig(
        cruise_speed_mps=float(planner_payload.get("cruise_speed_mps", DEFAULT_CONFIG.planner.cruise_speed_mps)),
        min_follow_distance_m=float(
            planner_payload.get(
                "min_follow_distance_m", DEFAULT_CONFIG.planner.min_follow_distance_m
            )
        ),
        max_steering_deg=float(planner_payload.get("max_steering_deg", DEFAULT_CONFIG.planner.max_steering_deg)),
    )

    return RuntimeConfig(detector=detector, planner=planner, fps=int(payload.get("fps", 20)))
