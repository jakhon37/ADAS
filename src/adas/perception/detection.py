"""Detector abstraction.

Production deployments should replace this with TensorRT/DeepStream-backed inference.
"""

from __future__ import annotations

from dataclasses import dataclass

from adas.core.models import BoundingBox


@dataclass(slots=True)
class ObjectDetector:
    confidence_threshold: float = 0.35

    def infer(self, frame: object, width: int, height: int) -> list[BoundingBox]:
        # Deterministic placeholder: one lead-vehicle-like box in the ego lane.
        conf = 0.8
        if conf < self.confidence_threshold:
            return []

        box_w, box_h = width * 0.12, height * 0.18
        center_x, bottom_y = width / 2, height * 0.7
        return [
            BoundingBox(
                x1=center_x - box_w / 2,
                y1=bottom_y - box_h,
                x2=center_x + box_w / 2,
                y2=bottom_y,
                confidence=conf,
                label="vehicle",
            )
        ]
