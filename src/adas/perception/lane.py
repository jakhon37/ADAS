"""Lane estimation abstraction.

Production deployments can replace this with learned lane segmentation + spline fitting.
"""

from __future__ import annotations

from adas.core.models import LaneModel


class LaneEstimator:
    def estimate(self, frame: object, width: int, height: int) -> LaneModel | None:
        left_base = width * 0.36
        right_base = width * 0.64
        lane_center = (left_base + right_base) / 2.0
        return LaneModel(
            left_coeffs=(0.0, 0.0, left_base),
            right_coeffs=(0.0, 0.0, right_base),
            lane_center_px=lane_center,
            curvature_m=220.0,
        )
