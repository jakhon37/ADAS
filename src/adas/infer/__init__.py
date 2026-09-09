"""Jetson-native inference helpers (TensorRT + CUDA runtime)."""

from __future__ import annotations

__all__ = ["TrtEngine"]


def __getattr__(name: str):
    if name == "TrtEngine":
        from adas.infer.trt_engine import TrtEngine

        return TrtEngine
    raise AttributeError(name)
