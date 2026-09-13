"""Jetson-native inference helpers (TensorRT + CUDA runtime).

Import cost
-----------
:class:`~adas.infer.trt_engine.TrtEngine` pulls in ``tensorrt`` and
``libcudart`` the moment it is *used*, not when this package is imported, so
``import adas.infer`` stays free on a machine with no GPU. That is what lets
the pure-numpy decode paths in :mod:`adas.perception` be unit-tested in CI.

:class:`~adas.infer.trt_engine.FakeTrtEngine` is the CPU test double with the
same surface; :class:`~adas.infer.trt_engine.EngineError` is what every loading
or contract failure raises.
"""

from __future__ import annotations

__all__ = ["Binding", "EngineError", "FakeTrtEngine", "TrtEngine", "cudart", "sha256_file"]

_LAZY = {
    "Binding": "adas.infer.trt_engine",
    "EngineError": "adas.infer.trt_engine",
    "FakeTrtEngine": "adas.infer.trt_engine",
    "TrtEngine": "adas.infer.trt_engine",
    "sha256_file": "adas.infer.trt_engine",
    "cudart": "adas.infer.cudart",
}


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    import importlib

    module = importlib.import_module(module_name)
    if name == "cudart":
        return module
    return getattr(module, name)


def __dir__():
    return sorted(set(list(globals()) + __all__))
