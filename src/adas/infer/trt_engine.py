"""Synchronous TensorRT engine wrapper using CUDA runtime (JetPack 5 / TRT 8.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from adas.infer import cudart


def _dtype_map():
    import tensorrt as trt

    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT8: np.int8,
        trt.DataType.INT32: np.int32,
        trt.DataType.BOOL: np.bool_,
    }
    if hasattr(trt.DataType, "INT64"):
        mapping[trt.DataType.INT64] = np.int64
    return mapping


@dataclass
class Binding:
    name: str
    is_input: bool
    dtype: np.dtype
    shape: Tuple[int, ...]
    nbytes: int
    host: np.ndarray
    device: int


class TrtEngine:
    """Load a serialized TensorRT engine and run synchronous inference."""

    def __init__(self, engine_path: str) -> None:
        import tensorrt as trt

        self.engine_path = engine_path
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        with open(engine_path, "rb") as f:
            blob = f.read()
        engine = self._runtime.deserialize_cuda_engine(blob)
        if engine is None:
            raise RuntimeError("failed to deserialize %s" % engine_path)
        self._engine = engine
        self._ctx = engine.create_execution_context()
        self._stream = cudart.stream_create()
        self._bindings: Dict[str, Binding] = {}
        dmap = _dtype_map()
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dt = dmap[engine.get_tensor_dtype(name)]
            shape = tuple(int(x) for x in engine.get_tensor_shape(name))
            if any(s < 0 for s in shape):
                raise RuntimeError("dynamic shape on %s — use a static engine" % name)
            nbytes = int(np.prod(shape) * np.dtype(dt).itemsize)
            host = np.empty(shape, dtype=dt)
            device = cudart.malloc(nbytes)
            self._bindings[name] = Binding(
                name=name,
                is_input=is_input,
                dtype=np.dtype(dt),
                shape=shape,
                nbytes=nbytes,
                host=host,
                device=device,
            )
            self._ctx.set_tensor_address(name, device)
        self._inputs = [b for b in self._bindings.values() if b.is_input]
        self._outputs = [b for b in self._bindings.values() if not b.is_input]

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self._inputs[0].shape

    @property
    def input_name(self) -> str:
        return self._inputs[0].name

    @property
    def output_names(self) -> List[str]:
        return [b.name for b in self._outputs]

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for b in self._inputs:
            arr = inputs.get(b.name)
            if arr is None:
                if len(self._inputs) == 1 and len(inputs) == 1:
                    arr = next(iter(inputs.values()))
                else:
                    raise KeyError("missing input %s" % b.name)
            arr = np.ascontiguousarray(arr, dtype=b.dtype)
            if arr.shape != b.shape:
                raise ValueError("input %s expected %s got %s" % (b.name, b.shape, arr.shape))
            np.copyto(b.host, arr)
            cudart.memcpy_async(
                b.device,
                int(b.host.ctypes.data),
                b.nbytes,
                cudart.cudaMemcpyHostToDevice,
                self._stream,
            )
        ok = self._ctx.execute_async_v3(self._stream)
        if not ok:
            raise RuntimeError("execute_async_v3 failed for %s" % self.engine_path)
        for b in self._outputs:
            cudart.memcpy_async(
                int(b.host.ctypes.data),
                b.device,
                b.nbytes,
                cudart.cudaMemcpyDeviceToHost,
                self._stream,
            )
        event = cudart.event_create()
        cudart.event_record(event, self._stream)
        cudart.event_synchronize(event)
        cudart.event_destroy(event)
        return {b.name: np.copy(b.host) for b in self._outputs}

    def close(self) -> None:
        for b in self._bindings.values():
            cudart.free(b.device)
        cudart.stream_destroy(self._stream)
        self._bindings.clear()

    def __enter__(self) -> "TrtEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
