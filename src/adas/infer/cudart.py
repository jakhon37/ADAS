"""Minimal CUDA runtime bindings via ctypes (no pycuda)."""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Optional

CUDA_SUCCESS = 0
CUDA_ERROR_NOT_READY = 600
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2


class CudaError(RuntimeError):
    pass


_lib: Optional[ctypes.CDLL] = None


def _load() -> ctypes.CDLL:
    for name in ("libcudart.so", "libcudart.so.11.0", "libcudart.so.11"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    found = ctypes.util.find_library("cudart")
    if found:
        return ctypes.CDLL(found)
    raise CudaError("libcudart.so not found")


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _load()
        _lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        _lib.cudaMalloc.restype = ctypes.c_int
        _lib.cudaFree.argtypes = [ctypes.c_void_p]
        _lib.cudaFree.restype = ctypes.c_int
        _lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        _lib.cudaMemcpyAsync.restype = ctypes.c_int
        _lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        _lib.cudaStreamCreate.restype = ctypes.c_int
        _lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        _lib.cudaStreamDestroy.restype = ctypes.c_int
        _lib.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        _lib.cudaEventCreate.restype = ctypes.c_int
        _lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _lib.cudaEventRecord.restype = ctypes.c_int
        _lib.cudaEventQuery.argtypes = [ctypes.c_void_p]
        _lib.cudaEventQuery.restype = ctypes.c_int
        _lib.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
        _lib.cudaEventSynchronize.restype = ctypes.c_int
        _lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
        _lib.cudaEventDestroy.restype = ctypes.c_int
        _lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        _lib.cudaGetErrorString.restype = ctypes.c_char_p
    return _lib


def check(err: int, what: str) -> None:
    if err != CUDA_SUCCESS:
        msg = lib().cudaGetErrorString(err)
        text = msg.decode("utf-8", "replace") if msg else str(err)
        raise CudaError("%s: %s (%s)" % (what, text, err))


def malloc(nbytes: int) -> int:
    ptr = ctypes.c_void_p()
    check(lib().cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
    return int(ptr.value or 0)


def free(ptr: int) -> None:
    if ptr:
        check(lib().cudaFree(ctypes.c_void_p(ptr)), "cudaFree")


def stream_create() -> int:
    s = ctypes.c_void_p()
    check(lib().cudaStreamCreate(ctypes.byref(s)), "cudaStreamCreate")
    return int(s.value or 0)


def stream_destroy(s: int) -> None:
    if s:
        check(lib().cudaStreamDestroy(ctypes.c_void_p(s)), "cudaStreamDestroy")


def event_create() -> int:
    e = ctypes.c_void_p()
    check(lib().cudaEventCreate(ctypes.byref(e)), "cudaEventCreate")
    return int(e.value or 0)


def event_record(event: int, stream: int) -> None:
    check(
        lib().cudaEventRecord(ctypes.c_void_p(event), ctypes.c_void_p(stream)),
        "cudaEventRecord",
    )


def event_synchronize(event: int) -> None:
    check(lib().cudaEventSynchronize(ctypes.c_void_p(event)), "cudaEventSynchronize")


def event_destroy(event: int) -> None:
    if event:
        check(lib().cudaEventDestroy(ctypes.c_void_p(event)), "cudaEventDestroy")


def memcpy_async(dst: int, src: int, nbytes: int, kind: int, stream: int) -> None:
    check(
        lib().cudaMemcpyAsync(
            ctypes.c_void_p(dst),
            ctypes.c_void_p(src),
            nbytes,
            kind,
            ctypes.c_void_p(stream),
        ),
        "cudaMemcpyAsync",
    )
