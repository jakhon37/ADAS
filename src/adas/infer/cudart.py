"""Minimal CUDA runtime bindings via ctypes (no pycuda).

Why ctypes
----------
``pycuda`` may not be installed on the target (it is an explicit project
prohibition on this Jetson), and the TensorRT Python bindings do not expose
memory management. Everything this project needs from the CUDA runtime is a
dozen entry points, so they are bound directly out of ``libcudart.so``.

Units and ownership
-------------------
* Every pointer is carried as a plain Python ``int`` holding the raw address.
  Zero means "no allocation"; freeing zero is a no-op.
* ``nbytes`` arguments are byte counts, never element counts.
* Allocation functions return the address and transfer ownership to the
  caller. Nothing here is garbage collected -- :class:`PinnedBuffer` is the
  only RAII-style helper and it must be closed explicitly (or used as a
  context manager).

Failure behaviour
-----------------
Every entry point checks the returned ``cudaError_t`` and raises
:class:`CudaError` carrying ``cudaGetErrorString`` plus the numeric code. No
call silently ignores a failure. A failure to load ``libcudart.so`` at all
raises :class:`CudaError` from :func:`lib`, so a host without CUDA fails at
first use with a clear message rather than at import time -- which is what
makes the pure-numpy decode paths in :mod:`adas.perception` unit-testable on a
machine with no GPU.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Optional, Tuple

import numpy as np

CUDA_SUCCESS = 0
CUDA_ERROR_NOT_READY = 600

cudaMemcpyHostToHost = 0
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3

#: ``cudaHostAllocDefault`` -- page-locked, not mapped, not write-combined.
cudaHostAllocDefault = 0
#: ``cudaStreamNonBlocking`` -- do not implicitly synchronise with the legacy
#: default stream. Without this every TensorRT enqueue serialises against any
#: other library in the process that uses stream 0.
cudaStreamNonBlocking = 1


class CudaError(RuntimeError):
    """A CUDA runtime call returned a non-zero ``cudaError_t``."""


_lib: Optional[ctypes.CDLL] = None


def _load() -> ctypes.CDLL:
    for name in ("libcudart.so", "libcudart.so.11.0", "libcudart.so.11", "libcudart.so.10.2"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    found = ctypes.util.find_library("cudart")
    if found:
        return ctypes.CDLL(found)
    raise CudaError(
        "libcudart.so not found. CUDA is required for TensorRT inference; "
        "the pure-numpy decode helpers do not need it."
    )


def _bind(handle: ctypes.CDLL) -> None:
    void_p = ctypes.c_void_p
    void_pp = ctypes.POINTER(ctypes.c_void_p)

    signatures = (
        ("cudaMalloc", [void_pp, ctypes.c_size_t]),
        ("cudaFree", [void_p]),
        ("cudaHostAlloc", [void_pp, ctypes.c_size_t, ctypes.c_uint]),
        ("cudaFreeHost", [void_p]),
        ("cudaMemcpy", [void_p, void_p, ctypes.c_size_t, ctypes.c_int]),
        ("cudaMemcpyAsync", [void_p, void_p, ctypes.c_size_t, ctypes.c_int, void_p]),
        ("cudaMemsetAsync", [void_p, ctypes.c_int, ctypes.c_size_t, void_p]),
        ("cudaStreamCreate", [void_pp]),
        ("cudaStreamCreateWithFlags", [void_pp, ctypes.c_uint]),
        ("cudaStreamDestroy", [void_p]),
        ("cudaStreamSynchronize", [void_p]),
        ("cudaDeviceSynchronize", []),
        ("cudaEventCreate", [void_pp]),
        ("cudaEventRecord", [void_p, void_p]),
        ("cudaEventQuery", [void_p]),
        ("cudaEventSynchronize", [void_p]),
        ("cudaEventElapsedTime", [ctypes.POINTER(ctypes.c_float), void_p, void_p]),
        ("cudaEventDestroy", [void_p]),
        ("cudaGetLastError", []),
        ("cudaMemGetInfo", [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]),
    )
    for name, argtypes in signatures:
        fn = getattr(handle, name, None)
        if fn is None:  # pragma: no cover - every listed symbol exists in CUDA >= 10
            continue
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
    handle.cudaGetErrorString.argtypes = [ctypes.c_int]
    handle.cudaGetErrorString.restype = ctypes.c_char_p


def lib() -> ctypes.CDLL:
    """Return the loaded ``libcudart`` handle, loading and binding it once."""
    global _lib
    if _lib is None:
        handle = _load()
        _bind(handle)
        _lib = handle
    return _lib


def check(err: int, what: str) -> None:
    """Raise :class:`CudaError` unless ``err`` is ``cudaSuccess``."""
    if err != CUDA_SUCCESS:
        msg = lib().cudaGetErrorString(err)
        text = msg.decode("utf-8", "replace") if msg else str(err)
        raise CudaError("%s: %s (cudaError_t=%d)" % (what, text, err))


# --------------------------------------------------------------------------- #
# Device memory
# --------------------------------------------------------------------------- #


def malloc(nbytes: int) -> int:
    """Allocate ``nbytes`` of device memory and return its address."""
    if nbytes <= 0:
        raise ValueError("cudaMalloc needs a positive size, got %r" % (nbytes,))
    ptr = ctypes.c_void_p()
    check(lib().cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "cudaMalloc(%d)" % nbytes)
    address = int(ptr.value or 0)
    if address == 0:
        raise CudaError("cudaMalloc returned a null pointer for %d bytes" % nbytes)
    return address


def free(ptr: int) -> None:
    """Free device memory. Freeing 0 is a no-op so ``close()`` stays idempotent."""
    if ptr:
        check(lib().cudaFree(ctypes.c_void_p(ptr)), "cudaFree")


def mem_get_info() -> Tuple[int, int]:
    """Return ``(free_bytes, total_bytes)`` of device memory."""
    free_b = ctypes.c_size_t()
    total_b = ctypes.c_size_t()
    check(lib().cudaMemGetInfo(ctypes.byref(free_b), ctypes.byref(total_b)), "cudaMemGetInfo")
    return int(free_b.value), int(total_b.value)


# --------------------------------------------------------------------------- #
# Pinned (page-locked) host memory
# --------------------------------------------------------------------------- #


def host_alloc(nbytes: int, flags: int = cudaHostAllocDefault) -> int:
    """Allocate ``nbytes`` of page-locked host memory and return its address.

    Page-locked memory is what makes ``cudaMemcpyAsync`` genuinely
    asynchronous. A copy issued against ordinary pageable memory is staged by
    the driver through an internal bounce buffer, so it neither overlaps
    compute nor reaches full PCIe/soc-fabric bandwidth.
    """
    if nbytes <= 0:
        raise ValueError("cudaHostAlloc needs a positive size, got %r" % (nbytes,))
    ptr = ctypes.c_void_p()
    check(
        lib().cudaHostAlloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(flags)),
        "cudaHostAlloc(%d)" % nbytes,
    )
    address = int(ptr.value or 0)
    if address == 0:
        raise CudaError("cudaHostAlloc returned a null pointer for %d bytes" % nbytes)
    return address


def free_host(ptr: int) -> None:
    """Free page-locked host memory. Freeing 0 is a no-op."""
    if ptr:
        check(lib().cudaFreeHost(ctypes.c_void_p(ptr)), "cudaFreeHost")


class PinnedBuffer:
    """A page-locked host allocation exposed as a numpy array.

    ``array`` is a view onto memory owned by the CUDA runtime, **not** by
    numpy. It stays valid until :meth:`close`; using it afterwards is a
    use-after-free, so callers that hand the array outward must either copy it
    or document the lifetime (see :meth:`adas.infer.trt_engine.TrtEngine.infer_views`).

    Falls back to an ordinary numpy allocation when ``cudaHostAlloc`` is
    unavailable, so a machine without CUDA can still exercise the surrounding
    code; :attr:`pinned` records which happened.
    """

    __slots__ = ("array", "nbytes", "pinned", "_ptr")

    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype, allow_fallback: bool = False) -> None:
        dtype = np.dtype(dtype)
        count = 1
        for dim in shape:
            count *= int(dim)
        nbytes = int(count) * int(dtype.itemsize)
        if nbytes <= 0:
            raise ValueError("PinnedBuffer needs a non-empty shape, got %r" % (shape,))
        self.nbytes = nbytes
        try:
            self._ptr = host_alloc(nbytes)
        except CudaError:
            if not allow_fallback:
                raise
            self._ptr = 0
            self.pinned = False
            self.array = np.zeros(shape, dtype=dtype)
            return
        self.pinned = True
        raw = (ctypes.c_byte * nbytes).from_address(self._ptr)
        self.array = np.frombuffer(raw, dtype=dtype, count=count).reshape(shape)
        self.array[...] = 0

    @property
    def address(self) -> int:
        """Host address of the first byte, for ``cudaMemcpy*``."""
        if self._ptr:
            return self._ptr
        return int(self.array.ctypes.data)

    def close(self) -> None:
        """Release the allocation. Idempotent; the array must not be used after."""
        if self._ptr:
            free_host(self._ptr)
            self._ptr = 0
        self.array = np.empty(0, dtype=self.array.dtype)

    def __enter__(self) -> "PinnedBuffer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Streams
# --------------------------------------------------------------------------- #


def stream_create(non_blocking: bool = True) -> int:
    """Create a stream. ``non_blocking`` avoids implicit sync with stream 0."""
    handle = ctypes.c_void_p()
    if non_blocking and hasattr(lib(), "cudaStreamCreateWithFlags"):
        check(
            lib().cudaStreamCreateWithFlags(ctypes.byref(handle), ctypes.c_uint(cudaStreamNonBlocking)),
            "cudaStreamCreateWithFlags",
        )
    else:
        check(lib().cudaStreamCreate(ctypes.byref(handle)), "cudaStreamCreate")
    return int(handle.value or 0)


def stream_destroy(stream: int) -> None:
    """Destroy a stream. Destroying 0 is a no-op."""
    if stream:
        check(lib().cudaStreamDestroy(ctypes.c_void_p(stream)), "cudaStreamDestroy")


def stream_synchronize(stream: int) -> None:
    """Block until every operation queued on ``stream`` has completed.

    This is the whole synchronisation story for a synchronous inference: one
    call, one driver round trip. Creating, recording, waiting on and destroying
    an event per frame costs four round trips and measures nothing.
    """
    check(lib().cudaStreamSynchronize(ctypes.c_void_p(stream)), "cudaStreamSynchronize")


def device_synchronize() -> None:
    """Block until every stream on the current device has completed."""
    check(lib().cudaDeviceSynchronize(), "cudaDeviceSynchronize")


# --------------------------------------------------------------------------- #
# Events (kept for optional on-device timing, not used per inference)
# --------------------------------------------------------------------------- #


def event_create() -> int:
    handle = ctypes.c_void_p()
    check(lib().cudaEventCreate(ctypes.byref(handle)), "cudaEventCreate")
    return int(handle.value or 0)


def event_record(event: int, stream: int) -> None:
    check(
        lib().cudaEventRecord(ctypes.c_void_p(event), ctypes.c_void_p(stream)),
        "cudaEventRecord",
    )


def event_query(event: int) -> bool:
    """True when the event has completed; False while it is still pending."""
    err = lib().cudaEventQuery(ctypes.c_void_p(event))
    if err == CUDA_SUCCESS:
        return True
    if err == CUDA_ERROR_NOT_READY:
        return False
    check(err, "cudaEventQuery")
    return False


def event_synchronize(event: int) -> None:
    check(lib().cudaEventSynchronize(ctypes.c_void_p(event)), "cudaEventSynchronize")


def event_elapsed_ms(start: int, end: int) -> float:
    """Milliseconds between two recorded events. Both must have completed."""
    out = ctypes.c_float()
    check(
        lib().cudaEventElapsedTime(ctypes.byref(out), ctypes.c_void_p(start), ctypes.c_void_p(end)),
        "cudaEventElapsedTime",
    )
    return float(out.value)


def event_destroy(event: int) -> None:
    if event:
        check(lib().cudaEventDestroy(ctypes.c_void_p(event)), "cudaEventDestroy")


# --------------------------------------------------------------------------- #
# Copies
# --------------------------------------------------------------------------- #


def memcpy_async(dst: int, src: int, nbytes: int, kind: int, stream: int) -> None:
    """Queue an asynchronous copy of ``nbytes`` on ``stream``."""
    if nbytes <= 0:
        return
    check(
        lib().cudaMemcpyAsync(
            ctypes.c_void_p(dst),
            ctypes.c_void_p(src),
            ctypes.c_size_t(nbytes),
            ctypes.c_int(kind),
            ctypes.c_void_p(stream),
        ),
        "cudaMemcpyAsync",
    )


def memcpy(dst: int, src: int, nbytes: int, kind: int) -> None:
    """Blocking copy of ``nbytes``."""
    if nbytes <= 0:
        return
    check(
        lib().cudaMemcpy(
            ctypes.c_void_p(dst),
            ctypes.c_void_p(src),
            ctypes.c_size_t(nbytes),
            ctypes.c_int(kind),
        ),
        "cudaMemcpy",
    )
