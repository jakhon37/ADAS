"""Synchronous TensorRT engine wrapper (JetPack 5 / TensorRT 8.5, static shapes).

Design
------
One engine, one execution context, one non-blocking CUDA stream, and one
page-locked host buffer plus one device buffer per binding, all allocated once
at construction. A call to :meth:`TrtEngine.infer` therefore performs **no**
host allocation: it copies the caller's input into the pinned staging buffer,
enqueues H2D copies, enqueues the network, enqueues D2H copies, and blocks on a
single ``cudaStreamSynchronize``.

Units
-----
``nbytes`` are bytes. ``last_*_ms`` timings are wall-clock milliseconds measured
on the host with ``time.perf_counter`` around the corresponding stage; the
inference figure includes the stream synchronisation, so it is end-to-end
latency, not isolated GPU compute.

Thread safety
-------------
None. A :class:`TrtEngine` owns a single execution context and a single set of
staging buffers, so exactly one thread may call :meth:`infer` at a time. Sharing
one instance across worker threads corrupts the staging buffers silently.

Failure behaviour
-----------------
* A missing file, an engine that fails to deserialise, or a dynamic (``-1``)
  dimension raises :class:`EngineError` at construction. Dynamic shapes are
  rejected rather than frozen to 1, which is how an earlier build of this
  project ended up with a ``1x3x1x1`` detector that ran without complaint.
* An engine whose sha256 differs from the digest ``models/MANIFEST.json``
  records for that filename raises :class:`EngineIntegrityError` **before** the
  blob is deserialised. This is on by default (``verify_manifest``): a swapped
  or truncated .engine in a safety-critical perception path must be refused,
  not run. An engine the manifest does not list cannot be verified; that is
  logged at WARNING and allowed, because an operator building a new model has
  nothing to check against yet.
* A wrong input name, rank or shape raises :class:`EngineError` before anything
  is copied to the device.
* Every CUDA call is checked; see :mod:`adas.infer.cudart`.
* :meth:`close` is idempotent and is also invoked from ``__del__`` as a
  backstop, but callers should close explicitly (or use the context manager) so
  that device memory is released deterministically -- this board has ~2.5 GiB
  of free RAM and several models competing for it.

Testing off-GPU
---------------
:class:`FakeTrtEngine` implements the same duck-typed surface backed by plain
numpy, so every decode path in :mod:`adas.perception` is unit-testable on a
machine with no CUDA, no TensorRT and no engine files.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from adas.infer import cudart

log = logging.getLogger("adas.infer")


class EngineError(RuntimeError):
    """Raised for any engine-loading, contract or inference failure."""


class EngineIntegrityError(EngineError):
    """An engine file is not the one ``models/MANIFEST.json`` records.

    A subclass of :class:`EngineError` so existing ``except EngineError``
    handlers keep working, but distinguishable for a caller that wants to treat
    "wrong bytes" differently from "will not deserialise".
    """


def sha256_file(path: str, chunk_bytes: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, read in 1 MiB chunks.

    Engines on this board range from 3 MB to 413 MB, so hashing is streamed.
    :class:`TrtEngine` calls this once per engine at construction to check the
    file against ``models/MANIFEST.json``. Measured on this Jetson: 10 ms for
    the 3 MB yolox_nano and 519 ms for the 413 MB UFLD-v2, once per process --
    :func:`verify_engine_file` memoises the result per (path, size, mtime), so a
    second construction of the same file costs 0.4 ms.
    """
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Manifest-backed integrity
# --------------------------------------------------------------------------- #

#: Repository root inferred from this file: ``src/adas/infer/trt_engine.py``.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

#: The digest inventory ``scripts/build_engines.py`` writes after a real build.
DEFAULT_MANIFEST_PATH = os.path.join(_REPO_ROOT, "models", "MANIFEST.json")

#: ``{absolute engine path: sha256}`` for every engine verified in this process.
#: :mod:`adas.io.health` publishes it on ``/healthz``, so an operator sees which
#: bytes are loaded rather than only which filename was configured. An engine
#: that could not be verified is absent from this map -- it is never listed with
#: an empty or invented digest.
VERIFIED_ENGINES: Dict[str, str] = {}

_manifest_cache: Dict[str, Dict[str, Any]] = {}
_digest_cache: Dict[Tuple[str, int, int], str] = {}


def load_manifest(manifest_path: Optional[str] = None) -> Dict[str, Any]:
    """Parse ``models/MANIFEST.json``, cached per path.

    Returns ``{}`` (and logs at WARNING) when the file is missing or malformed:
    a manifest that cannot be read means "nothing can be verified", which is
    reported, not silently treated as "everything is fine".
    """
    path = os.path.abspath(manifest_path or DEFAULT_MANIFEST_PATH)
    cached = _manifest_cache.get(path)
    if cached is not None:
        return cached
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        log.warning(
            "model manifest %s is unreadable (%s); engine digests cannot be checked",
            path, exc,
        )
        data = {}
    if not isinstance(data, dict):
        log.warning("model manifest %s is not a JSON object; ignoring it", path)
        data = {}
    _manifest_cache[path] = data
    return data


def manifest_entry(engine_path: str,
                   manifest_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The manifest record whose ``engine_file`` is this file's basename, or None.

    Matching is by basename on purpose: the manifest records what a file must
    *contain*, not where an operator put it, so a copy under another directory
    is still checked against the same digest.
    """
    name = os.path.basename(str(engine_path))
    models = load_manifest(manifest_path).get("models")
    if not isinstance(models, dict):
        return None
    for entry in models.values():
        if isinstance(entry, dict) and entry.get("engine_file") == name:
            return entry
    return None


def trt_version() -> str:
    """Running TensorRT version, or ``""`` when TensorRT is not importable."""
    try:
        import tensorrt as trt
    except Exception:  # pragma: no cover - the board always has TRT
        return ""
    return str(getattr(trt, "__version__", "") or "")


def verify_engine_file(engine_path: str, manifest_path: Optional[str] = None,
                       check_trt_version: bool = True) -> str:
    """Check one engine file against ``models/MANIFEST.json``.

    Returns:
        The file's sha256 when the manifest records a digest for that filename
        and the file matches it, or ``""`` when the manifest has no digest to
        check against (logged at WARNING -- "unverifiable" is a weaker claim
        than "verified" and the two are never conflated).

    Raises:
        EngineIntegrityError: the digest differs from the recorded one, or the
            manifest was built for a different TensorRT version than the one
            running (a serialized engine is version- and device-locked).
        EngineError: the file cannot be stat'ed or read.
    """
    path = os.path.abspath(str(engine_path))
    entry = manifest_entry(path, manifest_path)
    if entry is None:
        log.warning(
            "engine %s is not listed in %s: its integrity cannot be verified",
            path, os.path.abspath(manifest_path or DEFAULT_MANIFEST_PATH),
        )
        return ""
    expected = str(entry.get("engine_sha256") or "").strip().lower()
    if not expected:
        log.warning(
            "manifest entry for %s records no engine_sha256: integrity cannot be verified",
            os.path.basename(path),
        )
        return ""
    if check_trt_version:
        recorded = str(load_manifest(manifest_path).get("trt_version") or "")
        running = trt_version()
        if recorded and running and recorded != running:
            raise EngineIntegrityError(
                "engine %s was built against TensorRT %s but this process is running "
                "TensorRT %s. A serialized engine is version- and device-locked; "
                "rebuild it on this board with `python3 scripts/build_engines.py`."
                % (path, recorded, running)
            )
    try:
        stat = os.stat(path)
    except OSError as exc:
        raise EngineError("cannot stat engine %s: %s" % (path, exc)) from exc
    key = (path, int(stat.st_size), int(stat.st_mtime_ns))
    actual = _digest_cache.get(key)
    if actual is None:
        actual = sha256_file(path)
        _digest_cache[key] = actual
    if actual != expected:
        raise EngineIntegrityError(
            "engine %s does not match models/MANIFEST.json: sha256 is %s, expected "
            "%s (%d bytes on disk, the manifest records %s). Refusing to load a model "
            "that is not the one validated on this board -- restore the validated "
            "file or rebuild it with `python3 scripts/build_engines.py`."
            % (path, actual, expected, int(stat.st_size), entry.get("engine_bytes"))
        )
    VERIFIED_ENGINES[path] = actual
    log.debug("engine %s verified: sha256 %s", path, actual)
    return actual


def _dtype_map() -> Dict[object, object]:
    import tensorrt as trt

    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT8: np.int8,
        trt.DataType.INT32: np.int32,
        trt.DataType.BOOL: np.bool_,
    }
    for name, np_type in (("UINT8", np.uint8), ("INT64", np.int64), ("BF16", np.float32)):
        if hasattr(trt.DataType, name):
            mapping.setdefault(getattr(trt.DataType, name), np_type)
    return mapping


@dataclass
class Binding:
    """One engine IO tensor plus the host/device memory backing it."""

    name: str
    is_input: bool
    dtype: np.dtype
    shape: Tuple[int, ...]
    nbytes: int
    host: np.ndarray
    device: int

    @property
    def size(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 0


class TrtEngine:
    """Load a serialized TensorRT engine and run synchronous inference."""

    def __init__(
        self,
        engine_path: str,
        non_blocking_stream: bool = True,
        expected_sha256: Optional[str] = None,
        verify_manifest: bool = True,
    ) -> None:
        """Load ``engine_path``, refusing an engine that is not the validated one.

        Integrity is checked before the blob is deserialised:

        * ``expected_sha256`` pins the digest explicitly and always wins.
        * otherwise ``verify_manifest`` (the default) looks the file's basename
          up in ``models/MANIFEST.json`` and refuses a mismatch. An engine the
          manifest does not list is logged as unverifiable and loaded, so a
          freshly built model is not blocked by its own absence from the
          inventory.

        Pass ``verify_manifest=False`` only for a deliberately synthetic file
        (a fixture, a truncation test); there is no environment variable that
        turns the check off, because a supply-chain control with a silent
        bypass is not one.

        :attr:`sha256` is the verified digest, or ``""`` when the file could not
        be verified against anything.
        """
        self.engine_path = str(engine_path)
        self._closed = False
        self._stream = 0
        self._bindings: Dict[str, Binding] = {}
        self._pinned: List[cudart.PinnedBuffer] = []
        self._inputs: List[Binding] = []
        self._outputs: List[Binding] = []
        self.last_h2d_ms = 0.0
        self.last_exec_ms = 0.0
        self.last_d2h_ms = 0.0
        self.last_total_ms = 0.0

        if not os.path.isfile(self.engine_path):
            raise EngineError("TensorRT engine not found: %s" % self.engine_path)
        if expected_sha256:
            actual = sha256_file(self.engine_path)
            if actual.lower() != str(expected_sha256).strip().lower():
                raise EngineIntegrityError(
                    "engine %s has sha256 %s but %s was expected -- refusing to load a "
                    "model that is not the one that was validated"
                    % (self.engine_path, actual, expected_sha256)
                )
            self.sha256 = actual
            VERIFIED_ENGINES[os.path.abspath(self.engine_path)] = actual
        elif verify_manifest:
            self.sha256 = verify_engine_file(self.engine_path)
        else:
            self.sha256 = ""

        try:
            import tensorrt as trt
        except ImportError as exc:  # pragma: no cover - board always has TRT
            raise EngineError("tensorrt is not importable: %s" % exc) from exc

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        with open(self.engine_path, "rb") as handle:
            blob = handle.read()
        engine = self._runtime.deserialize_cuda_engine(blob)
        del blob
        if engine is None:
            raise EngineError(
                "failed to deserialize %s -- a TensorRT engine is version- and "
                "device-locked; rebuild it on this board with TensorRT %s"
                % (self.engine_path, getattr(trt, "__version__", "?"))
            )
        self._engine = engine
        self._ctx = engine.create_execution_context()
        if self._ctx is None:
            raise EngineError("could not create an execution context for %s" % self.engine_path)

        try:
            self._stream = cudart.stream_create(non_blocking=non_blocking_stream)
            self._build_bindings(trt)
        except Exception:
            self.close()
            raise

        self.device_memory_bytes = int(getattr(self._ctx, "device_memory_size", 0) or 0)

    # -- construction helpers --------------------------------------------- #

    def _build_bindings(self, trt) -> None:
        dmap = _dtype_map()
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            is_input = self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            trt_dtype = self._engine.get_tensor_dtype(name)
            if trt_dtype not in dmap:
                raise EngineError("binding %s has unsupported dtype %s" % (name, trt_dtype))
            dtype = np.dtype(dmap[trt_dtype])
            shape = tuple(int(x) for x in self._engine.get_tensor_shape(name))
            if not shape or any(dim < 0 for dim in shape):
                raise EngineError(
                    "binding %s of %s has a dynamic shape %s. Build a fully static "
                    "engine (--minShapes == --optShapes == --maxShapes)."
                    % (name, self.engine_path, shape)
                )
            buffer = cudart.PinnedBuffer(shape, dtype)
            self._pinned.append(buffer)
            device = cudart.malloc(buffer.nbytes)
            binding = Binding(
                name=name,
                is_input=is_input,
                dtype=dtype,
                shape=shape,
                nbytes=buffer.nbytes,
                host=buffer.array,
                device=device,
            )
            self._bindings[name] = binding
            self._ctx.set_tensor_address(name, device)
        self._inputs = [b for b in self._bindings.values() if b.is_input]
        self._outputs = [b for b in self._bindings.values() if not b.is_input]
        if not self._inputs:
            raise EngineError("%s exposes no input bindings" % self.engine_path)
        if not self._outputs:
            raise EngineError("%s exposes no output bindings" % self.engine_path)

    # -- contract ---------------------------------------------------------- #

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self._inputs[0].shape

    @property
    def input_name(self) -> str:
        return self._inputs[0].name

    @property
    def input_names(self) -> List[str]:
        return [b.name for b in self._inputs]

    @property
    def input_dtype(self) -> np.dtype:
        """Declared dtype of the primary input. ``float16`` for some engines."""
        return self._inputs[0].dtype

    @property
    def output_names(self) -> List[str]:
        return [b.name for b in self._outputs]

    @property
    def output_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {b.name: b.shape for b in self._outputs}

    @property
    def output_dtypes(self) -> Dict[str, np.dtype]:
        return {b.name: b.dtype for b in self._outputs}

    def binding(self, name: str) -> Binding:
        try:
            return self._bindings[name]
        except KeyError:
            raise EngineError(
                "%s has no binding %r; bindings are %s"
                % (self.engine_path, name, sorted(self._bindings))
            ) from None

    def describe(self) -> str:
        """One-line human-readable contract, for startup logs and errors."""
        parts = []
        for binding in list(self._inputs) + list(self._outputs):
            parts.append(
                "%s%s:%s%s"
                % ("in " if binding.is_input else "out ", binding.name, binding.dtype.str, binding.shape)
            )
        return "%s [%s]" % (self.engine_path, ", ".join(parts))

    # -- inference --------------------------------------------------------- #

    def _stage_inputs(self, inputs: Mapping[str, np.ndarray]) -> None:
        for binding in self._inputs:
            array = inputs.get(binding.name)
            if array is None:
                if len(self._inputs) == 1 and len(inputs) == 1:
                    array = next(iter(inputs.values()))
                else:
                    raise EngineError(
                        "missing input %r for %s; supplied %s, expected %s"
                        % (binding.name, self.engine_path, sorted(inputs), self.input_names)
                    )
            array = np.asarray(array)
            if array.shape != binding.shape:
                raise EngineError(
                    "input %r of %s expects shape %s, got %s"
                    % (binding.name, self.engine_path, binding.shape, array.shape)
                )
            if array.dtype.kind not in "fiub":
                raise EngineError(
                    "input %r must be numeric, got dtype %s" % (binding.name, array.dtype)
                )
            # One pass: numpy converts dtype and layout straight into pinned memory.
            np.copyto(binding.host, array, casting="unsafe")

    def _run(self, inputs: Mapping[str, np.ndarray]) -> None:
        if self._closed:
            raise EngineError("infer() called on a closed engine (%s)" % self.engine_path)
        start = time.perf_counter()
        self._stage_inputs(inputs)
        for binding in self._inputs:
            cudart.memcpy_async(
                binding.device,
                int(binding.host.ctypes.data),
                binding.nbytes,
                cudart.cudaMemcpyHostToDevice,
                self._stream,
            )
        after_h2d = time.perf_counter()
        if not self._ctx.execute_async_v3(self._stream):
            raise EngineError("execute_async_v3 failed for %s" % self.engine_path)
        after_exec = time.perf_counter()
        for binding in self._outputs:
            cudart.memcpy_async(
                int(binding.host.ctypes.data),
                binding.device,
                binding.nbytes,
                cudart.cudaMemcpyDeviceToHost,
                self._stream,
            )
        cudart.stream_synchronize(self._stream)
        end = time.perf_counter()
        self.last_h2d_ms = (after_h2d - start) * 1000.0
        self.last_exec_ms = (after_exec - after_h2d) * 1000.0
        self.last_d2h_ms = (end - after_exec) * 1000.0
        self.last_total_ms = (end - start) * 1000.0

    def infer(self, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run one inference and return **owned copies** of every output.

        Safe to keep past the next call. Costs one extra host memcpy per output;
        use :meth:`infer_views` in a hot loop that consumes the results
        immediately.
        """
        self._run(inputs)
        return {b.name: b.host.copy() for b in self._outputs}

    def infer_views(self, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run one inference and return **read-only views** of the pinned buffers.

        No copy is made, so this avoids a full output-sized memcpy per frame
        (8.6 MB for a 1x25200x85 float32 detector head). The returned arrays are
        invalidated by the next :meth:`infer`/:meth:`infer_views` call and by
        :meth:`close`; copy anything that must outlive them. They are marked
        non-writeable so an accidental in-place edit fails loudly instead of
        corrupting the staging buffer.
        """
        self._run(inputs)
        views = {}
        for binding in self._outputs:
            view = binding.host.view()
            view.flags.writeable = False
            views[binding.name] = view
        return views

    def infer_into(self, inputs: Mapping[str, np.ndarray], out: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run one inference, writing outputs into caller-owned arrays.

        ``out`` must already contain a correctly shaped array for every output
        name (extra keys are ignored). Nothing is allocated. Returns ``out``.
        """
        self._run(inputs)
        for binding in self._outputs:
            target = out.get(binding.name)
            if target is None:
                raise EngineError(
                    "infer_into() needs a destination for output %r; got %s"
                    % (binding.name, sorted(out))
                )
            if target.shape != binding.shape:
                raise EngineError(
                    "destination for %r has shape %s, engine produces %s"
                    % (binding.name, target.shape, binding.shape)
                )
            np.copyto(target, binding.host, casting="unsafe")
        return out

    # -- lifecycle --------------------------------------------------------- #

    def close(self) -> None:
        """Release context, device memory, pinned memory and the stream. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for binding in self._bindings.values():
            try:
                cudart.free(binding.device)
            except cudart.CudaError:
                # Nothing useful can be done during teardown, but never swallow
                # it silently: the caller needs to know memory leaked.
                log.exception(
                    "cudaFree failed for binding %s of %s", binding.name, self.engine_path
                )
        self._bindings.clear()
        self._inputs = []
        self._outputs = []
        for buffer in self._pinned:
            buffer.close()
        self._pinned = []
        if self._stream:
            try:
                cudart.stream_destroy(self._stream)
            finally:
                self._stream = 0
        # Drop TensorRT objects in dependency order.
        self._ctx = None
        self._engine = None
        self._runtime = None

    def __enter__(self) -> "TrtEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        try:
            self.close()
        except Exception:
            pass


class FakeTrtEngine:
    """CPU test double with the same surface as :class:`TrtEngine`.

    Construct it with the binding contract you want to emulate plus either a
    dict of fixed output arrays or a callable ``fn(inputs) -> dict``. This is
    what lets the YOLO/YOLOX/MiDaS decode paths be tested in CI with no Jetson,
    no CUDA and no engine file.

    Parameters
    ----------
    input_shapes:
        ``{name: shape}``. The first entry is what :attr:`input_name` and
        :attr:`input_shape` report.
    outputs:
        ``{name: ndarray}`` returned verbatim (copied) from :meth:`infer`, or a
        callable taking the staged input dict and returning such a mapping.
    output_shapes:
        Required only when ``outputs`` is a callable, so the contract is known
        before the first call.

    Failure behaviour matches :class:`TrtEngine`: a wrong shape or a missing
    input raises :class:`EngineError`, and calling after :meth:`close` raises.
    """

    def __init__(
        self,
        input_shapes: Mapping[str, Sequence[int]],
        outputs=None,
        output_shapes: Optional[Mapping[str, Sequence[int]]] = None,
        input_dtype: object = np.float32,
        engine_path: str = "<fake>",
    ) -> None:
        if not input_shapes:
            raise EngineError("FakeTrtEngine needs at least one input")
        self.engine_path = engine_path
        self._closed = False
        self._input_shapes = {k: tuple(int(d) for d in v) for k, v in input_shapes.items()}
        self._input_dtype = np.dtype(input_dtype)
        self._fn: Optional[Callable[[Dict[str, np.ndarray]], Mapping[str, np.ndarray]]] = None
        self._fixed: Dict[str, np.ndarray] = {}
        if callable(outputs):
            self._fn = outputs
            if not output_shapes:
                raise EngineError("FakeTrtEngine with a callable needs output_shapes")
            self._output_shapes = {k: tuple(int(d) for d in v) for k, v in output_shapes.items()}
        else:
            fixed = dict(outputs or {})
            if not fixed:
                raise EngineError("FakeTrtEngine needs outputs or an output callable")
            self._fixed = {k: np.asarray(v) for k, v in fixed.items()}
            self._output_shapes = {k: tuple(v.shape) for k, v in self._fixed.items()}
        self.last_h2d_ms = 0.0
        self.last_exec_ms = 0.0
        self.last_d2h_ms = 0.0
        self.last_total_ms = 0.0
        self.device_memory_bytes = 0
        #: Number of times :meth:`infer` (or a variant) has run. Handy for
        #: asserting a reduced-cadence channel really is reduced-cadence.
        self.call_count = 0
        self.last_inputs: Dict[str, np.ndarray] = {}

    @property
    def input_name(self) -> str:
        return next(iter(self._input_shapes))

    @property
    def input_names(self) -> List[str]:
        return list(self._input_shapes)

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self._input_shapes[self.input_name]

    @property
    def input_dtype(self) -> np.dtype:
        return self._input_dtype

    @property
    def output_names(self) -> List[str]:
        return list(self._output_shapes)

    @property
    def output_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return dict(self._output_shapes)

    def describe(self) -> str:
        return "%s [fake: in %s, out %s]" % (
            self.engine_path,
            self._input_shapes,
            self._output_shapes,
        )

    def _run(self, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self._closed:
            raise EngineError("infer() called on a closed engine (%s)" % self.engine_path)
        staged: Dict[str, np.ndarray] = {}
        for name, shape in self._input_shapes.items():
            array = inputs.get(name)
            if array is None:
                if len(self._input_shapes) == 1 and len(inputs) == 1:
                    array = next(iter(inputs.values()))
                else:
                    raise EngineError("missing input %r; supplied %s" % (name, sorted(inputs)))
            array = np.asarray(array)
            if array.shape != shape:
                raise EngineError("input %r expects shape %s, got %s" % (name, shape, array.shape))
            staged[name] = array.astype(self._input_dtype, copy=False)
        self.last_inputs = staged
        self.call_count += 1
        produced = self._fn(staged) if self._fn is not None else self._fixed
        result: Dict[str, np.ndarray] = {}
        for name, shape in self._output_shapes.items():
            value = produced.get(name)
            if value is None:
                raise EngineError("fake engine produced no output %r" % name)
            value = np.asarray(value)
            if value.shape != shape:
                raise EngineError("output %r expects shape %s, got %s" % (name, shape, value.shape))
            result[name] = value
        return result

    def infer(self, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {k: np.array(v, copy=True) for k, v in self._run(inputs).items()}

    def infer_views(self, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        views = {}
        for name, value in self._run(inputs).items():
            view = np.asarray(value).view()
            view.flags.writeable = False
            views[name] = view
        return views

    def infer_into(self, inputs: Mapping[str, np.ndarray], out: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for name, value in self._run(inputs).items():
            target = out.get(name)
            if target is None:
                raise EngineError("infer_into() needs a destination for output %r" % name)
            np.copyto(target, value, casting="unsafe")
        return out

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "FakeTrtEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
