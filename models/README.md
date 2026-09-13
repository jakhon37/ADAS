# ADAS inference models

Everything in this directory except `MANIFEST.json` and this file is a build
artifact and is **not** committed (`.gitignore` excludes `models/*.onnx`,
`models/*.engine`, `models/*.engine.build.log` and `*.log`).

`MANIFEST.json` is the contract. It records, per model: source URL, licence,
pinned ONNX SHA-256, built engine SHA-256, TensorRT version, exact binding
names/shapes/dtypes, preprocessing (resize mode, colour order, scale,
mean/std), output semantics, and the GPU compute time measured by `trtexec` on
this board. Read it before writing any pre- or post-processing code.

**As of 2026-09-13 the manifest is enforced at runtime, not just documentation.**
`adas.infer.trt_engine.verify_engine_file` checks an engine's sha256 and the
manifest's recorded TensorRT version *before* the plan is deserialised, and
`TrtEngine.__init__` does it by default, so every call site is covered. A mismatch
raises `EngineIntegrityError` and the process refuses to start; there is deliberately
no environment-variable bypass. An engine the manifest does not list logs a WARNING
and loads as *unverifiable*. The digests that were verified are published on
`/healthz` as `engine_sha256`. Until this change, nothing passed `expected_sha256`
anywhere in the production path and a swapped `.engine` loaded silently into the
detector that feeds AEB.

Consequence for the workflow: `scripts/build_engines.py` writes the digests after a
real build, so a normal rebuild is unaffected — but **an engine copied in by hand
without updating the manifest will now stop the process, by design.**

## Reproducing the model set

```bash
cd ~/myspace/ADAS
bash scripts/fetch_models.sh              # SHA-256-pinned downloads
python3 scripts/build_engines.py          # ONNX -> TensorRT FP16 + verify
```

Both scripts are driven entirely by `MANIFEST.json`; neither carries its own
copy of a URL or a hash. Individual models:

```bash
bash scripts/fetch_models.sh yolop
python3 scripts/build_engines.py --only yolop --force
python3 scripts/build_engines.py --list
```

`scripts/build_yolo_engine.py` still works and is now a thin shim over these
two, plus an optional "wait until the board is idle" gate.

### Shared-board rules

This Jetson Xavier NX is used by several projects at once (DMS, ADAS,
cv-research). Every TensorRT invocation in `build_engines.py` — build, verify
and introspection — is serialised behind `flock /tmp/jetson-gpu.lock`, and each
build is capped with `--memPoolSize=workspace:<MiB>` from the manifest so the
builder stays inside the ~2.5 GiB free-RAM ceiling. Do not call `trtexec`
directly without that lock.

## What is here and why

| Model | Role | Licence | Notes |
|---|---|---|---|
| `yolox_nano` | object detection | **Apache-2.0** | Licence-clean default detector |
| `yolox_tiny` | object detection | **Apache-2.0** | Higher accuracy, same pre/post-processing |
| `yolov5n` | object detection | **AGPL-3.0-only** | Development baseline only — see the licence warning below |
| `ufldv2_culane_res18` | lane detection | MIT | Real lane geometry; replaces the hardcoded 36 %/64 % mock |
| `yolop` | detection + drivable area + lane mask | MIT | Free-space source and cross-check; schedule below the detection rate |
| `midas_v21_small` | monocular relative depth | MIT | **Demoted.** Ordinal signal only; publishes no metric range at all — see below |

### Licence warning: yolov5n is AGPL-3.0

`pyproject.toml` declares this project MIT while `models/yolov5n.onnx` is
Ultralytics YOLOv5 v7.0 under **AGPL-3.0-only**. That is a genuine conflict for
any build that is distributed or served over a network. `yolox_nano` and
`yolox_tiny` (Apache-2.0) are drop-in alternatives at the same input scale and
were added specifically so the project can switch. `yolov5n` is kept as a
measured baseline; treat it as development-only.

Two model licences also carry dataset constraints that the code licence does
not cover: YOLOP is trained on BDD100K and UFLDv2 on CULane, both of which are
research/non-commercial datasets. The MIT code licence does not launder the
dataset terms — record this before any commercial deployment.

## Binding contract

Read back from each built engine with the TensorRT runtime, not copied from the
ONNX. `MANIFEST.json` → `models.<name>.bindings` carries the same data with
`source: "tensorrt_runtime"`.

| Model | Input | Outputs (in binding order) |
|---|---|---|
| `yolov5n` | `images` `1×3×640×640` **float16** | `output0` `1×25200×85` **float16** |
| `yolox_nano` | `images` `1×3×416×416` float32 | `output` `1×3549×85` float32 |
| `yolox_tiny` | `images` `1×3×416×416` float32 | `output` `1×3549×85` float32 |
| `yolop` | `images` `1×3×640×640` float32 | `det_out` `1×25200×6`, `drive_area_seg` `1×2×640×640`, `lane_line_seg` `1×2×640×640`, all float32 |
| `midas_v21_small` | `0` `1×3×256×256` float32 | `797` `1×256×256` float32 |
| `ufldv2_culane_res18` | `input` `1×3×320×1600` float32 | `loc_row` `1×200×72×4`, `loc_col` `1×100×81×4`, `exist_row` `1×2×72×4`, `exist_col` `1×2×81×4`, all float32 |

Two things to watch:

* **`yolov5n` binds float16 on both sides**; every other engine binds float32.
  TensorRT picked the IO format when that plan was built. `TrtEngine` casts in
  both directions, so callers can keep handing it float32 — but anything that
  reads a raw host buffer or assumes `float32` dtype must handle it.
* **MiDaS binding names are the literal strings `"0"` and `"797"`** — the v2.1
  ONNX export kept torch's numeric tensor ids. Use `TrtEngine.input_name` and
  `TrtEngine.output_names[0]` rather than hardcoding them.

## Preprocessing, in one place

Getting these wrong is silent: the engine still runs and still emits plausible
numbers. The manifest is authoritative; this is the summary.

| Model | Colour | Resize | Scale | mean / std |
|---|---|---|---|---|
| `yolov5n` | **RGB** | centred letterbox 640×640, pad 114 | `/255` | none |
| `yolox_*` | **BGR** | top-left letterbox 416×416, pad 114 | **none — raw 0…255** | none |
| `yolop` | **RGB** | centred letterbox 640×640, pad 114 | `/255` | ImageNet |
| `midas_v21_small` | **RGB** | stretch to 256×256 (aspect **not** preserved) | `/255` | ImageNet |
| `ufldv2_culane_res18` | **RGB** | stretch to 1600×533, then keep the **bottom 320 rows** | `/255` | ImageNet |

ImageNet = mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`, applied
after the `/255`. Build every blob as NCHW float32, batch 1; `TrtEngine` casts it
to whatever dtype the engine binds (see the binding contract above).

Three traps worth repeating:

* **YOLOX does not divide by 255 and expects BGR.** Its exported ONNX bakes no
  normalisation at all. Measured here on frame 150 of `example.mp4`, counting
  decoded detections above 0.30: BGR + raw `0…255` gives 10 (max conf 0.640),
  RGB + raw gives 9 (0.600), and **either `/255` variant gives 0**. Dividing by
  255 does not degrade this model, it silences it — with no error raised.

  **Correction, re-measured 2026-09-13:** those counts are **pre-NMS**, and
  `MANIFEST.json` → `yolox_nano.preprocessing.verified_empirically` does not say so.
  Running the *shipped* detector on that exact frame with
  `confidence_threshold=0.30` and `class_ids="all"` yields **3** detections after NMS
  (0.640, 0.340, 0.306), not 10. The max-confidence figure reproduces exactly, so the
  preprocessing conclusion — BGR, raw 0…255 — stands unchanged; only the count was
  misleading. `MANIFEST.json` is not owned by this file and still carries the
  unqualified number.
* **UFLDv2 is a stretch, not a letterbox**, so the x and y scale factors back
  to the original frame differ. The vendored
  `Ultra-Fast-Lane-Detection-v2/deploy/trt_infer.py` uses a different,
  demo-only crop and skips the ImageNet normalisation entirely — do not copy
  it. The canonical transform is `data/dataloader.py::get_test_loader`.
* **MiDaS output is inverse relative depth**, unitless, with an unknown affine
  scale and shift per frame. Larger means closer. It must never be published as
  a metric range without a per-frame alignment against a metric reference — and on
  this model, at this resolution, on this footage, **even with that alignment it must
  not be published as a metric range at all**. See the next section.

### MiDaS is not a range channel on this board

`adas.perception.depth.DepthRangeChannel` used to publish
`RangeEstimate(source=DEPTH_MODEL, distance_m=..., confidence≈0.70)` per detection,
and the safety arbiter substituted it into the lead range that TTC, RSS and AEB are
computed from. Measured on this board against the reference range over the replay
clip, that output carried almost no range information:

| what was measured | result |
|---|---|
| per-object Spearman rank correlation vs reference range | **0.14–0.25** across eight sampling strategies, plus a 3×-zoomed second inference pass |
| best variant found (road strip just below the box bottom edge) | 0.686 — still short of 0.80, still collapsing the far field, and it consumes `box.y2`, the same pixel row `ground_plane_range` already uses, so it forfeits the independence that was the channel's entire justification |
| true range spread vs reported spread | 5.3–61.5 m compressed into 3.7–13.4 m |
| single worst case | at frame 300 the 54 m car was reported *nearer* (8.51 m) than the 13 m car (10.18 m) |
| road-plane affine fit quality (this part works) | relative residual 0.041–0.076 |

The channel is therefore **demoted, not repaired**. `update()` returns
`RangeSource.UNAVAILABLE` with confidence 0 for every box.
`DepthRangeChannel(publish_metric=True)` is a *request*, not a switch: a rolling
self-audit computes Spearman over the last 240 (reference range, sampled disparity)
pairs and metres flow only while that measures ≥ 0.80 over ≥ 24 pairs. It fails
closed — too few pairs, or a NaN correlation from constant disparity, keeps the gate
shut. A separate `OrdinalDepth` record (disparity, rank, of, normalized) carries the
unitless signal; it has no `distance_m`, no confidence and no `source`, and it is not
a `RangeEstimate`, so the arbiter's range fusion cannot consume it by accident.

There is no config key plumbed through to `publish_metric`. That is deliberate: a
YAML flag that turns a demoted channel back on is the wrong affordance. Restoring
metric depth here needs a calibrated camera **and** a better model, not tuning.

## Measured performance

Filled in by `scripts/build_engines.py` from each `trtexec` build and verify
log; see `MANIFEST.json` → `models.<name>.build_latency` and `.verify` for the
full min/max/mean/median set. GPU compute time excludes H2D/D2H copies and all
CPU pre/post-processing.

<!-- LATENCY TABLE START -->
| Model | Engine | Size (MB) | GPU compute median (ms) | GPU compute mean (ms) | Throughput (qps) | Build (s) | Verified |
|---|---|---:|---:|---:|---:|---:|:--:|
| `yolov5n` | `yolov5n.engine` | 5.7 | 7.23 | 7.24 | 137.71 | 1541.56 | yes |
| `yolox_nano` | `yolox_nano.engine` | 3.2 | 4.71 | 4.71 | 212.11 | 1062.16 | yes |
| `yolox_tiny` | `yolox_tiny.engine` | 12.7 | 6.43 | 6.43 | 155.11 | 1096.10 | yes |
| `yolop` | `yolop_640.engine` | 20.2 | 26.98 | 26.98 | 36.74 | 1077.78 | yes |
| `midas_v21_small` | `midas_v21_small_256.engine` | 33.9 | 6.33 | 6.33 | 157.52 | 340.71 | yes |
| `ufldv2_culane_res18` | `ufldv2_culane_res18.engine` | 413.4 | 16.53 | 20.47 | 47.56 | 614.84 | yes |

Measured on jetson-xavier-nx, TensorRT 8.5.2.2, FP16, batch 1. Regenerate with `python3 scripts/build_engines.py`.
<!-- LATENCY TABLE END -->

### Reading the numbers

At 20 Hz the whole pipeline has a 50 ms frame budget, and decode plus CPU
post-processing already costs roughly 25–35 ms on this board. The GPU side of a
detector-plus-lanes configuration is therefore the only combination that fits
every frame:

| Configuration | GPU compute | Fits 20 Hz? |
|---|---:|---|
| `yolox_nano` + `ufldv2_culane_res18` | 21.2 ms | yes, with CPU headroom |
| `yolox_tiny` + `ufldv2_culane_res18` | 23.0 ms | yes, tighter |
| `yolov5n` + `ufldv2_culane_res18` | 23.8 ms | yes, but AGPL |
| adding `midas_v21_small` | +6.3 ms | yes at a reduced rate |
| adding `yolop` | +27.0 ms | **no** — schedule it at 2–5 Hz |

`yolop` is the most expensive engine here by a factor of four and is the wrong
choice for the per-frame detection path. Run it as a low-rate free-space and
cross-check channel. `midas_v21_small` is cheap enough to run every second or
third frame as an independent range channel.

`ufldv2_culane_res18` is 413 MB of resident engine. That is the real cost of
this model on a 6.7 GiB board — check `MemAvailable` before adding another
large engine alongside it.

## Verification performed

Every engine in the table was checked three ways on this board:

1. `trtexec --loadEngine` round-trip (recorded in `MANIFEST.json` → `verify`),
   which proves the serialized plan deserialises under this exact TensorRT build.
2. Deserialisation and one inference through the project's own
   `adas.infer.trt_engine.TrtEngine`, which is the path the runtime actually
   uses. All six load, run, and return finite outputs.
3. For `ufldv2_culane_res18`, a functional check on frames 30/150/300 of
   `Ultra-Fast-Lane-Detection-v2/example.mp4`: the two row-head ego-lane
   markings decode to roughly 260–410 px and 1160–1200 px on a 1280 px frame,
   so they bracket the vehicle with a plausible ~750–910 px lane width at the
   bottom of the image. The lane model produces real geometry, not noise.

### The UFLD preprocessing defect is fixed

A previous revision of this file recorded that `src/adas/perception/ufld.py` fed the
engine **BGR, `/255`, no ImageNet normalisation** with a non-canonical crop, which
drove the existence head into saturation (ego-lane markings reported present at all
72 of 72 row anchors, including anchors above the horizon, where the canonical
transform reports 44–47). **That is no longer the case.** `ufld.py` now implements
the canonical transform documented above — RGB, stretch to 1600×533, `/255` then
ImageNet mean/std (folded into a single `px * scale - shift` pass), bottom 320 rows —
and binds its output tensors by name. Verified by inspection of `ufld.py` on
2026-09-13; the 100%-lane-detection rate in every run in `docs/JETSON.md` is with the
canonical transform.

## When a model cannot be built

`build_engines.py` records the exact failure under `MANIFEST.json` → `blocked`
(reason, `trtexec` tail, log path) and keeps going with the remaining models.
A consuming workstream that finds its model under `blocked` must ship its
honest stub (`is_mock=True` / `source=UNAVAILABLE`) rather than a plausible
constant.

## UFLDv2 ONNX provenance

There is no direct per-file download for the UFLDv2 CULane ResNet-18 ONNX. The
manifest points at PINTO_model_zoo entry 324's 2.9 GB `resources.tar.gz`;
`fetch_models.sh` streams it through `tar` and keeps only
`ufldv2_culane_res18_320x1600.onnx`, so nothing else touches the disk. That
stream took about 8 minutes at ~4 MB/s from this board.

The file is 825 MB because UFLDv2's head is one dense layer of roughly 186 M
parameters (2048 → `(200·72 + 100·81 + 2·72 + 2·81) · 4` lanes). Budget for the
FP16 engine being resident in GPU memory at run time.
