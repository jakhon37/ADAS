# Local inference weights

Binaries in this directory are **not** committed (see `.gitignore`).

| File | How to get it |
|---|---|
| `yolov5n.onnx` | `python3 scripts/build_yolo_engine.py` (idle Jetson) |
| `yolov5n.engine` | Built by the same script via `trtexec` |
| UFLD engine | Not produced yet. Needed for `--lane ufld` |

On this Xavier NX (JetPack 5.1.7 / TensorRT 8.5) the YOLOv5n engine was verified: input `images` 1×3×640×640, output `output0` 1×25200×85, ~7.3 ms GPU.

Rebuild only when nothing else owns the GPU (no `dms.app`).
