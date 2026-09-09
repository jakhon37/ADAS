"""Frame sources: synthetic, video file, OpenCV camera, Jetson GStreamer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

from adas.core.exceptions import SensorError
from adas.core.logger import setup_logger

logger = setup_logger(__name__)

JETSON_CSI_PIPELINE = (
    "nvarguscamerasrc sensor-id={sensor} ! "
    "video/x-raw(memory:NVMM), width={width}, height={height}, "
    "framerate=30/1, format=NV12 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
)


@dataclass
class CapturedFrame:
    image: object
    width: int
    height: int


class FrameSource:
    def read(self) -> Optional[CapturedFrame]:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def frames(self) -> Iterator[CapturedFrame]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SyntheticSource(FrameSource):
    def __init__(self, width: int = 1280, height: int = 720, as_image: bool = False) -> None:
        self.width = width
        self.height = height
        self.as_image = as_image

    def read(self) -> Optional[CapturedFrame]:
        if self.as_image:
            import numpy as np

            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            image = {"width": self.width, "height": self.height}
        return CapturedFrame(image=image, width=self.width, height=self.height)


class OpenCVSource(FrameSource):
    def __init__(self, uri, width: Optional[int] = None, height: Optional[int] = None) -> None:
        import cv2

        self._cv2 = cv2
        self.cap = cv2.VideoCapture(uri)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise SensorError("failed to open video source: %s" % uri)
        logger.info("Opened video source %s", uri)

    def read(self) -> Optional[CapturedFrame]:
        ok, image = self.cap.read()
        if not ok or image is None:
            return None
        h, w = image.shape[:2]
        return CapturedFrame(image=image, width=int(w), height=int(h))

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def open_source(
    source_type: str,
    uri: str = "",
    width: int = 1280,
    height: int = 720,
    as_image: bool = False,
) -> FrameSource:
    """Open a frame source from config / CLI."""
    kind = (source_type or "synthetic").lower()
    if kind == "synthetic":
        return SyntheticSource(width=width, height=height, as_image=as_image)
    if kind == "video":
        if not uri:
            raise SensorError("video source requires a file path")
        return OpenCVSource(uri)
    if kind == "camera":
        if uri.startswith("nvargus") or uri.startswith("nvv4l2") or "!" in uri:
            return OpenCVSource(uri)
        if uri in ("", "csi", "csi:0"):
            pipeline = JETSON_CSI_PIPELINE.format(sensor=0, width=width, height=height)
            logger.info("Using Jetson CSI pipeline")
            return OpenCVSource(pipeline)
        if uri.startswith("csi:"):
            sensor = int(uri.split(":", 1)[1] or "0")
            pipeline = JETSON_CSI_PIPELINE.format(sensor=sensor, width=width, height=height)
            return OpenCVSource(pipeline)
        try:
            index = int(uri)
        except ValueError:
            index = uri
        return OpenCVSource(index, width=width, height=height)
    raise SensorError("unknown source type: %s" % source_type)


def parse_source_arg(value: Optional[str]) -> Tuple[str, str]:
    """Parse CLI --source into (type, uri)."""
    if not value or value == "synthetic":
        return "synthetic", ""
    if value in ("camera", "csi"):
        return "camera", "csi:0"
    if value.startswith("camera:"):
        return "camera", value.split(":", 1)[1]
    return "video", value
