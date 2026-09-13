# ADAS Core — CI / unit-test image.  x86_64 and arm64.  CPU ONLY.
#
# READ THIS BEFORE USING IT FOR ANYTHING ELSE
# -------------------------------------------
# This image contains NO CUDA, NO TensorRT and NO cuDNN, and its OpenCV is the
# pip headless build, not the L4T one with GStreamer. It therefore
# CANNOT run the real perception path: no .engine will load, no camera will open, and
# `--detector tensorrt` will fail. It exists for exactly one job — running the unit
# tests and the mock pipeline on a machine that is not a Jetson.
#
# The previous version of this file was `python:3.11-slim`, declared itself
# "production-ready containerization", and was recommended as the primary deployment
# in DEPLOYMENT.md. On the target board it could not have loaded a single model. The
# runtime image for the vehicle is deploy/Dockerfile.jetson, built on the L4T base
# that carries CUDA 11.4 and TensorRT 8.5.
#
# Python 3.8 to match the board (JetPack 5.1.6 ships 3.8.10), so a test that passes
# here has been run on the interpreter that will run in the vehicle.
#
# Build:  docker build -t adas-core:ci .
# Test:   docker run --rm adas-core:ci                       # the test suite
# Mock:   docker run --rm adas-core:ci python -m adas.cli --frames 20
FROM python:3.8-slim

LABEL org.opencontainers.image.title="adas-core (CI)"
LABEL org.opencontainers.image.description="ADAS reference pipeline: unit tests and mock pipeline. NO CUDA/TensorRT — cannot run real perception."
LABEL org.opencontainers.image.source="https://github.com/jakhon37/ADAS"
LABEL org.opencontainers.image.licenses="MIT"
LABEL com.adas.gpu="none"
LABEL com.adas.purpose="ci"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    ADAS_LOG_LEVEL=INFO \
    ADAS_LOG_FORMAT=text

WORKDIR /app

# numpy is a genuine runtime dependency (src/adas/perception, src/adas/tracking and
# src/adas/ros2 all import it at module scope); pytest and ruff are what this image
# is for.
#
# opencv-python-headless is a deliberate, bounded exception to the project's
# "never pip install opencv" rule. That rule protects the *board*, where a pip build
# would shadow the L4T OpenCV 4.5.4 that has GStreamer compiled in. Here there is no
# L4T OpenCV to shadow, and without cv2 a third of the suite (tests/test_ufld.py,
# tests/test_yolo_decode.py — every preprocessing-geometry test) is skipped with
# ModuleNotFoundError, which is a worse trade: those tests are exactly the ones that
# catch a letterbox or colour-order regression before it reaches a vehicle.
# `headless` because there is no display and it drags in no GTK/X11.
# Pinned below 4.6 to stay close to the board's 4.5.4 behaviour.
RUN pip install --no-cache-dir \
      "numpy>=1.19,<1.25" \
      "opencv-python-headless>=4.5.4,<4.6" \
      "pytest>=7.4,<8.4" \
      "pytest-cov>=4.0" \
      "ruff>=0.3"

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY config.example.json ./

# Fail the build if the package cannot even be imported on 3.8 — the cheapest
# possible guard against a 3.9+ syntax regression reaching the board.
RUN python -c "import adas, adas.io, adas.core.metrics, adas.core.logger, cv2, numpy; \
    print('adas', adas.__version__, 'cv2', cv2.__version__, 'np', numpy.__version__)"

RUN useradd --system --create-home --uid 1000 adas \
 && mkdir -p /data /var/lib/adas \
 && chown -R adas:adas /data /var/lib/adas /app
USER adas

# No HEALTHCHECK: the default command is a test runner, and a container whose job is
# to exit 0 has no steady state to probe. The runtime image
# (deploy/Dockerfile.jetson) has a real one against /healthz.

ENTRYPOINT []
CMD ["python", "-m", "pytest", "tests/", "-q"]
