"""Debugging tools: record a run, replay it, compare decisions.

A recording captures the perception output, the plan, the ARBITRATED command and
the arbitration result for every frame.  :func:`adas.tools.replayer.replay_with_pipeline`
pushes that recorded perception back through a pipeline on any machine -- GPU or
not -- which is the cheap way to tell whether a change to the tracker, the
planner or the arbiter alters behaviour on real data.
"""

from adas.tools.recorder import (
    DataRecorder,
    RECORD_VERSION,
    RecordingConfig,
    RecordingPipeline,
)
from adas.tools.replayer import (
    DataReplayer,
    RecordedDetector,
    RecordedLaneEstimator,
    ReplayConfig,
    build_replay_backends,
    replay_with_pipeline,
)

__all__ = [
    "DataRecorder",
    "DataReplayer",
    "RECORD_VERSION",
    "RecordedDetector",
    "RecordedLaneEstimator",
    "RecordingConfig",
    "RecordingPipeline",
    "ReplayConfig",
    "build_replay_backends",
    "replay_with_pipeline",
]
