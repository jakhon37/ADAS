"""Runtime orchestration: frame sources, the frame loop and the pipeline.

The three pieces are deliberately separable:

* :mod:`adas.runtime.capture` owns *where frames and ego speed come from*.
* :mod:`adas.runtime.pipeline` owns *what happens to one frame*, and ends with
  the safety arbiter, whose command is the only one that may be actuated.
* :mod:`adas.runtime.runner` owns *timing and failure policy* -- measured dt,
  pacing, reconnection, the fail-safe on a failed frame, cooperative shutdown.
"""

from adas.runtime.capture import (
    CapturedFrame,
    EgoSpeedSource,
    FrameSource,
    open_ego_source,
    open_source,
    parse_source_arg,
)
from adas.runtime.pipeline import ADASPipeline
from adas.runtime.runner import PipelineRunner, RunSummary, RunnerHooks, synthetic_frame

__all__ = [
    "ADASPipeline",
    "CapturedFrame",
    "EgoSpeedSource",
    "FrameSource",
    "PipelineRunner",
    "RunSummary",
    "RunnerHooks",
    "open_ego_source",
    "open_source",
    "parse_source_arg",
    "synthetic_frame",
]
