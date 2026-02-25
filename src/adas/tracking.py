"""Simple multi-object tracker with nearest-center association."""

from __future__ import annotations

from dataclasses import dataclass, field

from adas.types import BoundingBox, TrackedObject


@dataclass
class _TrackState:
    box: BoundingBox
    missed: int = 0


@dataclass
class MultiObjectTracker:
    max_missed: int = 5
    _next_track_id: int = 1
    _tracks: dict[int, _TrackState] = field(default_factory=dict)

    def update(self, detections: list[BoundingBox]) -> list[TrackedObject]:
        assigned: set[int] = set()

        for track_id, state in list(self._tracks.items()):
            best_idx = None
            best_dist = float("inf")
            tx = (state.box.x1 + state.box.x2) / 2.0
            ty = (state.box.y1 + state.box.y2) / 2.0

            for idx, det in enumerate(detections):
                if idx in assigned:
                    continue
                dx = (det.x1 + det.x2) / 2.0 - tx
                dy = (det.y1 + det.y2) / 2.0 - ty
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None and best_dist < 120:
                state.box = detections[best_idx]
                state.missed = 0
                assigned.add(best_idx)
            else:
                state.missed += 1
                if state.missed > self.max_missed:
                    self._tracks.pop(track_id)

        for idx, det in enumerate(detections):
            if idx in assigned:
                continue
            self._tracks[self._next_track_id] = _TrackState(box=det)
            self._next_track_id += 1

        tracked: list[TrackedObject] = []
        for track_id, state in self._tracks.items():
            box_height = max(1.0, state.box.y2 - state.box.y1)
            distance_m = 35.0 / box_height
            tracked.append(
                TrackedObject(
                    track_id=track_id,
                    box=state.box,
                    velocity_mps=0.0,
                    distance_m=distance_m,
                )
            )
        return tracked
