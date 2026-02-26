"""Simple multi-object tracker with nearest-center association."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from adas.core.exceptions import TrackingError, ValidationError
from adas.core.logger import setup_logger
from adas.core.models import BoundingBox, TrackedObject
from adas.core.validation import validate_bounding_box

logger = setup_logger(__name__)


@dataclass
class _TrackState:
    box: BoundingBox
    missed: int = 0


@dataclass
class MultiObjectTracker:
    """Multi-object tracker with data association and track management."""
    
    max_missed: int = 5
    association_threshold_px: float = 120.0
    focal_length_px: float = 35.0
    min_box_height_px: float = 1.0
    max_distance_m: float = 200.0
    
    _next_track_id: int = 1
    _tracks: dict[int, _TrackState] = field(default_factory=dict)

    def update(self, detections: list[BoundingBox]) -> list[TrackedObject]:
        """Update tracker with new detections.
        
        Args:
            detections: List of detected bounding boxes
            
        Returns:
            List of tracked objects with persistent IDs
            
        Raises:
            TrackingError: If tracking fails
        """
        try:
            # Validate all detections first
            valid_detections = []
            for det in detections:
                try:
                    validate_bounding_box(det)
                    valid_detections.append(det)
                except ValidationError as e:
                    logger.warning(f"Invalid detection skipped: {e}")
            
            assigned: set[int] = set()

            # Associate existing tracks to detections
            for track_id, state in list(self._tracks.items()):
                best_idx = None
                best_dist = float("inf")
                
                # Track center
                tx = (state.box.x1 + state.box.x2) / 2.0
                ty = (state.box.y1 + state.box.y2) / 2.0

                # Find nearest detection
                for idx, det in enumerate(valid_detections):
                    if idx in assigned:
                        continue
                    
                    # Detection center
                    dx = (det.x1 + det.x2) / 2.0 - tx
                    dy = (det.y1 + det.y2) / 2.0 - ty
                    
                    # Euclidean distance (optimized with math.hypot)
                    dist = math.hypot(dx, dy)
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx

                # Associate if within threshold
                if best_idx is not None and best_dist < self.association_threshold_px:
                    state.box = valid_detections[best_idx]
                    state.missed = 0
                    assigned.add(best_idx)
                    logger.debug(f"Track {track_id} associated with det {best_idx} (dist={best_dist:.1f})")
                else:
                    state.missed += 1
                    if state.missed > self.max_missed:
                        logger.debug(f"Track {track_id} deleted (missed {state.missed} frames)")
                        self._tracks.pop(track_id)

            # Create new tracks for unassigned detections
            for idx, det in enumerate(valid_detections):
                if idx in assigned:
                    continue
                track_id = self._next_track_id
                self._tracks[track_id] = _TrackState(box=det)
                self._next_track_id += 1
                logger.debug(f"New track {track_id} created")

            # Generate tracked objects output
            tracked: list[TrackedObject] = []
            for track_id, state in self._tracks.items():
                try:
                    distance_m = self._estimate_distance(state.box)
                    tracked.append(
                        TrackedObject(
                            track_id=track_id,
                            box=state.box,
                            velocity_mps=0.0,  # TODO: Compute from track history
                            distance_m=distance_m,
                        )
                    )
                except ValueError as e:
                    logger.warning(f"Track {track_id} distance estimation failed: {e}")
                    
            logger.debug(f"Tracking {len(tracked)} objects")
            return tracked
            
        except Exception as e:
            raise TrackingError(f"Tracker update failed: {e}") from e
    
    def _estimate_distance(self, box: BoundingBox) -> float:
        """Estimate distance from bounding box height.
        
        Args:
            box: Bounding box
            
        Returns:
            Estimated distance in meters
            
        Raises:
            ValueError: If estimation fails
        """
        box_height = box.y2 - box.y1
        
        # Ensure minimum height to avoid division issues
        box_height = max(self.min_box_height_px, box_height)
        
        distance_m = self.focal_length_px / box_height
        
        if not math.isfinite(distance_m):
            raise ValueError("Distance calculation resulted in non-finite value")
        
        # Clamp to maximum distance
        distance_m = min(distance_m, self.max_distance_m)
        
        return distance_m
    
    def reset(self) -> None:
        """Reset tracker state."""
        logger.info("Tracker reset")
        self._tracks.clear()
        self._next_track_id = 1
