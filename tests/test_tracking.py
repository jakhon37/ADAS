"""Tests for multi-object tracker."""


from adas.core.models import BoundingBox
from adas.tracking import MultiObjectTracker


def test_tracker_creates_new_track():
    """Test that tracker creates new track for detection."""
    tracker = MultiObjectTracker()
    
    detections = [
        BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.9, label="car")
    ]
    
    tracked = tracker.update(detections)
    
    assert len(tracked) == 1
    assert tracked[0].track_id == 1


def test_tracker_maintains_track_id():
    """Test that tracker maintains consistent IDs across frames."""
    tracker = MultiObjectTracker()
    
    # Frame 1
    detections1 = [
        BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.9, label="car")
    ]
    tracked1 = tracker.update(detections1)
    track_id = tracked1[0].track_id
    
    # Frame 2 - similar position
    detections2 = [
        BoundingBox(x1=105, y1=105, x2=205, y2=205, confidence=0.9, label="car")
    ]
    tracked2 = tracker.update(detections2)
    
    # Should maintain same ID
    assert len(tracked2) == 1
    assert tracked2[0].track_id == track_id


def test_tracker_creates_multiple_tracks():
    """Test tracking multiple objects."""
    tracker = MultiObjectTracker()
    
    detections = [
        BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.9, label="car"),
        BoundingBox(x1=400, y1=100, x2=500, y2=200, confidence=0.9, label="car"),
    ]
    
    tracked = tracker.update(detections)
    
    assert len(tracked) == 2
    assert tracked[0].track_id != tracked[1].track_id


def test_tracker_deletes_lost_tracks():
    """Test that lost tracks are deleted after max_missed frames."""
    tracker = MultiObjectTracker(max_missed=2)
    
    # Frame 1 - create track
    detections1 = [
        BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.9, label="car")
    ]
    tracker.update(detections1)
    
    # Frames 2-4 - no detections (exceeds max_missed)
    for _ in range(4):
        tracker.update([])
    
    # Track should be gone
    assert len(tracker._tracks) == 0


def test_tracker_distance_estimation():
    """Pinhole range: distance = (object_height * focal) / box_height."""
    tracker = MultiObjectTracker(focal_length_px=900.0, object_height_m=1.5)

    detections = [
        BoundingBox(x1=100, y1=100, x2=200, y2=235, confidence=0.9, label="car")  # height=135
    ]
    tracked = tracker.update(detections)

    # 1.5 * 900 / 135 = 10.0 m
    assert abs(tracked[0].distance_m - 10.0) < 0.05


def test_tracker_velocity_from_range_rate():
    tracker = MultiObjectTracker(focal_length_px=900.0, object_height_m=1.5)
    far = BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.9, label="car")  # h=100 → 13.5m
    close = BoundingBox(x1=90, y1=80, x2=210, y2=230, confidence=0.9, label="car")  # h=150 → 9.0m
    tracker.update([far], dt_s=0.1)
    tracked = tracker.update([close], dt_s=0.1)
    # closing: (13.5 - 9.0) / 0.1 = 45 m/s
    assert tracked[0].velocity_mps > 0.0


def test_tracker_reset():
    """Test tracker reset."""
    tracker = MultiObjectTracker()
    
    detections = [
        BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.9, label="car")
    ]
    tracker.update(detections)
    
    tracker.reset()
    
    assert len(tracker._tracks) == 0
    assert tracker._next_track_id == 1
