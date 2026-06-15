"""Simple IoU-based tracker for assigning persistent IDs to detected objects."""
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from app.services.inference.types import BBox


@dataclass
class Track:
    track_id: int
    bbox: BBox
    last_seen: float
    frames_seen: int = 1
    state: str = "pending"  # pending | processing | violation | safe
    processed: bool = False


class IoUTracker:
    """Assigns stable track IDs to detections across frames using IoU matching."""

    def __init__(self, iou_threshold: float = 0.3, max_age_seconds: float = 2.0):
        self.iou_threshold = iou_threshold
        self.max_age = max_age_seconds
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, bboxes: List[BBox]) -> List[Track]:
        """Match new detections to existing tracks, return updated tracks."""
        now = time.time()
        self._remove_stale(now)

        bboxes = self._nms_bboxes(bboxes)

        if not bboxes:
            return []

        matched_tracks = []
        unmatched_bboxes = list(range(len(bboxes)))

        # Match existing tracks to new detections by IoU
        for tid, track in list(self.tracks.items()):
            best_iou = 0.0
            best_idx = -1

            for i in unmatched_bboxes:
                iou = self._compute_iou(track.bbox, bboxes[i])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= self.iou_threshold and best_idx >= 0:
                # Update existing track
                track.bbox = bboxes[best_idx]
                track.last_seen = now
                track.frames_seen += 1
                matched_tracks.append(track)
                unmatched_bboxes.remove(best_idx)

        # Create new tracks for unmatched detections
        for i in unmatched_bboxes:
            track = Track(
                track_id=self._next_id,
                bbox=bboxes[i],
                last_seen=now,
            )
            self.tracks[self._next_id] = track
            matched_tracks.append(track)
            self._next_id += 1

        return matched_tracks

    def mark_processed(self, track_id: int, state: str = "safe"):
        """Mark a track as processed (no need to run stage2/3 again)."""
        if track_id in self.tracks:
            self.tracks[track_id].processed = True
            self.tracks[track_id].state = state

    def get_unprocessed(self, tracks: List[Track]) -> List[Track]:
        """Get tracks that haven't been processed through stage2/3 yet."""
        return [t for t in tracks if not t.processed and t.state == "pending"]

    def _remove_stale(self, now: float):
        expired = [tid for tid, t in self.tracks.items() if now - t.last_seen > self.max_age]
        for tid in expired:
            del self.tracks[tid]

    @staticmethod
    def _nms_bboxes(bboxes: List[BBox], iou_threshold: float = 0.5) -> List[BBox]:
        """Remove duplicate detections in the same frame (keep highest confidence)."""
        if len(bboxes) <= 1:
            return bboxes

        sorted_bboxes = sorted(bboxes, key=lambda b: b.confidence, reverse=True)
        keep: List[BBox] = []
        for b in sorted_bboxes:
            if all(IoUTracker._compute_iou(b, k) < iou_threshold for k in keep):
                keep.append(b)
        return keep

    @staticmethod
    def _compute_iou(a: BBox, b: BBox) -> float:
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
