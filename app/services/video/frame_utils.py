import cv2
import numpy as np
from typing import List

from app.services.inference.types import BBox
from app.services.realtime.tracker import Track
from app.core import config

# Colors: BGR
COLOR_PENDING = (0, 165, 255)   # orange - chưa xử lý
COLOR_SAFE = (0, 255, 0)        # green - có mũ
COLOR_VIOLATION = (0, 0, 255)   # red - vi phạm
COLOR_LINE = (0, 200, 255)      # yellow-orange


def draw_bboxes_tracked(frame: np.ndarray, tracks: List[Track], show_roi: bool = False) -> np.ndarray:
    """Draw tracked bboxes with state-based coloring + detection line."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Vùng phát hiện giữa frame (1/3 → 2/3)
    if config.ENABLE_DETECTION_ZONE:
        y_top = int(h * config.DETECTION_ZONE_START_RATIO)
        y_bot = int(h * config.DETECTION_ZONE_END_RATIO)
        dash_len = 20
        for y_line in (y_top, y_bot):
            for x in range(0, w, dash_len * 2):
                cv2.line(annotated, (x, y_line), (min(x + dash_len, w), y_line), COLOR_LINE, 2, cv2.LINE_AA)
        cv2.putText(
            annotated, "VUNG PHAT HIEN",
            (10, y_top + max(24, (y_bot - y_top) // 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LINE, 1, cv2.LINE_AA,
        )
    elif config.ENABLE_DETECTION_LINE:
        line_y = int(h * config.DETECTION_LINE_RATIO)
        dash_len = 20
        for x in range(0, w, dash_len * 2):
            cv2.line(annotated, (x, line_y), (min(x + dash_len, w), line_y), COLOR_LINE, 2, cv2.LINE_AA)
        cv2.putText(
            annotated, "VUNG PHAT HIEN",
            (10, line_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LINE, 1, cv2.LINE_AA,
        )

    # ROI overlay
    if show_roi and config.ENABLE_ROI:
        y1r = int(h * config.ROI_Y_START_RATIO)
        y2r = int(h * config.ROI_Y_END_RATIO)
        x1r = int(w * config.ROI_X_START_RATIO)
        x2r = int(w * config.ROI_X_END_RATIO)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1r, y1r), (x2r, y2r), (0, 255, 255), 2)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

    for track in tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

        if track.state == "violation":
            color = COLOR_VIOLATION
            label = f"#{track.track_id} VI PHAM {bbox.confidence:.0%}"
        elif track.state == "safe":
            color = COLOR_SAFE
            label = f"#{track.track_id} OK {bbox.confidence:.0%}"
        else:
            color = COLOR_PENDING
            label = f"#{track.track_id} {bbox.confidence:.0%}"

        thickness = 3 if track.state == "violation" else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


def resize_for_stream(frame: np.ndarray) -> np.ndarray:
    """Resize frame for WebSocket streaming."""
    h, w = frame.shape[:2]
    target_w = config.STREAM_WIDTH
    if w <= target_w:
        return frame
    scale = target_w / w
    new_h = int(h * scale)
    return cv2.resize(frame, (target_w, new_h), interpolation=cv2.INTER_AREA)


def crop_bbox(frame: np.ndarray, bbox: BBox) -> np.ndarray:
    """Crop a bbox region from frame."""
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox.x1))
    y1 = max(0, int(bbox.y1))
    x2 = min(w, int(bbox.x2))
    y2 = min(h, int(bbox.y2))
    return frame[y1:y2, x1:x2]
