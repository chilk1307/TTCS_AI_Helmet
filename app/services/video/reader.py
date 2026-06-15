import cv2
import numpy as np
from typing import Generator, Tuple, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)


class VideoReader:
    """Read video frames from file."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 25.0
        self.total_frames = 0
        self.width = 0
        self.height = 0

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            logger.error(f"Cannot open video: {self.video_path}")
            return False

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"Video opened: {self.video_path} | "
            f"{self.width}x{self.height} | {self.fps:.1f}fps | "
            f"{self.total_frames} frames"
        )
        return True

    def read_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield (frame_index, frame) tuples."""
        if self.cap is None:
            return

        idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
