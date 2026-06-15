import time
from collections import deque


class FPSCounter:
    """Track FPS using a sliding window."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps: deque = deque(maxlen=window_size)

    def tick(self):
        self.timestamps.append(time.time())

    @property
    def fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed
