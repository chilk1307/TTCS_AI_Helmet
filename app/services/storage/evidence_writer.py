import cv2
import os
import numpy as np
from typing import Optional

from app.core import config
from app.core.logger import get_logger

logger = get_logger(__name__)


class EvidenceWriter:
    """Save evidence images for violations."""

    def __init__(self):
        os.makedirs(config.EVIDENCE_DIR, exist_ok=True)

    def save_vehicle(self, violation_id: str, image: np.ndarray) -> Optional[str]:
        path = os.path.join(config.EVIDENCE_DIR, f"{violation_id}_vehicle.jpg")
        return self._save(path, image)

    def save_plate(self, violation_id: str, image: np.ndarray) -> Optional[str]:
        path = os.path.join(config.EVIDENCE_DIR, f"{violation_id}_plate.jpg")
        return self._save(path, image)

    def save_frame(self, violation_id: str, image: np.ndarray) -> Optional[str]:
        path = os.path.join(config.EVIDENCE_DIR, f"{violation_id}_frame.jpg")
        return self._save(path, image)

    def _save(self, path: str, image: np.ndarray) -> Optional[str]:
        try:
            if image is not None and image.size > 0:
                cv2.imwrite(path, image)
                return path
        except Exception as e:
            logger.error(f"Error saving evidence: {e}")
        return None
