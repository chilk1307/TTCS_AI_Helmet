"""Stage 3: OCR license plate via YOLO char detection + rule-based postprocessor."""
import torch
import numpy as np
from typing import List, Optional
from ultralytics import YOLO

from app.core import config
from app.core.logger import get_logger
from app.services.inference.model_loader import load_yolo_model
from app.services.inference.plate_postprocessor import (
    postprocess_plate, enhance_plate_crop, PlateResult,
)

logger = get_logger(__name__)


class Stage3HelmetOCR:

    def __init__(self):
        self.model: YOLO = load_yolo_model(config.STAGE3_MODEL_PATH)
        self._warmup()

    def _warmup(self):
        logger.info("Stage3/OCR warmup...")
        dummy = np.zeros((160, 320, 3), dtype=np.uint8)
        with torch.no_grad():
            self.model.predict(dummy, imgsz=config.STAGE3_IMGSZ, verbose=False, half=True)
        logger.info("Stage3/OCR warmup done.")

    @torch.no_grad()
    def _detect_chars(self, crop: np.ndarray) -> List[dict]:
        """Run YOLO char detection on a single crop. Returns list of char dicts."""
        results = self.model.predict(
            crop, conf=config.STAGE3_CONF,
            imgsz=config.STAGE3_IMGSZ, verbose=False, half=True,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        detections = []
        for i in range(len(xyxy)):
            detections.append({
                "char": self.model.names[clses[i]],
                "conf": float(confs[i]),
                "bbox": [float(xyxy[i][0]), float(xyxy[i][1]),
                         float(xyxy[i][2]), float(xyxy[i][3])],
            })
        return detections

    def ocr_plate(self, plate_crop: np.ndarray) -> Optional[str]:
        """OCR a plate crop with multi-variant enhancement. Returns best plate or None."""
        if plate_crop is None or plate_crop.size == 0:
            return None

        try:
            variants = enhance_plate_crop(plate_crop)

            best_result: Optional[PlateResult] = None
            best_score = -999

            for variant in variants:
                detections = self._detect_chars(variant)
                if not detections:
                    continue

                result = postprocess_plate(detections)

                if result.score > best_score:
                    best_score = result.score
                    best_result = result

            if best_result is None:
                return None

            if best_result.corrections:
                logger.info(
                    f"OCR: raw={best_result.raw_text} → "
                    f"fixed={best_result.normalized_text} "
                    f"score={best_result.score:.0f} valid={best_result.valid} "
                    f"corrections={best_result.corrections}"
                )

            if best_result.valid:
                return best_result.normalized_text
            else:
                return best_result.raw_text or None

        except Exception as e:
            logger.error(f"OCR error: {e}")
            return None

    def ocr_plate_full(self, plate_crop: np.ndarray) -> Optional[PlateResult]:
        """OCR with full result (for debug/logging)."""
        if plate_crop is None or plate_crop.size == 0:
            return None

        try:
            variants = enhance_plate_crop(plate_crop)
            best_result: Optional[PlateResult] = None
            best_score = -999

            for variant in variants:
                detections = self._detect_chars(variant)
                if not detections:
                    continue
                result = postprocess_plate(detections)
                if result.score > best_score:
                    best_score = result.score
                    best_result = result

            return best_result
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return None

    def ocr_batch(self, plate_crops: List[np.ndarray]) -> List[Optional[str]]:
        return [self.ocr_plate(crop) for crop in plate_crops]
