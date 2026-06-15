import torch
import numpy as np
from typing import List
from ultralytics import YOLO

from app.core import config
from app.core.logger import get_logger
from app.services.inference.model_loader import load_yolo_model
from app.services.inference.types import BBox, Stage2Result

logger = get_logger(__name__)

CLASS_HELMET = 0
CLASS_NOHELMET = 1
CLASS_PLATE = 2


class Stage2PartDetector:
    """Detects helmet/no-helmet/license plate on cropped motorcyclist."""

    def __init__(self):
        self.model: YOLO = load_yolo_model(config.STAGE2_MODEL_PATH)
        self._warmup()

    def _warmup(self):
        logger.info("Stage2 warmup...")
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        with torch.no_grad():
            self.model.predict(dummy, imgsz=config.STAGE2_IMGSZ, verbose=False, half=True)
        logger.info("Stage2 warmup done.")

    @torch.no_grad()
    def detect_batch(self, crops: List[np.ndarray], vehicle_bboxes: List[BBox]) -> List[Stage2Result]:
        if not crops:
            return []

        results_list = self.model.predict(
            crops,
            conf=config.STAGE2_CONF,
            imgsz=config.STAGE2_IMGSZ,
            verbose=False,
            half=True,
        )

        stage2_results = []
        for i, results in enumerate(results_list):
            helmet_bboxes = []
            nohelmet_bboxes = []
            plate_bboxes = []

            if results and results.boxes is not None and len(results.boxes) > 0:
                xyxy = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                clses = results.boxes.cls.cpu().numpy().astype(int)

                for j in range(len(xyxy)):
                    bbox = BBox(
                        x1=float(xyxy[j][0]), y1=float(xyxy[j][1]),
                        x2=float(xyxy[j][2]), y2=float(xyxy[j][3]),
                        confidence=float(confs[j]),
                        class_id=int(clses[j]),
                        class_name=self.model.names[int(clses[j])],
                    )
                    if clses[j] == CLASS_HELMET:
                        helmet_bboxes.append(bbox)
                    elif clses[j] == CLASS_NOHELMET:
                        nohelmet_bboxes.append(bbox)
                    elif clses[j] == CLASS_PLATE:
                        plate_bboxes.append(bbox)

            stage2_results.append(Stage2Result(
                vehicle_bbox=vehicle_bboxes[i],
                helmet_bboxes=helmet_bboxes,
                nohelmet_bboxes=nohelmet_bboxes,
                plate_bboxes=plate_bboxes,
            ))

        return stage2_results
