import torch
import numpy as np
from typing import List
from ultralytics import YOLO

from app.core import config
from app.core.logger import get_logger
from app.services.inference.model_loader import load_yolo_model
from app.services.inference.types import BBox, Stage1Result

logger = get_logger(__name__)


class Stage1MotorbikeDetector:
    """Detects motorcyclists in frames with ROI + detection line support."""

    def __init__(self):
        self.model: YOLO = load_yolo_model(config.STAGE1_MODEL_PATH)
        self.frame_count = 0
        self._warmup()

    def _warmup(self):
        """Warmup GPU with a dummy inference to avoid first-call latency."""
        logger.info("Stage1 warmup...")
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        with torch.no_grad():
            self.model.predict(dummy, imgsz=config.STAGE1_IMGSZ, verbose=False, half=True)
        logger.info("Stage1 warmup done.")

    @torch.no_grad()
    def detect(self, frame: np.ndarray, frame_index: int) -> Stage1Result:
        self.frame_count += 1
        use_full_frame = (
            not config.ENABLE_ROI
            or self.frame_count % config.FULL_FRAME_DETECT_EVERY_N_FRAMES == 0
        )

        if use_full_frame:
            roi = frame
            offset_x, offset_y = 0, 0
        else:
            h, w = frame.shape[:2]
            y1 = int(h * config.ROI_Y_START_RATIO)
            y2 = int(h * config.ROI_Y_END_RATIO)
            x1 = int(w * config.ROI_X_START_RATIO)
            x2 = int(w * config.ROI_X_END_RATIO)
            roi = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1

        results = self.model.predict(
            roi,
            conf=config.STAGE1_CONF,
            imgsz=config.STAGE1_IMGSZ,
            verbose=False,
            half=True,
        )

        frame_h = frame.shape[0]
        detection_line_y = frame_h * config.DETECTION_LINE_RATIO if config.ENABLE_DETECTION_LINE else 0

        bboxes: List[BBox] = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes_data = results[0].boxes
            if len(boxes_data) > 0:
                xyxy = boxes_data.xyxy.cpu().numpy()
                confs = boxes_data.conf.cpu().numpy()
                for i in range(len(xyxy)):
                    abs_x1 = float(xyxy[i][0] + offset_x)
                    abs_y1 = float(xyxy[i][1] + offset_y)
                    abs_x2 = float(xyxy[i][2] + offset_x)
                    abs_y2 = float(xyxy[i][3] + offset_y)

                    if config.ENABLE_DETECTION_LINE and abs_y2 < detection_line_y:
                        continue

                    bboxes.append(BBox(
                        x1=abs_x1, y1=abs_y1, x2=abs_x2, y2=abs_y2,
                        confidence=float(confs[i]),
                        class_id=0,
                        class_name="motorcyclist",
                    ))

        return Stage1Result(
            frame=frame,
            frame_index=frame_index,
            bboxes=bboxes,
            roi_offset=(offset_x, offset_y),
        )
