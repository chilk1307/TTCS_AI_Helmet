"""PyTorch (.pt) inference via ultralytics."""
from __future__ import annotations

from typing import List

import numpy as np

from metrics import Box


class PtRunner:
    def __init__(self, model_path: str, conf: float, imgsz: int):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def predict(self, bgr: np.ndarray) -> List[Box]:
        results = self.model.predict(
            bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )
        boxes: List[Box] = []
        if not results:
            return boxes
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return boxes
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            boxes.append(
                Box(
                    class_id=int(cls_ids[i]),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(confs[i]),
                )
            )
        return boxes
