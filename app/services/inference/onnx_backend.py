"""ONNX Runtime inference backend (C++ engine under the hood).

Replaces ultralytics predict() with raw ONNX Runtime for maximum speed.
No Python overhead from ultralytics preprocessing/postprocessing.
"""
import os
import sys
import glob
import ctypes

# Preload CUDA 12 shared libs so onnxruntime-gpu CUDAExecutionProvider works
def _preload_cuda_libs():
    lib_dirs = glob.glob(os.path.join(sys.prefix, "lib/python*/site-packages/nvidia/*/lib"))
    targets = ["libcublas.so", "libcublasLt.so", "libcudnn.so", "libcufft.so", "libcurand.so"]
    for d in sorted(lib_dirs):
        if not os.path.isdir(d):
            continue
        os.environ["LD_LIBRARY_PATH"] = d + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        for t in targets:
            p = os.path.join(d, t)
            if os.path.exists(p):
                try:
                    ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass

_preload_cuda_libs()

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Optional, Tuple

from app.core import config
from app.core.logger import get_logger

logger = get_logger(__name__)

# ONNX model paths
STAGE1_ONNX = config.STAGE1_MODEL_PATH.replace(".pt", ".onnx")
STAGE2_ONNX = config.STAGE2_MODEL_PATH.replace(".pt", ".onnx")
STAGE3_ONNX = config.STAGE3_MODEL_PATH.replace(".pt", ".onnx")

# Class name maps (from model inspection)
STAGE1_NAMES = {0: "motorcyclist"}
STAGE2_NAMES = {0: "helmet", 1: "nohelmet", 2: "licenseplate"}
STAGE3_NAMES = {
    0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8',
    8: '9', 9: 'A', 10: 'B', 11: 'C', 12: 'D', 13: 'E', 14: 'F',
    15: 'G', 16: 'H', 17: 'K', 18: 'L', 19: 'M', 20: 'N', 21: 'P',
    22: 'S', 23: 'T', 24: 'U', 25: 'V', 26: 'X', 27: 'Y', 28: 'Z', 29: '0',
}


def _create_session(model_path: str) -> ort.InferenceSession:
    """Create ONNX Runtime session with CUDA if available."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    providers = []
    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
        }))
    providers.append("CPUExecutionProvider")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4

    session = ort.InferenceSession(model_path, opts, providers=providers)
    active = session.get_providers()
    logger.info(f"ONNX loaded: {os.path.basename(model_path)} | providers={active}")
    return session


def _preprocess(image: np.ndarray, imgsz: int = 320) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Preprocess image for YOLO ONNX: letterbox + normalize.
    Returns (blob, scale, (pad_w, pad_h)).
    """
    h, w = image.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_w = (imgsz - new_w) // 2
    pad_h = (imgsz - new_h) // 2
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # HWC → CHW
    blob = np.expand_dims(blob, 0)  # Add batch dim
    blob = np.ascontiguousarray(blob)

    return blob, scale, (pad_w, pad_h)


def _postprocess_yolov8_end2end(output: np.ndarray, scale: float, pad: Tuple[int, int],
                                 conf_threshold: float) -> List[Tuple[np.ndarray, float, int]]:
    """Post-process end2end YOLOv8 output: shape (1, N, 6) where cols = [x1,y1,x2,y2,conf,cls].
    Returns list of (bbox_xyxy, conf, class_id).
    """
    if output.ndim == 3:
        output = output[0]  # Remove batch dim → (N, 6)

    results = []
    pad_w, pad_h = pad

    for det in output:
        x1, y1, x2, y2, conf, cls_id = det[:6]
        if conf < conf_threshold:
            continue

        # Remove padding and scale back to original image coords
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        results.append((np.array([x1, y1, x2, y2]), float(conf), int(cls_id)))

    return results


class OnnxDetector:
    """Generic ONNX-based YOLO detector."""

    def __init__(self, model_path: str, names: dict, imgsz: int = 320):
        self.session = _create_session(model_path)
        self.names = names
        self.imgsz = imgsz
        self.input_name = self.session.get_inputs()[0].name

        # Warmup
        logger.info(f"ONNX warmup: {os.path.basename(model_path)}...")
        dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})
        logger.info(f"ONNX warmup done: {os.path.basename(model_path)}")

    def detect(self, image: np.ndarray, conf: float = 0.4) -> List[Tuple[np.ndarray, float, int]]:
        """Run detection on a single image. Returns list of (bbox, conf, class_id)."""
        blob, scale, pad = _preprocess(image, self.imgsz)
        outputs = self.session.run(None, {self.input_name: blob})
        return _postprocess_yolov8_end2end(outputs[0], scale, pad, conf)

    def detect_batch(self, images: List[np.ndarray], conf: float = 0.4) -> List[List[Tuple[np.ndarray, float, int]]]:
        """Run detection on multiple images. Currently sequential (batch=1)."""
        return [self.detect(img, conf) for img in images]


# ---------------------------------------------------------------------------
# Stage-specific wrappers (drop-in replacements for ultralytics versions)
# ---------------------------------------------------------------------------
from app.services.inference.types import BBox, Stage1Result, Stage2Result


class Stage1OnnxDetector:
    """ONNX replacement for Stage1MotorbikeDetector."""

    def __init__(self):
        self.detector = OnnxDetector(STAGE1_ONNX, STAGE1_NAMES, config.STAGE1_IMGSZ)
        self.frame_count = 0

    def detect(self, frame: np.ndarray, frame_index: int) -> Stage1Result:
        self.frame_count += 1
        use_full = (
            not config.ENABLE_ROI
            or self.frame_count % config.FULL_FRAME_DETECT_EVERY_N_FRAMES == 0
        )

        if use_full:
            roi = frame
            off_x, off_y = 0, 0
        else:
            h, w = frame.shape[:2]
            y1 = int(h * config.ROI_Y_START_RATIO)
            y2 = int(h * config.ROI_Y_END_RATIO)
            x1 = int(w * config.ROI_X_START_RATIO)
            x2 = int(w * config.ROI_X_END_RATIO)
            roi = frame[y1:y2, x1:x2]
            off_x, off_y = x1, y1

        detections = self.detector.detect(roi, config.STAGE1_CONF)

        frame_h = frame.shape[0]
        zone_y1 = frame_h * config.DETECTION_ZONE_START_RATIO
        zone_y2 = frame_h * config.DETECTION_ZONE_END_RATIO

        bboxes = []
        for bbox, conf, cls_id in detections:
            ax1 = float(bbox[0] + off_x)
            ay1 = float(bbox[1] + off_y)
            ax2 = float(bbox[2] + off_x)
            ay2 = float(bbox[3] + off_y)

            if config.ENABLE_DETECTION_ZONE:
                cy = (ay1 + ay2) / 2.0
                if cy < zone_y1 or cy > zone_y2:
                    continue

            bboxes.append(BBox(
                x1=ax1, y1=ay1, x2=ax2, y2=ay2,
                confidence=conf, class_id=0, class_name="motorcyclist",
            ))

        return Stage1Result(frame=frame, frame_index=frame_index, bboxes=bboxes, roi_offset=(off_x, off_y))


class Stage2OnnxDetector:
    """ONNX replacement for Stage2PartDetector."""

    def __init__(self):
        self.detector = OnnxDetector(STAGE2_ONNX, STAGE2_NAMES, config.STAGE2_IMGSZ)

    def detect_batch(self, crops: List[np.ndarray], vehicle_bboxes: List[BBox]) -> List[Stage2Result]:
        if not crops:
            return []

        results = []
        for i, crop in enumerate(crops):
            dets = self.detector.detect(crop, config.STAGE2_CONF)
            helmet, nohelmet, plate = [], [], []

            for bbox, conf, cls_id in dets:
                b = BBox(
                    x1=float(bbox[0]), y1=float(bbox[1]),
                    x2=float(bbox[2]), y2=float(bbox[3]),
                    confidence=conf, class_id=cls_id,
                    class_name=STAGE2_NAMES.get(cls_id, "unknown"),
                )
                if cls_id == 0:
                    helmet.append(b)
                elif cls_id == 1:
                    nohelmet.append(b)
                elif cls_id == 2:
                    plate.append(b)

            results.append(Stage2Result(
                vehicle_bbox=vehicle_bboxes[i],
                helmet_bboxes=helmet,
                nohelmet_bboxes=nohelmet,
                plate_bboxes=plate,
            ))
        return results


class Stage3OnnxOCR:
    """ONNX replacement for Stage3HelmetOCR."""

    def __init__(self):
        self.detector = OnnxDetector(STAGE3_ONNX, STAGE3_NAMES, config.STAGE3_IMGSZ)

    def _detect_chars(self, crop: np.ndarray) -> List[dict]:
        dets = self.detector.detect(crop, config.STAGE3_CONF)
        return [
            {
                "char": STAGE3_NAMES.get(cls_id, "?"),
                "conf": conf,
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            }
            for bbox, conf, cls_id in dets
        ]

    def ocr_plate(self, plate_crop: np.ndarray) -> Optional[str]:
        from app.services.inference.plate_postprocessor import postprocess_plate, enhance_plate_crop, PlateResult

        if plate_crop is None or plate_crop.size == 0:
            return None

        try:
            variants = enhance_plate_crop(plate_crop)
            best: Optional[PlateResult] = None
            best_score = -999

            for v in variants:
                dets = self._detect_chars(v)
                if not dets:
                    continue
                result = postprocess_plate(dets)
                if result.score > best_score:
                    best_score = result.score
                    best = result

            if best is None:
                return None

            if best.corrections:
                logger.info(f"ONNX-OCR: {best.raw_text}→{best.normalized_text} score={best.score:.0f}")

            return best.normalized_text if best.valid else (best.raw_text or None)

        except Exception as e:
            logger.error(f"ONNX-OCR error: {e}")
            return None

    def ocr_batch(self, crops: List[np.ndarray]) -> List[Optional[str]]:
        return [self.ocr_plate(c) for c in crops]
