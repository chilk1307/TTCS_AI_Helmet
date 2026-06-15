"""Paths and thresholds for PT vs ONNX (C++) benchmark."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

# External models (read-only)
STAGE1_PT = os.path.join(PROJECT_ROOT, "models", "stage1", "detect_vehicle.pt")
STAGE1_ONNX = os.path.join(PROJECT_ROOT, "models", "stage1", "detect_vehicle.onnx")
STAGE2_PT = os.path.join(PROJECT_ROOT, "models", "stage2", "mu_bien_so_stage2.pt")
STAGE2_ONNX = os.path.join(PROJECT_ROOT, "models", "stage2", "mu_bien_so_stage2.onnx")
STAGE3_ONNX = os.path.join(PROJECT_ROOT, "models", "stage3", "ocr_plate.onnx")
CPP_LIB = os.path.join(PROJECT_ROOT, "cpp_inference", "build", "libtraffic_inference.so")

# Test sets inside compare-test
YOLO_1 = {
    "name": "yolo_1 (stage1 motorcyclist)",
    "images_dir": os.path.join(ROOT, "stage1", "test", "images"),
    "labels_dir": os.path.join(ROOT, "stage1", "test", "labels"),
    "pt_model": STAGE1_PT,
    "onnx_model": STAGE1_ONNX,
    "conf": 0.4,
    "classes": {0: "motorcyclist"},
}

YOLO_2 = {
    "name": "yolo_2 (stage2 helmet/plate)",
    "images_dir": os.path.join(ROOT, "stage2", "test", "images"),
    "labels_dir": os.path.join(ROOT, "stage2", "test", "labels"),
    "pt_model": STAGE2_PT,
    "onnx_model": STAGE2_ONNX,
    "conf": 0.35,
    "classes": {0: "helmet", 1: "nohelmet", 2: "licenseplate"},
}

IMGSZ = 320
IOU_THRESH = 0.5
WARMUP_IMAGES = 5
RESULTS_DIR = os.path.join(ROOT, "results")
