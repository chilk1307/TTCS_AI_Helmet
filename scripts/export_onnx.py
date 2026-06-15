"""Export all 3 YOLO models to ONNX format."""
import sys
sys.path.insert(0, "/teamspace/studios/this_studio")

from ultralytics import YOLO
from app.core import config

MODELS = [
    ("stage1", config.STAGE1_MODEL_PATH),
    ("stage2", config.STAGE2_MODEL_PATH),
    ("stage3", config.STAGE3_MODEL_PATH),
]

for name, path in MODELS:
    print(f"\n{'='*50}")
    print(f"Exporting {name}: {path}")
    print(f"{'='*50}")
    model = YOLO(path)
    model.export(format="onnx", imgsz=320, half=False, simplify=True, opset=17)
    print(f"{name} exported OK")

print("\nAll exports complete.")
