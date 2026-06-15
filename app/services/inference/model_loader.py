import torch
from ultralytics import YOLO
from app.core.logger import get_logger

logger = get_logger(__name__)


def load_yolo_model(model_path: str, device: str = "cuda") -> YOLO:
    """Load a YOLO model with GPU optimizations."""
    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)
    model.to(device)

    if device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    logger.info(f"Model loaded on {device}. Classes: {model.names}")
    return model
