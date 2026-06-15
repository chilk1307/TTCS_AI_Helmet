import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Auth ---
ADMIN_USERNAME = "1"
ADMIN_PASSWORD = "1"
SECRET_KEY = "traffic-monitor-demo-secret-key-2024"
TOKEN_EXPIRE_MINUTES = 480

# --- Model Paths ---
MODELS_DIR = os.path.join(BASE_DIR, "models")
STAGE1_MODEL_PATH = os.path.join(MODELS_DIR, "stage1", "detect_vehicle.pt")
STAGE2_MODEL_PATH = os.path.join(MODELS_DIR, "stage2", "mu_bien_so_stage2.pt")
STAGE3_MODEL_PATH = os.path.join(MODELS_DIR, "stage3", "ocr_plate.pt")

# --- Model Confidence Thresholds ---
STAGE1_CONF = 0.4
STAGE2_CONF = 0.72
STAGE3_CONF = 0.25

# --- Model Image Size (giảm stage1 cho nhanh, ROI đã crop nhỏ) ---
STAGE1_IMGSZ = 320
STAGE2_IMGSZ = 320
STAGE3_IMGSZ = 320

# --- ROI (Region of Interest) ---
ENABLE_ROI = True
ROI_Y_START_RATIO = 0.45
ROI_Y_END_RATIO = 1.0
ROI_X_START_RATIO = 0.0
ROI_X_END_RATIO = 1.0
FULL_FRAME_DETECT_EVERY_N_FRAMES = 60

# --- Detection Zone (vùng giữa 1/3 frame, dùng cho backend cpp/onnx) ---
# Chỉ giữ xe khi tâm bbox nằm trong [start, end] theo chiều cao frame
ENABLE_DETECTION_ZONE = True
DETECTION_ZONE_START_RATIO = 1.0 / 3.0
DETECTION_ZONE_END_RATIO = 2.0 / 3.0

# --- Detection Line (backend python, giữ nguyên) ---
ENABLE_DETECTION_LINE = True
DETECTION_LINE_RATIO = 0.55

# --- Pipeline Performance ---
TARGET_FPS = 30
VIOLATION_PROCESS_EVERY_N_FRAMES = 5
QUEUE_MAX_SIZE = 50

# --- Stream ---
STREAM_WIDTH = 960
STREAM_JPEG_QUALITY = 55

# --- Deduplication ---
DEDUP_SECONDS = 10

# --- Storage ---
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
UPLOADS_DIR = os.path.join(STORAGE_DIR, "uploads")
EVIDENCE_DIR = os.path.join(STORAGE_DIR, "evidence")
VIOLATIONS_DIR = os.path.join(STORAGE_DIR, "violations")
VIOLATIONS_FILE = os.path.join(VIOLATIONS_DIR, "violations.xlsx")

# --- Inference Backend ---
# "python" = ultralytics (PyTorch), "cpp" = ONNX Runtime (C++ engine)
INFERENCE_BACKEND = "python"

# --- Server ---
HOST = "0.0.0.0"
PORT = 8000

for d in [UPLOADS_DIR, EVIDENCE_DIR, VIOLATIONS_DIR]:
    os.makedirs(d, exist_ok=True)
