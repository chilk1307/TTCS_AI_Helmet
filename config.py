"""
Tập trung toàn bộ cấu hình hệ thống vào 1 file duy nhất.
Chỉ cần sửa ở đây, không cần lục tìm khắp project.
"""

# ── Đường dẫn Model ──────────────────────────────────────
MODEL_STAGE1 = "models/stage1.pt"
MODEL_STAGE2 = "models/stage2.pt"
MODEL_STAGE3 = "models/stage3.pt"

# ── Confidence Threshold ─────────────────────────────────
STAGE1_CONF = 0.45          # Phát hiện xe máy
STAGE2_CONF = 0.45          # Phát hiện mũ / biển số (tổng quát)
NOHELMET_MIN_CONF = 0.60    # Riêng cho "nohelmet" — cao hơn để tránh nhận nhầm tóc/mũ lưỡi trai
STAGE3_CONF = 0.4           # OCR ký tự biển số

# ── Kích thước inference ──────────────────────────────────
STAGE1_IMGSZ = 640          # Resize ảnh đầu vào cho Stage 1
STAGE3_IMGSZ = 640          # Resize biển số cho Stage 3 (tăng từ 320)

# ── Kích thước tối thiểu biển số (pixel) ──────────────────
MIN_PLATE_WIDTH = 30
MIN_PLATE_HEIGHT = 15

# ── Video Processing ──────────────────────────────────────
SKIP_FRAMES = 2             # Chỉ chạy Stage 2+3 mỗi N frame (Stage 1 track luôn chạy)
CSV_UPDATE_INTERVAL = 15    # Cập nhật bảng CSV trên UI mỗi N frame

# ── Deferred Logging (Video) ──────────────────────────────
# Số frame không thấy ID → coi là đã rời khung hình → ghi biên bản
LOST_ID_THRESHOLD = 30

# ── OCR ──────────────────────────────────────────────────
OCR_MIN_CHARS = 5           # Tối thiểu ký tự để coi là biển số hợp lệ
OCR_UPSCALE_FACTOR = 2      # Phóng to biển số trước khi OCR

# ── Màu sắc (BGR cho OpenCV) ─────────────────────────────
COLORS = {
    'motorcyclist': (255, 0, 0),    # Xanh dương: Khung xe máy
    'helmet':       (0, 255, 0),    # Xanh lá: Đội mũ an toàn
    'nohelmet':     (0, 0, 255),    # Đỏ: Không đội mũ (Vi phạm)
    'licenseplate': (0, 255, 255),  # Vàng: Khung biển số
}
