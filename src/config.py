"""
Tập trung toàn bộ cấu hình hệ thống vào 1 file duy nhất.
Chỉ cần sửa ở đây, không cần lục tìm khắp project.
"""

# ── Đường dẫn Model ──────────────────────────────────────
MODEL_STAGE1 = "data/models/stage1.pt"
MODEL_STAGE2 = "data/models/stage2.pt"
MODEL_STAGE3 = "data/models/stage3.pt"

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

# ── Kích thước crop tối thiểu (pixel) ────────────────────
# Crop xe máy < MIN_CROP_SIZE pixel (cạnh nhỏ nhất) → bỏ qua, không chạy Stage 2
MIN_CROP_SIZE = 50

# ── Helmet Detection Logic ───────────────────────────────
# Chỉ xét detection helmet/nohelmet nằm ở phần TRÊN crop (vùng đầu)
# Ví dụ: 0.55 = chỉ xét detection có center_y < 55% chiều cao crop
HELMET_REGION_RATIO = 0.55

# Khi có CẢ helmet VÀ nohelmet ở cùng vùng → nếu helmet.conf - nohelmet.conf > MARGIN
# → tin helmet (tránh tóc dài bị nhận nhầm thành nohelmet)
HELMET_CONF_MARGIN = 0.15

# ── Video Processing ──────────────────────────────────────
SKIP_FRAMES = 3             # Chỉ chạy Stage 2+3 mỗi N frame (Stage 1 track luôn chạy)
CSV_UPDATE_INTERVAL = 15    # Cập nhật bảng CSV trên UI mỗi N frame

# ★ Vùng Nhận Diện (Detection Zone) - Chỉ áp dụng cho video
# Xe chỉ được phân tích mũ/biển số khi tâm xe nằm trong vùng này (30% -> 80% chiều cao video)
# Giúp loại bỏ xe quá xa (mờ, khó đọc) và xe chuẩn bị ra khỏi màn hình
ZONE_Y_MIN_RATIO = 0.30
ZONE_Y_MAX_RATIO = 0.80

# ── Deferred Logging & Tracking Logic (Video) ─────────────
# Cần tối thiểu N frame nhận diện "không mũ" để bắt đầu hiện khung ĐỎ báo Phạt Nguội trên màn hình
MIN_VIOLATION_FRAMES = 3

# Tỷ lệ vi phạm tối thiểu để chốt hạ ghi biên bản (Post-processing)
# = (Số frame không mũ) / (Số frame có mũ + Số frame không mũ)
# Ví dụ 0.4 (40%) nghĩa là: nếu AI nhìn rõ đầu người trong 10 frame, phải có ít nhất 4 frame không mũ thì mới lập biên bản.
MIN_VIOLATION_RATIO = 0.40

# Số frame không thấy ID → coi là đã rời khung hình → tiến hành xét duyệt tỷ lệ để ghi biên bản
LOST_ID_THRESHOLD = 30

# ── OCR ──────────────────────────────────────────────────
OCR_MIN_CHARS = 5           # Tối thiểu ký tự để coi là biển số hợp lệ
OCR_UPSCALE_FACTOR = 2      # Phóng to biển số trước khi OCR
PLATE_PAD_RATIO = 0.10      # Padding 10% mỗi cạnh khi crop biển số (tránh cắt sát ký tự biên)
OCR_MULTI_SCALES = [2]       # Scale phóng to cho OCR (thêm scale sẽ chậm hơn nhưng chính xác hơn)

# ── Màu sắc (BGR cho OpenCV) ─────────────────────────────
COLORS = {
    'motorcyclist': (255, 0, 0),    # Xanh dương: Khung xe máy
    'helmet':       (0, 255, 0),    # Xanh lá: Đội mũ an toàn
    'nohelmet':     (0, 0, 255),    # Đỏ: Không đội mũ (Vi phạm)
    'licenseplate': (0, 255, 255),  # Vàng: Khung biển số
}
