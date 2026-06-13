import re
import cv2
from src.config import STAGE3_CONF, STAGE3_IMGSZ, OCR_MIN_CHARS, OCR_MULTI_SCALES

# ══════════════════════════════════════════════════════════
# BẢNG SỬA LỖI KÝ TỰ OCR THEO VỊ TRÍ
# ══════════════════════════════════════════════════════════
# Biển số VN: [SỐ SỐ] [CHỮ] [CHỮ/SỐ?] - [SỐ SỐ SỐ SỐ SỐ]
# Dòng 1: mã tỉnh (2 số) + series (1-2 chữ/số)
# Dòng 2: 4-5 số

# Ký tự HAY BỊ NHẦM khi đọc:
LETTER_TO_DIGIT = {
    'O': '0', 'Q': '0', 'D': '0',
    'I': '1', 'L': '1', 'J': '1',
    'Z': '2',
    'S': '5',
    'G': '6',
    'T': '7',
    'B': '8',
}

DIGIT_TO_LETTER = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B',
    '4': 'A',
    '3': 'E',
}

# Chữ cái hợp lệ trong biển số VN (không có I, O, Q, W, J vì dễ nhầm)
VN_SERIES_LETTERS = set('ABCDEFGHKLMNPRSTUVXYZ')

# ══════════════════════════════════════════════════════════
# REGEX VALIDATE BIỂN SỐ VIỆT NAM
# ══════════════════════════════════════════════════════════
# Format biển số dân dụng VN (2 dòng): "XXY-ZZZZZ" hoặc "XXY1-ZZZZZ"
#   XX  = mã tỉnh (2 số, 11-99)
#   Y   = series chữ (1-2 ký tự)
#   Z   = 4-5 số
# Format biển số 1 dòng (ít gặp): "XXYZZZZZ"

# Pattern cho text ĐÃ GHÉP (có dấu '-')
VN_PLATE_PATTERN_2LINE = re.compile(
    r'^[0-9]{2}[A-Z]{1,2}[0-9]?-[0-9]{3,5}$'
)

# Pattern cho biển 1 dòng (không dấu '-')
VN_PLATE_PATTERN_1LINE = re.compile(
    r'^[0-9]{2}[A-Z]{1,2}[0-9]?[0-9]{3,5}$'
)

# Pattern tổng quát: bắt ĐA SỐ biển số VN hợp lệ
VN_PLATE_LOOSE = re.compile(
    r'^[0-9]{2}[A-Z]{1,2}[0-9]?-?[0-9]{3,5}$'
)


def validate_vn_plate(plate_text):
    """
    Kiểm tra xem chuỗi có đúng format biển số VN không.
    Trả về: (is_valid, cleaned_text)
    """
    if not plate_text:
        return False, ""

    text = plate_text.upper().strip()

    # Bỏ khoảng trắng thừa
    text = text.replace(' ', '')

    # Kiểm tra tổng ký tự (trừ dấu -)
    char_count = len(text.replace('-', ''))
    if char_count < 7 or char_count > 10:
        return False, text

    # Kiểm tra pattern
    if VN_PLATE_PATTERN_2LINE.match(text):
        return True, text
    if VN_PLATE_PATTERN_1LINE.match(text):
        return True, text
    if VN_PLATE_LOOSE.match(text):
        return True, text

    return False, text


# ══════════════════════════════════════════════════════════
# HẬU XỬ LÝ: SỬA LỖI KÝ TỰ THEO VỊ TRÍ
# ══════════════════════════════════════════════════════════

def _correct_line1(text):
    """
    Sửa dòng 1: [SỐ SỐ] [CHỮ] [CHỮ hoặc SỐ]
    VD: 43K, 34A, 29B1, 81AL
    """
    if not text or len(text) < 2:
        return text

    corrected = list(text)

    # Vị trí 0-1: PHẢI là SỐ (mã tỉnh)
    for i in range(min(2, len(corrected))):
        ch = corrected[i]
        if ch in LETTER_TO_DIGIT:
            corrected[i] = LETTER_TO_DIGIT[ch]

    # Vị trí 2: PHẢI là CHỮ (series)
    if len(corrected) > 2:
        ch = corrected[2]
        if ch.isdigit() and ch in DIGIT_TO_LETTER:
            candidate = DIGIT_TO_LETTER[ch]
            if candidate in VN_SERIES_LETTERS:
                corrected[2] = candidate

    # Vị trí 3: có thể là SỐ (29B1) hoặc CHỮ (81AL)
    # → Chỉ sửa nếu rõ ràng:
    if len(corrected) > 3:
        ch = corrected[3]
        # Nếu vị trí 3 là ký tự và phần còn lại (dòng 2) toàn số
        # → khả năng cao vị trí 3 cũng nên là số
        if ch in LETTER_TO_DIGIT and len(corrected) == 4:
            # 4 ký tự dòng 1: XX + Y + Z → Z ở vị trí 3 nên là số nếu dòng 1 là XXY1
            corrected[3] = LETTER_TO_DIGIT[ch]

    return ''.join(corrected)


def _correct_line2(text):
    """Sửa dòng 2: toàn bộ PHẢI là SỐ."""
    if not text:
        return text
    return ''.join(LETTER_TO_DIGIT.get(ch, ch) for ch in text)


def correct_plate_text(str1, str2):
    """Sửa lỗi cả 2 dòng rồi ghép lại."""
    line1 = _correct_line1(str1)
    line2 = _correct_line2(str2)

    if line2:
        result = f"{line1}-{line2}"
    else:
        result = line1

    # ★ Validate format VN — nếu invalid, vẫn trả về nhưng ghi log
    is_valid, cleaned = validate_vn_plate(result)
    return cleaned


# ══════════════════════════════════════════════════════════
# LOẠI BỎ KÝ TỰ TRÙNG LẶP (IoU-based NMS cho ký tự)
# ══════════════════════════════════════════════════════════

def _compute_iou(box_a, box_b):
    """Tính IoU (Intersection over Union) giữa 2 box ký tự."""
    # Mỗi box: {'x': center_x, 'y': center_y, 'w': width, 'h': height}
    ax1 = box_a['x'] - box_a['w'] / 2
    ay1 = box_a['y'] - box_a['h'] / 2
    ax2 = box_a['x'] + box_a['w'] / 2
    ay2 = box_a['y'] + box_a['h'] / 2

    bx1 = box_b['x'] - box_b['w'] / 2
    by1 = box_b['y'] - box_b['h'] / 2
    bx2 = box_b['x'] + box_b['w'] / 2
    by2 = box_b['y'] + box_b['h'] / 2

    # Intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    # Union
    area_a = box_a['w'] * box_a['h']
    area_b = box_b['w'] * box_b['h']
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def _remove_duplicate_chars(chars, iou_threshold=0.35):
    """
    Loại bỏ detection trùng lặp bằng IoU-based NMS.

    Ưu điểm so với phương pháp cũ (chỉ so center X):
      - Xử lý chính xác khi 2 ký tự có kích thước khác nhau
      - Không bỏ sót trường hợp 2 ký tự gần nhau nhưng không chồng
      - Không loại nhầm ký tự hẹp đứng cạnh nhau (I, 1)
    """
    if len(chars) < 2:
        return chars

    # Sắp xếp theo confidence giảm dần
    chars_sorted = sorted(chars, key=lambda c: c['conf'], reverse=True)

    keep = []
    suppressed = set()

    for i in range(len(chars_sorted)):
        if i in suppressed:
            continue
        keep.append(chars_sorted[i])

        for j in range(i + 1, len(chars_sorted)):
            if j in suppressed:
                continue
            iou = _compute_iou(chars_sorted[i], chars_sorted[j])
            if iou > iou_threshold:
                suppressed.add(j)

    return keep


# ══════════════════════════════════════════════════════════
# CHIA DÒNG BIỂN SỐ — CHÍNH XÁC HƠN
# ══════════════════════════════════════════════════════════

def _split_plate_lines(chars):
    """
    Chia ký tự thành 1 hoặc 2 dòng dựa trên gap Y.

    Cải tiến:
      - Threshold 0.5 * avg_h (thay vì 0.3) → tránh chia sai khi chữ lệch nhẹ
      - Yêu cầu min_gap_pixels >= 5 → tránh chia dòng do noise position
      - Kiểm tra tỷ lệ ký tự mỗi dòng: nếu 1 dòng chỉ có 1 ký tự → sai → gộp lại
    """
    if len(chars) < 2:
        return chars, []

    chars.sort(key=lambda c: c['y'])

    max_gap = 0
    split_idx = 0
    for i in range(len(chars) - 1):
        gap = chars[i + 1]['y'] - chars[i]['y']
        if gap > max_gap:
            max_gap = gap
            split_idx = i + 1

    avg_h = sum(c['h'] for c in chars) / len(chars)

    # ★ Điều kiện chia dòng chặt hơn:
    #   1. Gap phải > 50% chiều cao trung bình ký tự
    #   2. Gap phải > 5 pixel tuyệt đối (tránh noise)
    #   3. Mỗi dòng phải có >= 2 ký tự (1 ký tự = outlier, không phải dòng riêng)
    should_split = (
        max_gap > avg_h * 0.5
        and max_gap > 5
        and split_idx >= 2
        and (len(chars) - split_idx) >= 2
    )

    if should_split:
        line1 = chars[:split_idx]
        line2 = chars[split_idx:]
    else:
        line1 = chars
        line2 = []

    return line1, line2


# ══════════════════════════════════════════════════════════
# ĐỌC BIỂN SỐ — SINGLE SCALE
# ══════════════════════════════════════════════════════════

def _ocr_single_scale(plate_img, model_s3, scale_factor):
    """
    Chạy OCR ở 1 scale cụ thể. Trả về (plate_text, char_count, avg_conf).
    """
    # 1. Phóng to
    if scale_factor > 1:
        img = cv2.resize(
            plate_img, None,
            fx=scale_factor, fy=scale_factor,
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        img = plate_img

    # 2. Chạy OCR
    results = model_s3.predict(img, conf=STAGE3_CONF, imgsz=STAGE3_IMGSZ, verbose=False)[0]

    chars = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model_s3.names[int(box.cls[0])]
        conf = float(box.conf[0])
        chars.append({
            'char': label,
            'x': (x1 + x2) / 2,
            'y': (y1 + y2) / 2,
            'w': x2 - x1,
            'h': y2 - y1,
            'conf': conf,
        })

    if len(chars) < OCR_MIN_CHARS:
        return "", 0, 0.0

    # 3. Chia dòng chính xác
    line1, line2 = _split_plate_lines(chars)

    # 4. Loại bỏ ký tự trùng lặp (IoU NMS)
    line1 = _remove_duplicate_chars(line1)
    line2 = _remove_duplicate_chars(line2)

    # 5. Sắp xếp trái → phải
    line1.sort(key=lambda c: c['x'])
    line2.sort(key=lambda c: c['x'])

    str1 = "".join(c['char'] for c in line1)
    str2 = "".join(c['char'] for c in line2)

    # 6. Hậu xử lý sửa lỗi ký tự theo vị trí
    plate_text = correct_plate_text(str1, str2)

    # 7. Kiểm tra kết quả
    char_count = len(plate_text.replace('-', ''))
    if char_count < OCR_MIN_CHARS:
        return "", 0, 0.0

    # Tính confidence trung bình
    all_chars = line1 + line2
    avg_conf = sum(c['conf'] for c in all_chars) / len(all_chars) if all_chars else 0.0

    return plate_text, char_count, avg_conf


# ══════════════════════════════════════════════════════════
# HÀM ĐỌC BIỂN SỐ CHÍNH — MULTI-SCALE + CHỌN KẾT QUẢ TỐT NHẤT
# ══════════════════════════════════════════════════════════

def read_plate_yolo26(plate_img, model_s3):
    """
    Đọc ký tự trên biển số bằng YOLO Stage-3.

    Cải tiến v2:
      - Multi-scale: thử OCR ở nhiều scale (2x, 3x), chọn kết quả tốt nhất
      - IoU-based NMS thay vì center-X distance
      - Chia dòng chính xác hơn (threshold cao hơn, check min chars/line)
      - Validate format biển số VN

    Tiêu chí chọn kết quả tốt nhất:
      1. Ưu tiên text valid theo format VN
      2. Ưu tiên text có nhiều ký tự hơn
      3. Nếu bằng ký tự → chọn avg_conf cao hơn
    """
    best_text = ""
    best_count = 0
    best_conf = 0.0
    best_valid = False

    for scale in OCR_MULTI_SCALES:
        text, count, conf = _ocr_single_scale(plate_img, model_s3, scale)
        if not text:
            continue

        is_valid, _ = validate_vn_plate(text)

        # Logic chọn kết quả tốt nhất:
        is_better = False

        if is_valid and not best_valid:
            # Valid luôn thắng invalid
            is_better = True
        elif is_valid == best_valid:
            # Cùng trạng thái valid → so sánh chi tiết
            if count > best_count:
                is_better = True
            elif count == best_count and conf > best_conf:
                is_better = True

        if is_better:
            best_text = text
            best_count = count
            best_conf = conf
            best_valid = is_valid

    return best_text