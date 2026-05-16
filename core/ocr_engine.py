import cv2
from config import STAGE3_CONF, STAGE3_IMGSZ, OCR_MIN_CHARS, OCR_UPSCALE_FACTOR

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

# Chữ cái hợp lệ trong biển số VN
VN_SERIES_LETTERS = set('ABCDEFGHKLMNPRSTUVXYZ')


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
    
    # Vị trí 3+: KHÔNG sửa (có thể là chữ 81AL hoặc số 29B1)
    
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
    return f"{line1}-{line2}" if line2 else line1


# ══════════════════════════════════════════════════════════
# LOẠI BỎ KÝ TỰ TRÙNG LẶP (NMS CHO KÝ TỰ)
# ══════════════════════════════════════════════════════════

def _remove_duplicate_chars(chars):
    """
    Loại bỏ các detection chồng chéo nhau trên cùng 1 vị trí.
    Nếu 2 ký tự có tâm x quá gần nhau (< 50% chiều rộng trung bình),
    chỉ giữ ký tự có confidence cao hơn.
    """
    if len(chars) < 2:
        return chars
    
    # Tính chiều rộng trung bình
    avg_w = sum(c.get('w', c['h']) for c in chars) / len(chars)
    min_dist = avg_w * 0.5  # Khoảng cách tối thiểu giữa 2 ký tự
    
    # Sắp xếp theo x, rồi loại bỏ trùng
    chars.sort(key=lambda c: c['x'])
    filtered = [chars[0]]
    
    for i in range(1, len(chars)):
        prev = filtered[-1]
        curr = chars[i]
        
        if abs(curr['x'] - prev['x']) < min_dist:
            # 2 ký tự chồng nhau → giữ cái conf cao hơn
            if curr['conf'] > prev['conf']:
                filtered[-1] = curr
        else:
            filtered.append(curr)
    
    return filtered


# ══════════════════════════════════════════════════════════
# HÀM ĐỌC BIỂN SỐ CHÍNH — ĐƠN GIẢN & CHÍNH XÁC
# ══════════════════════════════════════════════════════════

def read_plate_yolo26(plate_img, model_s3):
    """
    Đọc ký tự trên biển số bằng YOLO Stage-3.
    Pipeline: Upscale → OCR → Loại trùng → Chia dòng → Hậu xử lý.
    """
    # 1. Phóng to biển số
    if OCR_UPSCALE_FACTOR > 1:
        plate_img = cv2.resize(
            plate_img, None,
            fx=OCR_UPSCALE_FACTOR, fy=OCR_UPSCALE_FACTOR,
            interpolation=cv2.INTER_CUBIC
        )

    # 2. Chạy OCR
    results = model_s3.predict(plate_img, conf=STAGE3_CONF, imgsz=STAGE3_IMGSZ, verbose=False)[0]
    
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
        return ""

    # 3. Chia 2 dòng bằng gap lớn nhất theo y
    chars.sort(key=lambda c: c['y'])
    
    max_gap = 0
    split_idx = 0
    for i in range(len(chars) - 1):
        gap = chars[i + 1]['y'] - chars[i]['y']
        if gap > max_gap:
            max_gap = gap
            split_idx = i + 1
    
    avg_h = sum(c['h'] for c in chars) / len(chars)
    
    if max_gap > avg_h * 0.3:
        line1 = chars[:split_idx]
        line2 = chars[split_idx:]
    else:
        line1 = chars
        line2 = []
    
    # 4. ★ Loại bỏ ký tự trùng lặp TRƯỚC KHI ghép
    line1 = _remove_duplicate_chars(line1)
    line2 = _remove_duplicate_chars(line2)
    
    # 5. Sắp xếp trái → phải
    line1.sort(key=lambda c: c['x'])
    line2.sort(key=lambda c: c['x'])
    
    str1 = "".join(c['char'] for c in line1)
    str2 = "".join(c['char'] for c in line2)
    
    # 6. ★ Hậu xử lý sửa lỗi ký tự theo vị trí
    plate_text = correct_plate_text(str1, str2)
    
    # 7. Kiểm tra kết quả
    char_count = len(plate_text.replace('-', ''))
    if char_count >= OCR_MIN_CHARS:
        return plate_text
    
    return ""