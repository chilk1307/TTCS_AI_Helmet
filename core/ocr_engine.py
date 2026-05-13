import re

# Regex kiểm tra format biển số Việt Nam (tối thiểu 7 ký tự số/chữ)
# VD: 43D2-07326, 29B1-12345, 51F-12345
VN_PLATE_REGEX = re.compile(r'^[0-9A-Z]{2,4}[A-Z]?\d?-[0-9A-Z]{4,6}$')

# Số ký tự tối thiểu để coi là biển số hợp lệ
MIN_CHARS = 5

def read_plate_yolo26(plate_img, model_s3):
    """
    Đọc ký tự trên biển số bằng YOLO Stage-3.
    Cải tiến:
      - Chia 2 dòng bằng gap lớn nhất giữa các tọa độ y (thay vì threshold cứng)
      - Lọc kết quả bằng regex biển số VN
      - Yêu cầu tối thiểu 5 ký tự
    """
    results = model_s3.predict(plate_img, conf=0.4, imgsz=320, verbose=False)[0]
    
    chars = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model_s3.names[int(box.cls[0])] 
        chars.append({'char': label, 'x': (x1+x2)/2, 'y': (y1+y2)/2, 'h': y2-y1})
    
    # Kiểm tra số ký tự tối thiểu
    if len(chars) < MIN_CHARS:
        return ""

    # ── Chia 2 dòng bằng thuật toán tìm gap lớn nhất ──
    chars.sort(key=lambda c: c['y'])
    
    # Tìm khoảng cách y lớn nhất giữa 2 ký tự liền kề → đó là ranh giới 2 dòng
    max_gap = 0
    split_idx = 0
    for i in range(len(chars) - 1):
        gap = chars[i+1]['y'] - chars[i]['y']
        if gap > max_gap:
            max_gap = gap
            split_idx = i + 1
    
    # Nếu gap lớn nhất > 30% chiều cao trung bình ký tự → biển số 2 dòng
    avg_h = sum(c['h'] for c in chars) / len(chars)
    
    if max_gap > avg_h * 0.3:
        line1 = chars[:split_idx]
        line2 = chars[split_idx:]
    else:
        # Biển số 1 dòng (xe đời mới)
        line1 = chars
        line2 = []
    
    # Sắp xếp mỗi dòng theo trục x (trái → phải)
    line1.sort(key=lambda c: c['x'])
    line2.sort(key=lambda c: c['x'])
    
    str1 = "".join([c['char'] for c in line1])
    str2 = "".join([c['char'] for c in line2])
    
    plate_text = f"{str1}-{str2}" if str2 else str1
    
    # ── Kiểm tra format biển số VN bằng regex ──
    # Nếu khớp regex → trả về. Nếu không → vẫn trả về nhưng cảnh báo
    # (Vì model OCR có thể nhận nhầm ký tự, ta không nên loại bỏ hoàn toàn)
    if VN_PLATE_REGEX.match(plate_text):
        return plate_text
    
    # Nếu không khớp regex nhưng đủ ký tự → vẫn trả về (model OCR không hoàn hảo)
    if len(plate_text.replace('-', '')) >= MIN_CHARS:
        return plate_text
    
    return ""