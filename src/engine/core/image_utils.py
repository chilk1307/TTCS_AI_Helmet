import numpy as np
import math
import cv2

from src.config import PLATE_PAD_RATIO


# ══════════════════════════════════════════════════════════
# CÂN BẰNG SÁNG TỐI (CLAHE) — Chống lóa, tăng contrast
# ══════════════════════════════════════════════════════════
def changeContrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


# ══════════════════════════════════════════════════════════
# LÀM NÉT (Unsharp Mask) — Khôi phục chi tiết sau upscale
# ══════════════════════════════════════════════════════════
def sharpen_image(img):
    """
    Unsharp Mask: blur rồi trừ ra khỏi ảnh gốc → tăng cạnh.
    Hiệu quả hơn kernel sharpen thông thường vì kiểm soát được sigma.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    # ảnh_nét = ảnh_gốc * (1 + amount) - blur * amount
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return sharpened


# ══════════════════════════════════════════════════════════
# PADDING VIỀN TRẮNG — Tránh YOLO miss ký tự sát biên
# ══════════════════════════════════════════════════════════
def add_border_padding(img, pad_ratio=None):
    """
    Thêm viền trắng quanh ảnh biển số.
    YOLO hay bỏ sót ký tự đầu/cuối nếu chúng nằm sát cạnh ảnh.
    """
    if pad_ratio is None:
        pad_ratio = PLATE_PAD_RATIO

    h, w = img.shape[:2]
    pad_x = max(int(w * pad_ratio), 3)
    pad_y = max(int(h * pad_ratio), 3)

    # copyMakeBorder với viền trắng (255, 255, 255)
    padded = cv2.copyMakeBorder(
        img,
        top=pad_y, bottom=pad_y, left=pad_x, right=pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    return padded


# ══════════════════════════════════════════════════════════
# KHỬ NHIỄU NHẸ — Cho ảnh tối / nhiều noise
# ══════════════════════════════════════════════════════════
def denoise_if_needed(img):
    """
    Kiểm tra độ sáng trung bình. Nếu ảnh tối (mean < 80) → khử nhiễu nhẹ.
    Không khử nhiễu ảnh sáng bình thường vì sẽ làm mất chi tiết.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        if mean_brightness < 80:
            # bilateralFilter: nhanh hơn fastNlMeans ~60 lần cho ảnh nhỏ
            # d=7: đường kính pixel lân cận
            # sigmaColor=50: lọc màu (giữ cạnh)
            # sigmaSpace=50: lọc không gian
            return cv2.bilateralFilter(img, 7, 50, 50)
    except Exception:
        pass  # Nếu denoise lỗi → trả về ảnh gốc, không crash pipeline

    return img


# ══════════════════════════════════════════════════════════
# XOAY ẢNH + TÍNH GÓC NGHIÊNG (Deskew)
# ══════════════════════════════════════════════════════════
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)


def compute_skew(src_img):
    """Tìm góc nghiêng của biển số bằng Hough Lines."""
    img = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(img, threshold1=30, threshold2=100, apertureSize=3, L2gradient=True)
    h, w = img.shape[:2]
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 30, minLineLength=w / 1.5, maxLineGap=h / 3.0)

    if lines is None:
        return 0.0

    min_line = 100
    min_line_pos = 0
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            center_point = [((x1 + x2) / 2), ((y1 + y2) / 2)]
            if center_point[1] < min_line:
                min_line = center_point[1]
                min_line_pos = i

    angle = 0.0
    cnt = 0
    for x1, y1, x2, y2 in lines[min_line_pos]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        # Chuyển sang độ rồi mới so sánh (arctan2 trả về radian, max ±3.14)
        ang_degree = math.degrees(ang)
        if math.fabs(ang_degree) <= 30:  # Chỉ xoay nếu nghiêng ≤ 30°
            angle += ang_degree
            cnt += 1

    return (angle / cnt) if cnt > 0 else 0.0


# ══════════════════════════════════════════════════════════
# HÀM TỔNG HỢP: Pipeline tiền xử lý biển số đầy đủ
# ══════════════════════════════════════════════════════════
def preprocess_and_deskew(src_img):
    """
    Pipeline tiền xử lý biển số:
      1. Guard: bỏ qua ảnh quá nhỏ
      2. Denoise nếu ảnh tối
      3. CLAHE tăng contrast
      4. Deskew (nắn thẳng)
      5. Sharpen (làm nét)
      6. Padding viền trắng
    """
    h, w = src_img.shape[:2]
    if h < 10 or w < 10:
        return src_img

    # 1. Khử nhiễu nếu ảnh tối
    img = denoise_if_needed(src_img)

    # 2. Tăng contrast
    enhanced_img = changeContrast(img)

    # 3. Nắn thẳng
    angle = compute_skew(enhanced_img)
    if angle != 0.0:
        enhanced_img = rotate_image(enhanced_img, angle)

    # 4. Làm nét
    enhanced_img = sharpen_image(enhanced_img)

    # 5. Padding viền trắng
    enhanced_img = add_border_padding(enhanced_img)

    return enhanced_img