import numpy as np
import math
import cv2

# Bí kíp CLAHE: Cân bằng sáng tối, chống lóa cực mạnh
def changeContrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# Xoay ảnh
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

# Tìm góc nghiêng của biển số
def compute_skew(src_img):
    img = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(img, threshold1=30, threshold2=100, apertureSize=3, L2gradient=True)
    h, w = img.shape[:2]
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w/1.5, maxLineGap=h/3.0)
    
    if lines is None:
        return 0.0

    min_line = 100
    min_line_pos = 0
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            center_point = [((x1+x2)/2), ((y1+y2)/2)]
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

# HÀM TỔNG HỢP GỌI TỪ MAIN: Vừa chống lóa, vừa nắn thẳng
def preprocess_and_deskew(src_img):
    # Guard: ảnh quá nhỏ sẽ crash medianBlur/Canny
    h, w = src_img.shape[:2]
    if h < 10 or w < 10:
        return src_img
    enhanced_img = changeContrast(src_img)
    angle = compute_skew(enhanced_img)
    if angle != 0.0:
        return rotate_image(enhanced_img, angle)
    return enhanced_img