import cv2
import os
import csv
from datetime import datetime

def log_violation(plate_text, evidence_img, output_dir):
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    time_display = now.strftime("%Y-%m-%d %H:%M:%S")
    
    if not plate_text:
        plate_text = "KHONG_RO"

    # 1. Lưu ảnh bằng chứng
    evidence_filename = f"ViPham_{plate_text}_{timestamp_str}.jpg"
    evidence_path = os.path.join(output_dir, 'Bang_Chung', evidence_filename)
    cv2.imwrite(evidence_path, evidence_img)

    # 2. Ghi Excel (Tự động lưu vào đúng thư mục output_dir)
    log_file = os.path.join(output_dir, 'Danh_Sach_Phat_Nguoi.csv')
    
    # Kiểm tra xem file đã tồn tại chưa để tạo Tiêu đề (Header)
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Thời gian vi phạm', 'Biển số xe', 'Tên file Bằng chứng', 'Lỗi'])
        writer.writerow([time_display, plate_text, evidence_filename, 'Không đội mũ bảo hiểm'])