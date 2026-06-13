import cv2
import os
import csv
from datetime import datetime

def log_violation(plate_text, evidence_img, output_dir):
    """
    Ghi biên bản vi phạm: lưu ảnh bằng chứng + ghi vào CSV.
    
    Tự động phân biệt cấu trúc thư mục:
      - Giao diện Web  (output_dir = "outputs")  → ảnh vào images/,  CSV vào reports/
      - Terminal/CLI    (output_dir = "test_outputs") → ảnh vào Bang_Chung/, CSV ở root
    """
    now = datetime.now()
    # Thêm microsecond vào tên file để tránh trùng khi 2 vi phạm xảy ra cùng giây
    timestamp_str = now.strftime("%Y%m%d_%H%M%S_%f")
    time_display = now.strftime("%Y-%m-%d %H:%M:%S")
    
    if not plate_text:
        plate_text = "KHONG_RO"

    # Phân biệt đường dẫn theo nguồn gọi (Web hay Terminal)
    is_web = 'web_results' in output_dir or 'outputs' in output_dir
    
    if is_web:
        # Web UI: Ảnh → data/outputs/web_results/images/ | CSV → data/outputs/web_results/reports/
        img_dir = os.path.join(output_dir, 'images')
        csv_dir = os.path.join(output_dir, 'reports')
    else:
        # Terminal: Ảnh → data/outputs/cli_results/images/ | CSV → data/outputs/cli_results/
        img_dir = os.path.join(output_dir, 'images')
        csv_dir = output_dir
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # 1. Lưu ảnh bằng chứng
    evidence_filename = f"ViPham_{plate_text}_{timestamp_str}.jpg"
    evidence_path = os.path.join(img_dir, evidence_filename)
    cv2.imwrite(evidence_path, evidence_img)

    # 2. Ghi CSV danh sách phạt nguội
    log_file = os.path.join(csv_dir, 'Danh_Sach_Phat_Nguoi.csv')
    
    # Kiểm tra xem file đã tồn tại chưa để tạo Tiêu đề (Header)
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Thời gian vi phạm', 'Biển số xe', 'Tên file Bằng chứng', 'Lỗi'])
        writer.writerow([time_display, plate_text, evidence_filename, 'Không đội mũ bảo hiểm'])