import cv2
import os
import warnings
from ultralytics import YOLO

# CÁC MẢNH GHÉP TỪ THƯ MỤC CORE
from core.logger import log_violation
from core.image_utils import preprocess_and_deskew
from core.ocr_engine import read_plate_yolo26
from core.tracking_engine import ViolationTracker

# Tắt cảnh báo để Terminal luôn sạch sẽ, chuyên nghiệp
warnings.filterwarnings("ignore")

# CẤU HÌNH MÀU SẮC (Chuẩn BGR trong OpenCV)
COLORS = {
    'motorcyclist': (255, 0, 0),    # Xanh dương: Khung xe máy
    'helmet': (0, 255, 0),          # Xanh lá: Đội mũ an toàn
    'nohelmet': (0, 0, 255),        # Đỏ: Không đội mũ (Vi phạm)
    'licenseplate': (0, 255, 255)   # Vàng: Khung biển số
}

# ==========================================
# TRÁI TIM CỦA HỆ THỐNG: LUỒNG XỬ LÝ LOGIC
# ==========================================
def process_logic(img, model_s1, model_s2, model_s3, output_dir, tracker, is_video=False):
    # 1. BẬT CHẾ ĐỘ QUÉT THÔNG MINH THEO LOẠI ĐẦU VÀO
    if is_video:
        # Video: Cần Tracking để cấp ID và theo dõi (chống spam file Excel)
        res_s1 = model_s1.track(img, persist=True, conf=0.45, verbose=False)[0]
    else:
        # Ảnh tĩnh: Quét vét cạn (Predict) để không bỏ sót bất kỳ ai
        res_s1 = model_s1.predict(img, conf=0.45, verbose=False)[0]
    
    # Nếu khung hình không có xe máy nào, bỏ qua luôn cho nhẹ máy
    if res_s1.boxes is None:
        return img

    # 2. XỬ LÝ TỪNG ĐỐI TƯỢNG XE MÁY TÌM ĐƯỢC
    for index, box1 in enumerate(res_s1.boxes):
        x1, y1, x2, y2 = map(int, box1.xyxy[0])
        
        # Cấp ID: Nếu là video thì lấy ID do thuật toán cấp, ảnh thì tự đánh số 1, 2, 3...
        if is_video and box1.id is not None:
            track_id = int(box1.id[0]) 
        else:
            track_id = index + 1 
        
        # Luôn vẽ khung và hiển thị ID cho người lái xe
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS['motorcyclist'], 2)
        cv2.putText(img, f"ID: {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Cắt ảnh xe máy ra để đưa vào Stage 2
        crop_img = img[y1:y2, x1:x2]
        if crop_img.size == 0: continue
            
        # 3. NHẬN DIỆN MŨ BẢO HIỂM VÀ BIỂN SỐ (Stage 2)
        res_s2 = model_s2.predict(crop_img, conf=0.35, verbose=False)[0]
        
        violation_detected = False
        plate_box = None
        
        for box2 in res_s2.boxes:
            cx1, cy1, cx2, cy2 = map(int, box2.xyxy[0])
            label = model_s2.names[int(box2.cls[0])]
            
            if label == 'nohelmet': 
                violation_detected = True
            if label == 'licenseplate': 
                plate_box = (cx1, cy1, cx2, cy2)

            # Vẽ khung mũ và biển số lên hình lớn (Cộng dồn tọa độ)
            fx1, fy1 = cx1 + x1, cy1 + y1
            fx2, fy2 = cx2 + x1, cy2 + y1
            cv2.rectangle(img, (fx1, fy1), (fx2, fy2), COLORS.get(label, (255,255,255)), 2)

        # 4. ĐỌC CHỮ TRÊN BIỂN SỐ (Stage 3)
        final_plate_text = ""
        if plate_box is not None:
            px1, py1, px2, py2 = plate_box
            plate_crop = crop_img[py1:py2, px1:px2]
            
            if plate_crop.size > 0:
                # Gọi chuyên gia xử lý ảnh để nắn thẳng và khử lóa
                clean_plate = preprocess_and_deskew(plate_crop) 
                # Gọi chuyên gia OCR để đọc chữ
                final_plate_text = read_plate_yolo26(clean_plate, model_s3)
                
        # 5. GHI BIÊN BẢN & HIỂN THỊ KẾT QUẢ
        if final_plate_text:
            if violation_detected:
                # Hiển thị chữ cảnh báo nhấp nháy trên màn hình
                cv2.putText(img, f"PHAT NGUOI: {final_plate_text}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['nohelmet'], 2)
                
                # Logic ghi sổ: Nếu là Ảnh -> Ghi luôn. Nếu là Video -> Hỏi Tracker xem ghi chưa.
                if (not is_video) or (not tracker.is_logged(track_id)):
                    log_violation(final_plate_text, crop_img, output_dir)
                    print(f"🚨 ĐÃ LẬP BIÊN BẢN (ID {track_id}): {final_plate_text}")
                    
                    if is_video:
                        tracker.mark_as_logged(track_id) # Khóa ID lại không cho ghi trùng lặp
            else:
                # Người chấp hành tốt, hiển thị chữ màu xanh
                cv2.putText(img, f"AN TOAN: {final_plate_text}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['helmet'], 2)

    return img

# ==========================================
# HÀM CHẠY CHÍNH TỔNG HỢP (MAIN)
# ==========================================
def main():
    print("🚀 Đang khởi động Hệ thống AI Giao thông CHUYÊN NGHIỆP...")
    
    # 1. NẠP 3 MÔ HÌNH LÕI YOLO
    model_s1 = YOLO('models/stage1.pt')
    model_s2 = YOLO('models/stage2.pt')
    model_s3 = YOLO('models/stage3.pt') # Đảm bảo đúng tên file Stage 3 của bạn

    input_dir = 'test_inputs/'      
    output_dir = 'test_outputs/' 
    
    # Tạo sẵn thư mục lưu bằng chứng nếu chưa có
    os.makedirs(os.path.join(output_dir, 'Bang_Chung'), exist_ok=True)

    files = os.listdir(input_dir)
    if not files:
        print("⚠️ Thư mục trống! Vui lòng cho ảnh hoặc video vào test_inputs/")
        return

    # 2. BẮT ĐẦU QUÉT FILE
    for filename in files:
        file_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        # --------- NHÁNH 1: XỬ LÝ ẢNH ---------
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(file_path)
            if img is not None:
                print(f"🖼️ Đang quét Ảnh tĩnh: {filename}...")
                tracker = ViolationTracker() # Reset tracker
                # Chú ý: is_video=False
                img = process_logic(img, model_s1, model_s2, model_s3, output_dir, tracker, is_video=False)
                cv2.imwrite(out_path, img)
                print(f"✅ Đã xử lý xong Ảnh: {filename}")
        
        # --------- NHÁNH 2: XỬ LÝ VIDEO ---------
        elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(file_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))
            
            print(f"🎬 Đang quét Video động: {filename}...")
            tracker = ViolationTracker() # Reset tracker cho mỗi Video mới
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Chú ý: is_video=True
                frame = process_logic(frame, model_s1, model_s2, model_s3, output_dir, tracker, is_video=True)
                out.write(frame)
                
            cap.release()
            out.release()
            print(f"✅ Đã xử lý xong Video: {filename}")

    # Đường dẫn log file linh hoạt
    final_csv_path = os.path.join(output_dir, 'Danh_Sach_Phat_Nguoi.csv')
    print(f"\n🎉 HOÀN TẤT! Toàn bộ file Excel phạt nguội nằm ở: {final_csv_path}")

if __name__ == '__main__':
    main()