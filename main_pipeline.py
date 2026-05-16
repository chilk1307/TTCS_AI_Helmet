import cv2
import os
import warnings
from ultralytics import YOLO

# CÁC MẢNH GHÉP TỪ THƯ MỤC CORE
from core.logger import log_violation
from core.image_utils import preprocess_and_deskew
from core.ocr_engine import read_plate_yolo26
from core.tracking_engine import ViolationTracker

# CẤU HÌNH TẬP TRUNG
from config import (
    COLORS, STAGE1_CONF, STAGE2_CONF, NOHELMET_MIN_CONF,
    STAGE1_IMGSZ, MIN_PLATE_WIDTH, MIN_PLATE_HEIGHT,
)

# Tắt cảnh báo để Terminal luôn sạch sẽ, chuyên nghiệp
warnings.filterwarnings("ignore")


# ==========================================
# TRÁI TIM CỦA HỆ THỐNG: LUỒNG XỬ LÝ LOGIC
# ==========================================
def process_logic(img, model_s1, model_s2, model_s3, output_dir, tracker, is_video=False, frame_idx=0):
    """
    Luồng xử lý chính: Detect xe → Detect mũ/biển số → OCR → Ghi biên bản.
    
    Cải tiến so với v1:
      - Clone ảnh trước khi vẽ → crop sạch cho Stage 2
      - Lấy biển số diện tích lớn nhất
      - Confidence riêng cho nohelmet (chống nhận nhầm tóc)
      - Video: tích lũy bằng chứng tốt nhất, ghi biên bản khi xe rời khung hình
      - Ảnh: ghi biên bản ngay, kể cả khi biển số KHONG_RO
      - Error handling: 1 xe lỗi không làm sập toàn bộ
    
    Trả về: (img_đã_vẽ, stats_dict)
    """
    stats = {'total_vehicles': 0, 'violations_this_frame': 0}
    
    try:
        # 1. QUÉT XE MÁY (Stage 1)
        if is_video:
            res_s1 = model_s1.track(img, persist=True, conf=STAGE1_CONF, imgsz=STAGE1_IMGSZ, verbose=False)[0]
        else:
            res_s1 = model_s1.predict(img, conf=STAGE1_CONF, imgsz=STAGE1_IMGSZ, verbose=False)[0]
        
        if res_s1.boxes is None or len(res_s1.boxes) == 0:
            return img, stats

        stats['total_vehicles'] = len(res_s1.boxes)

        # ★ CLONE ảnh gốc TRƯỚC khi vẽ — crop sẽ lấy từ bản sạch
        img_clean = img.copy()

        # 2. XỬ LÝ TỪNG XE MÁY
        for index, box1 in enumerate(res_s1.boxes):
            try:
                x1, y1, x2, y2 = map(int, box1.xyxy[0])
                
                # Cấp ID
                if is_video and box1.id is not None:
                    track_id = int(box1.id[0]) 
                else:
                    track_id = index + 1 

                # Vẽ khung xe lên ảnh HIỂN THỊ (không phải ảnh sạch)
                cv2.rectangle(img, (x1, y1), (x2, y2), COLORS['motorcyclist'], 2)
                cv2.putText(img, f"ID: {track_id}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # ★ Crop từ ảnh SẠCH (không có bounding box)
                crop_img = img_clean[y1:y2, x1:x2]
                if crop_img.size == 0:
                    continue

                # Đánh dấu ID vẫn còn trong khung hình (cho video tracker)
                if is_video:
                    tracker.mark_seen(track_id, frame_idx)
                    
                # 3. NHẬN DIỆN MŨ & BIỂN SỐ (Stage 2)
                res_s2 = model_s2.predict(crop_img, conf=STAGE2_CONF, verbose=False)[0]
                
                has_nohelmet = False
                plate_box = None
                plate_area = 0
                
                for box2 in res_s2.boxes:
                    cx1, cy1, cx2, cy2 = map(int, box2.xyxy[0])
                    cls_id = int(box2.cls[0])
                    label = model_s2.names[cls_id]
                    conf = float(box2.conf[0])
                    
                    # ★ Nohelmet dưới ngưỡng → BỎ QUA HOÀN TOÀN (không vẽ, không đếm)
                    #   Tránh vẽ khung đỏ nhưng lại nói "AN TOÀN" → gây nhầm lẫn
                    if label == 'nohelmet' and conf < NOHELMET_MIN_CONF:
                        continue
                    
                    if label == 'nohelmet':
                        has_nohelmet = True
                    
                    # Lấy biển số DIỆN TÍCH LỚN NHẤT
                    if label == 'licenseplate':
                        area = (cx2 - cx1) * (cy2 - cy1)
                        if area > plate_area:
                            plate_box = (cx1, cy1, cx2, cy2)
                            plate_area = area

                    # Vẽ khung + LABEL text (chỉ vẽ detection đã qua lọc)
                    fx1, fy1 = cx1 + x1, cy1 + y1
                    fx2, fy2 = cx2 + x1, cy2 + y1
                    color = COLORS.get(label, (255, 255, 255))
                    cv2.rectangle(img, (fx1, fy1), (fx2, fy2), color, 2)
                    label_text = f"{label} {conf:.0%}"
                    cv2.putText(img, label_text, (fx1, fy1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                # ★ LOGIC PHÁN QUYẾT VI PHẠM:
                # CÓ nohelmet = VI PHẠM (dù tài xế có đội mũ, khách không đội vẫn phạt)
                # Confidence đã lọc bằng NOHELMET_MIN_CONF (0.60) để tránh nhận nhầm tóc
                violation_detected = has_nohelmet

                # 4. ĐỌC BIỂN SỐ (Stage 3)
                final_plate_text = ""
                if plate_box is not None:
                    px1, py1, px2, py2 = plate_box
                    pw, ph = px2 - px1, py2 - py1
                    
                    if pw >= MIN_PLATE_WIDTH and ph >= MIN_PLATE_HEIGHT:
                        plate_crop = crop_img[py1:py2, px1:px2]
                        if plate_crop.size > 0:
                            clean_plate = preprocess_and_deskew(plate_crop)
                            final_plate_text = read_plate_yolo26(clean_plate, model_s3)

                # 5. GHI BIÊN BẢN & HIỂN THỊ
                # ★ CHỈ ghi vi phạm khi ĐỌC ĐƯỢC biển số (không ghi KHONG_RO)
                #   Lý do: Không có biển số → không thể phạt nguội → ghi vô ích
                if violation_detected and final_plate_text:
                    cv2.putText(img, f"PHAT NGUOI: {final_plate_text}", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['nohelmet'], 2)
                    
                    if is_video:
                        tracker.update_violation(
                            track_id, final_plate_text, crop_img,
                            plate_area, frame_idx
                        )
                    else:
                        log_violation(final_plate_text, crop_img, output_dir)
                        print(f"🚨 ĐÃ LẬP BIÊN BẢN (ID {track_id}): {final_plate_text}")
                        stats['violations_this_frame'] += 1
                        
                elif final_plate_text:
                    # Người chấp hành tốt
                    cv2.putText(img, f"AN TOAN: {final_plate_text}", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['helmet'], 2)

            except Exception as e:
                # ★ 1 xe lỗi không làm sập toàn bộ pipeline
                print(f"⚠️ Lỗi xử lý xe #{index}: {e}")
                continue

        # ★ VIDEO: Ghi biên bản cho các xe đã rời khung hình
        if is_video:
            flushed = tracker.flush_lost_ids(frame_idx, output_dir)
            stats['violations_this_frame'] += flushed

    except Exception as e:
        print(f"⚠️ Lỗi pipeline frame: {e}")

    return img, stats


# ==========================================
# HÀM CHẠY CHÍNH TỔNG HỢP (MAIN)
# ==========================================
def main():
    from config import MODEL_STAGE1, MODEL_STAGE2, MODEL_STAGE3
    
    print("🚀 Đang khởi động Hệ thống AI Giao thông CHUYÊN NGHIỆP...")
    
    model_s1 = YOLO(MODEL_STAGE1)
    model_s2 = YOLO(MODEL_STAGE2)
    model_s3 = YOLO(MODEL_STAGE3)

    input_dir = 'test_inputs/'      
    output_dir = 'test_outputs/' 
    
    os.makedirs(os.path.join(output_dir, 'Bang_Chung'), exist_ok=True)

    files = os.listdir(input_dir)
    if not files:
        print("⚠️ Thư mục trống! Vui lòng cho ảnh hoặc video vào test_inputs/")
        return

    for filename in files:
        file_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(file_path)
            if img is not None:
                print(f"🖼️ Đang quét Ảnh tĩnh: {filename}...")
                tracker = ViolationTracker()
                img, _ = process_logic(img, model_s1, model_s2, model_s3, output_dir, tracker, is_video=False)
                cv2.imwrite(out_path, img)
                print(f"✅ Đã xử lý xong Ảnh: {filename}")
        
        elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(file_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                  (int(cap.get(3)), int(cap.get(4))))
            
            print(f"🎬 Đang quét Video động: {filename}...")
            tracker = ViolationTracker()
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                frame, _ = process_logic(frame, model_s1, model_s2, model_s3,
                                         output_dir, tracker, is_video=True, frame_idx=frame_idx)
                out.write(frame)
                
            # ★ Ghi biên bản cho các xe còn lại khi video kết thúc
            tracker.finalize(output_dir)
            
            cap.release()
            out.release()
            print(f"✅ Đã xử lý xong Video: {filename}")

    final_csv_path = os.path.join(output_dir, 'Danh_Sach_Phat_Nguoi.csv')
    print(f"\n🎉 HOÀN TẤT! Toàn bộ file Excel phạt nguội nằm ở: {final_csv_path}")

if __name__ == '__main__':
    main()