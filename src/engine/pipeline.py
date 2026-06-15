import cv2
import os
import warnings
from ultralytics import YOLO

# CÁC MẢNH GHÉP TỪ THƯ MỤC CORE
from src.engine.core.logger import log_violation
from src.engine.core.image_utils import preprocess_and_deskew
from src.engine.core.ocr_engine import read_plate_yolo26
from src.engine.core.tracking_engine import ViolationTracker

# CẤU HÌNH TẬP TRUNG
from src.config import (
    COLORS, STAGE1_CONF, STAGE2_CONF, NOHELMET_MIN_CONF,
    STAGE1_IMGSZ, MIN_PLATE_WIDTH, MIN_PLATE_HEIGHT,
    MIN_CROP_SIZE, HELMET_REGION_RATIO, HELMET_CONF_MARGIN,
    PLATE_PAD_RATIO, SKIP_FRAMES,
    ZONE_Y_MIN_RATIO, ZONE_Y_MAX_RATIO,
)

# Tắt cảnh báo để Terminal luôn sạch sẽ, chuyên nghiệp
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════
# HELPER: PHÁN QUYẾT MŨ BẢO HIỂM THÔNG MINH
# ══════════════════════════════════════════════════════════

def _judge_helmet_violation(detections, crop_h):
    """
    Phán quyết vi phạm mũ bảo hiểm dựa trên vị trí + confidence.

    Logic:
      1. Chỉ xét detection có center_y < HELMET_REGION_RATIO * crop_h (vùng đầu)
      2. Thu thập TẤT CẢ helmet/nohelmet detections ở vùng đầu
      3. Nếu có CẢ helmet và nohelmet ở cùng vùng:
         - Nếu max(helmet.conf) - max(nohelmet.conf) > HELMET_CONF_MARGIN → tin helmet (AN TOÀN)
         - Ngược lại → VI PHẠM (cẩn thận hơn: vẫn phạt khi không chắc chắn)
      4. Nếu chỉ có nohelmet (không có helmet nào) → VI PHẠM chắc chắn
      5. Nếu chỉ có helmet (không có nohelmet) → AN TOÀN
      6. Nếu KHÔNG detect được gì ở vùng đầu → KHÔNG phán quyết (trả về False)
         Lý do: góc khuất, ảnh mờ → không nên đoán bừa

    Trường hợp đặc biệt — NHIỀU NGƯỜI trên 1 xe:
      - Nếu có >= 1 nohelmet pass filter → VI PHẠM (dù tài xế có đội mũ)
      - Vì luật VN: TẤT CẢ người trên xe phải đội mũ

    Args:
        detections: list of dict {'label', 'conf', 'cy'} — các detection helmet/nohelmet
        crop_h: chiều cao crop xe máy (pixel)

    Returns:
        bool — True nếu VI PHẠM, False nếu AN TOÀN
    """
    head_region_limit = crop_h * HELMET_REGION_RATIO

    # Lọc detection ở vùng đầu
    helmets_in_head = []
    nohelmet_in_head = []

    for d in detections:
        if d['cy'] > head_region_limit:
            continue  # Bỏ qua detection ở phần thân/chân → không phải đầu

        if d['label'] == 'helmet':
            helmets_in_head.append(d)
        elif d['label'] == 'nohelmet':
            nohelmet_in_head.append(d)

    # Case 1: Không detect được gì ở vùng đầu → không đủ bằng chứng
    if not helmets_in_head and not nohelmet_in_head:
        return False

    # Case 2: Chỉ có helmet → AN TOÀN
    if helmets_in_head and not nohelmet_in_head:
        return False

    # Case 3: Chỉ có nohelmet → VI PHẠM chắc chắn
    if nohelmet_in_head and not helmets_in_head:
        return True

    # Case 4: CÓ CẢ HAI → cần so sánh confidence
    # Trường hợp nhiều người: nếu có nhiều nohelmet → khả năng cao có người không đội mũ
    max_helmet_conf = max(d['conf'] for d in helmets_in_head)
    max_nohelmet_conf = max(d['conf'] for d in nohelmet_in_head)

    # ★ Nếu có NHIỀU nohelmet detections (>=2) → gần như chắc chắn vi phạm
    #   (nhiều người không đội mũ, hoặc model rất tự tin)
    if len(nohelmet_in_head) >= 2:
        return True

    # ★ So sánh conf: nếu helmet conf VƯỢT TRỘI hơn nohelmet → tin helmet
    #   Điều này giải quyết trường hợp tóc dài bị nhận nhầm thành nohelmet
    #   (nohelmet conf thường ~0.50-0.60, trong khi helmet conf ~0.70-0.85)
    if max_helmet_conf - max_nohelmet_conf > HELMET_CONF_MARGIN:
        return False  # Helmet đáng tin hơn → AN TOÀN

    # Mặc định: khi không chắc chắn → VI PHẠM (cẩn thận hơn)
    return True


# ══════════════════════════════════════════════════════════
# HELPER: CROP BIỂN SỐ CÓ PADDING
# ══════════════════════════════════════════════════════════

def _crop_plate_with_padding(source_img, px1, py1, px2, py2):
    """
    Crop biển số với padding mở rộng viền.
    Tránh cắt sát ký tự đầu/cuối → YOLO OCR đọc đầy đủ hơn.
    """
    h, w = source_img.shape[:2]
    pw, ph = px2 - px1, py2 - py1

    # Tính padding
    pad_x = int(pw * PLATE_PAD_RATIO)
    pad_y = int(ph * PLATE_PAD_RATIO)

    # Mở rộng box nhưng không vượt quá biên ảnh
    new_x1 = max(0, px1 - pad_x)
    new_y1 = max(0, py1 - pad_y)
    new_x2 = min(w, px2 + pad_x)
    new_y2 = min(h, py2 + pad_y)

    return source_img[new_y1:new_y2, new_x1:new_x2]


# ══════════════════════════════════════════════════════════
# TRÁI TIM CỦA HỆ THỐNG: LUỒNG XỬ LÝ LOGIC
# ══════════════════════════════════════════════════════════

def process_logic(img, model_s1, model_s2, model_s3, output_dir, tracker, is_video=False, frame_idx=0):
    """
    Luồng xử lý chính: Detect xe → Detect mũ/biển số → OCR → Ghi biên bản.

    Cải tiến v2 (so với v1):
      - Clone ảnh trước khi vẽ → crop sạch cho Stage 2
      - Lấy biển số diện tích lớn nhất
      - ★ Logic helmet THÔNG MINH: kiểm tra vùng đầu + confidence weighting
      - ★ Plate padding: mở rộng crop biển số tránh cắt sát ký tự
      - ★ Filter crop nhỏ: bỏ qua crop < MIN_CROP_SIZE pixel
      - Video: tích lũy bằng chứng tốt nhất, ghi biên bản khi xe rời khung hình
      - Ảnh: ghi biên bản ngay
      - Error handling: 1 xe lỗi không làm sập toàn bộ

    Trả về: (img_đã_vẽ, stats_dict)
    """
    stats = {'total_vehicles': 0, 'violations_this_frame': 0}

    try:
        img_h, img_w = img.shape[:2]

        # ★ Vẽ Vùng Nhận Diện (Detection Zone) cho video
        if is_video:
            zone_y_min = int(img_h * ZONE_Y_MIN_RATIO)
            zone_y_max = int(img_h * ZONE_Y_MAX_RATIO)
            # Vẽ 2 vạch ngang nổi bật hơn (độ dày 3, màu Đỏ Cam) để dễ demo
            zone_color = (0, 140, 255) # Cam đậm / Đỏ cam BGR
            cv2.line(img, (0, zone_y_min), (img_w, zone_y_min), zone_color, 3, cv2.LINE_AA)
            cv2.line(img, (0, zone_y_max), (img_w, zone_y_max), zone_color, 3, cv2.LINE_AA)
            cv2.putText(img, "DETECTION ZONE - BAT DAU QUET", (10, zone_y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, zone_color, 2, cv2.LINE_AA)
            cv2.putText(img, "DETECTION ZONE - KET THUC", (10, zone_y_max + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, zone_color, 2, cv2.LINE_AA)

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

                # ★ KIỂM TRA VÙNG NHẬN DIỆN (Chỉ áp dụng cho Video)
                if is_video:
                    veh_cy = (y1 + y2) / 2
                    if veh_cy < zone_y_min or veh_cy > zone_y_max:
                        # Nằm ngoài vùng → Chỉ tracking, bỏ qua Stage 2+3
                        continue
                    
                    # Đánh dấu ID vẫn còn trong vùng nhận diện (cho video tracker)
                    tracker.mark_seen(track_id, frame_idx)

                    # Nếu xe đã được ghi biên bản → Không cần tốn công đọc lại
                    if tracker.is_logged(track_id):
                        cv2.putText(img, "LOGGED", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        continue

                # ★ Filter crop quá nhỏ — không đủ chi tiết cho Stage 2
                crop_w, crop_h = x2 - x1, y2 - y1
                if min(crop_w, crop_h) < MIN_CROP_SIZE:
                    continue

                # ★ Crop từ ảnh SẠCH (không có bounding box)
                crop_img = img_clean[y1:y2, x1:x2]
                if crop_img.size == 0:
                    continue

                # 3. NHẬN DIỆN MŨ & BIỂN SỐ (Stage 2)
                res_s2 = model_s2.predict(crop_img, conf=STAGE2_CONF, verbose=False)[0]

                # ★ Thu thập TẤT CẢ detections trước khi phán quyết
                helmet_detections = []  # {'label', 'conf', 'cy'}
                plate_box = None
                plate_area = 0

                for box2 in res_s2.boxes:
                    cx1, cy1, cx2, cy2 = map(int, box2.xyxy[0])
                    cls_id = int(box2.cls[0])
                    label = model_s2.names[cls_id]
                    conf = float(box2.conf[0])

                    # Center Y trong crop (để kiểm tra vùng đầu)
                    cy_center = (cy1 + cy2) / 2

                    # ★ Nohelmet dưới ngưỡng MIN → BỎ QUA HOÀN TOÀN
                    if label == 'nohelmet' and conf < NOHELMET_MIN_CONF:
                        continue

                    # Thu thập helmet/nohelmet detections cho logic phán quyết
                    if label in ('helmet', 'nohelmet'):
                        helmet_detections.append({
                            'label': label,
                            'conf': conf,
                            'cy': cy_center,
                        })

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

                # ★ LOGIC PHÁN QUYẾT VI PHẠM — THÔNG MINH
                violation_detected = _judge_helmet_violation(helmet_detections, crop_h)

                if is_video:
                    # Có nhìn thấy đầu người không?
                    has_head_detection = len(helmet_detections) > 0
                    if has_head_detection:
                        tracker.record_vote(track_id, is_violation=violation_detected)

                # Khoá trạng thái: Cần ít nhất N frames vi phạm để khóa khung đỏ (chống False Positive hiển thị)
                already_violated = is_video and tracker.is_confirmed_violator_ui(track_id)
                is_violator = already_violated if is_video else violation_detected

                # 4. ĐỌC BIỂN SỐ (Stage 3) - TỐI ƯU HÓA THỜI GIAN
                final_plate_text = ""
                
                # ★ CHỈ CHẠY OCR KHI:
                # 1. Là ảnh (luôn chạy)
                # 2. Hoặc là Video nhưng xe này ĐANG VI PHẠM ở frame này
                # 3. Hoặc là Video nhưng xe này ĐÃ BỊ XÁC NHẬN LÀ VI PHẠM từ trước (cần OCR liên tục để lấy biển nét nhất)
                run_ocr = not is_video or is_violator or violation_detected

                if run_ocr and plate_box is not None:
                    px1, py1_p, px2, py2_p = plate_box
                    pw, ph = px2 - px1, py2_p - py1_p

                    if pw >= MIN_PLATE_WIDTH and ph >= MIN_PLATE_HEIGHT:
                        # ★ Crop biển số CÓ PADDING — tránh cắt sát ký tự biên
                        plate_crop = _crop_plate_with_padding(
                            crop_img, px1, py1_p, px2, py2_p
                        )
                        if plate_crop.size > 0:
                            clean_plate = preprocess_and_deskew(plate_crop)
                            final_plate_text = read_plate_yolo26(clean_plate, model_s3)

                # 5. GHI BIÊN BẢN & HIỂN THỊ
                if is_violator and final_plate_text:
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
    from src.config import MODEL_STAGE1, MODEL_STAGE2, MODEL_STAGE3

    print("🚀 Đang khởi động Hệ thống AI Giao thông CHUYÊN NGHIỆP...")

    model_s1 = YOLO(MODEL_STAGE1)
    model_s2 = YOLO(MODEL_STAGE2)
    model_s3 = YOLO(MODEL_STAGE3)

    input_dir = 'data/inputs/'
    output_dir = 'data/outputs/cli_results/'

    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

    files = os.listdir(input_dir)
    if not files:
        print("⚠️ Thư mục trống! Vui lòng cho ảnh hoặc video vào data/inputs/")
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
            orig_fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # ★ Giảm FPS output để video chiếu chậm lại (tốt cho việc demo, nhìn rõ vi phạm)
            slow_fps = max(15, orig_fps // 2)
            if orig_fps >= 60:
                slow_fps = 20  # Nếu 60fps thì đưa về 20fps (chậm 3 lần) để dễ nhìn
            elif orig_fps == 30:
                slow_fps = 15  # Nếu 30fps thì đưa về 15fps (chậm 2 lần)
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), slow_fps,
                                  (int(cap.get(3)), int(cap.get(4))))

            print(f"🎬 Đang quét Video động: {filename} ({total_frames} frames, gốc {orig_fps}fps -> demo {slow_fps}fps)...")
            tracker = ViolationTracker()
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # ★ SKIP_FRAMES: Chỉ chạy Stage 2+3 mỗi N frame
                #   Frame khác: chỉ chạy Stage 1 track (nhẹ) để giữ tracking ID
                run_full = (frame_idx % SKIP_FRAMES == 0)

                if run_full:
                    frame, _ = process_logic(frame, model_s1, model_s2, model_s3,
                                             output_dir, tracker, is_video=True, frame_idx=frame_idx)
                else:
                    # Chỉ tracking nhẹ (Stage 1 only)
                    res_s1 = model_s1.track(frame, persist=True, conf=STAGE1_CONF, imgsz=STAGE1_IMGSZ, verbose=False)[0]
                    if res_s1.boxes is not None:
                        for box1 in res_s1.boxes:
                            if box1.id is not None:
                                tracker.mark_seen(int(box1.id[0]), frame_idx)

                out.write(frame)

                # In tiến trình mỗi 50 frame
                if frame_idx % 50 == 0:
                    pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
                    print(f"   ⏳ Frame {frame_idx}/{total_frames} ({pct:.0f}%)")

            # ★ Ghi biên bản cho các xe còn lại khi video kết thúc
            tracker.finalize(output_dir)

            cap.release()
            out.release()
            print(f"✅ Đã xử lý xong Video: {filename}")

    final_csv_path = os.path.join(output_dir, 'Danh_Sach_Phat_Nguoi.csv')
    print(f"\n🎉 HOÀN TẤT! Toàn bộ file Excel phạt nguội nằm ở: {final_csv_path}")

if __name__ == '__main__':
    main()