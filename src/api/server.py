"""
==========================================================
  🚀 AI PHẠT NGUỘI GIAO THÔNG — Flask Backend Server
  Phục vụ giao diện HTML/JS (thay thế Streamlit)
  Tận dụng 100% logic AI từ main_pipeline.py & core/
==========================================================
"""

import os
import cv2
import json
import uuid
import time
import glob
import base64
import threading
import warnings
from datetime import datetime
import io
import zipfile

from flask import (
    Flask, request, jsonify, Response,
    send_from_directory, send_file,
)
import pandas as pd
from ultralytics import YOLO

# ── Import AI pipeline & core (giữ nguyên 100%) ──────────
from src.engine.pipeline import process_logic
from src.engine.core.tracking_engine import ViolationTracker
from src.config import (
    MODEL_STAGE1, MODEL_STAGE2, MODEL_STAGE3,
    SKIP_FRAMES, STAGE1_CONF, STAGE1_IMGSZ,
    ZONE_Y_MIN_RATIO, ZONE_Y_MAX_RATIO,
)

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
# APP INIT
# ══════════════════════════════════════════════════════════
app = Flask(__name__, static_folder="../frontend", static_url_path="/static")

# Thư mục output (dùng chung cấu trúc với Streamlit)
WEB_OUTPUT_DIR = "data/outputs/web_results"
CSV_PATH = os.path.join(WEB_OUTPUT_DIR, "reports", "Danh_Sach_Phat_Nguoi.csv")
IMG_DIR = os.path.join(WEB_OUTPUT_DIR, "images")
UPLOAD_DIR = os.path.join(WEB_OUTPUT_DIR, "uploads")
for d in [os.path.join(WEB_OUTPUT_DIR, "reports"), IMG_DIR, UPLOAD_DIR]:
    os.makedirs(d, exist_ok=True)

# Lock để đảm bảo chỉ 1 task xử lý AI tại 1 thời điểm
processing_lock = threading.Lock()

# Task storage: task_id → {file_paths, type, status}
tasks = {}
task_cancelled = {}

# ══════════════════════════════════════════════════════════
# LOAD MODELS (1 lần duy nhất khi server khởi động)
# ══════════════════════════════════════════════════════════
print("🔄 Đang tải 3 model AI…")
model_s1 = YOLO(MODEL_STAGE1)
model_s2 = YOLO(MODEL_STAGE2)
model_s3 = YOLO(MODEL_STAGE3)
print("✅ Đã tải xong 3 model AI!")


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

def encode_frame(frame, quality=70):
    """Encode OpenCV frame → base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def read_csv():
    """Đọc CSV vi phạm, trả về DataFrame."""
    if os.path.isfile(CSV_PATH):
        try:
            return pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def sse_event(payload):
    """Format dữ liệu thành SSE event string."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ══════════════════════════════════════════════════════════
# ROUTE: TRANG CHỦ
# ══════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")


# ══════════════════════════════════════════════════════════
# API: UPLOAD FILE
# ══════════════════════════════════════════════════════════

@app.route("/api/upload", methods=["POST"])
def upload():
    """Nhận file ảnh/video từ frontend, lưu tạm, trả task_id."""
    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"error": "Chưa chọn file"}), 400

    task_id = str(uuid.uuid4())[:8]
    file_paths = []
    file_type = "image"

    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext in (".mp4", ".avi", ".mov"):
            file_type = "video"
        safe_name = f"{task_id}_{f.filename}"
        save_path = os.path.join(UPLOAD_DIR, safe_name)
        f.save(save_path)
        file_paths.append(save_path)

    tasks[task_id] = {
        "file_paths": file_paths,
        "type": file_type,
        "status": "pending",
    }

    return jsonify({
        "task_id": task_id,
        "type": file_type,
        "count": len(file_paths),
    })


# ══════════════════════════════════════════════════════════
# API: XỬ LÝ FILE (SSE Stream)
# ══════════════════════════════════════════════════════════

@app.route("/api/process/cancel/<task_id>", methods=["POST"])
def cancel_process(task_id):
    """API để client gọi huỷ quá trình đang xử lý."""
    task_cancelled[task_id] = True
    return jsonify({"status": "ok"})

@app.route("/api/process/<task_id>")
def process_stream(task_id):
    """SSE endpoint — stream tiến trình xử lý từng frame/ảnh."""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task không tồn tại"}), 404

    def generate():
        if not processing_lock.acquire(blocking=False):
            yield sse_event({"type": "error", "message": "Hệ thống đang xử lý task khác. Vui lòng chờ."})
            return

        try:
            task["status"] = "processing"
            task_cancelled[task_id] = False

            if task["type"] == "image":
                yield from _process_images(task, task_id)
            else:
                yield from _process_video(task, task_id)

            task["status"] = "done"
        except GeneratorExit:
            task["status"] = "cancelled"
        finally:
            processing_lock.release()
            # Dọn file upload còn sót
            for p in task.get("file_paths", []):
                if os.path.isfile(p):
                    try: os.unlink(p)
                    except: pass

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _process_images(task, task_id):
    """Generator xử lý nhiều ảnh, yield SSE events."""
    total = len(task["file_paths"])
    total_vehicles = 0
    total_violations = 0

    for i, path in enumerate(task["file_paths"]):
        if task_cancelled.get(task_id):
            yield sse_event({"type": "error", "message": "Đã dừng xử lý!"})
            break

        img = cv2.imread(path)
        if img is None:
            continue

        # Gửi stage info
        yield sse_event({"type": "stage", "stage": 1, "text": f"Quét ảnh {i+1}/{total}…"})

        # Đảm bảo reset trạng thái YOLO để tránh nhận diện chập chờn (state bleed)
        model_s1.predictor = None
        model_s2.predictor = None
        model_s3.predictor = None
        
        tracker = ViolationTracker()
        result, stats = process_logic(
            img, model_s1, model_s2, model_s3,
            WEB_OUTPUT_DIR, tracker, is_video=False,
        )

        total_vehicles += stats["total_vehicles"]
        total_violations += stats["violations_this_frame"]

        # Gửi frame kết quả
        b64 = encode_frame(result)
        yield sse_event({
            "type": "frame",
            "image": b64,
            "stats": {"vehicles": total_vehicles, "violations": total_violations},
            "progress": {"current": i + 1, "total": total},
            "filename": os.path.basename(path),
        })

        # Thông báo vi phạm mới
        if stats["violations_this_frame"] > 0:
            yield sse_event({
                "type": "violation_alert",
                "count": stats["violations_this_frame"],
            })

        try: os.unlink(path)
        except: pass

    yield sse_event({
        "type": "done",
        "total_vehicles": total_vehicles,
        "total_violations": total_violations,
    })


def _process_video(task, task_id):
    """Generator xử lý video, yield SSE events."""
    path = task["file_paths"][0]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        yield sse_event({"type": "error", "message": "Không mở được video"})
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Đảm bảo reset trạng thái YOLO để tránh rác state (đặc biệt khi đổi từ predict sang track)
    model_s1.predictor = None
    model_s2.predictor = None
    model_s3.predictor = None

    tracker = ViolationTracker()
    model_s1.predictor = None  # Reset YOLO tracker nội bộ

    frame_idx = 0
    all_seen_ids = set()  # ★ Tập hợp TẤT CẢ unique track IDs từng thấy trong video
    total_violations = 0

    yield sse_event({
        "type": "video_info",
        "fps": fps,
        "total_frames": total_frames,
    })

    try:
        while True:
            if task_cancelled.get(task_id):
                yield sse_event({"type": "error", "message": "Đã dừng xử lý!"})
                break

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            run_full = (frame_idx % SKIP_FRAMES == 0)

            if run_full:
                yield sse_event({"type": "stage", "stage": 3, "text": f"Frame {frame_idx}/{total_frames}"})
                processed, stats = process_logic(
                    frame, model_s1, model_s2, model_s3,
                    WEB_OUTPUT_DIR, tracker, is_video=True, frame_idx=frame_idx,
                )
                # ★ Thu thập unique IDs từ YOLO tracker boxes
                if hasattr(processed, '__class__'):  # processed là frame
                    pass  # IDs được thu thập qua res_s1 bên trong process_logic
                total_violations += stats["violations_this_frame"]

                if stats["violations_this_frame"] > 0:
                    yield sse_event({"type": "violation_alert", "count": stats["violations_this_frame"]})
            else:
                # Tracking nhẹ (Stage 1 only)
                res_s1 = model_s1.track(
                    frame, persist=True, conf=STAGE1_CONF,
                    imgsz=STAGE1_IMGSZ, verbose=False,
                )[0]
                processed = frame
                if res_s1.boxes is not None:
                    img_h = frame.shape[0]
                    zone_y_min = int(img_h * ZONE_Y_MIN_RATIO)
                    zone_y_max = int(img_h * ZONE_Y_MAX_RATIO)
                    for box1 in res_s1.boxes:
                        if box1.id is not None:
                            tid = int(box1.id[0])
                            all_seen_ids.add(tid)  # ★ Đếm unique ID
                            y1, y2 = box1.xyxy[0][1], box1.xyxy[0][3]
                            veh_cy = (y1 + y2) / 2
                            if zone_y_min <= veh_cy <= zone_y_max:
                                tracker.mark_seen(tid, frame_idx)

            # ★ Thu thập unique IDs từ tracker.last_seen (bao gồm cả run_full frames)
            all_seen_ids.update(tracker.last_seen.keys())
            all_seen_ids.update(tracker.logged_ids)

            # Chỉ gửi các frame đã được vẽ bounding box (run_full)
            if run_full:
                b64 = encode_frame(processed, quality=65)
                yield sse_event({
                    "type": "frame",
                    "image": b64,
                    "stats": {"vehicles": len(all_seen_ids), "violations": total_violations},
                    "progress": {"current": frame_idx, "total": total_frames},
                })
    finally:
        # Ghi biên bản cho xe còn lại
        finalized = tracker.finalize(WEB_OUTPUT_DIR)
        total_violations += finalized
        cap.release()
        import time
        for _ in range(5):
            try:
                os.unlink(path)
                break
            except Exception:
                time.sleep(0.2)

    yield sse_event({
        "type": "done",
        "total_vehicles": len(all_seen_ids),
        "total_violations": total_violations,
        "total_frames": frame_idx,
    })


# ══════════════════════════════════════════════════════════
# API: CAMERA GIẢ LẬP (SSE Stream)
# ══════════════════════════════════════════════════════════

@app.route("/api/camera/stream")
def camera_stream():
    """SSE endpoint giả lập luồng camera real-time."""
    def generate():
        if not processing_lock.acquire(blocking=False):
            yield sse_event({"type": "error", "message": "Hệ thống đang bận."})
            return

        try:
            # Tìm video mẫu trong data/camera_sample/
            video_path = None
            for ext in ("*.mp4", "*.avi", "*.mov"):
                found = glob.glob(os.path.join("data/camera_sample", ext))
                if found:
                    video_path = found[0]
                    break

            if not video_path:
                yield sse_event({"type": "error", "message": "Chưa có video mẫu. Hãy copy 1 video vào thư mục data/camera_sample/"})
                return

            cap = cv2.VideoCapture(video_path)
            tracker = ViolationTracker()
            model_s1.predictor = None
            model_s2.predictor = None
            model_s3.predictor = None

            frame_idx = 0
            all_seen_ids = set()  # ★ Unique IDs
            total_violations = 0

            yield sse_event({
                "type": "camera_start",
                "source": os.path.basename(video_path),
            })

            while True:
                ret, frame = cap.read()
                if not ret:
                    # Loop lại video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    tracker = ViolationTracker()
                    model_s1.predictor = None
                    model_s2.predictor = None
                    model_s3.predictor = None
                    frame_idx = 0
                    total_violations = 0
                    continue

                frame_idx += 1
                run_full = (frame_idx % SKIP_FRAMES == 0)

                if run_full:
                    processed, stats = process_logic(
                        frame, model_s1, model_s2, model_s3,
                        WEB_OUTPUT_DIR, tracker, is_video=True, frame_idx=frame_idx,
                    )
                    total_violations += stats["violations_this_frame"]

                    if stats["violations_this_frame"] > 0:
                        yield sse_event({"type": "violation_alert", "count": stats["violations_this_frame"]})
                else:
                    res_s1 = model_s1.track(
                        frame, persist=True, conf=STAGE1_CONF,
                        imgsz=STAGE1_IMGSZ, verbose=False,
                    )[0]
                    processed = frame
                    if res_s1.boxes is not None:
                        img_h = frame.shape[0]
                        zone_y_min = int(img_h * ZONE_Y_MIN_RATIO)
                        zone_y_max = int(img_h * ZONE_Y_MAX_RATIO)
                        for box1 in res_s1.boxes:
                            if box1.id is not None:
                                tid = int(box1.id[0])
                                all_seen_ids.add(tid)
                                y1, y2 = box1.xyxy[0][1], box1.xyxy[0][3]
                                veh_cy = (y1 + y2) / 2
                                if zone_y_min <= veh_cy <= zone_y_max:
                                    tracker.mark_seen(tid, frame_idx)

                # ★ Thu thập unique IDs
                all_seen_ids.update(tracker.last_seen.keys())
                all_seen_ids.update(tracker.logged_ids)

                # Chỉ gửi các frame đã được vẽ bounding box (run_full)
                if run_full:
                    b64 = encode_frame(processed, quality=60)
                    yield sse_event({
                        "type": "frame",
                        "image": b64,
                        "stats": {"vehicles": len(all_seen_ids), "violations": total_violations},
                        "frame_idx": frame_idx,
                    })

                time.sleep(0.02)  # Tránh quá tải CPU

        except GeneratorExit:
            pass
        finally:
            processing_lock.release()
            try: cap.release()
            except: pass

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ══════════════════════════════════════════════════════════
# API: VIOLATIONS CRUD
# ══════════════════════════════════════════════════════════

@app.route("/api/violations")
def get_violations():
    """Trả danh sách vi phạm dạng JSON."""
    df = read_csv()
    if df.empty:
        return jsonify([])
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/violations/image/<path:filename>")
def get_evidence_image(filename):
    """Phục vụ ảnh bằng chứng vi phạm."""
    return send_from_directory(os.path.abspath(IMG_DIR), filename)


@app.route("/api/violations", methods=["DELETE"])
def clear_violations():
    """Xóa toàn bộ lịch sử vi phạm."""
    if os.path.isfile(CSV_PATH):
        os.remove(CSV_PATH)
    if os.path.isdir(IMG_DIR):
        for f in os.listdir(IMG_DIR):
            fp = os.path.join(IMG_DIR, f)
            if os.path.isfile(fp):
                os.remove(fp)
    return jsonify({"status": "ok", "message": "Đã xóa toàn bộ lịch sử vi phạm"})


@app.route("/api/violations/download")
def download_zip():
    """Tải file ZIP vi phạm (CSV + Ảnh), có hỗ trợ filter ngày."""
    df = read_csv()
    if df.empty:
        return jsonify({"error": "Chưa có dữ liệu"}), 404

    date_from = request.args.get("from")
    date_to = request.args.get("to")

    if date_from or date_to:
        dates = df["Thời gian vi phạm"].str.split(" ").str[0]
        if date_from:
            df = df[dates >= date_from]
        if date_to:
            df = df[dates <= date_to]

    if df.empty:
        return jsonify({"error": "Không có vi phạm nào trong khoảng thời gian này"}), 404

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        csv_data = df.to_csv(index=False)
        zf.writestr("Danh_Sach_Phat_Nguoi.csv", csv_data.encode("utf-8-sig"))
        
        for img_name in df["Tên file Bằng chứng"].dropna().unique():
            img_path = os.path.join(IMG_DIR, str(img_name))
            if os.path.isfile(img_path):
                zf.write(img_path, f"Bang_Chung/{img_name}")

    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype="application/zip",
        as_attachment=True,
        download_name="HoSo_PhatNguoi.zip"
    )


@app.route("/api/violations/stats")
def violation_stats():
    """Thống kê vi phạm theo giờ & loại — cho Chart.js."""
    df = read_csv()
    if df.empty:
        return jsonify({"hourly": {}, "by_type": {}})

    hourly = {}
    by_type = {}

    for _, row in df.iterrows():
        try:
            ts = str(row.get("Thời gian vi phạm", ""))
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            hour = f"{dt.hour:02d}:00"
            hourly[hour] = hourly.get(hour, 0) + 1
        except Exception:
            pass

        vtype = str(row.get("Lỗi", "Không xác định"))
        by_type[vtype] = by_type.get(vtype, 0) + 1

    return jsonify({"hourly": hourly, "by_type": by_type})


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  🚀 AI PHẠT NGUỘI GIAO THÔNG — Web Server")
    print("  📁 Mở trình duyệt: http://localhost:5000")
    print("=" * 56 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
