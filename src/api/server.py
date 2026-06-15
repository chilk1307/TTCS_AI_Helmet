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
import queue
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

# Task storage: task_id → {file_paths, type, status, frame_queue, stats, alerts_queue, done_event}
tasks = {}
task_cancelled = {}

# Camera stream global state
camera_state = {
    "frame_queue": queue.Queue(maxsize=30),
    "alerts_queue": queue.Queue(),
    "stats": {"vehicles": 0, "violations": 0},
    "running": False,
    "thread": None
}

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

def encode_frame_bytes(frame, quality=80):
    """Encode OpenCV frame → raw JPEG bytes for MJPEG."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()

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
        "frame_queue": queue.Queue(maxsize=30),
        "alerts_queue": queue.Queue(),
        "stats": {"vehicles": 0, "violations": 0, "current_frame": 0, "total_frames": 1, "done": False},
        "event": threading.Event()
    }
    task_cancelled[task_id] = False

    return jsonify({
        "task_id": task_id,
        "type": file_type,
        "count": len(file_paths),
    })


# ══════════════════════════════════════════════════════════
# MULTITHREADING: AI WORKERS
# ══════════════════════════════════════════════════════════

def ai_worker_upload(task_id):
    """Luồng chạy AI độc lập cho file upload. Kết quả ném vào Queue."""
    task = tasks.get(task_id)
    if not task: return

    try:
        if task["type"] == "image":
            _worker_images(task, task_id)
        else:
            _worker_video(task, task_id)
    finally:
        task["stats"]["done"] = True
        task["event"].set()
        processing_lock.release()
        # Dọn dẹp
        for p in task.get("file_paths", []):
            if os.path.isfile(p):
                try: os.unlink(p)
                except: pass


def _worker_images(task, task_id):
    total = len(task["file_paths"])
    task["stats"]["total_frames"] = total
    total_vehicles = 0
    total_violations = 0

    for i, path in enumerate(task["file_paths"]):
        if task_cancelled.get(task_id): break

        img = cv2.imread(path)
        if img is None: continue

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

        task["stats"]["vehicles"] = total_vehicles
        task["stats"]["violations"] = total_violations
        task["stats"]["current_frame"] = i + 1

        if stats["violations_this_frame"] > 0:
            task["alerts_queue"].put(stats["violations_this_frame"])

        # Put high-res frame bytes into queue
        jpg_bytes = encode_frame_bytes(result)
        try:
            task["frame_queue"].put(jpg_bytes, timeout=1.0)
        except queue.Full:
            pass


def _worker_video(task, task_id):
    path = task["file_paths"][0]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    task["stats"]["total_frames"] = total_frames

    model_s1.predictor = None
    model_s2.predictor = None
    model_s3.predictor = None
    tracker = ViolationTracker()
    model_s1.predictor = None

    frame_idx = 0
    all_seen_ids = set()
    total_violations = 0

    try:
        while True:
            if task_cancelled.get(task_id): break

            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1

            run_full = (frame_idx % SKIP_FRAMES == 0)

            if run_full:
                processed, stats = process_logic(
                    frame, model_s1, model_s2, model_s3,
                    WEB_OUTPUT_DIR, tracker, is_video=True, frame_idx=frame_idx,
                )
                total_violations += stats["violations_this_frame"]
                if stats["violations_this_frame"] > 0:
                    task["alerts_queue"].put(stats["violations_this_frame"])
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

            all_seen_ids.update(tracker.last_seen.keys())
            all_seen_ids.update(tracker.logged_ids)

            task["stats"]["vehicles"] = len(all_seen_ids)
            task["stats"]["violations"] = total_violations
            task["stats"]["current_frame"] = frame_idx

            # Truyền mjpeg nét căng
            jpg_bytes = encode_frame_bytes(processed)
            try:
                task["frame_queue"].put(jpg_bytes, timeout=0.5)
            except queue.Full:
                pass # Bỏ qua nếu mạng chậm (Frame dropping)

    finally:
        finalized = tracker.finalize(WEB_OUTPUT_DIR)
        task["stats"]["violations"] += finalized
        cap.release()


# ══════════════════════════════════════════════════════════
# API: XỬ LÝ FILE (SSE & MJPEG Stream)
# ══════════════════════════════════════════════════════════

@app.route("/api/process/cancel/<task_id>", methods=["POST"])
def cancel_process(task_id):
    task_cancelled[task_id] = True
    return jsonify({"status": "ok"})

@app.route("/api/process/<task_id>")
def process_stream(task_id):
    """SSE endpoint: Chỉ truyền tiến độ (JSON), KHÔNG truyền ảnh Base64."""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task không tồn tại"}), 404

    def generate():
        if not processing_lock.acquire(blocking=False):
            yield sse_event({"type": "error", "message": "Hệ thống đang xử lý task khác."})
            return

        try:
            task["status"] = "processing"
            task_cancelled[task_id] = False
            
            # Start background AI worker
            t = threading.Thread(target=ai_worker_upload, args=(task_id,))
            t.daemon = True
            t.start()

            # Listen to stats
            while True:
                if task_cancelled.get(task_id):
                    yield sse_event({"type": "error", "message": "Đã dừng xử lý!"})
                    break

                stats = task["stats"]
                
                # Push alerts if any
                try:
                    while True:
                        alert_count = task["alerts_queue"].get_nowait()
                        yield sse_event({"type": "violation_alert", "count": alert_count})
                except queue.Empty:
                    pass

                # Push progress
                yield sse_event({
                    "type": "progress",
                    "stats": {"vehicles": stats["vehicles"], "violations": stats["violations"]},
                    "progress": {"current": stats["current_frame"], "total": stats["total_frames"]},
                })

                if stats["done"]:
                    yield sse_event({
                        "type": "done",
                        "total_vehicles": stats["vehicles"],
                        "total_violations": stats["violations"],
                    })
                    break

                time.sleep(0.1) # Update UI at 10Hz
        except GeneratorExit:
            task_cancelled[task_id] = True # Cancel AI if client disconnects

    return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.route("/api/video_feed/upload/<task_id>")
def video_feed_upload(task_id):
    """MJPEG endpoint: Truyền video HD mượt mà xuống thẻ <img>."""
    task = tasks.get(task_id)
    if not task: return "Not found", 404

    def generate():
        while True:
            if task_cancelled.get(task_id) or (task["stats"]["done"] and task["frame_queue"].empty()):
                break
            try:
                frame_bytes = task["frame_queue"].get(timeout=0.5)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except queue.Empty:
                pass
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ══════════════════════════════════════════════════════════
# API: CAMERA GIẢ LẬP (Đa luồng + MJPEG)
# ══════════════════════════════════════════════════════════

def ai_worker_camera():
    video_path = None
    for ext in ("*.mp4", "*.avi", "*.mov"):
        found = glob.glob(os.path.join("data/camera_sample", ext))
        if found:
            video_path = found[0]
            break

    if not video_path:
        camera_state["running"] = False
        processing_lock.release()
        return

    cap = cv2.VideoCapture(video_path)
    tracker = ViolationTracker()
    model_s1.predictor = None
    model_s2.predictor = None
    model_s3.predictor = None

    frame_idx = 0
    all_seen_ids = set()
    total_violations = 0

    try:
        while camera_state["running"]:
            ret, frame = cap.read()
            if not ret:
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
                    camera_state["alerts_queue"].put(stats["violations_this_frame"])
            else:
                res_s1 = model_s1.track(frame, persist=True, conf=STAGE1_CONF, imgsz=STAGE1_IMGSZ, verbose=False)[0]
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
                            if zone_y_min <= ((y1+y2)/2) <= zone_y_max:
                                tracker.mark_seen(tid, frame_idx)

            all_seen_ids.update(tracker.last_seen.keys())
            all_seen_ids.update(tracker.logged_ids)

            camera_state["stats"]["vehicles"] = len(all_seen_ids)
            camera_state["stats"]["violations"] = total_violations

            # Truyền mjpeg nét căng (giới hạn tốc độ khung hình thủ công để giống thật)
            jpg_bytes = encode_frame_bytes(processed)
            try:
                camera_state["frame_queue"].put(jpg_bytes, timeout=0.5)
            except queue.Full:
                pass

            time.sleep(0.03) # Camera thật thường ~30fps
    finally:
        cap.release()
        camera_state["running"] = False
        processing_lock.release()

@app.route("/api/camera/stream")
def camera_stream_stats():
    """SSE endpoint cho Camera: Truyền metadata (JSON)."""
    def generate():
        if camera_state["running"]:
            # Đang chạy rồi thì chỉ attach vào nghe thôi
            pass
        else:
            if not processing_lock.acquire(blocking=False):
                yield sse_event({"type": "error", "message": "Hệ thống đang bận."})
                return
            
            # Xóa sạch queue cũ
            while not camera_state["frame_queue"].empty(): camera_state["frame_queue"].get()
            while not camera_state["alerts_queue"].empty(): camera_state["alerts_queue"].get()

            camera_state["running"] = True
            t = threading.Thread(target=ai_worker_camera)
            t.daemon = True
            t.start()
            camera_state["thread"] = t

        yield sse_event({"type": "camera_start", "source": "Local Video File"})

        try:
            while camera_state["running"]:
                try:
                    while True:
                        alert = camera_state["alerts_queue"].get_nowait()
                        yield sse_event({"type": "violation_alert", "count": alert})
                except queue.Empty: pass

                yield sse_event({
                    "type": "stats",
                    "stats": {"vehicles": camera_state["stats"]["vehicles"], "violations": camera_state["stats"]["violations"]}
                })
                time.sleep(0.1)
        except GeneratorExit:
            # Client disconnect
            camera_state["running"] = False

    return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})

@app.route("/api/video_feed/camera")
def video_feed_camera():
    """MJPEG endpoint cho Camera."""
    def generate():
        while camera_state["running"]:
            try:
                frame_bytes = camera_state["frame_queue"].get(timeout=0.5)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except queue.Empty: pass
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ══════════════════════════════════════════════════════════
# API: VIOLATIONS CRUD
# ══════════════════════════════════════════════════════════

@app.route("/api/violations")
def get_violations():
    """Trả danh sách vi phạm dạng JSON."""
    df = read_csv()
    if df.empty: return jsonify([])
    return jsonify(df.to_dict(orient="records"))

@app.route("/api/violations/image/<path:filename>")
def get_evidence_image(filename):
    """Phục vụ ảnh bằng chứng vi phạm."""
    return send_from_directory(os.path.abspath(IMG_DIR), filename)

@app.route("/api/violations", methods=["DELETE"])
def clear_violations():
    """Xóa toàn bộ lịch sử vi phạm."""
    if os.path.isfile(CSV_PATH): os.remove(CSV_PATH)
    if os.path.isdir(IMG_DIR):
        for f in os.listdir(IMG_DIR):
            fp = os.path.join(IMG_DIR, f)
            if os.path.isfile(fp): os.remove(fp)
    return jsonify({"status": "ok", "message": "Đã xóa toàn bộ lịch sử vi phạm"})

@app.route("/api/violations/download")
def download_zip():
    """Tải file ZIP vi phạm (CSV + Ảnh), có hỗ trợ filter ngày."""
    df = read_csv()
    if df.empty: return jsonify({"error": "Chưa có dữ liệu"}), 404

    date_from = request.args.get("from")
    date_to = request.args.get("to")

    if date_from or date_to:
        dates = df["Thời gian vi phạm"].str.split(" ").str[0]
        if date_from: df = df[dates >= date_from]
        if date_to: df = df[dates <= date_to]

    if df.empty: return jsonify({"error": "Không có vi phạm nào trong khoảng thời gian này"}), 404

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        csv_data = df.to_csv(index=False)
        zf.writestr("Danh_Sach_Phat_Nguoi.csv", csv_data.encode("utf-8-sig"))
        for img_name in df["Tên file Bằng chứng"].dropna().unique():
            img_path = os.path.join(IMG_DIR, str(img_name))
            if os.path.isfile(img_path):
                zf.write(img_path, f"Bang_Chung/{img_name}")

    memory_file.seek(0)
    return send_file(memory_file, mimetype="application/zip", as_attachment=True, download_name="HoSo_PhatNguoi.zip")

@app.route("/api/violations/stats")
def violation_stats():
    """Thống kê vi phạm theo giờ & loại — cho Chart.js."""
    df = read_csv()
    if df.empty: return jsonify({"hourly": {}, "by_type": {}})
    hourly = {}
    by_type = {}
    for _, row in df.iterrows():
        try:
            ts = str(row.get("Thời gian vi phạm", ""))
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            hour = f"{dt.hour:02d}:00"
            hourly[hour] = hourly.get(hour, 0) + 1
        except Exception: pass
        vtype = str(row.get("Lỗi", "Không xác định"))
        by_type[vtype] = by_type.get(vtype, 0) + 1
    return jsonify({"hourly": hourly, "by_type": by_type})


if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  🚀 AI PHẠT NGUỘI GIAO THÔNG — Web Server")
    print("  📁 Mở trình duyệt: http://localhost:5000")
    print("=" * 56 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
