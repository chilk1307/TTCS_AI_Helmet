"""
==========================================================
  🚀 AI PHẠT NGUỘI GIAO THÔNG - Giao diện Web (Streamlit)
  Tác giả: Senior Frontend & AI Engineer
  Mô tả : Giao diện nhận diện vi phạm không đội mũ bảo hiểm
           sử dụng 3 model YOLO theo pipeline 3 giai đoạn.
==========================================================
"""

import streamlit as st
import cv2
import os
import tempfile
import zipfile
import io
import pandas as pd
from ultralytics import YOLO

# ── Import pipeline & tracker từ project ──────────────────
from main_pipeline import process_logic
from core.tracking_engine import ViolationTracker
from config import (
    MODEL_STAGE1, MODEL_STAGE2, MODEL_STAGE3,
    SKIP_FRAMES, CSV_UPDATE_INTERVAL,
)

# ══════════════════════════════════════════════════════════
# 0. CẤU HÌNH TRANG (phải gọi đầu tiên, trước mọi lệnh st)
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Phạt Nguội Giao Thông",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# 1. LOAD MODEL VÀO CACHE (chỉ load 1 lần duy nhất)
# ══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔄 Đang tải 3 model AI — chờ xíu nhé…")
def load_models():
    """Tải 3 model YOLO một lần và cache lại trong bộ nhớ."""
    s1 = YOLO(MODEL_STAGE1)
    s2 = YOLO(MODEL_STAGE2)
    s3 = YOLO(MODEL_STAGE3)
    return s1, s2, s3

model_s1, model_s2, model_s3 = load_models()

# ══════════════════════════════════════════════════════════
# 2. ĐƯỜNG DẪN OUTPUT RIÊNG CHO GIAO DIỆN WEB
# ══════════════════════════════════════════════════════════
WEB_OUTPUT_DIR = "outputs"
CSV_PATH       = os.path.join(WEB_OUTPUT_DIR, "reports", "Danh_Sach_Phat_Nguoi.csv")
IMG_DIR        = os.path.join(WEB_OUTPUT_DIR, "images")
os.makedirs(os.path.join(WEB_OUTPUT_DIR, "reports"), exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════
# 3. CUSTOM CSS – Giao diện tối, hiện đại
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Import font Google ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Toàn bộ app ──────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
}

/* ── Sidebar ───────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e0e0ff;
}

/* ── Metric cards ──────────────────────────────────── */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 24px;
    transition: transform 0.2s, box-shadow 0.2s;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(99,102,241,0.25);
}
div[data-testid="stMetric"] label {
    color: #a5b4fc !important;
    font-weight: 600;
    letter-spacing: 0.4px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 800;
    font-size: 2rem !important;
}

/* ── Dataframe / bảng ──────────────────────────────── */
div[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    overflow: hidden;
}

/* ── Nút bấm chính ────────────────────────────────── */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 0;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.5px;
    transition: all 0.25s;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35);
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(99,102,241,0.5);
    background: linear-gradient(135deg, #818cf8, #a78bfa);
}

/* ── File uploader ─────────────────────────────────── */
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 2px dashed rgba(99,102,241,0.35);
    border-radius: 16px;
    padding: 12px;
    transition: border-color 0.3s;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(139,92,246,0.6);
}

/* ── Header gradient text ──────────────────────────── */
.main-title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.sub-title {
    text-align: center;
    color: #94a3b8;
    font-size: 0.95rem;
    margin-bottom: 28px;
}

/* ── Divider ───────────────────────────────────────── */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent);
    margin: 24px 0;
    border: none;
}

/* ── Khung hiển thị ảnh / video ────────────────────── */
.video-container {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 8px;
    overflow: hidden;
}
.video-container img {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 4. HÀM TIỆN ÍCH
# ══════════════════════════════════════════════════════════

def load_web_csv():
    """Đọc file CSV danh sách phạt nguội từ thư mục giao diện web."""
    if os.path.isfile(CSV_PATH):
        try:
            return pd.read_csv(CSV_PATH, encoding="utf-8")
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def save_upload_to_temp(uploaded_file):
    """Lưu file người dùng upload vào thư mục tạm và trả về đường dẫn."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


def render_violation_table_with_gallery(df):
    """Hiển thị bảng vi phạm + gallery ảnh bằng chứng khi click mở rộng."""
    if df.empty:
        st.info("Chưa có dữ liệu vi phạm. Hãy upload file và bắt đầu xử lý.")
        return
    
    # Hiển thị bảng
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Gallery ảnh bằng chứng
    evidence_files = df['Tên file Bằng chứng'].tolist() if 'Tên file Bằng chứng' in df.columns else []
    
    if evidence_files:
        with st.expander("🖼️ Xem ảnh bằng chứng vi phạm", expanded=False):
            # Hiển thị ảnh theo grid 3 cột
            cols = st.columns(3)
            for i, fname in enumerate(evidence_files):
                img_path = os.path.join(IMG_DIR, fname)
                if os.path.isfile(img_path):
                    with cols[i % 3]:
                        img = cv2.imread(img_path)
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            # Lấy biển số từ tên file
                            plate = fname.replace("ViPham_", "").rsplit("_", 2)[0]
                            st.image(img_rgb, caption=f"🚨 {plate}", use_container_width=True)


# ══════════════════════════════════════════════════════════
# 5. SIDEBAR – Cài đặt hệ thống
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Cài đặt Hệ thống")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Hiển thị trạng thái model
    st.markdown("#### 🤖 Trạng thái Model")
    st.success("✅ Stage 1 — Phát hiện xe", icon="🔍")
    st.success("✅ Stage 2 — Mũ & Biển số", icon="🪖")
    st.success("✅ Stage 3 — OCR biển số", icon="🔡")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Upload file (★ hỗ trợ nhiều ảnh cùng lúc)
    st.markdown("#### 📂 Tải lên Ảnh / Video")
    uploaded_files = st.file_uploader(
        "Kéo thả hoặc chọn file",
        type=["png", "jpg", "jpeg", "mp4", "avi", "mov"],
        help="Hỗ trợ nhiều ảnh cùng lúc (PNG, JPG) hoặc 1 video (MP4, AVI, MOV)",
        accept_multiple_files=True,
    )

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Nút xử lý
    btn_process = st.button("🚀 Bắt đầu Xử lý", use_container_width=True)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Nút xóa lịch sử vi phạm
    if st.button("🗑️ Xóa lịch sử vi phạm", use_container_width=True):
        if os.path.isfile(CSV_PATH):
            os.remove(CSV_PATH)
        if os.path.isdir(IMG_DIR):
            for f in os.listdir(IMG_DIR):
                fp = os.path.join(IMG_DIR, f)
                if os.path.isfile(fp):
                    os.remove(fp)
        st.session_state.total_vehicles = 0
        st.session_state.total_violations = 0
        st.sidebar.success("✅ Đã xóa toàn bộ lịch sử vi phạm!")
        st.rerun()

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # ★ Nút tải ZIP (CSV + ảnh bằng chứng)
    has_csv = os.path.isfile(CSV_PATH)
    has_imgs = os.path.isdir(IMG_DIR) and len(os.listdir(IMG_DIR)) > 0
    if has_csv or has_imgs:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            if has_csv:
                zf.write(CSV_PATH, "Danh_Sach_Phat_Nguoi.csv")
            if has_imgs:
                for fname in os.listdir(IMG_DIR):
                    fpath = os.path.join(IMG_DIR, fname)
                    if os.path.isfile(fpath):
                        zf.write(fpath, f"images/{fname}")
        zip_buf.seek(0)
        st.download_button(
            "📦 Tải toàn bộ (CSV + Ảnh)",
            data=zip_buf,
            file_name="KetQua_Vi_Pham.zip",
            mime="application/zip",
            use_container_width=True,
        )
    
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Hướng dẫn nhanh
    st.markdown("#### 📌 Hướng dẫn nhanh")
    st.caption(
        "1️⃣ Upload ảnh hoặc video\n\n"
        "2️⃣ Nhấn **Bắt đầu Xử lý**\n\n"
        "3️⃣ Theo dõi kết quả real-time\n\n"
        "4️⃣ Xem bảng vi phạm bên dưới"
    )

# ══════════════════════════════════════════════════════════
# 6. MAIN PANEL – Hiển thị kết quả
# ══════════════════════════════════════════════════════════

# ── Tiêu đề chính ─────────────────────────────────────────
st.markdown('<p class="main-title">🚦 HỆ THỐNG AI NHẬN DIỆN PHẠT NGUỘI GIAO THÔNG</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Phát hiện vi phạm không đội mũ bảo hiểm · Nhận diện biển số tự động · Lập biên bản tức thì</p>', unsafe_allow_html=True)

# ── Hàng 1: Metrics (placeholder cập nhật real-time) ──────
col_m1, col_m2, col_m3 = st.columns(3)

if "total_vehicles" not in st.session_state:
    st.session_state.total_vehicles = 0
if "total_violations" not in st.session_state:
    st.session_state.total_violations = 0
if "system_status" not in st.session_state:
    st.session_state.system_status = "⏳ Chờ dữ liệu"

metric_vehicles   = col_m1.empty()
metric_violations = col_m2.empty()
metric_status     = col_m3.empty()

metric_vehicles.metric(label="🏍️ Tổng số xe phát hiện", value=st.session_state.total_vehicles)
metric_violations.metric(label="🚨 Số xe vi phạm", value=st.session_state.total_violations)
metric_status.metric(label="📡 Trạng thái hệ thống", value=st.session_state.system_status)

st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

# ── Hàng 2: Khung hiển thị ảnh / video ────────────────────
st.markdown("### 🎥 Khu vực Hiển thị Kết quả")
display_area = st.empty()

# ── Hàng 3: Bảng vi phạm + gallery ────────────────────
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
st.markdown("### 📋 Danh sách Vi phạm Phạt nguội")
table_area = st.empty()

# Hiển thị dữ liệu ban đầu
df_initial = load_web_csv()
with table_area.container():
    render_violation_table_with_gallery(df_initial)

# ══════════════════════════════════════════════════════════
# 7. XỬ LÝ KHI NHẤN NÚT
# ══════════════════════════════════════════════════════════
if btn_process:
    if not uploaded_files:
        st.sidebar.error("⚠️ Vui lòng upload ảnh hoặc video trước!")
        st.stop()

    st.session_state.system_status = "🔄 Đang xử lý…"
    st.session_state.total_vehicles = 0
    st.session_state.total_violations = 0

    # Phân loại file: ảnh vs video
    img_files = [f for f in uploaded_files if os.path.splitext(f.name)[1].lower() in (".png", ".jpg", ".jpeg")]
    vid_files = [f for f in uploaded_files if os.path.splitext(f.name)[1].lower() in (".mp4", ".avi", ".mov")]

    # ────────────────────────────────────────────────────
    # 7A. XỬ LÝ NHIỀU ẢNH
    # ────────────────────────────────────────────────────
    if img_files:
        total_vehicles_all = 0
        total_violations_all = 0

        for idx, uploaded_file in enumerate(img_files):
            st.session_state.system_status = f"🖼️ Đang quét ảnh {idx+1}/{len(img_files)}…"
            metric_status.metric(label="📡 Trạng thái hệ thống", value=f"🖼️ Ảnh {idx+1}/{len(img_files)}")

            tmp_path = save_upload_to_temp(uploaded_file)
            img = cv2.imread(tmp_path)
            if img is None:
                os.unlink(tmp_path)
                continue

            tracker = ViolationTracker()
            result_img, stats = process_logic(
                img, model_s1, model_s2, model_s3,
                WEB_OUTPUT_DIR, tracker, is_video=False
            )

            total_vehicles_all += stats['total_vehicles']
            total_violations_all += stats['violations_this_frame']

            # Cập nhật metrics
            st.session_state.total_vehicles = total_vehicles_all
            st.session_state.total_violations = total_violations_all
            metric_vehicles.metric(label="🏍️ Tổng số xe phát hiện", value=total_vehicles_all)
            metric_violations.metric(label="🚨 Số xe vi phạm", value=total_violations_all)

            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            display_area.image(result_rgb, caption=f"📸 Ảnh {idx+1}/{len(img_files)}: {uploaded_file.name}", use_container_width=True)

            os.unlink(tmp_path)

        # Cập nhật bảng cuối cùng
        st.session_state.system_status = "✅ Hoàn tất"
        metric_status.metric(label="📡 Trạng thái hệ thống", value="✅ Hoàn tất")
        with table_area.container():
            render_violation_table_with_gallery(load_web_csv())
        st.success(f"✅ Đã xử lý xong **{len(img_files)}** ảnh! Phát hiện **{total_vehicles_all}** xe, **{total_violations_all}** vi phạm mới.")

    # ────────────────────────────────────────────────────
    # 7B. XỬ LÝ VIDEO
    # ────────────────────────────────────────────────────
    if vid_files:
        uploaded_file = vid_files[0]  # Chỉ xử lý 1 video
        if len(vid_files) > 1:
            st.warning("⚠️ Hệ thống chỉ hỗ trợ 1 video mỗi lần. Đang xử lý video đầu tiên.")

        st.session_state.system_status = "🎬 Đang quét video…"
        metric_status.metric(label="📡 Trạng thái hệ thống", value="🎬 Đang quét…")

        tmp_path = save_upload_to_temp(uploaded_file)
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            st.error("❌ Không mở được video. Vui lòng kiểm tra file.")
            os.unlink(tmp_path)
            st.stop()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

        # ★ Tạo tracker MỚI + reset YOLO tracker nội bộ
        tracker = ViolationTracker()
        model_s1.predictor = None

        progress_bar = st.progress(0, text="⏳ Chuẩn bị xử lý video…")
        
        if "stop_video" not in st.session_state:
            st.session_state.stop_video = False
        stop_btn_area = st.empty()
        stop_btn_area.button("⏹ Dừng xử lý", on_click=lambda: st.session_state.update(stop_video=True),
                  use_container_width=True, key="stop_video_btn")
        
        frame_idx = 0
        max_vehicles = 0
        total_violations = st.session_state.total_violations  # Kế thừa từ ảnh nếu có

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if st.session_state.stop_video:
                st.warning("⚠️ Đã dừng xử lý video theo yêu cầu!")
                st.session_state.stop_video = False
                break

            frame_idx += 1

            run_full = (frame_idx % SKIP_FRAMES == 0)
            
            if run_full:
                processed, stats = process_logic(
                    frame, model_s1, model_s2, model_s3,
                    WEB_OUTPUT_DIR, tracker, is_video=True, frame_idx=frame_idx
                )
                max_vehicles = max(max_vehicles, stats['total_vehicles'])
                total_violations += stats['violations_this_frame']
            else:
                res_s1 = model_s1.track(frame, persist=True, conf=0.45, verbose=False)[0]
                processed = frame
                if res_s1.boxes is not None:
                    max_vehicles = max(max_vehicles, len(res_s1.boxes))

            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            display_area.image(
                processed_rgb,
                caption=f"🎬 Frame {frame_idx}/{total_frames}  |  FPS gốc: {fps}",
                use_container_width=True,
            )

            st.session_state.total_vehicles = max_vehicles
            st.session_state.total_violations = total_violations
            metric_vehicles.metric(label="🏍️ Tổng số xe phát hiện", value=max_vehicles)
            metric_violations.metric(label="🚨 Số xe vi phạm", value=total_violations)

            if frame_idx % CSV_UPDATE_INTERVAL == 0:
                with table_area.container():
                    render_violation_table_with_gallery(load_web_csv())

            pct = frame_idx / total_frames if total_frames > 0 else 0
            progress_bar.progress(
                min(pct, 1.0),
                text=f"🔄 Đang xử lý frame {frame_idx}/{total_frames} ({pct*100:.1f}%)",
            )

        cap.release()

        finalized = tracker.finalize(WEB_OUTPUT_DIR)
        total_violations += finalized
        st.session_state.total_violations = total_violations

        progress_bar.progress(1.0, text="✅ Hoàn tất xử lý video!")
        stop_btn_area.empty()
        
        st.session_state.system_status = "✅ Hoàn tất"
        metric_violations.metric(label="🚨 Số xe vi phạm", value=total_violations)
        metric_status.metric(label="📡 Trạng thái hệ thống", value="✅ Hoàn tất")

        with table_area.container():
            render_violation_table_with_gallery(load_web_csv())

        os.unlink(tmp_path)
        st.success(f"✅ Đã xử lý xong video! **{frame_idx}** frames, **{total_violations}** vi phạm.")

    if not img_files and not vid_files:
        st.error("❌ Định dạng file không được hỗ trợ.")

# ══════════════════════════════════════════════════════════
# 8. FOOTER
# ══════════════════════════════════════════════════════════
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:0.8rem; padding:16px 0;">
        🚦 <strong>AI Phạt Nguội Giao Thông</strong> · Phiên bản 2.0 · 
        Powered by YOLOv8 &amp; Streamlit · © 2026
    </div>
    """,
    unsafe_allow_html=True,
)
