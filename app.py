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
import time
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# ── Import pipeline & tracker từ project ──────────────────
from main_pipeline import process_logic
from core.tracking_engine import ViolationTracker

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
    s1 = YOLO("models/stage1.pt")   # Stage 1: Phát hiện xe / người lái
    s2 = YOLO("models/stage2.pt")   # Stage 2: Phát hiện mũ, biển số
    s3 = YOLO("models/stage3.pt")   # Stage 3: OCR – đọc ký tự biển số
    return s1, s2, s3

model_s1, model_s2, model_s3 = load_models()

# ══════════════════════════════════════════════════════════
# 2. ĐƯỜNG DẪN OUTPUT CỐ ĐỊNH
# ══════════════════════════════════════════════════════════
OUTPUT_DIR  = "test_outputs"
CSV_PATH    = os.path.join(OUTPUT_DIR, "Danh_Sach_Phat_Nguoi.csv")
os.makedirs(os.path.join(OUTPUT_DIR, "Bang_Chung"), exist_ok=True)

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

def load_csv():
    """Đọc file CSV danh sách phạt nguội (nếu tồn tại)."""
    if os.path.isfile(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH, encoding="utf-8")
            return df
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


def count_vehicles_in_frame(frame, model):
    """Đếm nhanh số xe (bounding box) trong 1 frame bằng Stage-1."""
    try:
        res = model.predict(frame, conf=0.5, verbose=False)[0]
        if res.boxes is not None:
            return len(res.boxes)
    except Exception:
        pass
    return 0

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

    # Upload file
    st.markdown("#### 📂 Tải lên Ảnh / Video")
    uploaded_file = st.file_uploader(
        "Kéo thả hoặc chọn file",
        type=["png", "jpg", "jpeg", "mp4", "avi", "mov"],
        help="Hỗ trợ ảnh (PNG, JPG) và video (MP4, AVI, MOV)",
    )

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Nút xử lý
    btn_process = st.button("🚀 Bắt đầu Xử lý", use_container_width=True)

    # Thông tin thêm
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
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

# ── Hàng 1: Metrics ───────────────────────────────────────
col_m1, col_m2, col_m3 = st.columns(3)

# Khởi tạo session_state nếu chưa có
if "total_vehicles" not in st.session_state:
    st.session_state.total_vehicles = 0
if "total_violations" not in st.session_state:
    st.session_state.total_violations = 0
if "system_status" not in st.session_state:
    st.session_state.system_status = "⏳ Chờ dữ liệu"

with col_m1:
    st.metric(label="🏍️ Tổng số xe phát hiện", value=st.session_state.total_vehicles)
with col_m2:
    st.metric(label="🚨 Số xe vi phạm", value=st.session_state.total_violations)
with col_m3:
    st.metric(label="📡 Trạng thái hệ thống", value=st.session_state.system_status)

st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

# ── Hàng 2: Khung hiển thị ảnh / video (placeholder) ──────
st.markdown("### 🎥 Khu vực Hiển thị Kết quả")
display_area = st.empty()

# ── Hàng 3: Bảng vi phạm (placeholder) ────────────────────
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
st.markdown("### 📋 Danh sách Vi phạm Phạt nguội")
table_area = st.empty()

# Hiển thị dữ liệu CSV ban đầu (nếu đã có sẵn)
df_initial = load_csv()
if not df_initial.empty:
    table_area.dataframe(df_initial, use_container_width=True, hide_index=True)
else:
    table_area.info("Chưa có dữ liệu vi phạm. Hãy upload file và bắt đầu xử lý.")

# ══════════════════════════════════════════════════════════
# 7. XỬ LÝ KHI NHẤN NÚT
# ══════════════════════════════════════════════════════════
if btn_process:
    # Kiểm tra đã upload file chưa
    if uploaded_file is None:
        st.sidebar.error("⚠️ Vui lòng upload ảnh hoặc video trước!")
        st.stop()

    # Lưu file upload vào thư mục tạm
    tmp_path = save_upload_to_temp(uploaded_file)
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    # Cập nhật trạng thái
    st.session_state.system_status = "🔄 Đang xử lý…"
    st.session_state.total_vehicles = 0
    st.session_state.total_violations = 0

    # ────────────────────────────────────────────────────
    # 7A. XỬ LÝ ẢNH
    # ────────────────────────────────────────────────────
    if file_ext in (".png", ".jpg", ".jpeg"):
        st.session_state.system_status = "🖼️ Đang quét ảnh…"

        img = cv2.imread(tmp_path)
        if img is None:
            st.error("❌ Không đọc được ảnh. Vui lòng kiểm tra file.")
            st.stop()

        # Đếm số xe trước khi xử lý
        vehicle_count = count_vehicles_in_frame(img, model_s1)
        st.session_state.total_vehicles = vehicle_count

        # Đếm số vi phạm trước xử lý (đọc CSV trước)
        df_before = load_csv()
        rows_before = len(df_before) if not df_before.empty else 0

        # Tạo tracker mới cho ảnh đơn lẻ
        tracker = ViolationTracker()

        # Gọi pipeline xử lý
        result_img = process_logic(img, model_s1, model_s2, model_s3, OUTPUT_DIR, tracker)

        # Chuyển BGR → RGB để hiển thị đúng màu trên Streamlit
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        # Hiển thị ảnh kết quả
        display_area.image(result_rgb, caption="📸 Kết quả phân tích ảnh", use_container_width=True)

        # Cập nhật số vi phạm mới
        df_after = load_csv()
        rows_after = len(df_after) if not df_after.empty else 0
        new_violations = rows_after - rows_before
        st.session_state.total_violations = new_violations

        # Cập nhật bảng vi phạm
        if not df_after.empty:
            table_area.dataframe(df_after, use_container_width=True, hide_index=True)

        st.session_state.system_status = "✅ Hoàn tất"

        # Dọn file tạm
        os.unlink(tmp_path)

        st.success(f"✅ Đã xử lý xong ảnh! Phát hiện **{vehicle_count}** xe, **{new_violations}** vi phạm mới.")

    # ────────────────────────────────────────────────────
    # 7B. XỬ LÝ VIDEO (real-time từng frame)
    # ────────────────────────────────────────────────────
    elif file_ext in (".mp4", ".avi", ".mov"):
        st.session_state.system_status = "🎬 Đang quét video…"

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            st.error("❌ Không mở được video. Vui lòng kiểm tra file.")
            st.stop()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

        # Đếm vi phạm trước xử lý
        df_before = load_csv()
        rows_before = len(df_before) if not df_before.empty else 0

        # Tạo tracker mới cho video
        tracker = ViolationTracker()

        # Thanh tiến trình
        progress_bar = st.progress(0, text="⏳ Chuẩn bị xử lý video…")
        frame_idx = 0
        cumulative_vehicles = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Đếm xe trong frame hiện tại
            v_count = count_vehicles_in_frame(frame, model_s1)
            cumulative_vehicles = max(cumulative_vehicles, v_count)

            # Gọi pipeline chính
            processed = process_logic(frame, model_s1, model_s2, model_s3, OUTPUT_DIR, tracker)

            # Chuyển BGR → RGB
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

            # Hiển thị frame real-time
            display_area.image(
                processed_rgb,
                caption=f"🎬 Frame {frame_idx}/{total_frames}  |  FPS gốc: {fps}",
                use_container_width=True,
            )

            # Cập nhật metrics
            st.session_state.total_vehicles = cumulative_vehicles

            # Cập nhật bảng vi phạm liên tục
            df_live = load_csv()
            if not df_live.empty:
                rows_now = len(df_live)
                st.session_state.total_violations = rows_now - rows_before
                table_area.dataframe(df_live, use_container_width=True, hide_index=True)

            # Cập nhật thanh tiến trình
            pct = frame_idx / total_frames if total_frames > 0 else 0
            progress_bar.progress(
                min(pct, 1.0),
                text=f"🔄 Đang xử lý frame {frame_idx}/{total_frames} ({pct*100:.1f}%)",
            )

        cap.release()
        progress_bar.progress(1.0, text="✅ Hoàn tất xử lý video!")

        # Cập nhật trạng thái cuối cùng
        df_final = load_csv()
        final_violations = (len(df_final) - rows_before) if not df_final.empty else 0
        st.session_state.total_violations = final_violations
        st.session_state.system_status = "✅ Hoàn tất"

        if not df_final.empty:
            table_area.dataframe(df_final, use_container_width=True, hide_index=True)

        # Dọn file tạm
        os.unlink(tmp_path)

        st.success(f"✅ Đã xử lý xong video! **{frame_idx}** frames, **{final_violations}** vi phạm mới.")

    else:
        st.error("❌ Định dạng file không được hỗ trợ.")

# ══════════════════════════════════════════════════════════
# 8. FOOTER
# ══════════════════════════════════════════════════════════
st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:0.8rem; padding:16px 0;">
        🚦 <strong>AI Phạt Nguội Giao Thông</strong> · Phiên bản 1.0 · 
        Powered by YOLOv8 &amp; Streamlit · © 2026
    </div>
    """,
    unsafe_allow_html=True,
)
