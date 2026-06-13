# 🚦 AI Phạt Nguội Giao Thông — Nhận diện vi phạm không đội mũ bảo hiểm

Hệ thống AI sử dụng **3 model YOLO** theo pipeline 3 giai đoạn để tự động phát hiện người đi xe máy không đội mũ bảo hiểm, nhận diện biển số xe và lập biên bản vi phạm.

## 📋 Kiến trúc Pipeline

```
Ảnh / Video đầu vào
      │
      ▼
┌─────────────────┐
│  Stage 1 (YOLO) │  → Phát hiện xe máy / người lái
│   stage1.pt     │
└────────┬────────┘
         │ Crop từng xe
         ▼
┌─────────────────┐
│  Stage 2 (YOLO) │  → Nhận diện: mũ bảo hiểm / không mũ / biển số
│   stage2.pt     │
└────────┬────────┘
         │ Crop biển số
         ▼
┌─────────────────┐
│  Stage 3 (YOLO) │  → OCR: đọc ký tự trên biển số
│   stage3.pt     │
└────────┬────────┘
         │
         ▼
  📋 Ghi CSV + 📸 Lưu ảnh bằng chứng
```

## 🗂️ Cấu trúc thư mục

```
TTCS_AI_Helmet/
├── src/                          # 💻 Toàn bộ Source Code
│   ├── api/                      # Tầng API & Server
│   │   ├── __init__.py
│   │   └── server.py             # Flask API router
│   ├── engine/                   # Tầng AI & Xử lý 
│   │   ├── __init__.py
│   │   ├── pipeline.py           # Logic pipeline chính
│   │   └── core/                 # Các module thuật toán
│   │       ├── __init__.py
│   │       ├── image_utils.py
│   │       ├── ocr_engine.py
│   │       ├── tracking_engine.py
│   │       └── logger.py
│   ├── frontend/                 # Tầng Giao diện (HTML5, JS)
│   │   ├── index.html
│   │   └── app.js
│   └── config.py                 # Cấu hình tham số chung
│
├── data/                         # 📁 Quản lý Dữ liệu in/out
│   ├── models/                   # Trọng số YOLO (*.pt)
│   ├── inputs/                   # Ảnh/Video đầu vào để test CLI
│   └── outputs/                  # Nơi xuất kết quả
│       ├── cli_results/          # File xuất ra khi chạy CLI
│       └── web_results/          # File xuất ra khi chạy Web
│
├── run_server.py                 # 🚀 Entry point chạy Web
├── run_cli.py                    # 🚀 Entry point chạy Terminal
├── requirements.txt
└── README.md
```

## 🚀 Cài đặt & Chạy

### 1. Cài đặt môi trường

```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 2. Tải model

Tải 3 file model từ Google Drive và đặt vào thư mục `data/models/`:
- `stage1.pt` — Phát hiện xe máy (~158 MB)
- `stage2.pt` — Nhận diện mũ & biển số (~18 MB)
- `stage3.pt` — OCR ký tự biển số (~20 MB)

### 3. Chạy giao diện Web (Flask) ⭐ Khuyên dùng

```bash
python run_server.py
```

Mở trình duyệt tại **http://localhost:5000**

Giao diện bao gồm:
- **Trang chủ**: Chế độ Camera giả lập real-time + Chế độ Upload ảnh/video
- **Lịch sử Vi phạm**: Bảng dữ liệu, Biểu đồ thống kê, Xem ảnh bằng chứng, Lọc/Tải/Xóa
- **Thông báo Toast**: Cảnh báo vi phạm mới theo thời gian thực

### 4. Chạy giao diện Web (Streamlit) — Phiên bản cũ

```bash
streamlit run app.py
```

Mở trình duyệt tại **http://localhost:8501**

### 5. Chạy bằng Terminal (CLI)

Đặt ảnh/video vào `data/inputs/` rồi chạy:

```bash
python run_cli.py
```

Kết quả sẽ lưu vào `data/outputs/cli_results/`

## ⚙️ Yêu cầu hệ thống

- Python 3.10+
- GPU (khuyến nghị) hoặc CPU
- RAM tối thiểu 8 GB
- Dung lượng ổ cứng: ~200 MB cho model

## 📄 Giấy phép

Dự án phục vụ mục đích học tập — Thực tập cơ sở (TTCS) 2026.
