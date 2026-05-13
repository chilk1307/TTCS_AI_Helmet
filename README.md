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
├── app.py                 # Giao diện Web (Streamlit)
├── main_pipeline.py       # Pipeline xử lý chính + CLI
├── core/
│   ├── image_utils.py     # Tiền xử lý ảnh (CLAHE, deskew)
│   ├── logger.py          # Ghi CSV + lưu ảnh bằng chứng
│   ├── ocr_engine.py      # Đọc ký tự biển số (YOLO OCR)
│   └── tracking_engine.py # Chống lặp ID video
├── models/                # Chứa 3 file .pt (tải riêng)
├── test_inputs/           # Ảnh/video đầu vào test
├── test_outputs/          # Kết quả từ CLI (terminal)
└── outputs/               # Kết quả từ Web UI
    ├── reports/            # CSV danh sách vi phạm
    └── images/             # Ảnh bằng chứng vi phạm
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

Tải 3 file model từ Google Drive và đặt vào thư mục `models/`:
- `stage1.pt` — Phát hiện xe máy (~158 MB)
- `stage2.pt` — Nhận diện mũ & biển số (~18 MB)
- `stage3.pt` — OCR ký tự biển số (~20 MB)

### 3. Chạy giao diện Web

```bash
streamlit run app.py
```

Mở trình duyệt tại **http://localhost:8501**

### 4. Chạy bằng Terminal (CLI)

Đặt ảnh/video vào `test_inputs/` rồi chạy:

```bash
python main_pipeline.py
```

Kết quả sẽ lưu vào `test_outputs/`

## ⚙️ Yêu cầu hệ thống

- Python 3.10+
- GPU (khuyến nghị) hoặc CPU
- RAM tối thiểu 8 GB
- Dung lượng ổ cứng: ~200 MB cho model

## 📄 Giấy phép

Dự án phục vụ mục đích học tập — Thực tập cơ sở (TTCS) 2026.
