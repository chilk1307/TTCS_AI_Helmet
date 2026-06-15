# Hệ thống Giám sát Giao thông AI - Realtime

Demo hệ thống phát hiện vi phạm giao thông (xe máy không đội mũ bảo hiểm) từ video, sử dụng AI/Deep Learning.

## Tính năng

- **Phát hiện xe máy realtime** (YOLO Stage 1)
- **Kiểm tra mũ bảo hiểm** (YOLO Stage 2)
- **OCR biển số xe** (YOLO Stage 3 - character detection)
- **Dashboard admin tiếng Việt** với video stream realtime
- **Lưu trữ vi phạm** (Excel + ảnh minh chứng)
- **Chống ghi trùng** (dedup theo biển số + thời gian)
- **ROI tối ưu** (chỉ detect vùng đường phía dưới)

## Kiến trúc

```
Pipeline: Video → Frame Queue → Stage1 (detect xe máy)
                                    ├── Nhánh A: Overlay bbox → WebSocket → UI (realtime)
                                    └── Nhánh B: Stage2 → Stage3 → OCR → Lưu vi phạm (background)
```

## Cách chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy server
python run.py
```

Server sẽ chạy tại: http://localhost:8000

## Đăng nhập

- Username: `1`
- Password: `1`

## API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|--------|
| POST | /api/auth/login | Đăng nhập |
| POST | /api/videos/upload | Upload video |
| POST | /api/videos/{id}/start | Bắt đầu phân tích |
| POST | /api/videos/{id}/stop | Dừng phân tích |
| GET | /api/violations | Danh sách vi phạm |
| GET | /api/violations/{id}/evidence/{type} | Ảnh minh chứng |
| GET | /api/status/{video_id} | Trạng thái pipeline |
| WS | /ws/stream/{video_id} | Stream video realtime |

## Cấu trúc thư mục

```
app/
├── main.py              # FastAPI app
├── core/
│   ├── config.py        # Cấu hình toàn bộ
│   └── logger.py        # Logging
├── api/
│   ├── auth.py          # Xác thực
│   ├── video.py         # Upload & quản lý video
│   ├── violations.py    # Danh sách vi phạm
│   └── webrtc.py        # WebSocket stream + status
├── services/
│   ├── inference/
│   │   ├── pipeline.py          # Pipeline manager
│   │   ├── stage1_motorbike_detector.py
│   │   ├── stage2_part_detector.py
│   │   ├── stage3_helmet_ocr.py
│   │   ├── model_loader.py
│   │   └── types.py
│   ├── video/
│   │   ├── reader.py
│   │   └── frame_utils.py
│   ├── storage/
│   │   ├── sheet_writer.py
│   │   └── evidence_writer.py
│   └── realtime/
│       ├── fps_counter.py
│       └── queues.py
├── frontend/
│   ├── templates/
│   │   ├── index.html       # Trang login
│   │   └── dashboard.html   # Dashboard chính
│   └── static/
│       ├── css/style.css
│       └── js/app.js
models/
├── stage1/detect_vehicle.pt
├── stage2/mu_bien_so_stage2.pt
└── stage3/ocr_plate.pt
storage/
├── uploads/
├── evidence/
└── violations/violations.xlsx
```

## Yêu cầu hệ thống

- Python 3.10+
- GPU NVIDIA (CUDA) để chạy model
- RAM 16GB+

## Cấu hình

Các tham số cấu hình nằm trong `app/core/config.py`:

- `STAGE1_CONF`: Confidence threshold Stage 1 (mặc định 0.4)
- `STAGE2_CONF`: Confidence threshold Stage 2 (mặc định 0.72)
- `STAGE3_CONF`: Confidence threshold Stage 3/OCR (mặc định 0.25)
- `TARGET_FPS`: FPS mục tiêu (mặc định 30)
- `ENABLE_ROI`: Bật/tắt ROI (mặc định True)
- `ROI_Y_START_RATIO`: Tỉ lệ bắt đầu quét Y (mặc định 0.45)
- `DEDUP_SECONDS`: Thời gian dedup (mặc định 10s)
- `QUEUE_MAX_SIZE`: Kích thước queue tối đa (mặc định 50)

## Tải model

Model không nằm trên GitHub. Tải từ Hugging Face:

```bash
bash scripts/download_models.sh
```

Repo model: https://huggingface.co/2vhoc/helmet-detection-traffic

## Dataset training (legacy)

Thư mục `dataset_plate/`, `helmet.yaml`, `plate.yaml` — dữ liệu huấn luyện YOLO gốc.
