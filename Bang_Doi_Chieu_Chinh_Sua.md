# BẢNG ĐỐI CHIẾU CHI TIẾT TỪNG CHỮ CÁC PHẦN CHỈNH SỬA

Dưới đây là đối chiếu y hệt nguyên văn giữa **Bản cũ** và **Bản mới** để bạn dễ dàng kiểm tra. Toàn bộ phần Training (Chương 3) và các phần khác không có trong danh sách này đều được giữ nguyên 100%.

## Mục 2.6. Công nghệ xây dựng giao diện ứng dụng

### ❌ BẢN CŨ (Trong file `Báo cáo TTCS - Nhóm 6.md`)
```text
## 2.6. Công nghệ xây dựng giao diện ứng dụng (Streamlit)

Để đóng gói hệ thống AI thành một sản phẩm phần mềm hoàn chỉnh (Dashboard), nhóm sử dụng **Streamlit** – một Framework mã nguồn mở dựa trên Python.

- Khác với các công nghệ Frontend truyền thống (HTML/JS) đòi hỏi kiến trúc API phức tạp, Streamlit cho phép render (kết xuất) giao diện và xử lý luồng dữ liệu thời gian thực (Real-time Video Streaming) ngay trong cùng một tiến trình Python.
- Streamlit hỗ trợ tải file, quản lý luồng ảnh qua OpenCV, và cập nhật tự động bảng dữ liệu Pandas DataFrame, cực kỳ phù hợp để xây dựng các hệ thống giám sát và báo cáo trực quan cho bài toán Computer Vision.
```

### ✅ BẢN MỚI (Trong file `Báo_cáo_TTCS_Bản_Mới.md`)
```text
## 2.6. Kiến trúc Client-Server với Flask và Tailwind CSS

Để đóng gói hệ thống AI thành một sản phẩm phần mềm hoàn chỉnh, nhóm sử dụng mô hình Client-Server hiện đại, thay thế cho các framework tích hợp nguyên khối nhằm đạt hiệu năng tối đa:

- **Backend (Flask):** Đóng vai trò là Máy chủ ứng dụng. Flask cung cấp một Web Server gọn nhẹ, xử lý đa luồng (Multi-threading). Cốt lõi của kiến trúc này là sử dụng kỹ thuật `Multipart/x-mixed-replace` để truyền luồng Video theo thời gian thực (Video Streaming) từ OpenCV thẳng lên trình duyệt web mà không gây nghẽn cổ chai. Đồng thời cung cấp các API JSON để giao tiếp dữ liệu.
- **Frontend (HTML/JS & Tailwind CSS):** Giao diện được thiết kế độc lập. Tailwind CSS mang lại một bộ giao diện (Dashboard) giám sát an ninh cực kỳ chuyên nghiệp và mượt mà. JavaScript (AJAX) được sử dụng để gọi API cập nhật các thông số (Số xe vi phạm, Tổng xe) theo thời gian thực (Real-time) mà không cần tải lại trang (No-reload), khắc phục triệt để hiện tượng giật lag màn hình của các công nghệ cũ.
```

---

## Mục 4.3. Tổ chức không gian mã nguồn

### ❌ BẢN CŨ (Trong file `Báo cáo TTCS - Nhóm 6.md`)
```text
## 4.3. Tổ chức không gian mã nguồn (Source Code Workspace)

Dựa trên bản vẽ kiến trúc 3 lớp đã thiết kế ở mục 4.2, toàn bộ mã nguồn của dự án được quy hoạch thành các thư mục chuyên biệt. Việc tổ chức cấu trúc tuân thủ nghiêm ngặt nguyên tắc mã sạch (Clean Code), tách biệt rõ ràng giữa cấu hình, giao diện và logic xử lý.

Dưới đây là sơ đồ cây thư mục (Directory Tree) chính thức của dự án, làm tiền đề cho việc lập trình chi tiết ở Chương 5:

**_Hình 4.3: Cấu trúc thư mục của hệ thống._**
```

### ✅ BẢN MỚI (Trong file `Báo_cáo_TTCS_Bản_Mới.md`)
```text
## 4.3. Tổ chức không gian mã nguồn (Source Code Workspace)

Dựa trên bản vẽ kiến trúc 3 lớp đã thiết kế ở mục 4.2, toàn bộ mã nguồn của dự án được quy hoạch thành các thư mục chuyên biệt. Việc tổ chức cấu trúc tuân thủ nghiêm ngặt nguyên tắc mã sạch (Clean Code), tách biệt rõ ràng giữa cấu hình, giao diện và logic xử lý.

```text
data/                      # Lưu trữ dữ liệu hệ thống
├── inputs/                # Ảnh tĩnh đầu vào cho CLI
├── camera_sample/         # Video mẫu để quét Web
├── models/                # Chứa 3 tệp trọng số YOLO26 (.pt)
└── outputs/               # Kết quả xuất ra (Ảnh & CSV)
    ├── cli_results/       # Chạy từ dòng lệnh
    └── web_results/       # Chạy từ giao diện Web

src/                       # Mã nguồn hệ thống
├── api/                   # Phân hệ Web (Lớp Giao diện)
│   ├── server.py          # Máy chủ Flask & Điều phối API
│   ├── static/            # CSS, JavaScript, Asset (Tailwind)
│   └── templates/         # HTML Dashboard
├── engine/                # Phân hệ Cốt lõi (Lớp Xử lý)
│   ├── core/              # Các Module Logic tùy chỉnh
│   │   ├── image_utils.py    # Xử lý OpenCV, Khử lóa, Nắn thẳng
│   │   ├── logger.py         # Quản lý File, xuất CSV đa môi trường
│   │   ├── ocr_engine.py     # Luật bóc tách ký tự, NMS
│   │   └── tracking_engine.py# Quản lý ID, Tracking, Hybrid Voting
│   └── pipeline.py        # Kịch bản nối 3 luồng YOLO
└── config.py              # Thông số cấu hình tập trung

run_cli.py                 # File khởi chạy chế độ Terminal
run_server.py              # File khởi chạy chế độ Web Dashboard
```
```

---

## Mục 5.1.1. Tối ưu hóa và Quản lý cấu hình tập trung

### ❌ BẢN CŨ (Trong file `Báo cáo TTCS - Nhóm 6.md`)
```text
### 5.1.1. Tối ưu hóa và Quản lý cấu hình tập trung (config.py)

Trong các hệ thống phần mềm chuyên nghiệp, việc gán cứng (hard-code) các tham số rải rác khắp nơi là một "anti-pattern" (mẫu thiết kế lỗi) gây rủi ro lớn khi vận hành. Do đó, hệ thống đã gom toàn bộ các hằng số, ngưỡng AI và thông số không gian vào tệp config.py.

Sự tinh tế trong việc triển khai cấu hình được thể hiện qua các thiết lập chuyên sâu:

- **Tách biệt ngưỡng tin cậy (Confidence Threshold):** Mặc dù ngưỡng chung để nhận diện xe máy và vùng mũ được đặt ở mức an toàn là STAGE1_CONF = 0.45, nhưng riêng trạng thái vi phạm (Không đội mũ) lại được định nghĩa bằng một hằng số khắt khe hơn: NOHELMET_MIN_CONF = 0.60. Cải tiến này là chốt chặn quan trọng giúp AI tránh được hiện tượng False Positive (nhận diện nhầm tóc đen dày hoặc mũ lưỡi trai thành trạng thái không đội mũ).
- **Quy hoạch không gian tĩnh và Tiền xử lý OCR:** Các thông số như kích thước tối thiểu của biển số (MIN_PLATE_WIDTH = 30) và hệ số phóng to (OCR_UPSCALE_FACTOR = 2) được định nghĩa rõ ràng. Điều này giúp bộ điều phối tự động lọc bỏ các "vật thể rác" ở quá xa camera, đồng thời nội suy tăng cường độ nét cho ảnh trước khi đưa vào mô hình bóc tách chữ (Stage 3).
- **Tối ưu hóa hiệu năng (Performance Tuning):** Tệp cấu hình định nghĩa các tham số SKIP_FRAMES = 2 và CSV_UPDATE_INTERVAL = 15. Trong xử lý luồng Video, việc không phải render lại giao diện đồ họa ở mọi khung hình giúp tiết kiệm tài nguyên CPU/GPU, đảm bảo hệ thống duy trì được chỉ số FPS (Frames per Second) ổn định.
```

### ✅ BẢN MỚI (Trong file `Báo_cáo_TTCS_Bản_Mới.md`)
```text
### 5.1.1. Tối ưu hóa và Quản lý cấu hình tập trung (config.py)

Trong các hệ thống phần mềm chuyên nghiệp, việc gán cứng (hard-code) các tham số rải rác khắp nơi là một "anti-pattern" gây rủi ro lớn. Toàn bộ các hằng số, ngưỡng AI và thông số không gian được gom vào tệp `src/config.py`:

- **Tách biệt ngưỡng tin cậy (Confidence Threshold):** Mặc dù ngưỡng chung để nhận diện xe máy được đặt ở mức `STAGE1_CONF = 0.45`, trạng thái vi phạm (Không đội mũ) lại được định nghĩa khắt khe hơn: `NOHELMET_MIN_CONF = 0.60`. Cải tiến này là chốt chặn quan trọng giúp AI tránh được hiện tượng False Positive (nhận diện nhầm tóc đen dày thành không đội mũ).
- **Thiết lập Vùng Nhận Diện (Detection Zone):** Nhằm tối ưu hóa phần cứng, hệ thống không quét toàn bộ khung hình mà chỉ tập trung vào vùng trung tâm `ZONE_Y_MIN_RATIO = 0.30` đến `ZONE_Y_MAX_RATIO = 0.80`. Việc này giúp loại bỏ xe quá xa (mờ, khó đọc), chỉ phân tích khi phương tiện vào vùng rõ nét nhất, tăng vọt FPS cho máy tính.
- **Cấu hình Cơ chế Lai (Hybrid Rules):** Cấu hình `MIN_VIOLATION_FRAMES = 3` (Tối thiểu 3 frame vi phạm để khóa khung đỏ hiển thị) và `MIN_VIOLATION_RATIO = 0.40` (Tỷ lệ 40% frame không mũ để chốt hạ phạt nguội). Đây là thông số vàng giúp hệ thống triệt tiêu gần như 100% độ nhiễu.
- **Quy hoạch không gian OCR:** Kích thước tối thiểu `MIN_PLATE_WIDTH = 30` và hệ số phóng to `OCR_UPSCALE_FACTOR = 2` giúp bộ điều phối tự động lọc bỏ các "vật thể rác" ở xa, nội suy nét ảnh trước khi bóc tách chữ.
```

---

## Mục 5.2.3. Cấu trúc Bám vết trì hoãn

### ❌ BẢN CŨ (Trong file `Báo cáo TTCS - Nhóm 6.md`)
```text
### 5.2.3. Cấu trúc Bám vết trì hoãn và Quản lý file (tracking_engine.py & logger.py)

Đây là module cốt lõi biến hệ thống từ một "AI nhận diện khung hình" thành một "Phần mềm phạt nguội" thực thụ, giải quyết triệt để lỗi "bùng nổ dữ liệu" (Data Spamming) đã phân tích ở Chương 4.

- **Cơ chế Ghi log trì hoãn (Deferred Logging):** Thay vì vội vã ghi biên bản ngay ở khung hình đầu tiên phát hiện vi phạm (khi xe còn ở xa, biển số mờ), lớp đối tượng ViolationTracker sử dụng chiến lược "nuôi ID".
    - Hệ thống liên tục lưu trữ định danh track_id vào bộ nhớ đệm. Ở mỗi khung hình tiếp theo, nếu thuật toán tìm thấy một bức ảnh cắt có số lượng ký tự biển số dài hơn và rõ ràng hơn (new_text_len), nó sẽ tự động ghi đè bằng chứng tốt nhất vào biến tạm.
    - Hành động xuất file biên bản thực sự chỉ được kích hoạt (Flush) khi phương tiện đó đi khuất khỏi camera (Vượt quá ngưỡng LOST_ID_THRESHOLD).
- **Định tuyến đa môi trường và Tính toàn vẹn dữ liệu:** Hàm log_violation trong logger.py có khả năng tự động phân tích cấu trúc thư mục (os.path.isdir) để nhận diện môi trường khởi chạy (Web UI hay Terminal), từ đó định tuyến ảnh và file .csv về đúng thư mục đích.
- **Chống xung đột đa luồng (Thread-safety):** Để tránh hiện tượng xung đột dữ liệu khi có 2 phương tiện vi phạm cùng một lúc, hàm datetime.now() được mã hóa định dạng thời gian gắn kèm hậu tố _microsecond_ (%Y%m%d_%H%M%S_%f). Điều này đảm bảo mỗi tệp tin bằng chứng đều sở hữu một mã định danh duy nhất (Unique Timestamp), loại bỏ hoàn toàn nguy cơ ghi đè dữ liệu kế thừa.
```

### ✅ BẢN MỚI (Trong file `Báo_cáo_TTCS_Bản_Mới.md`)
```text
### 5.2.3. Cấu trúc Bám vết trì hoãn và Quản lý file (tracking_engine.py & logger.py)

Đây là module cốt lõi biến hệ thống từ một "AI nhận diện thô" thành một "Phần mềm phạt nguội" cực kỳ thông minh, giải quyết triệt để lỗi "bùng nổ dữ liệu" (Data Spamming) ở mô hình Baseline.

- **Cơ chế Bỏ phiếu Lai (Hybrid Voting Mechanism):** Để chống lại việc lóa camera trong 1-2 khung hình làm sai lệch kết quả, lớp `ViolationTracker` sử dụng cơ chế đếm tỷ lệ khắt khe:
  - *Tầng Giao diện (UI Lock):* Hệ thống chỉ vẽ khung màu ĐỎ cảnh báo lên màn hình khi chiếc xe bị bắt lỗi đủ 3 lần (`violation_votes >= 3`).
  - *Tầng Ghi sổ (Post-processing):* Khi xe đi khuất khỏi vùng nhận diện, máy tính tiến hành tính tỷ lệ `Ratio = Số khung hình không mũ / Tổng số khung hình nhìn rõ đầu`. Nếu tỷ lệ này `< 40%`, hệ thống lặng lẽ xóa hồ sơ (coi như nhận diện nhầm). Nếu `>= 40%`, lúc đó biên bản mới chính thức được lập. Cơ chế này đạt độ chuẩn xác tuyệt đối về mặt toán học bất chấp tốc độ khung hình (FPS) của Video gốc là 30 hay 60.
- **Cơ chế Lưu trữ bằng chứng xuất sắc:** Hệ thống luôn săn lùng bức ảnh tốt nhất. Ở mỗi khung hình, nó so sánh độ dài chuỗi ký tự và diện tích vùng biển số. Bức ảnh nào có biển số to nhất và chữ đọc đầy đủ nhất sẽ được ghi đè vào biến tạm làm bằng chứng pháp lý.
- **Định tuyến Đa Môi Trường (Multi-environment I/O):** Hàm `log_violation` trong `logger.py` được thiết kế linh hoạt để hỗ trợ hai chế độ khởi chạy:
  - Khi chạy ngầm từ Terminal (`run_cli.py`): Biên bản và ảnh đẩy về `data/outputs/cli_results`.
  - Khi chạy từ Web (`run_server.py`): Dữ liệu đồng bộ vào `data/outputs/web_results`. 
  Tất cả ảnh bằng chứng đều được mã hóa tên file kèm theo `microsecond` (Ví dụ: `ViPham_43D112345_20260518_101530_456789.jpg`), triệt tiêu hoàn toàn nguy cơ ghi đè khi 2 xe cùng vi phạm ở một giây.
```

---

## Mục 5.3. Triển khai Giao diện Người dùng

### ❌ BẢN CŨ (Trong file `Báo cáo TTCS - Nhóm 6.md`)
```text
## 5.3. Triển khai Giao diện Người dùng (Web UI) với Streamlit

Để biến các thuật toán AI phức tạp thành một sản phẩm phần mềm giám sát an ninh thân thiện, toàn bộ khối xử lý lõi (Backend) được móc nối với Giao diện Người dùng (Frontend) thông qua tệp app.py. Nhóm nghiên cứu lựa chọn **Streamlit** – một framework Python mạnh mẽ – nhằm đồng bộ hóa luồng video thời gian thực lên trình duyệt mà không cần thiết lập kiến trúc API (Client-Server) rườm rà.

### 5.3.1. Thiết kế Bố cục và Trải nghiệm Người dùng (UI/UX Design)

Giao diện được thiết lập ở chế độ mở rộng, lấy cảm hứng từ các bảng điều khiển (Dashboard) giám sát an ninh chuyên nghiệp. Nhằm thoát khỏi giao diện mặc định nhàm chán của Streamlit, mã nguồn đã áp dụng kỹ thuật **CSS Injection**:

- Sử dụng thẻ &lt;style&gt; để ghi đè (override) các thuộc tính hiển thị: Tạo hiệu ứng chữ phát sáng (Glowing Text) cho tiêu đề, thiết kế các dải phân cách (Divider) tinh tế và ẩn đi menu hệ thống mặc định nhằm mang lại cảm giác của một phần mềm độc lập.

Bố cục được chia làm 3 phân hệ tương tác chính:

- **Thanh điều hướng và Cấu hình (Sidebar):** Nằm ở bên trái, chứa module st.file_uploader cho phép linh hoạt tải lên ảnh (.jpg, .png) hoặc video (.mp4). Tích hợp nút điều khiển luồng (Bắt đầu / Dừng xử lý).
- **Khu vực Giám sát (Live-view & Metrics):** Ở trung tâm, bao gồm màn hình phát luồng video đang được AI vẽ Bounding Box. Phía trên là bộ đếm số liệu động (Live Metrics) hiển thị: _Tổng số xe_, _Số xe vi phạm_ và _Trạng thái hệ thống_.
- **Phân hệ Thống kê (Data & Gallery):** Nằm phía dưới, hiển thị danh sách vi phạm dưới dạng bảng tính. Tích hợp tính năng "Xem ảnh bằng chứng" dạng lưới (Grid), cho phép cán bộ giám sát nhấp chuột để xem trực tiếp các bức ảnh cận cảnh biển số xe ngay trên Web.

**_Hình 5.1: Giao diện tổng quan của Hệ thống_**

### 5.3.2. Quản lý Tài nguyên và Tối ưu hóa Luồng (Resource & Flow Optimization)

Để duy trì độ mượt mà khi xử lý các video giao thông dung lượng lớn, mã nguồn app.py đã ứng dụng các kỹ thuật quản lý tài nguyên khắt khe ở mức hệ thống:

- **Tối ưu hóa Bộ nhớ đệm (AI Caching):** Đặc thù của Streamlit là sẽ chạy lại toàn bộ mã nguồn mỗi khi người dùng tương tác. Để tránh việc tràn RAM do tải lại 3 mô hình YOLO26, hàm load_models() được bọc bởi Decorator @st.cache_resource. Nhờ đó, các tệp trọng số khổng lồ chỉ được nạp vào GPU/CPU duy nhất một lần khi khởi động.
- **Quản lý Tệp tin Tạm thời (Tempfile Management):** Khi người dùng tải lên một video động, hệ thống không lưu trực tiếp vào ổ cứng mà sử dụng thư viện tempfile.NamedTemporaryFile để tạo vùng nhớ tạm. Sau khi vòng lặp OpenCV (cv2.VideoCapture) quét xong khung hình cuối cùng, lệnh os.unlink() được gọi để tự động dọn rác, giải phóng bộ nhớ, đảm bảo máy chủ không bị quá tải lưu trữ khi vận hành 24/7.
- **Kết xuất Video Thời gian thực (Real-time Rendering):** Để phát video trực tiếp, mã nguồn khởi tạo một không gian ảo bằng lệnh video_area = st.empty(). Mỗi khung hình sau khi được AI xử lý sẽ được chuyển đổi sang chuẩn màu RGB và ghi đè liên tục vào vùng này, kết hợp với thanh tiến trình st.progress giúp người dùng dễ dàng theo dõi tốc độ quét.

### 5.3.3. Trực quan hóa Dữ liệu và Xuất Hồ sơ Pháp lý (Evidence Archiving)

Đây là phân hệ nâng tầm ứng dụng từ một "công cụ demo" thành một "phần mềm nghiệp vụ" hoàn chỉnh. Dữ liệu không chỉ được hiển thị mà còn được đóng gói thông minh:

- **Bảng báo cáo trực tuyến:** Hệ thống đọc tệp Danh_Sach_Phat_Nguoi.csv bằng thư viện Pandas và hiển thị lên web bằng hàm st.dataframe. Tính năng này cho phép cán bộ lọc, sắp xếp và tìm kiếm biển số xe vi phạm theo thời gian thực.
- **Đóng gói Hồ sơ tự động (ZIP Export):** Mã nguồn tích hợp module zipfile và io.BytesIO để tạo ra chức năng "Tải xuống Báo cáo & Bằng chứng". Khi người dùng nhấn nút, hệ thống sẽ tự động quét toàn bộ hình ảnh vi phạm cùng với tệp Excel, nén chúng lại về máy tính của người dùng. Chức năng này đáp ứng hoàn hảo quy trình bàn giao hồ sơ phạt nguội của cơ quan chức năng.
```

### ✅ BẢN MỚI (Trong file `Báo_cáo_TTCS_Bản_Mới.md`)
```text
## 5.3. Triển khai Ứng dụng Máy chủ (Web App) với Flask

Để biến các thuật toán AI phức tạp thành một sản phẩm phần mềm giám sát an ninh thân thiện và dễ triển khai, toàn bộ khối xử lý lõi (Backend) được móc nối với Giao diện (Frontend) thông qua framework **Flask**. Việc chuyển từ ứng dụng nguyên khối sang mô hình Web Server đa luồng là một bước tiến lớn của dự án.

### 5.3.1. Truyền luồng Video (Video Streaming) thời gian thực

Khác với ảnh tĩnh, luồng video đòi hỏi phải liên tục truyền hàng ngàn khung hình từ Máy chủ xuống Trình duyệt mà không được làm sập bộ nhớ. Mã nguồn `server.py` giải quyết bài toán này bằng cơ chế `Multipart/x-mixed-replace`:
- Hàm Generator `gen_frames()` liên tục lấy khung hình từ OpenCV sau khi đã qua lớp xử lý của AI Pipeline.
- Khung hình được mã hóa trực tiếp thành chuỗi byte JPEG ( `cv2.imencode('.jpg', frame)` ) và đẩy liên tục qua route `/video_feed` dưới dạng các luồng dữ liệu độc lập. Trình duyệt web sẽ tự động thay thế bức ảnh cũ bằng bức ảnh mới với tốc độ chớp nhoáng, tạo ra ảo giác của một Video mượt mà.

### 5.3.2. Quản lý trạng thái và Đồng bộ dữ liệu bất đồng bộ (AJAX)

Trong các giao diện giám sát an ninh, số lượng xe vi phạm cần nhảy liên tục. Tuy nhiên, nếu bắt trình duyệt tải lại (Reload) trang web để cập nhật số thì luồng Video sẽ bị đứt quãng.
- Backend cung cấp API `/api/stats` trả về chuỗi JSON chứa thông số `total_vehicles` và `violations_this_frame`.
- Dưới Frontend (`index.html`), hệ thống triển khai một đoạn mã JavaScript sử dụng hàm `fetch()` (AJAX) gọi API này mỗi 1000 mili-giây (1 giây). Dữ liệu JSON trả về sẽ tự động đắp vào các thẻ HTML (DOM Update), giúp bộ đếm xe vi phạm nhảy số tức thời (Real-time) ngay trong khi Video vẫn đang mượt mà phát hình bên cạnh.

### 5.3.3. Thiết kế Bố cục Giao diện (Tailwind CSS)

Giao diện được thiết kế ở chế độ tối (Dark Mode), lấy cảm hứng từ các bảng điều khiển (Dashboard) giám sát an ninh chuyên nghiệp. Để giảm thiểu việc viết CSS thủ công phức tạp, **Tailwind CSS** được nhúng trực tiếp qua mạng CDN.
- Bố cục được chia sử dụng CSS Grid/Flexbox vô cùng linh hoạt.
- Phân hệ bên trái: Khu vực tải lên (Upload) đa luồng (chọn file trên máy tính) và mô phỏng Camera Real-time.
- Phân hệ trung tâm: Trình phát Video lớn sắc nét, tích hợp lưới thông số thống kê nổi bật.
- Cấu trúc độc lập của HTML giúp dự án dễ dàng tích hợp thêm các công nghệ web khác trong tương lai mà không bị trói buộc vào nền tảng cốt lõi của Python.
```

---

