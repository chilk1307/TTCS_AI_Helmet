**BỘ KHOA HỌC VÀ CÔNG NGHỆ**

**HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG**

**\--------\*\*\*--------**

**BÁO CÁO THỰC TẬP CƠ SỞ**

Phát hiện người đi xe máy vi phạm không đội mũ bảo hiểm

**Giảng viên hướng dẫn**: TS. Đào Thị Thuý Quỳnh

**Lớp học phần:** Thực Tập Cơ Sở - Nhóm 13

**Nhóm thực hiện:** Nhóm 6

**Danh sách thành viên:**

**\-** Bùi Vũ Minh Phi – B23DCKH086

**\-** Vũ Văn Học – B23DCKH048

**\-** Lê Khắc Chí – B23DCKH010

**HÀ NỘI, 3-2026**

**MỤC LỤC**

[**CHƯƠNG 1: TỔNG QUAN VỀ ĐỀ TÀI**](#_7is1ofqfhhv1) **4**

[1.1. Đặt vấn đề và Lý do chọn đề tài](#_3bag0zezdh7g) 4

[1.2. Mục tiêu của dự án](#_eafhl5qmtb9) 5

[1.3. Đối tượng và Phạm vi nghiên cứu](#_5lgzlzo171f9) 5

[**CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ CÔNG NGHỆ ÁP DỤNG**](#_7bmptndqdss2) **6**

[2.1. Tổng quan về bài toán Nhận diện vật thể (Object Detection)](#_mxk4xfomutmn) 6

[2.2. Kiến trúc mạng YOLO và Sự lựa chọn YOLO26](#_bs0jojr41dyg) 6

[2.3. Thuật toán Theo dõi đối tượng (Object Tracking)](#_28jidixb8aej) 6

[2.4. Kỹ thuật Tiền xử lý ảnh (Image Preprocessing) với OpenCV](#_ju4o5dnmtpy1) 7

[2.5. Nhận dạng ký tự (OCR) và Thuật toán Hậu xử lý (Post-processing)](#_fr07k3gl9i5m) 7

[2.6. Công nghệ xây dựng giao diện ứng dụng (Flask và Tailwind CSS)](#_6pihs4lvqivj) 8

[**CHƯƠNG 3: XÂY DỰNG DỮ LIỆU VÀ HUẤN LUYỆN MÔ HÌNH**](#_ofuwti6dxm0j) **8**

[3.1. Xây dựng và Chuẩn bị Dữ liệu (Dataset Preparation)](#_u1ihrljni5lj) 8

[3.2. Luồng xử lý qua Kiến trúc 3 Cấp độ (3-Stage Pipeline)](#_l7ap3znkaq9o) 9

[3.3. Môi trường và Thiết lập Huấn luyện (Training Configuration)](#_axni66iiemfh) 9

[3.4. Quá trình Huấn luyện và Biểu đồ Trực quan](#_aj3jd0ehmvcu) 10

[3.5. Đánh giá Tổng thể Mô hình (Model Evaluation)](#_s8zjxhhanyuk) 10

[**CHƯƠNG 4: PHÂN TÍCH VÀ THIẾT KẾ KIẾN TRÚC HỆ THỐNG**](#_wd5ikjoqt3v6) **10**

[4.1. Đánh giá giới hạn của luồng AI cơ sở (Baseline Pipeline)](#_mmcisfnwnzqr) 10

[4.2. Kiến trúc hệ thống tổng thể (System Architecture)](#_r2v420qtawhd) 11

[4.3. Tổ chức không gian mã nguồn (Source Code Workspace)](#_bsf29oxqqnp3) 11

[**CHƯƠNG 5: TRIỂN KHAI MÃ NGUỒN VÀ XÂY DỰNG ỨNG DỤNG**](#_ybydfa8ntkcg) **12**

[5.1. Triển khai Cấu hình và Điều phối luồng trung tâm](#_91o15tt1xtp4) 12

[5.1.1. Tối ưu hóa và Quản lý cấu hình tập trung](#_y93ruf2enw5b) 12

[5.1.2. Giải thuật điều phối và Xử lý ngoại lệ](#_8emr1tcyymf1) 13

[5.2. Triển khai các Module xử lý cốt lõi (Middleware Core)](#_fodw66p6gl7e) 14

[5.2.1. Module Tiền xử lý và Nắn thẳng hình học](#_afaz18ho5ljl) 14

[5.2.2. Động cơ OCR và Khử nhiễu văn bản](#_m22rguwfioh) 15

[5.2.3. Cấu trúc Bám vết trì hoãn và Quản lý file](#_hse5xp2030gu) 15

[5.3. Triển khai Giao diện Người dùng (Web UI) với Flask và Tailwind CSS](#_uiwl9rf5pak7) 16

[5.3.1. Thiết kế Bố cục và Trải nghiệm Người dùng](#_j2g8qignkzbn) 16

[5.3.2. Quản lý Tài nguyên và Tối ưu hóa Luồng](#_5feiywv0u81l) 17

[5.3.3. Trực quan hóa Dữ liệu và Xuất Hồ sơ Pháp lý](#_grnr7m5gct3c) 17

[**CHƯƠNG 6: ĐÁNH GIÁ THỰC NGHIỆM VÀ TỔNG KẾT**](#_g3us2x7z5628) **17**

[6.1. Kết quả nhận diện trên giao diện ứng dụng](#_u0h0xt8j0xg7) 18

[6.1.1. Thực nghiệm xử lý dữ liệu ảnh tĩnh](#_vg17xylv8c75) 18

[6.1.2. Thực nghiệm xử lý luồng video động](#_vq7hzrjtlb5) 18

[6.1.3. Đánh giá năng lực xử lý](#_5fl0tbah974x) 19

[6.2. Đánh giá tính ổn định hệ thống I/O và Quản lý dữ liệu](#_i2fjwuva64t1) 19

[6.2.1. Kiểm định cơ chế chống bùng nổ dữ liệu](#_i8rzatlzrof2) 20

[6.2.2. Kiểm định chất lượng hồ sơ chứng cứ và Phân định đa nguồn](#_de0ytbpwzex2) 20

[6.3. Kết luận chung và Thành tựu của Đồ án](#_qrvezs9x73a9) 21

[6.4. Hạn chế còn tồn đọng và Định hướng phát triển tương lai](#_rluiuvpm2bg5) 21

[6.4.1. Hạn chế khách quan còn tồn đọng](#_okpnm59y8ixw) 21

[6.4.2. Định hướng phát triển tương lai](#_2sefr927a9b5) 22

# CHƯƠNG 1: TỔNG QUAN VỀ ĐỀ TÀI

## 1.1. Đặt vấn đề và Lý do chọn đề tài

Hiện nay, Việt Nam là một trong những quốc gia có tỷ lệ sử dụng xe gắn máy cao nhất thế giới. Đi kèm với sự phát triển mạnh mẽ về số lượng phương tiện tham gia giao thông là những thách thức to lớn trong công tác quản lý và đảm bảo an toàn giao thông đường bộ. Một trong những vi phạm phổ biến và trực tiếp gây nguy hiểm đến tính mạng người tham gia giao thông là hành vi **không đội mũ bảo hiểm**.

Mặc dù các cơ quan chức năng đã áp dụng nhiều biện pháp tuyên truyền và xử phạt, nhưng tình trạng vi phạm vẫn thường xuyên tiếp diễn, đặc biệt là ở những khu vực hoặc khung giờ vắng bóng lực lượng Cảnh sát Giao thông. Phương pháp tuần tra và xử phạt thủ công hiện tại bộc lộ nhiều hạn chế:

- **Tiêu tốn nguồn nhân lực:** Lực lượng mỏng không thể bao quát toàn bộ các tuyến đường 24/7.
- **Dễ bỏ sót vi phạm:** Trong những khung giờ cao điểm, việc quan sát bằng mắt thường và ghi nhận thủ công rất dễ xảy ra sai sót hoặc không kịp xử lý.
- **Khó khăn trong việc thu thập bằng chứng:** Việc lập biên bản xử phạt nguội cần có hình ảnh chứng minh rõ ràng, minh bạch và có tính pháp lý cao.

Trong những năm gần đây, sự bùng nổ của **Trí tuệ Nhân tạo (AI)**, đặc biệt là lĩnh vực **Thị giác máy tính (Computer Vision)** và **Học sâu (Deep Learning)**, đã mở ra hướng đi mới để giải quyết triệt để bài toán này. Các mô hình mạng nơ-ron tích chập (CNN), tiêu biểu là kiến trúc YOLO (You Only Look Once), đã chứng minh được khả năng nhận diện vật thể theo thời gian thực với độ chính xác vượt trội.

Xuất phát từ thực tiễn đó, nhóm chúng em quyết định thực hiện đề tài: **"Nghiên cứu và Xây dựng Hệ thống AI Tự động Giám sát, Nhận diện Hành vi Không đội Mũ bảo hiểm và Trích xuất Biển số xe"**. Hệ thống được kỳ vọng sẽ đóng vai trò như một "trợ lý ảo" đắc lực, hoạt động bền bỉ 24/7, tự động bám bắt đối tượng vi phạm, trích xuất dữ liệu biển số và lập hồ sơ phạt nguội, góp phần nâng cao ý thức của người tham gia giao thông và số hóa quy trình quản lý đô thị thông minh.

## 1.2. Mục tiêu của dự án

Mục tiêu cốt lõi của đồ án là xây dựng thành công một pipeline (luồng xử lý) tự động hoàn toàn, tiếp nhận đầu vào là luồng video/hình ảnh từ camera giao thông và cho ra kết quả cuối cùng là danh sách các phương tiện vi phạm kèm bằng chứng. Để đạt được điều này, dự án phân rã thành các mục tiêu cụ thể sau:

- **Về mặt Nhận diện (Detection):** Xây dựng và huấn luyện thành công các mô hình AI có khả năng định vị chính xác người đi xe máy, phân loại trạng thái đội mũ bảo hiểm (Có/Không) trong điều kiện môi trường thực tế.
- **Về mặt Bóc tách ký tự (OCR):** Phát triển module nhận diện và đọc biển số xe máy chuẩn định dạng Việt Nam, có khả năng chống chịu được các yếu tố nhiễu như góc chụp nghiêng, lóa sáng hay bóng râm.
- **Về mặt Tối ưu hệ thống (System Architecture):** Thiết lập kiến trúc xử lý nối tiếp 3 giai đoạn (3-Stage Pipeline) để tối ưu hóa tài nguyên phần cứng và tăng độ chính xác.
    - Tích hợp thuật toán theo dõi vết (Object Tracking) nhằm cấp định danh (ID) duy nhất cho mỗi phương tiện, loại bỏ tình trạng ghi nhận trùng lặp nhiều lần cho cùng một vi phạm trên video.
- **Về mặt Ứng dụng:** Tự động hóa khâu lưu trữ, kết xuất file nhật ký (Excel/CSV) và lưu trữ hình ảnh cắt cận cảnh phương tiện vi phạm phục vụ công tác đối soát pháp lý.

## 1.3. Đối tượng và Phạm vi nghiên cứu

Để đảm bảo tính khả thi và tập trung chuyên sâu cho đồ án, phạm vi nghiên cứu được xác định rõ như sau:

- **Đối tượng nghiên cứu:** Các thuật toán nhận diện vật thể (Object Detection) và theo dõi đối tượng (Object Tracking), cụ thể là họ mô hình YOLO26.
    - Kỹ thuật xử lý ảnh số (Digital Image Processing) ứng dụng trong nhận dạng ký tự quang học (OCR).
- **Phạm vi dữ liệu đầu vào:** Hệ thống tập trung xử lý dữ liệu hình ảnh/video ban ngày hoặc trong điều kiện ánh sáng đô thị tương đối rõ nét.
    - Góc quay camera giả định là góc quay từ trên cao chiếu chéo xuống đường (tương tự các góc lắp đặt camera giao thông hiện hành).
- **Phạm vi nhận diện:**
    - Tập trung giới hạn vào phương tiện là **Xe gắn máy / Xe mô tô 2 bánh**.
    - Biển số xe được nhận diện tuân theo chuẩn định dạng biển số xe máy của Việt Nam (gồm biển 1 dòng và biển 2 dòng).

# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ CÔNG NGHỆ ÁP DỤNG

## 2.1. Tổng quan về bài toán Nhận diện vật thể (Object Detection)

Nhận diện vật thể là một trong những bài toán cốt lõi của Thị giác máy tính (Computer Vision), có nhiệm vụ không chỉ phân loại (Classification) xem trong ảnh có đối tượng gì, mà còn phải định vị (Localization) chính xác đối tượng đó nằm ở đâu thông qua các hộp giới hạn (Bounding boxes).

Hiện nay, các mô hình học sâu giải quyết bài toán này được chia làm hai trường phái chính:

- **Mô hình hai giai đoạn (Two-stage Detectors):** Tiêu biểu là họ R-CNN. Mô hình trích xuất vùng đề xuất trước rồi mới phân loại. Nhược điểm là tốc độ chậm, không phù hợp cho quét video thời gian thực.
- **Mô hình một giai đoạn (One-stage Detectors):** Tiêu biểu là họ YOLO (You Only Look Once). Mô hình dự đoán trực tiếp tọa độ khung bao và xác suất lớp trong một lần quét toàn bộ mạng, mang lại tốc độ cực nhanh. Đây là nền tảng cốt lõi được lựa chọn cho hệ thống giám sát giao thông này.

## 2.2. Kiến trúc mạng YOLO và Sự lựa chọn YOLO26

**YOLO26** là phiên bản tiên tiến nhất được phát hành bởi Ultralytics. Trong dự án này, nhóm quyết định sử dụng YOLO26 với cấu hình **Large (26l)** cho cả 3 giai đoạn xử lý vì các lý do sau:

- **Kiến trúc Anchor-Free:** Trái với các phiên bản cũ phụ thuộc vào Anchor Box cố định, YOLO26 linh hoạt hơn trong việc nhận diện các vật thể có tỷ lệ khung hình đa chiều (như biển số dài, biển số vuông, người đi xe máy góc nghiêng).
- **Sức mạnh của cấu hình Large (26l):** Việc sử dụng bản Large với mạng nơ-ron sâu giúp hệ thống trích xuất được những đặc trưng vi mô nhất (như nét chữ bị mờ, biển số ở xa), đẩy độ chính xác (mAP) lên mức tối đa để phục vụ cho các tác vụ mang tính pháp lý như phạt nguội.

## 2.3. Thuật toán Theo dõi đối tượng (Object Tracking)

Xử lý luồng Video giao thông luôn đối mặt với vấn đề "Ghi nhận trùng lặp". Một phương tiện vi phạm xuất hiện trong 90 khung hình sẽ bị lập biên bản 90 lần nếu chỉ dùng Object Detection tĩnh.

Hệ thống giải quyết bài toán này bằng việc ứng dụng thuật toán **Object Tracking** (tiêu biểu là BoT-SORT / ByteTrack tích hợp trong YOLO26). Cơ chế hoạt động dựa trên nguyên lý _Tracking by Detection_:

- Khi YOLO phát hiện một đối tượng, thuật toán Tracking sẽ trích xuất đặc trưng không gian và hướng di chuyển để cấp một định danh duy nhất (Track ID).
- Sử dụng bộ lọc động học (Kalman Filter), hệ thống dự đoán và nối liền quỹ đạo của ID đó qua các khung hình tiếp theo. Nhờ vậy, nền tảng phần mềm có thể quản lý vòng đời của một phương tiện và ngăn chặn việc ghi log trùng lặp.

## 2.4. Kỹ thuật Tiền xử lý ảnh (Image Preprocessing) với OpenCV

Ảnh cắt từ camera giao thông thường gặp hai vấn đề lớn: lóa sáng và méo phối cảnh. Nhóm áp dụng các thuật toán xử lý ảnh không gian bằng thư viện OpenCV để chuẩn hóa dữ liệu trước khi nhận dạng chữ:

- **Cân bằng sáng cục bộ (CLAHE):** Sử dụng thuật toán _Contrast Limited Adaptive Histogram Equalization_. Khi chuyển ảnh sang không gian màu LAB, CLAHE được áp dụng lên kênh L (Độ sáng) nhằm san phẳng các vùng lóa sáng cục bộ mà không làm tăng nhiễu, giúp viền chữ đen nổi bật trên nền trắng.
- **Biến đổi Hough và Nắn thẳng góc nghiêng (Deskewing):** Biển số thường bị nghiêng ngẫu nhiên. Hệ thống kết hợp thuật toán phát hiện cạnh **Canny** và biến đổi **HoughLinesP** để trích xuất hệ số góc của các đoạn thẳng nằm ngang trên biển số. Dựa vào góc này, một ma trận xoay (Rotation Matrix) được tạo ra để nắn thẳng ảnh về trạng thái vuông góc (Warp Affine).

## 2.5. Nhận dạng ký tự (OCR) và Thuật toán Hậu xử lý (Post-processing)

Thay vì sử dụng các công cụ OCR thương mại, hệ thống sử dụng YOLO để bóc tách từng ký tự. Tuy nhiên, YOLO chỉ trả về các Bounding Box rời rạc. Để ghép thành biển số hoàn chỉnh, hệ thống áp dụng nền tảng lý thuyết của các thuật toán Hậu xử lý (Post-processing):

- **Giải thuật Không gian hình học (Spatial Heuristics):** Dựa trên tọa độ Y của các hộp giới hạn, hệ thống tính toán khoảng cách (Gap) lớn nhất để tự động chia biển số thành các dòng (Line 1, Line 2). Sau đó, tại mỗi dòng, các ký tự được sắp xếp tuần tự theo trục X từ trái qua phải.
- **Thuật toán Khử trùng lặp (Custom Non-Maximum Suppression - NMS):** Trong quá trình OCR, có trường hợp một ký tự bị YOLO nhận diện thành hai hộp Box đè lên nhau. Thuật toán NMS được tinh chỉnh dựa trên tâm X: Nếu hai tâm quá gần nhau, ký tự có độ tự tin (Confidence) thấp hơn sẽ bị triệt tiêu.
- **Hệ luật suy diễn (Rule-based Mapping):** Để khắc phục điểm yếu kinh điển của AI là nhầm lẫn các ký tự giống nhau (Ví dụ: 8 và B, D và 0), hệ thống ứng dụng kỹ thuật Map Từ điển (Dictionary Mapping). Dựa vào định dạng chuẩn của biển số xe máy Việt Nam (Ví dụ: 2 ký tự đầu bắt buộc là Số), hệ thống sẽ ép kiểu các ký tự AI dự đoán nhầm về định dạng chuẩn, đẩy độ chính xác của chuỗi OCR lên mức gần như tuyệt đối.

## 2.6. Kiến trúc Client-Server với Flask và Tailwind CSS

Để đóng gói hệ thống AI thành một sản phẩm phần mềm hoàn chỉnh, nhóm sử dụng mô hình Client-Server hiện đại, thay thế cho các framework tích hợp nguyên khối nhằm đạt hiệu năng tối đa:

- **Backend (Flask):** Đóng vai trò là Máy chủ ứng dụng. Flask cung cấp một Web Server gọn nhẹ, xử lý đa luồng (Multi-threading). Cốt lõi của kiến trúc này là sử dụng kỹ thuật `Multipart/x-mixed-replace` để truyền luồng Video theo thời gian thực (Video Streaming) từ OpenCV thẳng lên trình duyệt web mà không gây nghẽn cổ chai. Đồng thời cung cấp các API JSON để giao tiếp dữ liệu.
- **Frontend (HTML/JS & Tailwind CSS):** Giao diện được thiết kế độc lập. Tailwind CSS mang lại một bộ giao diện (Dashboard) giám sát an ninh cực kỳ chuyên nghiệp và mượt mà. JavaScript (AJAX) được sử dụng để gọi API cập nhật các thông số (Số xe vi phạm, Tổng xe) theo thời gian thực (Real-time) mà không cần tải lại trang (No-reload), khắc phục triệt để hiện tượng giật lag màn hình của các công nghệ cũ.

# CHƯƠNG 3: XÂY DỰNG DỮ LIỆU VÀ HUẤN LUYỆN MÔ HÌNH

Để xây dựng một hệ thống nhận diện giao thông có khả năng hoạt động thực tế với độ chính xác cao, cốt lõi nằm ở chất lượng dữ liệu và phương pháp huấn luyện. Chương này trình bày chi tiết về quá trình xây dựng bộ dữ liệu (Dataset), luồng xử lý qua kiến trúc 3 cấp độ (3-Stage Pipeline), cũng như các thiết lập và kết quả huấn luyện cho 3 mô hình YOLO26-Large.

## 3.1. Xây dựng và Chuẩn bị Dữ liệu (Dataset Preparation)

Dữ liệu là "nguồn nhiên liệu" quyết định sự thông minh của AI. Vì hệ thống bao gồm 3 tác vụ chuyên biệt, nhóm đã tiến hành thu thập, làm sạch và dán nhãn (labeling) thành 3 bộ dữ liệu độc lập:

- **Dataset 1 (Phát hiện phương tiện):** Gồm hàng ngàn hình ảnh trích xuất từ camera giao thông đô thị. Dữ liệu được dán nhãn tập trung vào đối tượng tổng thể là motorcyclist (người điều khiển xe máy).
- **Dataset 2 (Đánh giá vi phạm):** Bao gồm các hình ảnh đã được cắt cận cảnh (crop) vào khu vực người lái xe. Nhóm tiến hành dán 3 nhãn phân lớp: helmet (đội mũ bảo hiểm), nohelmet (không đội mũ) và licenseplate (vùng chứa biển số).
- **Dataset 3 (Nhận dạng ký tự - OCR):** Bộ dữ liệu "Vietnamese License Plate" chuẩn, chứa các ký tự được cắt trực tiếp từ biển số xe máy Việt Nam thực tế, phân bổ đều cho 36 lớp (từ 0-9 và A-Z).

Thay vì dùng một mô hình khổng lồ xử lý mọi thứ gây ra nhiễu loạn, hệ thống đưa dữ liệu đi qua một đường ống (Pipeline) gồm 3 trạm kiểm duyệt nối tiếp nhau. Mỗi trạm đảm nhận một tác vụ duy nhất nhằm tối ưu hóa độ chính xác:

- **Trạm 1 (Stage 1 - Quét diện rộng):** Hình ảnh/Khung hình video gốc được đưa vào mô hình thứ nhất. Mô hình này làm nhiệm vụ quét toàn cảnh để tìm ra tất cả các motorcyclist. Nếu tìm thấy, hệ thống sẽ sử dụng tọa độ Bounding Box để cắt (Crop) tách riêng từng người đi xe máy ra khỏi khung cảnh nền (background).
- **Trạm 2 (Stage 2 - Đánh giá cục bộ):** Các bức ảnh cắt cận cảnh người đi xe máy tiếp tục được đưa vào mô hình thứ hai. Mô hình này quét tìm nohelmet để xác định hành vi vi phạm, đồng thời khoanh vùng licenseplate. Tọa độ vùng biển số lại tiếp tục được dùng để cắt ra một bức ảnh chỉ chứa biển số xe.
- **Trạm 3 (Stage 3 - Bóc tách ký tự):** Bức ảnh biển số (sau khi đã trải qua bước tiền xử lý chống lóa và xoay thẳng bằng OpenCV) được đưa vào mô hình cuối cùng. Mô hình này quét qua bề mặt biển số, bóc tách từng ký tự và trả về tọa độ của chúng. Sau đó, thuật toán không gian (Spatial Heuristics) sẽ sắp xếp các ký tự này thành chuỗi định dạng chuẩn (VD: 43D1-12345).

**_Hình 3.1: Sơ đồ pipeline 3 giai đoạn của mô hình_**_._

## 3.2. Thiết lập môi trường và tham số huấn luyện (Training Configuration)

Để quá trình huấn luyện các mô hình diễn ra hiệu quả và đạt độ chính xác cao nhất, nhóm nghiên cứu đã tiến hành thiết lập môi trường phần cứng và tinh chỉnh các siêu tham số (Hyperparameters) phù hợp với đặc thù của từng bài toán nhận diện.

### 3.2.1. Môi trường phần cứng (Hardware Environment)

Toàn bộ quá trình huấn luyện 3 mô hình (Stage 1, 2 và 3) đều được thực hiện trên nền tảng điện toán đám mây Google Colab. Việc sử dụng các bộ xử lý đồ họa (GPU) hiệu năng cao được cung cấp trên nền tảng này giúp giải quyết triệt để bài toán thiếu hụt tài nguyên phần cứng cục bộ. Năng lực xử lý ma trận của GPU cho phép hệ thống nạp các bộ dữ liệu lớn, tăng kích thước lô (batch size) lên mức tối đa, từ đó rút ngắn đáng kể thời gian huấn luyện mô hình từ vài ngày xuống chỉ còn vài giờ.

### 3.2.2. Cấu hình siêu tham số chung (Global Hyperparameters)

Mặc dù giải quyết 3 tác vụ khác nhau, các mô hình đều chia sẻ một bộ khung thiết lập thông số cốt lõi nhằm đảm bảo tính đồng bộ:

- **Số chu kỳ huấn luyện (epochs):** Là số lần toàn bộ tập dữ liệu được đưa qua mạng nơ-ron để học tập. Thông số này được thiết lập đủ lớn (từ 100 đến 200 chu kỳ) để đảm bảo mô hình có thời gian tìm kiếm được trọng số tối ưu nhất, giúp hàm mất mát (Loss) hội tụ ổn định.
- **Kích thước lô (batch size):** Là số lượng hình ảnh được mô hình xử lý đồng thời trong một bước cập nhật trọng số. Batch size được thiết lập ở mức tối đa mà bộ nhớ VRAM của GPU cho phép (như 32 hoặc 64). Việc này không chỉ giúp tăng tốc độ tính toán mà còn làm cho các đường gradient cập nhật trở nên mượt mà và bớt nhiễu hơn.
- **Thuật toán tối ưu (optimizer):** Hệ thống sử dụng các thuật toán tối ưu tiên tiến như AdamW hoặc SGD (Stochastic Gradient Descent) kết hợp với tham số đà (momentum). Chức năng của tham số này là điều hướng quá trình cập nhật trọng số một cách thông minh, giúp mô hình nhanh chóng tiến về điểm cực tiểu toàn cục và tránh bị mắc kẹt tại các cực tiểu cục bộ.
- **Kích thước ảnh đầu vào (imgsz):** Định nghĩa độ phân giải của ma trận ảnh trước khi đưa vào mạng (ví dụ: imgsz=640). Thiết lập kích thước đủ lớn là điều kiện bắt buộc để mạng nơ-ron không làm mất đi các đặc trưng vi mô của các vật thể nhỏ (như xe máy ở xa camera).

### 3.2.3. Tinh chỉnh chuyên biệt cho từng giai đoạn

Tùy thuộc vào đặc điểm của đối tượng cần nhận diện, một số tham số được can thiệp riêng rẽ để ép mô hình học theo đúng kịch bản mong muốn:

- **Giai đoạn 1 & 2 (Phát hiện phương tiện và Phân loại mũ bảo hiểm):** Ở 2 giai đoạn này, mục tiêu là nhận diện các đối tượng có kích thước và hình dáng phức tạp trên toàn bối cảnh (Full-frame). Do đó, thuật toán Tăng cường dữ liệu (Data Augmentation) được áp dụng mạnh mẽ (như thay đổi độ sáng, độ bão hòa màu, thêm nhiễu) để giúp AI làm quen với các điều kiện thời tiết thực tế đa dạng.
- **Giai đoạn 3 (Bóc tách ký tự biển số - OCR):** Bài toán đọc chữ có tính chất hình học vô cùng đặc thù và nhạy cảm, do đó hệ thống buộc phải can thiệp hai thiết lập quan trọng:
    - **Tắt lật ảnh ngang (fliplr = 0.0):** Trong khi xe máy lật ngược lại vẫn là xe máy, thì việc lật ngang một chữ cái (ví dụ chữ C hoặc E) sẽ biến nó thành một ký tự vô nghĩa không tồn tại. Tham số fliplr bị ép về 0 để cấm mô hình tự động lật ảnh trong quá trình tăng cường dữ liệu, bảo toàn tính chính xác tuyệt đối của hình dáng chữ cái.
    - **Thu nhỏ kích thước đầu vào (imgsz = 320):** Do dữ liệu đưa vào Giai đoạn 3 chỉ là những bức ảnh cận cảnh biển số đã được cắt gọn (Crop) từ trước, việc duy trì độ phân giải 640 là dư thừa. Hạ imgsz xuống 320 giúp tiết kiệm hơn 50% tài nguyên tính toán nhưng vẫn thừa sức giữ lại độ nét của các ký tự.

## 3.3. Quá trình Huấn luyện và Biểu đồ Trực quan

Các mô hình được huấn luyện qua hàng trăm chu kỳ (Epochs). Trong suốt quá trình này, hàm mất mát (Loss Function) cho cả việc khoanh vùng (Box Loss) và phân loại (Class Loss) liên tục giảm và hội tụ, chứng minh mô hình đang học hỏi đúng hướng.

### 3.3.1. Giai đoạn 1: Phát hiện phương tiện

a. Diễn biến quá trình huấn luyện

Mô hình thể hiện khả năng học tập ổn định xuyên suốt 182 epochs. Khác với hiện tượng hội tụ nhanh ở giai đoạn đọc ký tự, bài toán phát hiện biển số đòi hỏi mô hình phải học lâu hơn để tinh chỉnh chính xác tọa độ.

Tại những epoch cuối cùng (cụ thể quanh epoch 182), hệ thống ghi nhận các chỉ số hiệu suất rất ấn tượng và ổn định trên tập Kiểm thử (Validation):

- Độ phủ (Recall): 95.74% – Mô hình bắt được hầu hết các biển số có trong ảnh mà không bị bỏ sót.
- Độ chính xác (Precision): 77.05% – Tỷ lệ nhận diện đúng khá tốt.
- mAP50: 93.48% – Độ chính xác trung bình với ngưỡng IoU 50% ở mức rất cao, đáp ứng tốt yêu cầu thực tế.
- mAP50-95: 89.31% – Một con số cực kỳ xuất sắc cho thang đo khắt khe này, cho thấy bounding box bao quanh biển số rất vừa vặn.

Về mặt giá trị lỗi (Loss curves):

- train/box_loss  
    giảm sâu từ $1.09$ (epoch 1) xuống chỉ còn khoảng $0.38$ ở cuối quá trình.
- val/box_loss  
    cũng ghi nhận đà giảm tương tự, từ $0.65$ xuống còn $0.33$, và không có dấu hiệu bị overfitting (quá khớp), minh chứng bằng việc val_loss vẫn giữ được mức thấp và ổn định.

b. Biểu đồ trực quan

_Biểu đồ Hàm mất mát (Loss Curves):_

Biểu đồ cho thấy cả Training Loss và Validation Loss đều có xu hướng giảm dần đều qua các epoch và tiệm cận dần về một giá trị giới hạn ở cuối quá trình. Sự đồng pha này cho thấy mô hình không gặp hiện tượng Overfitting và học được đúng các đặc trưng quan trọng để phát hiện biển số.

_Biểu đồ Đánh giá Hiệu suất (mAP, Precision, Recall):_

Các đường cong đo lường hiệu suất (đặc biệt là Recall và mAP50-95) tăng trưởng rõ rệt ở 50 epoch đầu tiên, sau đó duy trì mức cao ổn định kéo dài đến epoch 182. Biểu đồ mAP50 tiệm cận mức 0.93, khẳng định thuật toán nhận diện có độ tin cậy lớn trong việc bóc tách vùng chứa biển số ra khỏi các bối cảnh nền phức tạp của môi trường.

_Ma trận Nhầm lẫn (Confusion Matrix):_

Độ chính xác tổng thể rất cao: Đạt mAP50 = 93.49% và mAP50-95 = 89.32%, cho thấy khung dự đoán (bounding box) bám rất sát vào đối tượng. Bắt vết cực kỳ nhạy bén: Tỷ lệ Recall đạt 95.75% minh chứng mô hình gần như không bỏ sót bất kỳ người đi xe máy nào trong khung hình (rất ít lỗi False Negative). Độ chuẩn xác (Precision): Đạt 77.06%, mức khá. Tuy đôi lúc còn nhầm lẫn một vài đối tượng nền thành xe máy (False Positive), nhưng trong bài toán an toàn giao thông, việc ưu tiên Recall (thà nhận diện nhầm còn hơn bỏ sót) là hoàn toàn phù hợp.  
Mô hình nhận diện Motorcyclist hoạt động cực kỳ đáng tin cậy, định vị khung chuẩn xác và sẵn sàng để ứng dụng vào hệ thống nhận diện thực tế.

### 3.3.2.Giai đoạn 2: Phân loại hành vi và Định vị biển số

a. Diễn biến Quá trình Huấn luyện

Mô hình thể hiện khả năng học tập ổn định xuyên suốt 145 epochs. Khác với hiện tượng hội tụ nhanh ở giai đoạn đọc ký tự, bài toán phát hiện đa lớp đối tượng đòi hỏi mô hình phải học lâu hơn để tinh chỉnh chính xác tọa độ bounding box cũng như phân loại chính xác các thuộc tính của mũ bảo hiểm và biển số.

Tại điểm tốt nhất (cụ thể ở epoch 115), hệ thống ghi nhận các chỉ số hiệu suất rất ấn tượng và ổn định trên tập Validation:

- Độ phủ (Recall): 94.08% – Mô hình bắt được hầu hết các đối tượng mục tiêu (mũ bảo hiểm và biển số) trong ảnh mà không bị bỏ sót.
- Độ chính xác (Precision): 96.09%– Tỷ lệ nhận diện đúng cực kỳ cao, giảm thiểu tối đa các cảnh báo sai/nhầm lẫn.
- mAP@50: 96.80% – Độ chính xác trung bình với ngưỡng IoU 50% ở mức rất cao, khẳng định độ tin cậy lớn khi phát hiện vùng mục tiêu.
- mAP@50-95: 77.99% – Chỉ số chất lượng định vị bounding box ở mức rất tốt đối với bài toán nhận diện các đối tượng có kích thước nhỏ và trung bình trong môi trường phức tạp.

Về mặt giá trị lỗi (Loss curves):

- train/box_loss giảm sâu từ 1.4630 (epoch 1) xuống chỉ còn khoảng 0.7138 tại epoch 115 và 0.6014 ở epoch 145.
- val/box_loss giảm từ 1.2447 xuống còn 0.7798 tại epoch 115, và tiếp tục đi ngang ổn định quanh mức 0.7718 ở epoch cuối, cho thấy không có dấu hiệu bị overfitting (quá khớp).
- train/cls_loss giảm từ 1.9441 xuống còn 0.3340 (epoch 115) and 0.2724 (epoch 145).
- val/cls_loss giảm từ 0.7382 xuống còn 0.3460 (epoch 115) and 0.3592 (epoch 145).

2\. Biểu đồ Trực quan (Visualizations)

_Biểu đồ Hàm mất mát (Loss Curves)_

Biểu đồ cho thấy cả Training Loss và Validation Loss đều có xu hướng giảm dần đều qua các epoch và tiệm cận dần về một giá trị giới hạn ở cuối quá trình. Sự đồng pha này cho thấy mô hình không gặp hiện tượng Overfitting và học được đúng các đặc trưng quan trọng để phân loại và định vị mũ bảo hiểm cũng như biển số.

_Biểu đồ Đánh giá Hiệu suất (mAP, Precision, Recall):_

Các đường cong đo lường hiệu suất (đặc biệt là Recall, Precision và mAP@50) tăng trưởng rõ rệt ở 40 epoch đầu tiên, sau đó duy trì mức cao ổn định kéo dài đến epoch 145. Biểu đồ mAP@50 tiệm cận mức 0.968, khẳng định thuật toán nhận diện có độ tin cậy lớn trong việc bóc tách vùng chứa biển số và phân loại trạng thái đội mũ bảo hiểm của người đi xe máy.

_Ma trận Nhầm lẫn (Confusion Matrix)_

Ma trận nhầm lẫn cho thấy tỷ lệ nhận diện đúng (True Positive) rất cao ở cả ba lớp đối tượng helmet, nohelmet, và licenseplate. Sự nhầm lẫn giữa các lớp là vô cùng hạn chế, khẳng định tính hiệu quả vượt trội của việc huấn luyện mô hình YOLO26l.

### 3.3.3. Giai đoạn 3: Bóc tách ký tự OCR

a. Diễn biến quá trình huấn luyện

Mô hình bắt đầu quá trình hội tụ rất nhanh ngay từ những epoch đầu tiên. Do tính năng Early Stopping được thiết lập ở mức 50, quá trình huấn luyện đã tự động kết thúc ở epoch thứ 97 (hoàn thành trong khoảng 0.765 giờ) vì không ghi nhận thêm sự cải thiện đáng kể nào ở các thang đo đánh giá sau đó.

Điểm hội tụ tốt nhất (Best Model) được hệ thống ghi nhận tại epoch 47 với các chỉ số hiệu suất trên tập Validation vô cùng ấn tượng:

- Độ chính xác (Precision): 96.76% (0.9676)
- Độ phủ (Recall): 95.33% (0.9533)
- mAP50: 98.55% (0.9855)
- mAP50-95: 79.37% (0.7937)

Về mặt giá trị lỗi (Loss), train/box_loss giảm đều từ mức 1.04 ở epoch 1 xuống còn 0.59 ở epoch 97. Tương tự, val/box_loss cũng ổn định và đạt mức 0.752 tại epoch tốt nhất (epoch 47). Việc cả training loss và validation loss đều giảm đồng pha cho thấy mô hình học được các đặc trưng tốt mà không bị overfitting (quá khớp).

b. Biểu đồ trực quan

_Biểu đồ Hàm mất mát (Loss Curves):_

Dựa vào biểu đồ Loss, có thể thấy hàm mất mát cho việc phân loại (Classification Loss) và dự đoán bounding box (Box Loss) giảm dốc mạnh trong 10 epoch đầu, sau đó giảm thoai thoải dần và đi ngang từ epoch 40 trở đi. Khoảng cách giữa Train Loss và Val Loss được duy trì ở mức hẹp, chứng tỏ mô hình không học lệch.

_Biểu đồ Đánh giá Hiệu suất (mAP, Precision, Recall):_

Biểu đồ cho thấy đường cong mAP50 tiệm cận mốc 1.0 (gần 99%) từ rất sớm và duy trì ổn định. Điều này thể hiện mô hình OCR có độ tự tin cực cao khi khoanh vùng và nhận diện đúng các ký tự (chữ/số) ngay cả khi chúng nằm sát nhau trên biển số xe. Chỉ số mAP50-95 dao động quanh mức 75% - 79%, là một mức rất tốt cho bài toán có đến 30 lớp đối tượng nhỏ (small objects).

_Ma trận Nhầm lẫn (Confusion Matrix):_

Ma trận nhầm lẫn cho thấy đường chéo chính có mật độ điểm ảnh tập trung rất đậm, chứng minh tỷ lệ dự đoán chính xác tuyệt đối (True Positive) chiếm ưu thế ở hầu hết các lớp. Tuy nhiên, vẫn tồn tại một số ít nhầm lẫn cục bộ giữa các cặp ký tự có hình dáng tương đồng nhau, đặc thù của font chữ biển số (ví dụ: số 8 và chữ B, hoặc chữ D và số 0). Dù vậy, tỷ lệ này rất nhỏ và không làm suy giảm hiệu suất tổng thể của hệ thống.

## 3.4. Đánh giá tổng thể mô hình (Overall Model Evaluation)

Bên cạnh các chỉ số đo lường hiệu suất (Metrics) bằng con số, việc đánh giá tổng thể mô hình bắt buộc phải dựa trên kết quả nghiệm thu trực quan từ tập dữ liệu kiểm thử (Validation Set). Các hình ảnh val_batch_pred được hệ thống tự động xuất ra sau khi kết thúc quá trình huấn luyện, minh chứng cho năng lực thực tế của AI khi phải đối mặt với những hình ảnh mà nó chưa từng nhìn thấy trước đó.

Dựa trên các kết quả xuất ra, năng lực tổng thể của 3 mô hình được đánh giá như sau:

### **3.4.1. Đánh giá tổng thể Giai đoạn 1: Phát hiện phương tiện**

Kết quả trực quan từ các lô ảnh kiểm thử của Giai đoạn 1 cho thấy mô hình YOLO26-Large đã hoàn thành xuất sắc nhiệm vụ rà quét toàn cảnh:

- **Khả năng nhận diện đa tỷ lệ (Multi-scale Detection):** Mô hình không chỉ bắt được các phương tiện xe máy ở rất gần camera (kích thước pixel lớn) mà còn định vị chính xác những chiếc xe ở tít phía xa trong hậu cảnh. Khung Bounding Box bao trọn vẹn cả người điều khiển và phương tiện, tạo tiền đề cắt ảnh (Crop) chuẩn xác cho Giai đoạn 2.
- **Xử lý hiện tượng che khuất (Occlusion Handling):** Trong các khung hình có mật độ giao thông đông đúc, các phương tiện đi sát nhau và che khuất lẫn nhau, mô hình vẫn bóc tách được từng đối tượng riêng biệt mà không bị hiện tượng vẽ gộp hai xe thành một.

**_Hình 3.4.1: Đánh giá tổng thể kết quả dự đoán của mô hình Giai đoạn 1_**_._

### **3.4.2. Đánh giá tổng thể Giai đoạn 2: Phân loại hành vi và Định vị biển số**

Các lô ảnh kiểm thử của Giai đoạn 2 đánh giá năng lực hoạt động trên vùng không gian hẹp (chỉ tập trung vào phương tiện). Kết quả cho thấy độ nhạy bén rất cao của mạng nơ-ron:

- **Tính chính xác trong phân loại đa lớp (Multi-class Classification):** Mô hình phân định rạch ròi và dán nhãn chính xác trạng thái người điều khiển có đội mũ bảo hiểm hay không. Đáng chú ý, AI không bị đánh lừa bởi các yếu tố nhiễu vật lý (như người vi phạm đội mũ lưỡi trai, hoặc có mái tóc đen dày, búi tóc cao).
- **Độ khít của khung bao biển số (Localization Accuracy):** Đối với nhãn biển số (license_plate), mô hình thể hiện khả năng khoanh vùng cực kỳ ấn tượng. Các khung Bounding Box bám sát vào 4 viền của biển số xe máy, loại bỏ hoàn toàn các chi tiết thừa xung quanh (như dè chắn bùn, bánh xe). Điều này đảm bảo tính vẹn toàn dữ liệu cho module xử lý chữ ở giai đoạn cuối.

**_Hình 3.4.2: Đánh giá tổng thể năng lực phân loại và định vị biển số ở Giai đoạn 2_**_._

### **3.4.3. Đánh giá tổng thể Giai đoạn 3: Bóc tách ký tự OCR**

Đây là bài toán đòi hỏi độ tỉ mỉ cao nhất. Các hình ảnh kiểm thử đánh giá năng lực đọc hiểu trên các biển số đã được cắt cận cảnh. Kết quả quan sát cho thấy:

- **Khả năng bắt viền ký tự sắc nét:** Mô hình YOLO26-Large đã phát hiện và vẽ khung bao chính xác cho từng ký tự chữ (A-Z) và số (0-9) đơn lẻ. Các khung bao được vẽ rất khít, không có hiện tượng một khung bao chứa hai ký tự.
- **Độ bền bỉ trước biến dạng (Robustness):** Trong các lô ảnh kiểm thử, có sự xuất hiện của những biển số bị mờ do chuyển động (Motion blur) hoặc bị lóa sáng. Tuy nhiên, mô hình vẫn nhận diện thành công và không bị nhầm lẫn giữa các ký tự có hình dáng sinh trắc học giống nhau (như 8 và B, 0 và D, Z và 2). Các nhãn (Label) hiển thị trên đầu mỗi ký tự hoàn toàn khớp với hình ảnh thực tế.

**_Hình 3.4.3: Đánh giá tổng thể năng lực tách ký tự của mô hình OCR Giai đoạn 3_**_._

# CHƯƠNG 4: PHÂN TÍCH VÀ THIẾT KẾ KIẾN TRÚC HỆ THỐNG

Sau khi huấn luyện thành công các mô hình AI độc lập ở Chương 3, nhiệm vụ tiếp theo là tích hợp chúng vào một hệ thống phần mềm hoàn chỉnh. Một mô hình học sâu dù có độ chính xác cao đến đâu cũng không thể ứng dụng thực tiễn nếu thiếu đi một kiến trúc phần mềm điều phối dữ liệu hợp lý. Chương này sẽ phân tích các giới hạn của luồng AI thô (Baseline), từ đó đề xuất bản vẽ kiến trúc tổng thể và quy hoạch cấu trúc mã nguồn cho toàn bộ dự án.

## 4.1. Đánh giá giới hạn của luồng AI cơ sở (Baseline Pipeline)

Luồng xử lý cơ sở (Baseline) được định nghĩa là việc ghép nối trực tiếp 3 mô hình YOLO (Stage 1 → Stage 2 → Stage 3) chạy nối tiếp nhau thuần túy, chưa có sự can thiệp của các bộ lọc dữ liệu hay trình quản lý trạng thái.

Qua chạy thử nghiệm Baseline trên video thực tế, hệ thống bộc lộ 3 giới hạn (Bottlenecks) nghiêm trọng về mặt thiết kế:

- **Rối loạn định dạng chuỗi OCR (Spatial Disorder):** Do YOLO là mô hình phát hiện vật thể, nó trả về tọa độ các ký tự một cách ngẫu nhiên. Điều này dẫn đến việc biển số thực tế là 43D1-12345 bị hệ thống in ra màn hình thành chuỗi vô nghĩa như 123D45431.
- **Suy giảm độ tin cậy do môi trường vật lý:** Biển số xe thường bị lóa sáng do đèn pha hoặc bị méo/nghiêng do góc máy camera giao thông. Khi đưa bức ảnh thô này vào quét, mô hình AI dễ dàng nhận diện sai các ký tự có hình dáng tương đồng (như nhầm 8 thành B, nhầm 0 thành D).
- **Hiện tượng bùng nổ dữ liệu thừa (Data Spamming):** Đây là lỗi nghiêm trọng nhất. Khi quét video ở tốc độ 30 FPS, một chiếc xe vi phạm xuất hiện trong 3 giây sẽ bị AI nhận diện 90 lần. Do không có bộ nhớ quản lý trạng thái, Baseline ghi đè 90 dòng giống hệt nhau vào file báo cáo Excel và lưu ra 90 bức ảnh cho cùng một phương tiện, gây sập hệ thống lưu trữ.

**_Hình 4.1: Các giới hạn của mô hình Baseline khi chưa có kiến trúc điều phối_**_._

## 4.2. Kiến trúc hệ thống tổng thể (System Architecture)

Để giải quyết triệt để 3 điểm yếu trên, nhóm nghiên cứu đã thiết kế một hệ thống phần mềm bao bọc lấy khối AI. Kiến trúc được xây dựng theo mô hình 3 lớp (3-Tier Architecture), giúp phân tách rõ ràng trách nhiệm của từng thành phần trong hệ thống:

- **Lớp Đầu vào và Giao diện (Input & UI Layer):** Đóng vai trò tương tác trực tiếp với người dùng. Tiếp nhận dữ liệu đầu vào (Ảnh tĩnh/Video động) thông qua giao diện Web. Lớp này cũng chịu trách nhiệm render bảng điều khiển và phát video thời gian thực (Real-time Streaming).
- **Lớp Xử lý Cốt lõi (Middleware Core Engine):** Đây là "Trái tim" của hệ thống, bao gồm 2 khối động cơ chạy song song và hỗ trợ lẫn nhau:
    - _Động cơ AI (AI Engine):_ Chứa 3 mô hình YOLO26-Large chịu trách nhiệm tìm kiếm và bóc tách vật thể.
    - _Động cơ Logic (Logic Engine):_ Chứa các thuật toán tùy chỉnh nhằm can thiệp vào AI. Bao gồm: Bộ tiền xử lý ảnh OpenCV (Khử lóa, nắn thẳng), Bộ hậu xử lý OCR (Sắp xếp không gian, ép luật định dạng) và Bộ theo dõi đối tượng (Tracking) chống ghi đè dữ liệu.
- **Lớp Đầu ra và Lưu trữ (Output & Storage Layer):** Chịu trách nhiệm quản lý vòng đời của bằng chứng vi phạm. Tự động định tuyến tệp tin và xuất báo cáo ra định dạng bảng tính (.csv) kèm theo hình ảnh cận cảnh đã được mã hóa thời gian (Timestamp).

**_Hình 4.2: Sơ đồ kiến trúc 3 lớp tổng thể của hệ thống_**_._

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

# CHƯƠNG 5: TRIỂN KHAI MÃ NGUỒN VÀ XÂY DỰNG ỨNG DỤNG

Dựa trên bản vẽ kiến trúc tổng thể và sơ đồ quy hoạch mã nguồn đã được thiết kế tại Chương 4, bước tiếp theo của đồ án là tiến hành lập trình và hiện thực hóa các ý tưởng logic thành mã nguồn Python. Chương này sẽ đi sâu vào việc giải thích cách các tệp tin trong hệ thống giao tiếp với nhau, từ việc quản lý cấu hình tập trung, triển khai các thuật toán can thiệp lõi (Middleware Core), cho đến việc đóng gói toàn bộ hệ thống thành một ứng dụng Web hoàn chỉnh.

## 5.1. Triển khai Cấu hình và Điều phối luồng trung tâm

Để hệ thống hoạt động ổn định, dễ dàng bảo trì và có khả năng chịu lỗi (Fault Tolerance) cao, nguyên tắc tách biệt mối quan tâm (Separation of Concerns) được áp dụng triệt để. Toàn bộ các thông số tĩnh được cô lập, trong khi logic điều hướng được gom vào một bộ điều phối duy nhất.

### 5.1.1. Tối ưu hóa và Quản lý cấu hình tập trung (config.py)

Trong các hệ thống phần mềm chuyên nghiệp, việc gán cứng (hard-code) các tham số rải rác khắp nơi là một "anti-pattern" gây rủi ro lớn. Toàn bộ các hằng số, ngưỡng AI và thông số không gian được gom vào tệp `src/config.py`:

- **Tách biệt ngưỡng tin cậy (Confidence Threshold):** Mặc dù ngưỡng chung để nhận diện xe máy được đặt ở mức `STAGE1_CONF = 0.45`, trạng thái vi phạm (Không đội mũ) lại được định nghĩa khắt khe hơn: `NOHELMET_MIN_CONF = 0.60`. Cải tiến này là chốt chặn quan trọng giúp AI tránh được hiện tượng False Positive (nhận diện nhầm tóc đen dày thành không đội mũ).
- **Thiết lập Vùng Nhận Diện (Detection Zone):** Nhằm tối ưu hóa phần cứng, hệ thống không quét toàn bộ khung hình mà chỉ tập trung vào vùng trung tâm `ZONE_Y_MIN_RATIO = 0.30` đến `ZONE_Y_MAX_RATIO = 0.80`. Việc này giúp loại bỏ xe quá xa (mờ, khó đọc), chỉ phân tích khi phương tiện vào vùng rõ nét nhất, tăng vọt FPS cho máy tính.
- **Cấu hình Cơ chế Lai (Hybrid Rules):** Cấu hình `MIN_VIOLATION_FRAMES = 3` (Tối thiểu 3 frame vi phạm để khóa khung đỏ hiển thị) và `MIN_VIOLATION_RATIO = 0.40` (Tỷ lệ 40% frame không mũ để chốt hạ phạt nguội). Đây là thông số vàng giúp hệ thống triệt tiêu gần như 100% độ nhiễu.
- **Quy hoạch không gian OCR:** Kích thước tối thiểu `MIN_PLATE_WIDTH = 30` và hệ số phóng to `OCR_UPSCALE_FACTOR = 2` giúp bộ điều phối tự động lọc bỏ các "vật thể rác" ở xa, nội suy nét ảnh trước khi bóc tách chữ.

### 5.1.2. Giải thuật điều phối và Xử lý ngoại lệ (main_pipeline.py)

Tệp main_pipeline.py đóng vai trò là "Nhạc trưởng", chứa hàm cốt lõi process_logic() để điều phối luồng dữ liệu đi qua 3 mô hình AI. Mã nguồn tại đây không chỉ gọi hàm đơn thuần mà được lập trình với 4 kỹ thuật phần mềm nâng cao:

- **Phân nhánh động (Dynamic Branching) và Vét lưới (Sweeping):** Hàm điều phối tự động nhận diện dữ liệu đầu vào thông qua cờ is_video.
    - Nếu là ảnh tĩnh: Lệnh model.predict() được gọi để tìm kiếm vét cạn vật thể.
    - Nếu là video động: Lệnh model.track(persist=True) được kích hoạt. Đặc biệt, khi luồng video kết thúc, hệ thống chủ động gọi phương thức tracker.finalize() để "vét lưới", đảm bảo xuất biên bản cho toàn bộ các xe vi phạm còn kẹt trong bộ nhớ đệm mà chưa kịp đi khuất khỏi camera.
- **Kỹ thuật trích xuất ảnh sạch (Clean Cropping):** Một lỗi kinh điển khi triển khai AI đa tầng là mô hình đi sau bị nhiễu loạn bởi các khung Bounding Box màu mè do mô hình đi trước vẽ lên. Để khắc phục, thuật toán luôn sử dụng lệnh img_clean = img.copy() để tạo ra một bản sao nguyên thủy. Việc cắt ảnh (Crop) đưa cho Stage 2 và Stage 3 luôn được trích xuất từ bản sao "sạch" này _trước khi_ các lệnh vẽ đồ họa (cv2.rectangle) được kích hoạt.
- **Logic chọn lọc không gian (Spatial Selection):** Trong một số khung hình phức tạp, AI ở Stage 2 có thể nhận diện ra 2 vùng cùng được cho là "Biển số xe" trên cùng một phương tiện (do nhiễu bóng đổ). Thay vì gửi cả 2 vùng vào Stage 3 gây lãng phí, mã nguồn tự động tính toán diện tích hình học bằng công thức (cx2 - cx1) \* (cy2 - cy1) và chỉ giữ lại Bounding Box có diện tích lớn nhất (bởi đây thường là biển số rõ nét nhất).
- **Cơ chế chịu lỗi (Fault Tolerance) và Luật ghi biên bản:** Toàn bộ quá trình quét hàng trăm phương tiện được bọc trong các chốt chặn an toàn.
    - _Kiểm soát biên (Boundary Check):_ Sử dụng lệnh if crop_img.size == 0: continue. Nhờ đó, nếu một chiếc xe máy đi sát ra mép camera khiến tọa độ cắt bị lỗi (rỗng), hệ thống sẽ chủ động bỏ qua đối tượng đó thay vì bị sập (Crash) toàn bộ phần mềm.
    - _Kiểm soát rác dữ liệu:_ Hệ thống áp dụng luật cứng: Nếu phát hiện vi phạm nhưng mô hình OCR trả về chuỗi rỗng (if not plate_text), hệ thống sẽ từ chối ghi log. Điều này giúp cơ sở dữ liệu luôn sạch sẽ, chỉ chứa các biên bản có giá trị pháp lý.

## 5.2. Triển khai các Module xử lý cốt lõi (Middleware Core)

Nếu main_pipeline.py là nhạc trưởng điều phối, thì thư mục core/ chính là "nhà máy" gia công dữ liệu. Các module tại đây can thiệp trực tiếp vào dữ liệu đầu vào và đầu ra của YOLO, biến các dự đoán thô thành dữ liệu có độ chính xác và tính pháp lý cao.

### 5.2.1. Module Tiền xử lý và Nắn thẳng hình học (image_utils.py)

Mục tiêu của module này là chuẩn hóa dữ liệu đầu vào cho mô hình bóc tách chữ (Stage 3), khắc phục hiện tượng mất nét do lóa sáng và sai lệch góc chụp phối cảnh từ camera giao thông.

- **Thuật toán Cân bằng sáng cục bộ (CLAHE):** Thay vì dùng các bộ lọc sáng thông thường dễ gây nhiễu, hàm changeContrast tiến hành chuyển đổi không gian màu của ảnh từ chuẩn BGR sang không gian LAB. Thuật toán CLAHE (cv2.createCLAHE) được áp dụng độc quyền lên kênh L (Luminance - Độ sáng), giữ nguyên kênh A và B. Kỹ thuật này giúp san phẳng các vùng lóa do đèn pha mà không làm biến dạng màu sắc, làm cho viền chữ đen nổi bật trên nền trắng.
- **Tự động tính toán và Nắn thẳng góc nghiêng (Auto-Deskewing):** Biển số cắt từ camera thường bị nghiêng ngẫu nhiên. Hàm compute_skew sử dụng bộ lọc cv2.medianBlur để khử nhiễu, tiếp nối bằng thuật toán phát hiện cạnh Canny. Khung ảnh sau đó được đưa qua phép biến đổi không gian cv2.HoughLinesP để dò tìm các đoạn thẳng nằm ngang. Từ hệ số góc của các đoạn thẳng này (thông qua hàm np.arctan2), hệ thống tính toán ra góc nghiêng trung bình (Angle). Cuối cùng, hàm cv2.warpAffine thực hiện phép xoay ma trận ảnh để đưa biển số về trạng thái vuông vức hoàn hảo.

### 5.2.2. Động cơ OCR và Khử nhiễu văn bản (ocr_engine.py)

YOLO26 ở Stage 3 trả về các hộp giới hạn (Bounding Boxes) chứa ký tự độc lập cùng độ tin cậy (Confidence). Để định hình lại chuỗi ký tự chuẩn mực, module ocr_engine.py thực thi một chuỗi giải thuật không gian và logic quy tắc nghiêm ngặt:

- **Giải thuật Khử trùng lặp cục bộ (Custom NMS):** Trong thực tế, AI thỉnh thoảng nhận diện một ký tự thành hai hộp Box chồng lên nhau. Hàm \_remove_duplicate_chars được thiết kế thủ công để đo khoảng cách trục X giữa tâm của các hộp. Nếu hai ký tự quá gần nhau (nhỏ hơn 50% chiều rộng), thuật toán tự động giữ lại ký tự có độ tin cậy (conf) cao hơn và triệt tiêu ký tự nhiễu.
- **Thuật toán Phân dòng động (Dynamic Line Splitting):** Khác với phương pháp chia dòng theo tỷ lệ % cố định dễ bị sai số khi biển số bị nghiêng, mã nguồn phát triển thuật toán dò khoảng trống lớn nhất. Bằng cách tính khoảng cách (gap) giữa các tọa độ Y của ký tự liên tiếp, tọa độ có gap lớn nhất (vượt quá 30% chiều cao trung bình của chữ) sẽ được chọn làm vạch phân cách chia biển số thành Line 1 và Line 2 một cách hoàn toàn tự động.
- **Hệ luật hậu xử lý (Rule-based Post-processing):** Để khắc phục điểm mù AI khi nhầm lẫn các ký tự tương đồng (như 8/B, 0/D, 5/S), hệ thống ánh xạ tập quy tắc định dạng biển số xe máy Việt Nam vào bộ từ điển (LETTER_TO_DIGIT và DIGIT_TO_LETTER).
    - Tại Dòng 1: Ép 2 ký tự đầu bắt buộc là SỐ, ký tự thứ ba bắt buộc là CHỮ.
    - Tại Dòng 2: Ép toàn bộ các ký tự bắt buộc là SỐ. Nhờ cơ chế nội suy thông minh này, tỷ lệ sai lệch chuỗi OCR đầu ra gần như được triệt tiêu hoàn toàn.

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

# CHƯƠNG 6: ĐÁNH GIÁ THỰC NGHIỆM VÀ TỔNG KẾT

Sau khi hoàn thiện toàn bộ quy trình từ nghiên cứu lý thuyết, huấn luyện các mô hình học sâu cho đến thiết lập kiến trúc phần mềm điều phối và xây dựng giao diện ứng dụng, nhóm nghiên cứu tiến hành triển khai thử nghiệm hệ thống trong môi trường thực nghiệm. Chương này tập trung trình bày các kết quả nhận diện trực quan thu được trên giao diện Web Dashboard, phân tích hiệu năng và tính ổn định của hệ thống quản lý dữ liệu đầu ra (I/O), từ đó đưa ra những kết luận tổng quan và định hướng phát triển phần mềm trong tương lai.

## 6.1. Kết quả nhận diện trên giao diện ứng dụng

Hệ thống được khởi chạy trên môi trường máy chủ cục bộ tích hợp giao diện tương tác Web. Tập dữ liệu thử nghiệm bao gồm các hình ảnh tĩnh và luồng video động thu thập từ camera giám sát giao thông đô thị tại Việt Nam với các góc máy chéo từ trên cao, mật độ phương tiện phức tạp và điều kiện ánh sáng thay đổi. Kết quả thực nghiệm ghi nhận trên giao diện phần mềm cho thấy sự phối hợp đồng bộ và chính xác giữa khối xử lý lõi và tầng giao diện đồ họa.

### 6.1.1. Thực nghiệm xử lý dữ liệu ảnh tĩnh (Static Image Evaluation)

Khi người dùng tải lên một tệp tin hình ảnh tĩnh thông qua bảng điều khiển Sidebar, bộ điều phối trung tâm lập tức kích hoạt luồng xử lý song song thông qua phương thức .predict(). Kết quả kết xuất đồ họa (Rendering) trên màn hình trung tâm hiển thị độ phân tách hành vi vô cùng rõ rệt:

- **Đối với các phương tiện chấp hành nghiêm chỉnh luật giao thông:** Hệ thống tự động bao quanh đối tượng người điều khiển xe máy bằng một hộp giới hạn (Bounding Box) màu xanh lá cây chuẩn. Trên đỉnh hộp hiển thị nhãn AN TOAN đi kèm điểm số tin cậy (Confidence Score) dao động từ (0.85) đến (0.98).
- **Đối với các phương tiện vi phạm hành vi không đội mũ bảo hiểm:** Hộp giới hạn lập tức chuyển sang sắc đỏ chủ đạo để thu hút sự chú ý của người giám sát. Nhãn hiển thị trên khung bao được cấu trúc theo định dạng tường minh: PHAT NGUOI - \[Chuỗi ký tự biển số đã OCR\].
- **Tương tác số liệu:** Ngay khi bức ảnh được xử lý xong, hệ thống tiến hành cập nhật tức thời các cấu phần thông số số liệu (Metrics Dashboard). Biến _Tổng số xe phát hiện_ và _Số xe vi phạm_ nhảy số chính xác, đồng thời bảng Pandas DataFrame phía dưới tự động nạp thêm thông tin chi tiết của phương tiện vi phạm mà không có độ trễ.

**_Hình 6.1: Kết quả nhận diện phân loại hành vi trên ảnh tĩnh_**_._

### 6.1.2. Thực nghiệm xử lý luồng video động (Dynamic Video Stream Evaluation)

Thực nghiệm trên luồng dữ liệu video động là bài toán phức tạp nhất, đòi hỏi tính ổn định và tốc độ phản hồi cao. Khi kích hoạt chế độ quét video, giao diện phần mềm hoạt động theo cơ chế phản hồi động:

- **Tính liên tục của luồng hiển thị:** Nhờ cơ chế giải phóng bộ nhớ và kết xuất đè khung hình thông qua vùng không gian chứa động, luồng video phát trên nền Web đạt sự mượt mà tối ưu. Thanh tiến trình (Progress Bar) chạy tịnh tiến theo thời gian thực, hiển thị chính xác tỷ lệ phần trăm khung hình đã xử lý trên tổng số khung hình của video.
- **Độ ổn định của khung bao bám vết (Tracking Stability):** Bộ lọc bám vết hoạt động hiệu quả khi liên tục duy trì một định danh (Track ID) duy nhất cho mỗi chiếc xe máy di chuyển xuyên suốt góc quay của camera. Khung bao màu đỏ bám sát theo quỹ đạo chuyển động của người vi phạm mà không xảy ra hiện tượng rung lắc hay nhảy cóc khung bao. Kể cả khi phương tiện bị che khuất một phần bởi các phương tiện khác hoặc đi qua vùng bóng râm, hệ thống vẫn giữ vững ID và duy trì trạng thái cảnh báo PHAT NGUOI.

**_Hình 6.2: Kết quả nhận diện phân loại hành vi trên video động_**_._

### 6.1.3. Đánh giá năng lực xử lý các góc chết vật lý (Edge-Case Validation)

Điểm vượt trội của hệ thống hoàn chỉnh so với luồng mô hình thô (Baseline) ban đầu được chứng minh rõ rệt thông qua khả năng xử lý các bức ảnh thuộc diện khó (Edge cases):

- **Khắc phục lỗi lóa sáng cục bộ:** Đối với các khung hình bị ánh nắng gắt chiếu trực tiếp hoặc lóa đèn pha từ phương tiện đối diện, vùng ảnh cắt biển số thô ban đầu bị mờ nhạt nét chữ. Tuy nhiên, sau khi đi qua bộ lọc cân bằng tương phản cục bộ kênh độ sáng (CLAHE) trong module xử lý ảnh, hình ảnh hiển thị trên giao diện kiểm duyệt (Gallery) cho thấy các nét chữ và số màu đen đã được đẩy bật lên rõ rệt trên nền trắng của biển số, giúp mô hình OCR bóc tách trơn tru.
- **Khắc phục lỗi lệch phối cảnh hình học:** Khi phương tiện rẽ cua hoặc camera đặt ở góc quá chéo, biển số bị biến dạng thành hình bình hành hoặc hình thang. Giao diện lưu trữ ghi nhận các hình ảnh bằng chứng đã được thuật toán xoay ma trận tự động nắn thẳng về một mặt phẳng trực diện hoàn hảo trước khi đọc chữ. Do đó, các chuỗi ký tự hiển thị trên màn hình không còn hiện tượng xáo trộn vị trí, đảm bảo cấu trúc chuẩn của biển số xe máy Việt Nam.

**_Hình 6.3: Hình ảnh so sánh trước và sau khi đi qua bộ lọc xử lý ảnh lõi_**_._

## 6.2. Đánh giá tính ổn định hệ thống I/O và Quản lý dữ liệu

Một hệ thống AI ứng dụng trong thực tế không chỉ cần năng lực nhận diện xuất sắc mà còn phải sở hữu cơ chế quản lý dữ liệu đầu ra thông minh, bảo toàn tài nguyên bộ nhớ và đảm bảo tính vẹn toàn thông tin pháp lý. Quá trình nghiệm thu thư mục lưu trữ tự động của phần mềm đã chứng minh tính hiệu quả của tư duy kiến trúc hệ thống đã thiết kế.

### 6.2.1. Kiểm định cơ chế chống bùng nổ dữ liệu (Anti-Spamming Validation)

Ở mô hình thô ban đầu, việc một chiếc xe vi phạm di chuyển qua camera tạo ra hàng chục dòng log trùng lặp trong file báo cáo, gây rác dữ liệu. Khi tiến hành kiểm tra tệp tin cơ sở dữ liệu Danh_Sach_Phat_Nguoi.csv được sinh ra tự động, kết quả đạt mức lý tưởng:

- **Tối ưu hóa dữ liệu dạng bảng:** Nhờ cơ chế ghi log trì hoãn (Deferred Logging), hệ thống quản lý trạng thái đã hoạt động như một bộ lọc thông minh. Mỗi một Track ID vi phạm duy nhất, dù xuất hiện trong luồng video kéo dài hàng trăm khung hình, cũng chỉ được hệ thống chốt hạ và ghi nhận đúng 01 dòng duy nhất vào tệp báo cáo.
- **Tính chính xác của nội dung biên bản:** Cấu trúc bảng tính được tổ chức ngay ngắn, phân tách rõ ràng các trường dữ liệu nghiệp vụ bao gồm: _Số thứ tự_, _Thời gian ghi nhận_, _Biển số xe hệ thống bóc tách_, _Loại vi phạm (Không đội mũ bảo hiểm)_ và _Đường dẫn lưu trữ hình ảnh bằng chứng_. Hoàn toàn không ghi nhận hiện tượng rỗng dòng hoặc ghi đè sai lệch dữ liệu.

**_Hình 6.4: Tệp tin báo cáo Danh_Sach_Phat_Nguoi.csv_**_._

### 6.2.2. Kiểm định chất lượng hồ sơ chứng cứ và Phân định đa nguồn

Tiến hành rà soát thư mục lưu trữ hình ảnh tự động của ứng dụng, các tiêu chuẩn về mặt pháp lý đều được đáp ứng nghiêm ngặt:

- **Chất lượng hình ảnh bằng chứng tối ưu:** Các bức ảnh cắt cận cảnh phương tiện vi phạm được lưu lại không phải là những bức ảnh mờ nhòe lúc xe ở quá xa. Thuật toán chọn lọc động đã giữ đúng cam kết: Bức ảnh được trích xuất là khoảnh khắc chiếc xe di chuyển đến vùng lân cận camera nhất, cho ra độ phân giải vùng biển số to nhất và chứa chuỗi OCR dài nhất.
- **Cơ chế mã hóa thời gian chống xung đột:** Định dạng tên tệp tin được lưu theo quy chuẩn nghiêm ngặt (Ví dụ: ViPham_43D112345_20260518_101530_456789.jpg). Việc tích hợp thành công hậu tố chuỗi thời gian chính xác đến mức **mili-giây (microsecond)** đã hoạt động như một lớp bảo mật dữ liệu. Thử nghiệm cho 2 phương tiện cùng vi phạm lọt vào ống kính camera tại cùng một giây cho thấy hệ thống vẫn sinh ra 2 tệp tin bằng chứng độc lập, triệt tiêu hoàn toàn lỗi xung đột ghi đè file (File Collision Error).
- **Tính hữu dụng của chức năng xuất hồ sơ:** Khi người dùng sử dụng tính năng "Tải xuống Báo cáo & Bằng chứng", hệ thống thực hiện nén tệp tin ngầm với tốc độ cao. File kết quả .ZIP tải về máy tính giải nén ra chứa trọn vẹn tệp CSV và thư mục ảnh chứng cứ, sẵn sàng đưa vào quy trình xử lý nghiệp vụ của cơ quan chức năng.

**_Hình 6.5: Cấu trúc thư mục ảnh và tệp tin nén ZIP tải về thành công_**_._

## 6.3. Kết luận chung và Thành tựu của Đồ án

Trải qua toàn bộ chu kỳ nghiên cứu, phát triển và triển khai thử nghiệm thực tế, đồ án đã hoàn thành xuất sắc tất cả các mục tiêu chiến lược đề ra ban đầu. Dự án đã chứng minh tính khả thi to lớn khi kết hợp giữa mô hình học sâu hiện đại và tư duy kiến trúc phần mềm tối ưu.

**Các thành tựu cốt lõi đạt được bao gồm:**

- **Xây dựng thành công Động cơ AI hiệu năng cao:** Huấn luyện thành công 3 mô hình YOLO26 trên cấu hình mạng Large (26l) với các tập dữ liệu tinh chỉnh riêng biệt. Mô hình đạt độ hội tụ lý tưởng trên các biểu đồ loss, đạt chỉ số mAP cao vượt trội và giải quyết tốt bài toán phân lớp nhầm lẫn ký tự thông qua ma trận nhầm lẫn (Confusion Matrix).
- **Kiến trúc luồng xử lý 3 cấp độ (3-Stage Pipeline) tối ưu:** Thiết lập đường ống dẫn dữ liệu phân cấp rạch ròi, áp dụng cơ chế _Clean Cropping_ giúp bảo toàn độ nguyên bản của điểm ảnh, ngăn chặn hiện tượng nhiễu chéo đồ họa giữa các mô hình học sâu nằm nối tiếp nhau.
- **Phát triển thành công các Module Logic cốt lõi (Middleware Engine):** Không phụ thuộc hoàn toàn vào AI, đồ án đã khẳng định chất xám lập trình thông qua các bộ thuật toán tự thiết kế: Cân bằng tương phản LAB-CLAHE, nắn thẳng hình học Deskewing bằng biến đổi Hough, giải thuật sắp xếp chữ cái OCR theo không gian 2D, hệ luật suy diễn ép quy tắc biển số Việt Nam và cơ chế bám vết trì hoãn _Deferred Logging_.
- **Đóng gói sản phẩm phần mềm thương mại hoàn chỉnh:** Chuyển đổi thành công các thuật toán phức tạp thành một ứng dụng Web Dashboard trực quan bằng Flask và Tailwind CSS. Ứng dụng tích hợp các kỹ thuật lập trình Frontend nâng cao như AI Caching chống tràn RAM, quản lý tệp tạm thời chống đầy ổ cứng, tiêm mã CSS tùy biến giao diện và đóng gói hồ sơ tự động sang định dạng ZIP.

## 6.4. Hạn chế còn tồn đọng và Định hướng phát triển tương lai

Dù đạt được những kết quả vô cùng khả quan và có tính ứng dụng cao, nhóm nghiên cứu vẫn khách quan nhìn nhận những giới hạn công nghệ hiện tại để vạch ra lộ trình nâng cấp hệ thống một cách rõ ràng.

### 6.4.1. Hạn chế khách quan còn tồn đọng

- **Rào cản về tài nguyên phần cứng:** Việc vận hành liên tiếp 3 mô hình học sâu thuộc cấu hình mạng Large (26l) đòi hỏi năng lực điện toán rất lớn để xử lý các phép toán ma trận phức tạp. Khi triển khai trên các dòng máy tính sử dụng CPU thuần túy, tốc độ xử lý khung hình (FPS) sẽ bị sụt giảm đáng kể. Hệ thống bắt buộc phải có sự hỗ trợ của các bộ xử lý đồ họa chuyên dụng (GPU) hoặc chip tăng tốc Tensor nếu muốn đạt tốc độ xử lý Real-time hoàn hảo trên luồng video độ phân giải cao.
- **Giới hạn môi trường vật lý cực đoan:** Trong điều kiện mật độ giao thông quá dày gắt (ùn tắc nghiêm trọng giờ cao điểm), các phương tiện che khuất lẫn nhau trong thời gian dài khiến bộ lọc bám vết thỉnh thoảng xảy ra hiện tượng hoán đổi định danh (ID Switch). Ngoài ra, vào ban đêm thiếu sáng nghiêm trọng hoặc khi biển số xe bị bùn đất che lấp hoàn toàn, hệ thống (cũng như mắt người) sẽ gặp khó khăn trong việc bóc tách chính xác ký tự chữ.

### 6.4.2. Định hướng phát triển tương lai

Dựa trên các hạn chế nêu trên, nhóm nghiên cứu hoạch định 3 mục tiêu chiến lược nhằm nâng cấp hệ thống lên quy mô doanh nghiệp:

- **Tối ưu hóa tốc độ suy luận bằng TensorRT/ONNX:** Tiến hành đóng gói và biên dịch (Export) các tệp trọng số mạng nơ-ron từ định dạng gốc .pt sang các định dạng trung gian tối ưu hóa phần cứng như **ONNX** hoặc **TensorRT**. Đồng thời áp dụng kỹ thuật lượng tử hóa dữ liệu (Quantization sang định dạng FP16 hoặc INT8). Giải pháp này sẽ giúp giảm dung lượng mô hình, nhân băng thông FPS lên gấp nhiều lần, cho phép hệ thống chạy mượt mà trên cả các thiết bị nhúng vi bìa nhỏ gọn (Edge AI) như Nvidia Jetson.
- **Chuyển đổi sang Hệ quản trị Cơ sở dữ liệu đám mây (Cloud Database):** Thay thế cơ chế ghi tệp tin cục bộ (file .csv thô) bằng việc tích hợp các hệ quản trị cơ sở dữ liệu quan hệ chuyên nghiệp như **PostgreSQL** hoặc **MySQL** cấu hình trên Cloud. Sự nâng cấp này cho phép hệ thống lưu trữ hàng triệu biên bản vi phạm, hỗ trợ cơ chế đồng bộ đa luồng an toàn và cho phép các cơ quan chức năng truy xuất, kết xuất báo cáo phạt nguội từ xa thông qua hệ thống tài khoản phân quyền bảo mật.
- **Mở rộng kịch bản nhận diện hành vi giao thông thông minh:** Tận dụng kiến trúc Pipeline linh hoạt sẵn có của phần mềm, nhóm sẽ tiếp tục thu thập dữ liệu và huấn luyện thêm các mô hình phụ trợ để tích hợp các tính năng thực tế khác như: Phát hiện phương tiện vượt đèn đỏ, phát hiện hành vi đi ngược chiều, lấn làn đường hoặc chở quá số người quy định, biến phần mềm thành một giải pháp Trung tâm Giám sát Giao thông Toàn diện (Smart Traffic City Control Center).