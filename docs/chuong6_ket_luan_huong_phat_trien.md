# CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

> Chương 5 đã trình bày các đóng góp và giải pháp kỹ thuật nổi bật rút ra
> trong quá trình thực hiện đề tài. Chương 6 tổng kết lại toàn bộ quá trình:
> mục 6.1 đánh giá những gì đã đạt được, đặt cạnh các hệ thống/công trình
> tương tự, nhìn nhận trung thực những hạn chế còn tồn tại và rút ra bài học
> kinh nghiệm; mục 6.2 đề xuất hướng phát triển, trước hết là các phần việc
> cần thiết để hoàn thiện các chức năng đã làm, sau đó là các hướng nâng cấp
> và mở rộng mới.

---

## 6.1 Kết luận

Đề tài đã xây dựng và vận hành thử một **hệ thống camera AI giám sát giao
thông** trên video CCTV thật tại giao lộ Phố Huế – Trần Khát Chân (Hà Nội),
hiện thực hoá đầy đủ bốn use case: phát hiện và theo dõi phương tiện trong từng
luồng video (UC1, YOLO11s_vn + ByteTrack), phát hiện vi phạm vượt đèn đỏ/vượt
vạch dừng/đi sai làn theo hệ luật (UC2), dự đoán hướng di chuyển dài hạn của
phương tiện trên bản đồ đường thực (UC3, Gradient Boosting + Route Predictor
trên đồ thị OpenStreetMap), và cấu hình hệ thống (UC4). Toàn bộ pipeline được
kiểm chứng trực tiếp trên dữ liệu CCTV thật, không chỉ trong môi trường mô
phỏng — đáp ứng đúng mục tiêu ban đầu là một hệ thống triển khai được trên hạ
tầng camera giao thông thực tế của Việt Nam, đặc biệt với cơ cấu phương tiện
có tỷ trọng xe máy rất cao.

**So sánh với các hệ thống/công trình tương tự.** Các hệ thống "phạt nguội"
đang triển khai tại Việt Nam tập trung vào *ghi nhận vi phạm và đọc biển số để
xử phạt* tại một điểm cố định, nhưng không theo dõi liên tục hành trình phương
tiện và không dự đoán hướng đi tương lai. Ở chiều nghiên cứu, công trình gần
nhất cùng bối cảnh giao thông hỗn loạn nhiều xe máy tại Việt Nam là Chau et al.
[1], dự đoán *toạ độ ngắn hạn* của phương tiện trong phạm vi một camera bằng
CNN-LSTM/CNN-GRU, nhưng không suy diễn hành trình qua nhiều ngã tư. Đề tài này
đi xa hơn ở chỗ kết hợp ba khối thường tách rời trong các hệ thống trên thành
một luồng hoàn chỉnh chạy thật: phát hiện vi phạm theo hệ luật minh bạch (UC2),
phân loại hướng rẽ tại từng ngã tư rồi *suy diễn liên tiếp qua nhiều ngã tư*
trên đồ thị đường thực và vẽ lên bản đồ (UC3), tất cả trên cùng một hạ tầng
theo dõi đa luồng thời gian thực (UC1) — và quan trọng hơn, mỗi thành phần đều
được đánh giá định lượng *trên dữ liệu thật* kèm thừa nhận trung thực về độ
tin cậy của từng con số, thay vì chỉ báo cáo kết quả mô phỏng.

**Những gì đã làm được.** Về định lượng (Bảng 4.5): YOLO11s_vn đạt mAP50 =
89,5%; Goal Classifier real-world đạt accuracy = 60,12%, balanced accuracy =
54,18% trên tập kiểm thử tách riêng khỏi tập huấn luyện. Về kỹ thuật, đề tài
giải quyết được bốn vấn đề/đóng góp đáng chú ý (Chương 5): (1) vượt rào cản
thiếu dữ liệu bằng quy trình tự động gán nhãn + kiểm định mẫu (15.957 ảnh/
623.329 box); (2) phát hiện và sửa lệch tham số horizon của Goal Classifier
(accuracy 59,73% → 60,12%) kèm so sánh kiến trúc và hiệu chỉnh xác suất; (3)
làm cho dự đoán hành trình *minh bạch* bằng cách sửa lỗi tìm đường và thay giả
định "100% đi thẳng" bằng prior; (4) xây dựng quy trình đo MOTA/IDF1 trên CCTV
thật cùng việc chỉ rõ giới hạn cấu trúc của nó.

**Những gì chưa làm được (ghi nhận trung thực).** Một số hạn chế vẫn còn,
phần lớn nằm ở *quy mô dữ liệu kiểm thử thật* hơn là ở kiến trúc hệ thống:

- **Đánh giá phát hiện chưa độc lập.** mAP50 = 89,5% được đo lại trên 800 ảnh
  lấy ngẫu nhiên từ chính pool huấn luyện (không lưu được val split gốc), nên
  có thể lạc quan hơn năng lực tổng quát hoá thật; chưa có tập đánh giá độc lập
  gán nhãn tay ở địa điểm/thời điểm khác. Đường cong Loss/mAP theo epoch của
  YOLO11s_vn cũng chưa trình bày được do nhật ký huấn luyện gốc (`results.csv`)
  không được lưu lại.
- **MOTA/IDF1 thật cho ByteTrack chưa hoàn tất.** Quy trình và công cụ đã xây
  xong (mục 5.4) nhưng ground-truth bán tự động có hạn chế cấu trúc (FN/FP ≈ 0
  → MOTA bị thổi phồng, chỉ IDF1/số lần đổi ID có ý nghĩa); cần hoàn tất gán
  nhãn toàn bộ video và bổ sung một bộ ground-truth độc lập để đo đúng tỷ lệ
  bỏ sót phát hiện.
- **Hai trong ba loại vi phạm UC2 chưa bật trên camera Phố Huế.** Đi sai làn
  (`LANE_VIOLATION`) đã hoạt động và minh hoạ được trên dữ liệu thật (Hình
  4.15); còn vượt đèn đỏ/vượt vạch dừng (`RED_LIGHT_VIOLATION`) tạm thời bị tắt
  do vùng quan tâm (ROI) vạch dừng/đèn tín hiệu được hiệu chỉnh thủ công từ một
  khung hình mẫu đang đặt chưa đúng vị trí, cần xác định lại từ một đoạn video
  đủ thoáng trước khi bật lại (xem 6.2.1).
- **Recall lớp "quay đầu" của Goal Classifier còn thấp (36,11%)** và đo trên
  tập kiểm thử rất nhỏ (36 mẫu) — chưa đủ tin cậy thống kê; nguyên nhân kỹ
  thuật đã biết là tín hiệu chuyển động đặc trưng của hành vi quay đầu chỉ rõ
  khi phương tiện đã đi vào trong lòng giao lộ.
- **Demo chưa diễn tập trực tiếp trước hội đồng** dù đã kiểm tra đầu-cuối
  (mục 4.5).

**Bài học kinh nghiệm.** Xuyên suốt quá trình, hai bài học nổi lên rõ nhất.
Thứ nhất, *một hệ thống "chạy không lỗi" không có nghĩa là chạy đúng*: nhiều
lỗi đáng kể nhất (lệch tham số horizon giữa huấn luyện/suy luận, giả định hiển
thị tin cậy 100% cho dự đoán xa) không hề gây exception, chỉ lộ ra khi đối
chiếu với dữ liệu/hành vi thật. Thứ hai, *thừa nhận đúng mức bất định và giới
hạn của từng phép đo* (mAP đo trên chính pool huấn luyện, MOTA bị thổi phồng
theo cấu trúc của ground-truth bán tự động) là điều kiện để báo cáo trung
thực, quan trọng không kém việc đạt được con số đẹp.

## 6.2 Hướng phát triển

Các hướng dưới đây chia làm ba nhóm: (A) hoàn thiện các chức năng đã làm —
xuất phát trực tiếp từ các hạn chế ở mục 6.1; (B) cải thiện và nâng cấp các
chức năng hiện có; (C) các hướng đi mới mở rộng phạm vi hệ thống.

### 6.2.1 Hoàn thiện các chức năng đã làm

- **Hoàn tất đo MOTA/IDF1 thật cho ByteTrack:** hoàn thành gán nhãn (sửa cột
  ID) cho toàn bộ video Phố Huế, đồng thời xây thêm một bộ ground-truth *độc
  lập* (gán nhãn box từ đầu, không tái dùng track dự đoán) cho ít nhất một đoạn
  ngắn để đo đúng tỷ lệ bỏ sót phát hiện (FN) — khắc phục hạn chế cấu trúc đã
  nêu ở mục 5.4.
- **Xây tập đánh giá phát hiện độc lập:** thu thêm 2–3 video ngắn ở địa
  điểm/thời điểm khác với dữ liệu huấn luyện, gán nhãn tay (không qua
  auto-label), để đo khả năng tổng quát hoá thật của YOLO11s_vn và thay con số
  mAP đo-trên-pool-huấn-luyện hiện tại bằng một con số held-out đáng tin. Lưu
  lại đầy đủ nhật ký huấn luyện (`results.csv`) ở lần huấn luyện tới để bổ sung
  đường cong Loss/mAP cho báo cáo.
- **Hoàn thiện phát hiện vượt đèn đỏ/vượt vạch dừng (UC2):** xác định lại đúng
  vùng vạch dừng và đèn tín hiệu chính điều khiển làn xe — quay hoặc tìm một
  đoạn video tại đúng camera này lúc đường thoáng đủ lâu để dò được đúng vạch
  sơn/vị trí đèn — rồi bật lại `RED_LIGHT_VIOLATION` (hiện đang tắt tạm), kèm
  bổ sung dữ liệu có gán nhãn đúng/sai để hiệu chỉnh và kiểm chứng luật theo
  từng camera trước khi tin dùng.
- **Bổ sung mẫu "quay đầu" cho Goal Classifier:** quay thêm video tại các vị
  trí có dải phân cách cho phép quay đầu, nơi hành vi này xuất hiện thường
  xuyên hơn, để tăng số mẫu kiểm thử (hiện 36 mẫu) lên mức đủ tin cậy thống kê.
- **Diễn tập demo trước hội đồng** trên đúng video Phố Huế – Trần Khát Chân.

### 6.2.2 Cải thiện và nâng cấp các chức năng hiện có

- **YOLO11 — tăng đa dạng domain huấn luyện:** mở rộng thêm các bộ dữ liệu
  camera giám sát công khai (VisDrone, UA-DETRAC...) và kiểm tra chéo trên
  nhiều địa điểm/góc camera khác nhau để nâng khả năng tổng quát hoá của
  detector, giảm rủi ro mô hình "học vẹt" đặc điểm của các địa điểm đã quen.
- **Goal Classifier — cải thiện recall "quay đầu" và "rẽ trái":** hướng khả
  thi nhất cho "quay đầu" là dự đoán *trong* giao lộ (in-junction) thay vì chỉ
  dựa vào giai đoạn tiếp cận; với "rẽ trái", bổ sung đặc trưng cấu trúc đường
  (ID làn vào, kết nối giao lộ) thay vì chỉ dựa vào động học quan sát được.
- **Dự đoán quỹ đạo bằng học sâu trên dữ liệu thật:** bổ sung một mô hình dự
  đoán quỹ đạo tuần tự (CNN-LSTM/CNN-GRU) huấn luyện trực tiếp trên toạ độ
  *pixel-space* trích từ chính pipeline (YOLO11s_vn + ByteTrack) — đúng dạng
  dữ liệu mà Chau et al. [1] đã dùng trên video CCTV Việt Nam — để dự đoán
  toạ độ ngắn hạn chính xác hơn bộ lọc Kalman hiện tại, bổ trợ cho dự đoán
  hướng rẽ dài hạn của Goal Classifier.
- **Ước lượng prior của Route Predictor trên dữ liệu thật:** prior phân bố
  hướng đi tại các ngã tư xa (mục 5.3) cần được đo trên dữ liệu quỹ đạo qua
  nhiều giao lộ liên tiếp của giao thông thật; khi đã có dữ liệu theo dõi xuyên
  nhiều giao lộ (xem hướng Re-ID ở 6.2.3), có thể ước lượng prior này trực tiếp
  tại khu vực triển khai thay cho giá trị tạm hiện nay.
- **Mở rộng quy mô và tối ưu hiệu năng:** xác minh mục tiêu ~15–20 FPS trên
  đúng cấu hình 6 camera đồng thời (hiện đã áp dụng phát hiện theo lô và điều
  tiết tần suất — mục 4.5), cân nhắc lượng tử hóa/triển khai biên (edge) để
  giảm tải máy chủ khi nhân rộng số camera.

### 6.2.3 Các hướng đi mới

- **Đánh dấu đối tượng và lưu biển số (mark & plate capture):** đề tài đã thử
  nghiệm OCR biển số qua nhiều vòng nhưng *chưa tích hợp vào sản phẩm cuối* do
  số liệu còn thấp. Kết luận khảo sát: với phương tiện đủ gần/đủ lớn (ô tô,
  một phần xe máy), một mô hình chuyên biệt cho biển Việt Nam đọc được ở mức
  đáng kể, và kỹ thuật hợp nhất nhiều khung hình theo track (multi-frame
  fusion) nâng tỷ lệ đọc đúng theo-track lên ~25% cho ô tô; nhưng phần lớn vùng
  biển xe máy phát hiện được chỉ rộng 14–33 px — xác nhận là *sàn vật lý cảm
  biến* (không đủ điểm ảnh để phân giải ký tự), chỉ camera zoom/PTZ chuyên chụp
  biển số mới giải quyết triệt để. Đây là hướng mở rộng tự nhiên cho năng lực
  "đánh dấu đối tượng nghi vấn", phụ thuộc vào việc nâng cấp phần cứng camera
  hoặc huấn luyện lại trên dữ liệu đúng độ phân giải.
- **Theo dõi xuyên camera (Re-Identification):** mở rộng UC1 từ một luồng sang
  nhiều camera liền tuyến để truy vết phương tiện qua các giao lộ liên tiếp —
  cần quay đồng thời ít nhất hai video tại hai giao lộ liền kề (chưa có trong
  dữ liệu hiện có) để kiểm chứng. Cần lưu ý một hạn chế đã biết: khả năng phân
  biệt lại của đặc trưng hình ảnh với *xe máy* còn yếu (do thiếu bộ dữ liệu
  Re-ID chuyên cho xe máy — các bộ công khai chủ yếu là ô tô), nên hướng này
  cần đầu tư thêm dữ liệu xe máy trước khi tin dùng cho loại phương tiện chiếm
  tỷ trọng lớn nhất tại Việt Nam.
- **Phát hiện vi phạm bằng học máy không giám sát:** khi đã tích lũy đủ dữ liệu
  vận hành thật, bổ sung một hướng phát hiện bất thường không giám sát
  (Isolation Forest, autoencoder — mục 3.4) song song với hệ luật, để bắt các
  hành vi bất thường chưa được mô hình hoá thành luật cụ thể.

---

Đề tài đã hoàn thành một hệ thống camera AI giám sát giao thông đa luồng hoạt
động được trên video CCTV thật, với cả bốn use case (phát hiện & theo dõi,
phát hiện vi phạm, dự đoán hướng di chuyển trên bản đồ, cấu hình) đều có minh
hoạ và số liệu đo trên dữ liệu thật — không dừng ở kết quả mô phỏng. Các hạn
chế còn tồn tại chủ yếu nằm ở quy mô dữ liệu kiểm thử thật và việc hiệu chỉnh
tham số theo từng camera, hơn là ở kiến trúc hệ thống; và các hướng phát triển
đề xuất ở mục 6.2 đều là những phần việc cụ thể, khả thi để tiếp tục hoàn thiện
sau đề tài này.

---

## TÀI LIỆU THAM KHẢO (Chương 6)

[1] T. Chau, D. V. Ngo, M. T. Nguyen, A. D. Nguyen-Tran, T. H. Do, "Leverage
Deep Learning Methods for Vehicle Trajectory Prediction in Chaotic Traffic,"
in *Intelligence of Things: Technologies and Applications (ICIT 2023)*,
Lecture Notes on Data Engineering and Communications Technologies, vol. 187,
Springer, Cham, 2023. DOI: 10.1007/978-3-031-46573-4_24.
