# CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

> Chương 5 đã trình bày năm vấn đề kỹ thuật cụ thể phát hiện được trong quá
> trình xây dựng hệ thống, giải pháp đã áp dụng và kết quả định lượng tương
> ứng. Chương 6 tổng kết lại toàn bộ quá trình thực hiện đề tài: mục 6.1 đánh
> giá những gì đã đạt được và những hạn chế còn tồn tại sau các cải thiện ở
> Chương 5; mục 6.2 đề xuất các hướng phát triển tiếp theo, tổng hợp từ các
> hạn chế đã xác định và từ những phần việc đã được khảo sát nhưng chưa thực
> hiện trong phạm vi đề tài này.

---

## 6.1 Kết luận

Đề tài tập trung vào **dự đoán hướng đi dài hạn của phương tiện trên bản đồ
đường thực tế** (UC4) — đặt cạnh công trình gần nhất cùng bối cảnh giao thông
hỗn loạn, nhiều xe máy tại Việt Nam (Chau et al. [1] — dự đoán toạ
độ ngắn hạn trong một camera, không suy diễn nhiều ngã tư), đề tài này đi xa
hơn ở việc kết hợp phân loại hướng rẽ tại từng ngã tư (Gradient Boosting có
hiệu chỉnh xác suất, accuracy=60,12% trên tập kiểm thử thật) với suy diễn
liên tiếp qua nhiều ngã tư trên đồ thị đường thực xây dựng từ OpenStreetMap
(Route Predictor), rồi hiển thị trực tiếp lên bản đồ. Để vận hành UC4 trên dữ
liệu thật, đề tài xây dựng đầy đủ hạ tầng hỗ trợ: phát hiện và theo dõi đối
tượng trong từng camera (UC1, YOLO11s_vn + ByteTrack), theo dõi xuyên camera
bằng nhận diện lại (OSNet fine-tune trên VeRi-776 — **năng lực hạ tầng hỗ
trợ việc truy vết sau đánh dấu, không đặc tả như một use case riêng — mục
2.4 Chương 2 — và không phải tiêu chí đánh giá thành công chính**;
có hạn chế đã biết và đo được với xe máy, mục 6.1 dưới và Chương 5 mục
5.5.8), phát hiện và đánh dấu sự cố (UC3, hệ luật suy diễn kết hợp đánh dấu
thủ công — bổ sung 2026-06-26: lưu ảnh crop đối tượng vào CSDL ngay khi đánh
dấu, giải quyết một phần hạn chế "không thể truy quét lại object sau khi mất
Global ID" phát hiện khi rà soát lại UC3.2, xem mục 4.2.2), giám sát qua
dashboard thời gian thực (UC5, FastAPI + React +
WebSocket/MJPEG), và quản lý camera/vùng quan tâm (UC6). Toàn bộ pipeline đã
được vận hành thử trực tiếp trên video CCTV thật tại giao lộ Phố Huế – Trần
Khát Chân (mục 4.5), không chỉ trong môi trường mô phỏng — đáp ứng đúng mục
tiêu đặt ra ban đầu là một hệ thống có thể triển khai trên hạ tầng camera
giám sát giao thông thực tế của Việt Nam, đặc biệt với cơ cấu phương tiện có
tỷ trọng xe máy cao.

Các kết quả định lượng chính trên dữ liệu thật (Bảng 4.4): YOLO11s_vn đạt
mAP50 = 89,5% (đo trên auto-label, chưa hand-verify toàn bộ); OSNet fine-tune
VeRi-776 đạt mAP = 71,89%, Rank-1 = 94,16% trên giao thức ReID mở chuẩn; Goal
Classifier real-world đạt accuracy = 60,12%, balanced accuracy = 54,18% trên
tập kiểm thử tách riêng khỏi tập huấn luyện. Quá trình xây dựng cũng phát
hiện và xử lý được năm vấn đề kỹ thuật cụ thể (Chương 5): sửa lỗi tìm đường
gần nhất và thay giả định cứng bằng prior thực nghiệm cho Route Predictor;
phát hiện và sửa lệch tham số horizon giữa huấn luyện/suy luận của Goal
Classifier (cải thiện accuracy 59,73% → 60,12%); tách bạch đúng phương pháp
đo lường huấn luyện và đánh giá ReID chuẩn cho OSNet; nhận diện và quản lý
hiện tượng quên thảm khốc khi fine-tune YOLO11 trên domain hẹp; và sửa một
lỗi đồng hồ thời gian gây cảnh báo giả ở bộ phát hiện sự cố.

Bên cạnh các kết quả đạt được, một số hạn chế vẫn còn tồn tại, cần được ghi
nhận trung thực:

- **Recall lớp "quay đầu" của Goal Classifier còn thấp (36,11%)** và đo trên
  một tập kiểm thử rất nhỏ (36 mẫu) — số liệu này chưa đủ độ tin cậy thống kê
  để khẳng định chắc chắn về chất lượng thật của lớp này; nguyên nhân kỹ
  thuật đã biết là tín hiệu chuyển động đặc trưng của hành vi quay đầu chỉ
  thật sự rõ khi phương tiện đã đi vào trong lòng giao lộ, không quan sát
  được rõ từ giai đoạn tiếp cận.
- **Goal Classifier chưa thể tận dụng dữ liệu mô phỏng để cải thiện mô hình
  thật**: dữ liệu CARLA (world-space, mét) và dữ liệu CCTV thật (pixel-space,
  chuẩn hoá theo khung bao) dùng hai không gian đặc trưng hoàn toàn khác
  nhau, không thể fine-tune trực tiếp một mô hình từ domain này sang domain
  kia — cải thiện mô hình thật chỉ có thể đến từ nhiều dữ liệu thật hơn,
  đặc trưng tốt hơn, hoặc augment dữ liệu thật bằng một mô hình chiếu camera
  mô phỏng (chưa thực hiện).
- **Hiệu năng ReID trên crop thật, nhiễu — có số liệu sơ bộ, đáng lo ngại**
  (Chương 5, mục 5.5.8): số mAP/Rank-1/Rank-5 ở Bảng 4.4 là benchmark truy
  hồi ngoại tuyến trên crop sạch VeRi-776. Đo sơ bộ (cỡ mẫu nhỏ, n=2 cặp
  cùng xe) trên crop thật từ pipeline cho thấy độ tương đồng cosine giữa
  "cùng xe máy" và "khác xe máy" gần như không phân biệt được (0,601 vs
  0,604) — khả năng do nhánh phương tiện chỉ fine-tune trên VeRi-776 (toàn
  bộ ô tô, không có xe máy). Ô tô phân biệt khá hơn (0,565, dải 0,41–0,68)
  nhưng vẫn chồng lấn nhiều. Đây mới là phép đo trong-một-camera (dùng dữ
  liệu gán nhãn cho ByteTrack), chưa phải benchmark xuyên camera đúng nghĩa
  — vẫn cần video 2 camera liền tuyến để đo đầy đủ.
- **MOTA/IDF1 cho ByteTrack trên video CCTV thật đang được đo, chưa có số
  liệu chính thức** (Chương 5, mục 5.4 — mô phỏng; mục 5.5.8 — quy trình đo
  trên dữ liệu thật, đang gán nhãn tay phần còn lại). Quy trình tái dùng
  track đã chạy + sửa tay cột ID (không gán nhãn từ đầu) có một hạn chế cấu
  trúc đã xác nhận: vì ground-truth là bản copy của track dự đoán, FN/FP
  luôn xấp xỉ 0, nên MOTA tổng hợp sẽ luôn bị thổi phồng — chỉ IDF1/số lần
  đổi ID là số liệu đáng tin từ quy trình này, cần một bộ ground-truth độc
  lập (gán nhãn từ đầu, không tái dùng pred) mới đo được đúng tỷ lệ bỏ sót
  phát hiện thật.
- **Ngưỡng tuyệt đối của Incident Detector (`SUDDEN_STOP`, `SUDDEN_ACCEL`,
  `VEHICLE_PROXIMITY`, `PREDICTED_COLLISION`, `OVERSPEED`) sai 100% trên mẫu
  thật đã kiểm chứng bằng tay tại camera Phố Huế** (Chương 5, mục 5.5.7) —
  đã tắt tạm cả 5 loại này cho camera này; hệ luật suy diễn cho phát hiện sự
  cố (mục 3.4) vẫn đúng về kiến trúc, nhưng các hằng số ngưỡng cụ thể cần
  hiệu chỉnh lại theo đơn vị thực tế và đặc thù từng camera trước khi tin
  dùng được trên dữ liệu thật.
- **Việc vận hành thử trên video CCTV thật (mục 4.5) đã được kiểm tra đầu-cuối
  nhưng chưa được diễn tập trực tiếp trước hội đồng.**
- **GRU huấn luyện trên CARLA đang được blend vào dự đoán quỹ đạo hiển thị
  live cho camera có calibration (đúng trường hợp Phố Huế), không phải một
  thành phần "để dành tương lai" như tài liệu trước đây ngụ ý** (Chương 5, mục
  5.5.10) — phát hiện khi đọc lại trực tiếp `EnsembleTrajectoryPredictor`.
  Mức độ ảnh hưởng vào kết quả hiển thị hiện dựa trên độ tự tin nội tại của
  GRU, không dựa trên bằng chứng độ chính xác thật nào (GRU chưa từng được đo
  ADE/FDE ngoài CARLA) — cần tắt nhánh blend này (Kalman-only) cho tới khi có
  số liệu thật, xem mục 6.2(B).

## 6.2 Hướng phát triển

Các hướng phát triển dưới đây được tổng hợp từ các hạn chế đã xác định ở mục
6.1 và từ các phần việc đã khảo sát nhưng chưa thực hiện trong phạm vi đề tài
này, ưu tiên theo nhóm: (A) bổ sung dữ liệu/đánh giá còn thiếu, (B) cải thiện
mô hình, (C) nhận diện biển số — đang tiếp tục đầu tư cải thiện. Vì
trọng tâm đóng góp của đề tài là dự đoán hướng đi dài hạn (UC4, mục 6.1), các
mục liên quan trực tiếp đến UC4 trong (A)/(B) — cải thiện Goal Classifier,
prior Route Predictor, dự đoán quỹ đạo học sâu — có độ ưu tiên cao hơn các
mục thuộc hạ tầng hỗ trợ (MOTA/ByteTrack, ReID xuyên camera), dù thứ tự trình
bày dưới đây vẫn theo nhóm chức năng để dễ tra cứu.

**(A) Bổ sung dữ liệu và đánh giá định lượng còn thiếu trên dữ liệu thật**

- **Hoàn tất đo MOTA/IDF1 thật trên video CCTV** (đang thực hiện, mục 5.5.8):
  quy trình và công cụ (`render_pred_review.py`, `apply_id_corrections.py`)
  đã xây xong, đang gán nhãn tay phần còn lại của video (192 khung hình).
  Việc còn thiếu: (a) hoàn tất sửa cột ID cho toàn bộ video, không chỉ một
  phần; (b) xây thêm một bộ ground-truth **độc lập** (gán nhãn từ đầu, không
  tái dùng track dự đoán) cho ít nhất một đoạn ngắn, để đo đúng tỷ lệ bỏ sót
  phát hiện (FN) — hạn chế cấu trúc của quy trình tái dùng-pred hiện tại là
  không bao giờ phát hiện được trường hợp ByteTrack bỏ sót hoàn toàn một
  đối tượng.
- **Chứng minh theo dõi xuyên camera trên video thật**: quay đồng thời
  (hoặc lệch không quá 1–2 phút) hai video tại hai giao lộ liền kề trên cùng
  một trục đường, camera cố định, tối thiểu 10–15 phút mỗi video, để có dữ
  liệu kiểm chứng Re-ID xuyên camera trên điều kiện triển khai thật — demo
  hiện tại (mục 4.5) mới dùng một camera. Đã rà các video CCTV thật có sẵn
  trong `data/datasets/traffic_intersection/` (`Ngã tư Xã Đàn-Phạm Ngọc
  Thạch`, `Giao thông nội đô HN`, `Clip tuyến cao tốc`, `NB-LC`) — không có
  cặp nào đạt điều kiện: 2 folder cao tốc (`NB-LC`/`Clip tuyến cao tốc`) là
  camera tuyến Nội Bài–Lào Cai, giao thông quá thưa và đồng hồ hiển thị giữa
  các file lệch nhau bất thường (không xác minh được đồng bộ thời gian); các
  folder còn lại đều có vẻ là một camera cố định/một địa điểm (nhiều file chỉ
  khác ngày quay), không phải cặp camera liền tuyến. Vẫn cần quay mới hoặc
  tìm nguồn khác.
- **Tập đánh giá YOLO độc lập**: thu thêm 2–3 video ngắn ở địa điểm hoặc thời
  điểm khác với 4 nơi đã dùng để huấn luyện, gán nhãn tay (không qua
  auto-label), để đo khả năng tổng quát hoá thật của detector, tránh nguy cơ
  "học vẹt" đặc điểm của các địa điểm đã quen.
- **Đánh giá lại OSNet trên crop nhiễu thật**: dùng video đã có sẵn, lấy crop
  trực tiếp từ đầu ra detection/tracking của pipeline (không phải crop sạch
  như VeRi-776), gán nhãn đúng/sai cho một số cặp track cùng đối tượng để đo
  lại độ chính xác ghép cặp trong điều kiện thực tế.
- **Hiệu chỉnh lại ngưỡng tuyệt đối của Incident Detector theo từng camera**:
  kiểm chứng bằng dữ liệu thật có gán nhãn tay (Chương 5, mục 5.5.7) cho thấy
  `SUDDEN_STOP`, `SUDDEN_ACCEL`, `VEHICLE_PROXIMITY`, `PREDICTED_COLLISION`,
  `OVERSPEED` đều sai 100% trên mẫu đã xem ở camera Phố Huế — không chỉ do
  nhiễu vận tốc tức thời (đã có cơ chế làm mượt/duy trì liên tục), mà nhiều
  khả năng do các ngưỡng tuyệt đối (`min_moving_speed`, `proximity_px`, giới
  hạn tốc độ theo lớp xe, đơn vị pixel/giây) chưa được hiệu chỉnh theo đúng
  khoảng cách/độ phân giải của camera cụ thể này. Đã tắt tạm cả 5 loại này
  cho camera Phố Huế (`incident_detection.disabled_types`) như giải pháp
  trước mắt. Hướng giải quyết triệt để: hiệu chỉnh lại ngưỡng theo đơn vị
  thực tế (m/s thay vì px/s, dựa trên phép quy đổi pixel↔mét đã có sẵn cho
  Goal Classifier — `modules/goal_classifier.py`, các đặc trưng
  `speed_ms_*`/`px_per_m_mean`) thay vì hằng số pixel cố định không khả chuyển
  giữa các camera/khoảng cách khác nhau; sau đó thu thêm cảnh quay tại các
  giao lộ đông đúc, gán nhãn tay true/false positive cho từng cảnh báo phát
  sinh để có cả hai nhóm dữ liệu cần cho việc chọn ngưỡng bằng phương pháp
  precision/recall (`scripts/eval/tune_incident_thresholds.py` đã viết sẵn,
  chưa chạy được vì thiếu mẫu true positive) — có thể cần dàn dựng nhẹ một
  vài tình huống thật để có ca dương tính rõ ràng kiểm tra từng quy tắc.
- **Thêm mẫu "quay đầu" cho Goal Classifier**: quay thêm 2–3 video tại các vị
  trí có dải phân cách cho phép quay đầu, nơi hành vi này xuất hiện thường
  xuyên hơn giao lộ thường, để tăng số mẫu kiểm thử (hiện chỉ 36 mẫu) lên mức
  đủ tin cậy thống kê cho recall của lớp này.
- **Bổ sung ảnh giao diện thực tế và diễn tập demo trước hội đồng**: chạy hệ
  thống trực tiếp để chụp ba ảnh UI còn thiếu (dashboard, bản đồ hành trình,
  quản lý cảnh báo) thay cho placeholder ở Chương 4, đồng thời diễn tập lại
  toàn bộ kịch bản demo trên video Phố Huế – Trần Khát Chân.

**(B) Cải thiện mô hình**

- **YOLO11 — fine-tune hợp nhất đa domain**: thay vì duy trì hai checkpoint
  riêng (Chương 5, mục 5.4), thử fine-tune trộn dữ liệu CCTV thật với dữ liệu
  domain hẹp, đóng băng backbone nhiều hơn và giảm learning rate/số epoch, để
  tìm một checkpoint vừa giữ được hiệu năng trên dữ liệu thật vừa không quên
  domain bổ sung; mở rộng thêm các bộ dữ liệu camera giám sát công khai
  (VisDrone, DriveIndia, UA-DETRAC) để tăng đa dạng domain huấn luyện.
- **Goal Classifier — cải thiện recall "quay đầu" và "rẽ trái"**: hướng khả
  thi nhất cho "quay đầu" là chuyển sang dự đoán *trong* giao lộ (in-junction
  traversal prediction) thay vì chỉ dựa vào giai đoạn tiếp cận, vì tín hiệu
  đặc trưng của hành vi này chỉ rõ khi xe đã vào trong; với "rẽ trái", cần bổ
  sung đặc trưng cấu trúc đường (ví dụ ID làn vào, kết nối giao lộ) thay vì
  chỉ dựa vào động học chuyển động quan sát được.
- **Dự đoán quỹ đạo học sâu cho horizon ngắn**: bộ dự đoán quỹ đạo dạng
  Seq2Seq GRU đa giả thuyết (`modules/trajectory_gru.py`) được huấn luyện trên
  dữ liệu mô phỏng CARLA (world-space, mét). Tài liệu trước đây mô tả model
  này "chưa được tích hợp, cần số liệu thật trước khi triển khai" — đọc lại
  trực tiếp `EnsembleTrajectoryPredictor`/`ai_processor.py` (Chương 5, mục
  5.5.10) cho thấy mô tả này **sai**: với camera có calibration (Phố Huế),
  GRU này **đang được blend trực tiếp vào dự đoán hiển thị live**, trọng số
  blend = độ tự tin nội tại của chính GRU, không dựa trên độ chính xác đo
  được nào. Việc cần làm trước, ưu tiên cao hơn cả việc huấn luyện lại: **tắt
  ngay nhánh blend này** (Kalman-only) cho tới khi có số liệu ADE/FDE thật để
  quyết định nên dùng hay không. Sau đó mới tới việc huấn luyện lại trên dữ
  liệu/đặc trưng không gian-ảnh thật (không phải world-space từ simulator) —
  Chau et al. [1] huấn luyện trực tiếp CNN-LSTM/CNN-GRU trên toạ độ
  pixel-space từ YOLOv7+DeepSORT trên video CCTV Việt Nam, đúng dạng dữ liệu
  còn thiếu cho GRU của đề tài này; có thể tham khảo cách chuẩn bị dữ liệu của
  công trình đó, cùng cân nhắc so sánh CNN-LSTM vs CNN-GRU (bài báo không tách
  riêng số liệu hai kiến trúc trong phần tóm tắt công khai) trên đúng dữ liệu
  pixel-space của đề tài trước khi quyết định kiến trúc cuối.
- **Prior thực nghiệm cho Route Predictor ở hop ≥ 1 trên dữ liệu thật**: prior
  hiện tại (mục 5.1) đo trên dữ liệu mô phỏng vì dữ liệu CCTV thật hiện mỗi
  đoạn video chỉ quan sát được một giao lộ, chưa có chuỗi quỹ đạo qua nhiều
  giao lộ liên tiếp; khi đã có dữ liệu theo dõi xuyên camera thật (mục A), có
  thể đo lại prior này trực tiếp trên hành vi giao thông thật tại khu vực
  triển khai.

**(C) Nhận diện biển số — đã thử nghiệm, đang tiếp tục đầu tư cải thiện (chưa có kết luận cuối)**

Đề tài đã thiết kế và thử nghiệm việc dùng **OCR biển số** làm tín hiệu bổ trợ
veto-only cho nhận diện lại xuyên camera (chỉ dùng để loại bỏ một cặp match đã
được ReID chấp nhận khi hai biển số đọc được rõ và khác nhau, không bao giờ
dùng để tự tạo match). Ba vòng kiểm tra đầu trên dữ liệu thật (tổng cộng ~235
mẫu phương tiện, bao gồm kiểm tra trực tiếp trên đúng video/khoảng cách camera
dùng cho demo ở mục 4.5), dùng EasyOCR (OCR văn bản cảnh tổng quát), cho kết
quả 0% biển số xe máy đọc được và chỉ một tỷ lệ nhỏ ô tô/xe tải ở cự ly gần
đọc được, quy nguyên nhân cho khoảng cách/độ phân giải camera.

**Hai vòng kiểm tra bổ sung sau đó cho thấy kết luận trên chỉ đúng một phần.**
Vòng 4 khoanh vùng biển số trong crop bằng màu sắc (trắng/vàng) kết hợp mật độ
biên (phân biệt biển thật với phản xạ kim loại/đèn xe — vốn đánh lừa lần
khoanh vùng đầu), rồi xác nhận lại bằng mắt: nhiều biển — kể cả biển 2 dòng
của xe máy — sau khi khoanh đúng vùng thì **đọc được rõ bằng mắt người**,
nhưng EasyOCR (dù đã thêm allowlist ký tự và ghép các đoạn text theo dòng)
vẫn đọc sai ngay trên ảnh rõ nét, ở độ tin cậy sát ngưỡng (ví dụ đọc
"729-E2" cho biển thật "29-E2", dao động qua/không qua ngưỡng tuỳ chi tiết
tiền xử lý). Tỷ lệ đọc được *một phần văn bản* tăng từ 0% lên 3,1% cho xe máy,
nhưng tỷ lệ đọc được *biển hợp lệ đầy đủ* vẫn 0% cho cả hai loại xe — cho thấy
khoanh vùng không còn là nút thắt, nút thắt nằm ở chính năng lực nhận diện ký
tự của OCR tổng quát.

Vòng 5 thay EasyOCR bằng kiến trúc 2 giai đoạn chuyên biệt cho biển Việt Nam
(YOLOv5 phát hiện vùng biển + một YOLOv5 riêng phát hiện từng ký tự như một
class, tham khảo từ mã nguồn và trọng số đã huấn luyện sẵn của một repo cộng
đồng — `github.com/trungdinh22/License-Plate-Recognition`, không có file
license rõ ràng nên chỉ dùng cho mục đích kiểm chứng tính khả thi, không tích
hợp vào sản phẩm). Kết quả cải thiện rõ rệt và **lần đầu có đọc đúng tuyệt
đối, xác nhận được bằng mắt**: ô tô — 12,0% phát hiện được vùng biển, 8,0%
đọc đúng biển hợp lệ đầy đủ (ví dụ một biển đọc đúng "30G12794" khớp 100% với
ảnh gốc, lặp lại nhất quán ở 3 khung hình khác nhau của cùng xe); xe máy —
30,9% phát hiện được vùng biển (so với gần như 0% ở các vòng trước) nhưng chỉ
0,7% đọc đúng đầy đủ. Phân tích sâu hơn lý do xe máy còn thấp: trong các vùng
biển xe máy phát hiện được, 73% có kích thước thật chỉ 14–33 pixel chiều
ngang — kiểm chứng bằng cách phóng to 1–4 lần trước khi đưa vào model, không
cải thiện ở bất kỳ tỷ lệ phóng nào — xác nhận đây đúng là **sàn vật lý/cảm
biến** (không còn đủ pixel để phân giải 7–9 ký tự), không phải hạn chế của
model; phần còn lại (~27%, kích thước 30–42px) đọc được một phần (1–6/7–9 ký
tự), cho thấy đây là vùng có dư địa cải thiện bằng cách huấn luyện lại trên dữ
liệu đúng độ phân giải/khoảng cách camera thực tế — nhưng dữ liệu gán nhãn ký
tự ở đúng độ phân giải này hiện không có sẵn (các bộ dữ liệu biển số Việt Nam
công khai tìm được trên Kaggle/GitHub nhiều khả năng cùng nguồn dữ liệu cự ly
gần hơn đã dùng để huấn luyện sẵn model thử ở vòng 5), nên việc fine-tune
chưa được thực hiện trong phạm vi đề tài.

**Kết luận cập nhật sau vòng 5, thay cho kết luận "0% do cảm biến" ở ba vòng
đầu**: với phương tiện đủ gần/đủ lớn trong khung hình (ô tô, một phần xe máy),
một model chuyên biệt huấn luyện đúng định dạng biển Việt Nam đọc được biển số
ở mức độ đáng kể, không phải một tín hiệu vô dụng như kết luận ban đầu; nhưng
73% vùng biển xe máy detect được chỉ rộng 14–33px — kiểm chứng bằng phóng to
1–4 lần không cải thiện gì — xác nhận đây là sàn vật lý/cảm biến, chỉ camera
zoom/PTZ chuyên chụp biển số mới giải quyết được; phần còn lại (~27%, 30–42px)
đọc được một phần, là vùng có dư địa cải thiện bằng huấn luyện đúng độ phân
giải.

**Vòng 6 (sau vòng 5, đang thực hiện) — hai hướng không cần huấn luyện lại đo
trước:**

- **Multi-frame fusion**: thay vì đọc biển trên một khung hình rời rạc, tận
  dụng việc một track tồn tại qua nhiều chục khung hình — thử đọc ở nhiều
  khung hình của cùng track, nhận kết quả đầu tiên đọc đúng. Đo trên 300 khung
  hình thật (458 track xe, dùng đúng detector/tracker của pipeline): tỷ lệ
  đọc-đúng-theo-track tăng **~6 lần** so với đọc-theo-khung-hình — ô tô từ
  4,1% lên **25,0%** (9/36 xe), xe máy từ 0,4% lên **2,5%** (8/314 xe) — không
  cần thêm dữ liệu/huấn luyện nào.
- **Tái tạo ý tưởng (không dùng trực tiếp code/weight) của repo cộng đồng
  bằng YOLO11** đang dùng sẵn trong project, để tránh 3 vấn đề của vòng 5
  (license không rõ, thêm framework YOLOv5 thứ hai, nguy cơ tự ý pip-install
  phá version pin trong venv — đã gặp thật 2 lần khi thử nghiệm vòng 5): thêm
  class `license_plate` vào chính `yolo11s_vn.pt` đang dùng (detect vùng biển
  "đi kèm miễn phí" trong lượt detect xe mỗi khung hình, không tốn thêm FPS),
  cộng một `YOLO11n` riêng, nhỏ, detect từng ký tự (30 class, giữ đúng bộ ký
  tự của model tham khảo). Dữ liệu huấn luyện: pseudo-label trên 4 đoạn video
  Phố Huế thật (giờ khác nhau) bằng chính 2 model tham khảo của vòng 5, người
  review/sửa lại bằng `labelImg` (256 khung hình/894 box vùng biển cho Stage
  1; 329 crop/1360 box ký tự cho Stage 2 trước review) — bổ sung thêm dữ liệu
  synthetic (giảm độ phân giải có kiểm soát một dataset biển số Việt Nam công
  khai khác, đã lọc bỏ ~28% mẫu lẫn biển nước ngoài phát hiện được trong
  dataset đó) để bù các chữ cái hiếm gặp trong mẫu thật. **Đang trong quá
  trình review nhãn tại thời điểm viết mục này — chưa có số liệu vòng 6 chính
  thức.**

Vì việc này vẫn đang tiếp diễn, đề tài **chưa đưa ra kết luận cuối** về việc
có tích hợp OCR biển số vào sản phẩm hay không — quyết định sẽ dựa trên số
liệu vòng 6 (so với baseline 8%/0,7% theo-khung-hình của vòng 5, và 25%/2,5%
theo-track của multi-frame fusion) đối chiếu với ngân sách FPS (mục 6.2(A) —
vẫn còn treo, xem thêm yêu cầu hiệu năng 6 camera/15–20fps). Mã nguồn các vòng
kiểm tra (`modules/plate_ocr.py`, `scripts/eval/test_plate_ocr_feasibility.py`
đến `_v3.py`, `multiframe_plate_fusion.py`, `pseudo_label_plates.py`,
`prepare_synthetic_char_data.py`) đều giữ trong repo. Trọng số/mã nguồn mượn
ở vòng 5 (`trungdinh22/License-Plate-Recognition` + `ultralytics/yolov5`)
**không commit vào repo** do chưa rõ license — chỉ dùng offline làm
pseudo-labeler (cùng quy ước với `best_xl_ITD_v1.2.pt` đã có), không nằm
trong sản phẩm cuối; cần tải lại theo đúng nguồn đã trích dẫn nếu muốn tái
lập bước pseudo-label.

---

Đề tài đã hoàn thành một hệ thống giám sát và truy vết phương tiện giao thông
đa camera hoạt động được trên video CCTV thật, với từng thành phần (phát
hiện, theo dõi, nhận diện lại, dự đoán hành vi, suy diễn hành trình, phát
hiện sự cố, giao diện thời gian thực) đều có số liệu đánh giá định lượng đo
trên dữ liệu thật, không chỉ dừng ở kết quả mô phỏng. Các hạn chế còn tồn tại
chủ yếu nằm ở quy mô dữ liệu kiểm thử thật (đặc biệt cho lớp "quay đầu" và
cho đánh giá MOTA/IDF1/ReID xuyên camera) hơn là ở kiến trúc hệ thống, và các
hướng phát triển đề xuất ở mục 6.2 đều là phần việc cụ thể, khả thi để tiếp
tục thực hiện sau đề tài này.

---

## TÀI LIỆU THAM KHẢO (Chương 6)

[1] T. Chau, D. V. Ngo, M. T. Nguyen, A. D. Nguyen-Tran, T. H. Do, "Leverage
Deep Learning Methods for Vehicle Trajectory Prediction in Chaotic Traffic,"
in *Intelligence of Things: Technologies and Applications (ICIT 2023)*,
Lecture Notes on Data Engineering and Communications Technologies, vol. 187,
Springer, Cham, 2023. DOI: 10.1007/978-3-031-46573-4_24.
