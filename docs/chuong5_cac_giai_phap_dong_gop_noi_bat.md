# CHƯƠNG 5. CÁC GIẢI PHÁP VÀ ĐÓNG GÓP NỔI BẬT

> Chương 4 đã trình bày kiến trúc, thiết kế và kết quả định lượng tổng quan
> của hệ thống trên dữ liệu thật. Chương này đi sâu vào những đóng góp và giải
> pháp kỹ thuật đáng chú ý nhất được rút ra trong suốt quá trình thực hiện đồ
> án — phần lớn xuất phát từ việc kiểm thử trực tiếp trên video CCTV thật và
> đối chiếu kết quả với hành vi/hình ảnh thực tế, thay vì chỉ dựa vào việc hệ
> thống "chạy không lỗi". Mỗi đóng góp được trình bày độc lập theo cấu trúc:
> *vấn đề đặt ra → giải pháp → kết quả*. Bốn đóng góp được chọn trình bày,
> bám sát các use case của hệ thống: vượt rào cản dữ liệu phát hiện bằng tự
> động gán nhãn (5.1, UC1); dự đoán hướng đi trên dữ liệu thật với hiệu chỉnh
> xác suất (5.2, UC3); suy diễn hành trình *minh bạch* trên bản đồ thực (5.3,
> UC3); và xây dựng quy trình đo MOTA/IDF1 trên CCTV thật cùng giới hạn cấu
> trúc của nó (5.4, UC1).

---

## 5.1 Vượt rào cản dữ liệu: xây dựng tập huấn luyện phát hiện giao thông Việt Nam bằng tự động gán nhãn

**Vấn đề.** Không tồn tại một bộ dữ liệu phát hiện đối tượng đủ lớn, gán nhãn
sẵn, đúng đặc thù giao thông Việt Nam quan sát từ camera CCTV trên cao (mật độ
xe máy rất cao, vật thể nhỏ và bị che khuất nhiều — mục 2.1). Các bộ công khai
như COCO/VisDrone hoặc không có đúng phân bố lớp/góc nhìn, hoặc thiếu xe máy
ở mật độ thực tế. Trong khi đó, gán nhãn tay toàn bộ một tập đủ lớn (hàng trăm
nghìn khung bao) là bất khả thi trong phạm vi một đồ án.

**Giải pháp.** Đề tài xây dựng một quy trình **tự động gán nhãn (auto-label)**:
dùng một mô hình phát hiện mạnh làm "giáo viên" sinh nhãn giả (pseudo-label)
cho kho video CCTV thật thu thập được, hợp nhất nhiều nguồn dữ liệu về cùng một
bộ nhãn 5 lớp giao thông (người, ô tô, xe máy, xe buýt, xe tải) qua script
`merge_detection_datasets.py` (hỗ trợ ánh xạ tên lớp tuỳ ý để tránh lỗi lệch
nhãn), rồi **kiểm định chất lượng bằng spot-check thủ công** thay cho việc xác
minh từng box: rà 120 ảnh grid đại diện (1 ảnh/video nguồn trên 124 video),
kết luận >85% khung bao hợp lệ trước khi đưa vào huấn luyện (chi tiết quy trình
QA ở `docs/qa_auto_label.md`). Cách tiếp cận này biến bài toán "gán nhãn từ đầu"
thành bài toán "sinh nhãn tự động + kiểm định mẫu", giảm chi phí gán nhãn xuống
nhiều bậc trong khi vẫn kiểm soát được rủi ro nhãn sai làm lệch huấn luyện.

**Kết quả.** Quy trình tạo ra một tập huấn luyện gồm **15.957 ảnh / 623.329
khung bao** (Bảng 4.4), đủ để fine-tune YOLO11s_vn đạt mAP50 = 89,5% trên mẫu
đánh giá (Bảng 4.5). Đề tài ghi nhận trung thực một caveat quan trọng về số
liệu này: vì không lưu lại val split gốc, mAP được đo lại trên 800 ảnh lấy
ngẫu nhiên *từ chính pool huấn luyện*, không phải một tập kiểm thử độc lập —
nên con số có thể lạc quan hơn năng lực tổng quát hoá thật, và việc xây một
tập đánh giá độc lập (gán nhãn tay, khác địa điểm/thời điểm) được ghi nhận là
phần việc còn thiếu ở Chương 6. Dù vậy, chính quy trình auto-label là điều
kiện tiên quyết để có một mô hình phát hiện vận hành được trên dữ liệu thật
trong điều kiện không có sẵn dữ liệu gán nhãn — nền tảng cho cả ba use case
phía sau.

## 5.2 Dự đoán hướng đi trên dữ liệu thật: khớp tham số, so sánh kiến trúc và hiệu chỉnh xác suất

**Vấn đề.** Mô-đun dự đoán hướng rẽ (Goal Classifier, mục 3.5) phục vụ UC3
được huấn luyện trên dữ liệu CCTV thật. Quá trình rà soát phát hiện một lỗi
lệch *âm thầm* giữa huấn luyện và suy luận: mã suy luận hardcode horizon = 1,5s
("kết quả tốt nhất theo horizon"), nhưng trọng số đang triển khai lại được
huấn luyện với giá trị mặc định 1,0s — mô hình suy luận trên một cửa sổ đặc
trưng *khác* với cửa sổ nó được huấn luyện. Lỗi này không gây exception nào nên
hệ thống vẫn "chạy được" và không bị phát hiện.

**Giải pháp.** (1) Huấn luyện lại đúng horizon = 1,5s để khớp suy luận (sao
lưu trọng số cũ). (2) Nhân dịp đã có dữ liệu thật ổn định, so sánh trực tiếp
ba kiến trúc trên cùng dữ liệu và cùng cách chia tập: Gradient Boosting (đang
dùng), Random Forest và mạng nơ-ron nhiều lớp (MLP). (3) Giữ bước hiệu chỉnh
xác suất (`CalibratedClassifierCV`, Platt scaling — mục 3.5) để mức tin cậy
hiển thị phản ánh đúng tần suất đúng thực tế, đo bằng ECE.

**Kết quả.** Khớp lại horizon cải thiện rõ: accuracy 59,73% → **60,12%**,
balanced accuracy 51,96% → **54,18%**, recall "rẽ trái" 53,7% → 60,5%, "rẽ
phải" 55,2% → 58,4% — xác nhận lệch train/inference thực sự làm giảm chất lượng
dù không gây lỗi runtime. Trong so sánh ba kiến trúc, không kiến trúc nào vượt
được accuracy tổng thể của Gradient Boosting trên dữ liệu hiện có; Random Forest
cho hiệu chỉnh xác suất tốt hơn rõ rệt (ECE ≈ 0,021 so với ≈ 0,12 của Gradient
Boosting) và recall lớp "quay đầu" cao hơn (44,8% so với ~36%), nhưng đổi lại
accuracy tổng thể thấp hơn — được ghi nhận là lựa chọn thay thế nếu vận hành về
sau ưu tiên độ tin cậy hiển thị hơn độ chính xác thô. Mô hình production
(Gradient Boosting + hiệu chỉnh, horizon 1,5s) đạt accuracy 60,12%, balanced
accuracy 54,18% trên 1.334 track kiểm thử tách riêng khỏi 5.332 track huấn
luyện (Bảng 4.5), recall theo lớp: đi thẳng 61,67%, rẽ trái 60,54%, rẽ phải
58,4%, quay đầu 36,11% (lớp yếu nhất, chỉ 36 mẫu kiểm thử — chưa đủ tin cậy
thống kê, xem hạn chế ở Chương 6). Đóng góp đáng chú ý ở đây không nằm ở con số
accuracy tuyệt đối, mà ở **phương pháp luận trung thực**: tách bạch rõ
train/test, đo ECE để không phóng đại độ tin cậy, và lựa chọn kiến trúc cổ điển
phù hợp với quy mô dữ liệu thật thay vì chạy theo học sâu khi dữ liệu còn ít.

## 5.3 Suy diễn hành trình *minh bạch*: sửa lỗi tìm đường và thay giả định cứng bằng prior

**Vấn đề.** Trong UC3 (suy diễn hành trình nhiều ngã tư trên bản đồ thực),
`route_predictor.py` bộc lộ ba vấn đề. Thứ nhất, chỉ mục không gian tìm đoạn
đường gần nhất bao gồm cả các đoạn nối nội bộ giao lộ (junction-connector
stub) dài dưới 1m do bước chuyển đổi OSM → OpenDRIVE (mục 3.5) sinh ra; khi
đoạn gần nhất tìm được là một stub, hành trình suy diễn "chết" ngay từ ngã tư
đầu. Thứ hai, hướng dự đoán thật từ Goal Classifier (mục 5.2) được gán theo
*chỉ số vòng lặp* thay vì "ngã tư thật đầu tiên gặp được" — trên bản đồ có
nhiều đoạn nối ngắn giữa camera và ngã tư (như Phố Huế, mục 4.5), điều này
khiến dự đoán thật bị bỏ qua hoàn toàn nếu đoạn đường gần nhất ban đầu chưa
phải một ngã tư thật. Thứ ba, với mọi ngã tư *sau* ngã tư đầu tiên (hop ≥ 1),
hệ thống dùng giả định cứng `direction='straight', confidence=1.0` — luôn hiển
thị "đi thẳng, tin cậy 100%" cho các bước suy diễn xa, vi phạm trực tiếp yêu
cầu phi chức năng "tính minh bạch của dự đoán" (mục 2.4): không được hiển thị
một dự đoán xa như thể chắc chắn tuyệt đối.

**Giải pháp.** (1) Lọc bỏ khỏi chỉ mục không gian mọi đoạn đường ngắn hơn 1,5m
trước khi tìm đoạn gần nhất — đây chỉ là artifact nội bộ, không phải đoạn đường
thật. (2) Sửa logic gán dự đoán thật theo đúng "ngã tư thật đầu tiên" thay vì
theo chỉ số vòng lặp, để dự đoán từ Goal Classifier được áp dụng đúng chỗ.
(3) Thay giả định cứng `confidence=1.0` ở các ngã tư xa (hop ≥ 1) bằng một
**prior phân bố hướng đi** — và quan trọng hơn cả con số cụ thể, **hiển thị
đúng mức bất định** thay vì luôn báo "100% chắc chắn".

**Kết quả.** Sau khi sửa: (a) hành trình suy diễn không còn "chết" ngay từ ngã
tư đầu do vướng đoạn nối nội bộ; (b) dự đoán thật từ Goal Classifier được áp
dụng đúng tại ngã tư thật đầu tiên, thay vì bị bỏ qua như trước; (c) mức tin
cậy hiển thị cho các ngã tư xa không còn là "100%" sai lệch mà phản ánh một
prior phân bố hướng đi — đáp ứng đúng yêu cầu minh bạch (mục 2.4). Đóng góp cốt
lõi của mục này là một nguyên tắc thiết kế: **một hệ dự đoán dài hạn phải hiển
thị đúng mức bất định của nó**, thà thừa nhận "chưa chắc chắn" còn hơn phóng
đại "100%". Việc ước lượng prior phân bố hướng đi này một cách đáng tin cậy cần
dữ liệu quỹ đạo *qua nhiều giao lộ liên tiếp trên dữ liệu CCTV thật* — điều mà
mỗi đoạn video một camera hiện chưa cung cấp được — nên đây là phần để hoàn
thiện khi đã có dữ liệu theo dõi xuyên nhiều giao lộ (Chương 6).

## 5.4 Xây dựng quy trình đo MOTA/IDF1 trên CCTV thật và giới hạn cấu trúc của ground-truth bán tự động

**Vấn đề.** Đánh giá chất lượng theo dõi (UC1) bằng MOTA/IDF1 (mục 4.4) cần
ground-truth track đồng bộ theo từng khung hình — thứ video CCTV thật không tự
có, và gán nhãn tay từng khung là rất tốn công. Ngoài ra, khi bắt tay xây quy
trình này còn phát hiện `main.py` luôn khởi tạo `ByteTrackWrapper` với
`frame_rate = 10` (hằng số) cho *mọi* nguồn, kể cả video CCTV thật (thực tế 3
FPS); ByteTrack dùng `frame_rate` để quy đổi `track_buffer` sang thời gian
(`max_time_lost = frame_rate/30 × track_buffer`), nên fps sai làm sai khoảng
thời gian "chờ" một track quay lại sau khi mất dấu tạm thời.

**Giải pháp.** (1) Thêm property `fps` cho `VideoSource`/`FileVideoSource`
(đọc đúng `cv2.CAP_PROP_FPS`), sửa `main.py` đặt `frame_rate` theo từng camera
thay vì hằng số. (2) Xây một quy trình lấy ground-truth **bán tự động** không
phải vẽ box từ đầu: tái dùng chính track ByteTrack đã chạy ra (`pred_*.txt`),
render video có vẽ box + đúng định danh (`render_pred_review.py`, đọc lại dữ
liệu đã có để ID vẽ lên khớp tuyệt đối với file pred) để người xem phát hiện
đoạn đổi ID, ghi vào một file sửa nhỏ dễ viết tay
(`id_corrections.csv`: `old_id,new_id,frame_start,frame_end`), áp bằng
`apply_id_corrections.py` để ra `gt_*.txt`, rồi chạy `evaluate_tracking.py`.

**Kết quả & giới hạn cấu trúc (phát hiện trung thực).** Vì `gt` là bản sao
1:1 từ `pred` (chỉ sửa cột `id`), FN và FP luôn xấp xỉ 0 *theo cấu trúc* —
không bao giờ có hộp ground-truth nào mà pred "không phát hiện được", vì gt
vốn được tạo từ chính các hộp đó. Do đó **MOTA tính theo cách này luôn bị thổi
phồng**, chỉ phản ánh tỉ lệ đổi ID chứ không phản ánh tỉ lệ bỏ sót phát hiện
thật; phép đo thử trên video Phố Huế cho MOTA = 99,5% và IDF1 = 100,8% — IDF1
vượt 100% còn cho thấy thêm một artifact khớp cặp của `py-motmetrics` khi
gt/pred gần giống tuyệt đối. Kết luận phương pháp luận: với loại GT bán tự
động này, **chỉ IDF1/số lần đổi ID mới có ý nghĩa**, không nên dùng MOTA tổng
hợp như con số cuối — và cần hoàn tất gán nhãn toàn bộ video (hoặc gán nhãn box
độc lập) trước khi báo cáo số chính thức (Chương 6). Tận dụng cùng dữ liệu,
một thử nghiệm A/B nhanh (dùng số định danh sinh ra trên 192 khung hình thật
làm proxy, không cần nhãn tay) cho thấy: `frame_rate = 10` (lỗi) → 390 ID;
`frame_rate = 3` (đúng) → 383 ID (giảm ~1,8%); `frame_rate = 3` + tăng
`track_buffer` 30→90 → 390 ID (không cải thiện). Giả thuyết: giao thông quá
đông khiến buffer dài giữ lại nhiều track "ma" cạnh tranh ghép cặp với vật thật
ở gần — một nhược điểm đã biết của họ tracker kiểu SORT/ByteTrack trong cảnh
đông. Quyết định: giữ `track_buffer = 30`, chỉ áp dụng sửa `frame_rate` vì đúng
về bản chất tham số. Đóng góp ở đây là một **quy trình đo tracking khả thi cho
CCTV thật** cùng việc *chỉ rõ giới hạn của chính nó* — quan trọng để không báo
cáo một con số MOTA đẹp nhưng sai bản chất.

---

Chương này đã trình bày bốn đóng góp/giải pháp nổi bật, có thể quy về hai nhóm
bài học xuyên suốt. Thứ nhất, **lỗi đo lường và giả định ẩn** (lệch tham số
horizon của Goal Classifier, giả định "100% đi thẳng" của Route Predictor)
không gây crash, hệ thống vẫn "chạy được" nhưng cho kết quả sai lệch âm thầm —
chỉ lộ ra khi rà soát kỹ và đo lại bằng phương pháp đúng. Thứ hai, **một số rào
cản phải quản lý bằng chiến lược chứ không "sửa" trực tiếp được** — như rào cản
thiếu dữ liệu được vượt qua bằng tự động gán nhãn kèm kiểm định mẫu, hay giới
hạn cấu trúc của ground-truth bán tự động phải được thừa nhận thay vì che giấu
sau một con số MOTA đẹp. Xuyên suốt cả hai nhóm là một nguyên tắc làm việc
trung thực: ưu tiên kiểm chứng trên dữ liệu thật, thừa nhận đúng mức bất định
và giới hạn của từng phép đo, thay vì chạy theo con số đẹp. Chương 6 tổng kết
kết quả đạt được, các hạn chế còn tồn tại sau những cải thiện ở chương này
(hoàn tất ground-truth tracking, xây tập đánh giá phát hiện độc lập) và đề xuất
hướng phát triển tiếp theo.
