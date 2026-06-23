# Tomorrow — Task list để tăng độ tin cậy mô hình

> Tạo ngày: 2026-06-23, cho công việc đi quay/gán nhãn ngày mai.
> Mục tiêu: vá các lỗ hổng kiểm thử đã xác định, để các model thật sự có cơ sở
> số liệu khi áp dụng vào CCTV thật — đặc biệt là phần "theo dõi liên camera"
> và "dự đoán hướng đi dài hạn" mà mục tiêu đề tài yêu cầu.

---

## Ưu tiên cao

### 1. Quay 2 video tại 2 giao lộ liền kề — chứng minh cross-camera Re-ID thật

**Vì sao quan trọng nhất:** đây là phần lõi của mục tiêu đề tài ("theo dõi đối
tượng liên camera") nhưng hiện chưa từng được chứng minh trên video thật — demo
hiện tại (Phố Huế–Trần Khát Chân) chỉ dùng 1 camera.

**Số lượng:** 2 video.

**Ở đâu:**
- Chọn 2 giao lộ **liền kề trên cùng một trục đường**, đủ gần để một xe đi từ
  giao lộ A có thể tới giao lộ B trong khoảng vài chục giây đến vài phút.
- Ưu tiên trục **Phố Huế – Trần Khát Chân** hoặc lân cận, vì map
  `Map/pho_hue_tkc.xodr` đã có sẵn **101 junctions / 742 roads** — kiểm tra
  trước xem giao lộ thứ 2 đã nằm trong map này chưa. Nếu có → không cần build
  map mới (OSM→SUMO→OpenDRIVE), chỉ cần chạy `Map/fit_camera_pose.py` thêm 1
  lần để calibrate camera thứ 2.
- Nếu giao lộ thứ 2 nằm ngoài map hiện tại, cần build map mới theo đúng quy
  trình ở `docs/demo.md` §8 — tốn thêm thời gian, nên tránh nếu có lựa chọn
  khác trong map cũ.

**Video cần như thế nào:**
- **Thời gian quay: đồng thời hoặc lệch nhau rất ít** (lý tưởng là quay cùng
  lúc 2 máy, hoặc lệch không quá 1-2 phút) — nếu không đồng bộ thời gian thì
  không thể kiểm tra xe nào đi từ camera 1 sang camera 2.
- **Độ dài:** tối thiểu 10-15 phút mỗi video, để có đủ số lượt xe đi qua cả 2
  điểm làm test case (giao thông Hà Nội đông nên 10-15 phút thường đủ vài chục
  lượt xe khớp hướng).
- **Góc đặt camera:** trên cao (cột điện, tầng 2-3 trở lên, hoặc ban công) nhìn
  xuống giao lộ, góc nghiêng ~20-30°, giống kiểu CCTV — khớp với phong cách
  video đã dùng để train/calibrate trước đó.
- **Camera phải cố định hoàn toàn** trong suốt video (đặt tripod hoặc tựa chắc
  chắn) — không cầm tay, không pan/zoom giữa video, vì bước calibration
  (`fit_camera_pose.py`) giả định 1 pose camera cố định cho toàn bộ clip.
- **Độ phân giải:** tối thiểu 1280×720 (720p), ưu tiên 1080p nếu máy quay hỗ
  trợ — xe máy ở xa cần đủ pixel để model nhận ra (đây là điểm yếu đã biết của
  detector hiện tại).
- **Thời điểm trong ngày:** ban ngày, trời sáng rõ, tránh mưa/sương mù cho lần
  thu đầu tiên (tránh thêm biến số gây nhiễu kết quả).
- **Định dạng:** .mp4 (H.264) là đủ, không cần raw.

---

### 2. Gán nhãn ground-truth track cho ByteTrack (đo MOTA/IDF1)

**Số lượng video cần:** không bắt buộc quay mới — có thể dùng **1 video đã có
sẵn**, miễn là chưa được dùng để train YOLO (kiểm tra lại danh sách 120 video
trong `data/auto_label/` xem có video nào bị bỏ qua hoặc giữ lại làm held-out
chưa). Nếu tất cả video hiện có đều đã dùng để train, cần quay thêm **1 video
mới** (~5 phút) theo đặc điểm như mục 1 ở trên (cùng style CCTV, cố định, ban
ngày).

**Việc cần làm (không phải quay, mà là gán nhãn):**
- Lấy một đoạn ~300-1000 khung hình từ video đó.
- Gán nhãn tay: mỗi xe/người xuất hiện được gán 1 ID cố định xuyên suốt đoạn
  video (dùng CVAT hoặc Label Studio, hoặc sửa tay từ track ByteTrack đã chạy
  sẵn — review từng đoạn xem có ID-switch hay không rồi sửa).
- Dùng `custom_tracking_system/scripts/eval/evaluate_tracking.py` (đã có, chạy qua
  `py-motmetrics`) — chỉ cần đổi nguồn GT từ CARLA sang file GT gán tay này.

---

### 3. Eval set YOLO độc lập, không overlap với training pool

**Số lượng video:** 2-3 video ngắn (~2-3 phút mỗi video).

**Ở đâu:** có thể là các tuyến/giao lộ **khác** với 4 nơi đã dùng để train
(Phố Huế, Giảng Võ, Cửa Nam, Lê Duẩn) — để kiểm tra khả năng tổng quát hoá thật
của model, không chỉ là "học vẹt" đặc điểm của 4 nơi cũ. Nếu không quay được
nơi khác, ít nhất phải là **thời điểm khác** (giờ khác trong ngày, ngày khác)
tại cùng 1 trong 4 nơi cũ, và đảm bảo đoạn video này **không** được đưa vào
tập train lần sau.

**Video cần như thế nào:** giống đặc điểm ở mục 1 (cố định, trên cao, ban
ngày, ≥720p).

**Việc cần làm sau khi có video:** gán nhãn tay (không dùng auto-label) cho
vài trăm khung hình sample ra từ video này, làm test set sạch. Dùng lại
`custom_tracking_system/scripts/eval_yolo11s_autolabel.py` nhưng đổi nguồn ảnh
sang set mới này.

---

## Ưu tiên trung bình

### 4. Re-eval OSNet trên crop nhiễu thật (không cần quay video mới)

Dùng video đã có sẵn — chạy pipeline thật, lấy crop output từ detection/
tracking thật (không phải crop sạch như VeRi-776), gán tay đúng/sai cho một
số cặp track cùng object, tính lại accuracy match. Không tốn thời gian quay,
chỉ tốn thời gian gán nhãn + chạy script.

### 5. Thu video cho Incident Detector validation

**Số lượng:** lý tưởng 1-2 giờ video tổng (có thể nhiều clip ngắn gộp lại), HOẶC
5-10 clip ngắn (~1-2 phút) có sự cố/near-miss rõ ràng.

**Ở đâu:** giao lộ đông, có khả năng xảy ra tình huống thật (dừng đột ngột, xe
áp sát, đi ngược chiều) — các giao lộ đã quay trước đó (Phố Huế, Giảng Võ...)
thường đủ đông để tự nhiên xuất hiện vài sự cố nếu quay đủ lâu.

**Video cần như thế nào:** giống mục 1. Nếu không tìm được sự cố tự nhiên
trong thời gian ngắn, có thể **dàn dựng nhẹ** 1-2 case rõ ràng (ví dụ tự lái
xe/đi bộ vào khung hình rồi dừng đột ngột) để có ca dương tính chắc chắn kiểm
tra rule `SUDDEN_STOP`.

---

## Ưu tiên thấp

### 6. Thêm mẫu u_turn cho Goal Classifier

**Số lượng:** 2-3 video (~10-15 phút mỗi video).

**Ở đâu:** chọn các điểm có dải phân cách cho phép quay đầu ("chỗ quay đầu")
trên các trục đã quen thuộc — đây là nơi xe u_turn xuất hiện thường xuyên hơn
giao lộ thường, giúp tăng nhanh số mẫu (hiện chỉ có ~29 mẫu test, quá ít để
tin được số liệu recall).

**Video cần như thế nào:** giống mục 1.

---

## Tổng số video cần quay mới (nếu làm đủ cả 6 mục)

| Mục | Số video mới cần quay | Có thể tái dùng video cũ không? |
|---|---|---|
| 1. Cross-camera Re-ID | 2 (đồng bộ thời gian) | Không |
| 2. MOTA/IDF1 GT | 0-1 | Có, nếu còn video chưa dùng để train |
| 3. YOLO eval set độc lập | 2-3 | Một phần — ưu tiên nơi/giờ mới |
| 4. OSNet crop nhiễu | 0 | Có, dùng toàn bộ video cũ |
| 5. Incident Detector | 0-vài clip ngắn | Có, cộng thêm dàn dựng nhẹ nếu cần |
| 6. Goal Classifier u_turn | 2-3 | Không (cần địa điểm đặc thù) |

**Tổng tối thiểu nếu ưu tiên cao trước:** ~3-5 video mới (mục 1 + mục 3),
phần còn lại có thể dùng video đã có sẵn trong `data/auto_label/`.
