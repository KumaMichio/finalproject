# Plan: Tinh chỉnh ngưỡng IncidentDetector dựa trên dữ liệu thật có nhãn

> Tạo ngày: 2026-06-25. Bối cảnh: Chương 5 mục 5.5.4/5.5.6 phát hiện
> `IncidentDetector` báo quá nhiều cảnh báo (157 sự cố/4 phút, 36 CRITICAL,
> evidence/ phình ~32GB/4 phút) trên video thật Phố Huế sau khi sửa lỗi phân
> mảnh Global ID — nguyên nhân gốc là ngưỡng các luật (`min_moving_speed`,
> `sudden_stop_ratio`, `sudden_accel_ratio`, `proximity_px`...) chưa được
> kiểm chứng trên dữ liệu thật có nhãn true/false-positive. Plan này chia
> việc thành phần **tự động hoá được ngay** (Bước 1, 5) và phần **cần làm
> tay** (Bước 2 — gán nhãn).

---

## Bước 0 — Việc cần làm trước (5 phút)

- Đảm bảo có ít nhất 1 video CCTV thật chưa qua xử lý (đã có:
  `custom_tracking_system/data/datasets/traffic_intersection/02. Dữ liệu
  camera giao thông/Phố Huế Trần Khát Chân/12h.10.9.22.mp4`).
- Kiểm tra dung lượng ổ đĩa trước khi chạy (`df -h` trên E:) — chiến dịch thu
  dữ liệu ở Bước 1 sẽ ghi ảnh, nhẹ hơn evidence video clip nhiều nhưng vẫn
  nên có vài GB trống.

## Bước 1 — Sửa code: ghi log đầy đủ context cho MỌI cảnh báo (tự động hoá được)

**Mục tiêu:** hiện tại chỉ severity CRITICAL mới được `EvidencePackage` lưu
clip video; WARNING (`SUDDEN_STOP`, `OVERSPEED`...) chỉ có 1 dòng log text,
không có gì để xem lại bằng mắt. Cần một cơ chế nhẹ hơn evidence (không lưu
video 30s/incident, sẽ lại đầy đĩa) nhưng đủ để gán nhãn tay.

**File cần sửa:** `custom_tracking_system/modules/incident_detector.py`,
phương thức `update()` (nơi tổng hợp `incidents` trước khi return — xem dòng
~127-160) và `_make()` (helper tạo dict incident, dùng chung cho mọi luật).

**Việc cần làm:**
1. Thêm 1 tham số mới cho `IncidentDetector.__init__`, ví dụ
   `snapshot_dir: str | None = None`. Nếu được set, sau khi tạo mỗi incident
   (mọi severity, không chỉ CRITICAL), lưu:
   - 1 ảnh JPEG của khung hình hiện tại (đã có sẵn `frame` truyền vào
     `update()`) — không cần video clip, ảnh tĩnh đủ để soát.
   - 1 file JSON nhỏ cạnh ảnh, chứa: `incident_id` (dùng timestamp+gid),
     `type`, `severity`, `gid`, `camera_id`, `message`, và **toàn bộ
     `details` dict đã có sẵn trong code hiện tại** (`speed_before`,
     `speed_after` cho SUDDEN_STOP; `speed_prev`, `speed_now` cho
     SUDDEN_ACCEL; `distance_px`, `speed_1`, `speed_2` cho
     VEHICLE_PROXIMITY; `predicted_distance`, `vehicle_speed` cho
     PREDICTED_COLLISION) — các trường này đã tồn tại trong `_check_*()`,
     chỉ cần xuất ra file thay vì chỉ giữ trong RAM.
2. Vẽ thêm bounding box của (các) đối tượng liên quan lên ảnh trước khi lưu
   (tái dùng `utils/visualization.py:Visualizer`, đã có sẵn, dùng trong
   main.py) — giúp người gán nhãn xem nhanh không cần tra lại video.
3. Gọi `IncidentDetector(snapshot_dir="docs/assets/incident_review")` (hoặc
   tương tự) khi chạy thử nghiệm — KHÔNG bật mặc định trong production
   (chỉ dùng cho campaign thu dữ liệu này).

**Lưu ý disk:** ảnh JPEG ~vài trăm KB, JSON vài trăm byte — với ước lượng
~150-200 incident/4 phút như đã đo, tổng dữ liệu cho một lượt chạy đầy đủ
video (~vài chục phút nếu loop) chỉ cỡ vài trăm MB, an toàn hơn evidence clip
rất nhiều.

**Ước lượng thời gian:** 30-45 phút code + test.

## Bước 2 — Gán nhãn tay true/false-positive (CẦN NGƯỜI LÀM, không tự động hoá được)

> Đã cập nhật theo cách Bước 1 thực tế được code (2026-06-25): **không cần
> file CSV riêng** — mỗi file `.json` đã có sẵn field `"label": null`, chỉ
> cần mở và sửa trực tiếp thành `"tp"` hoặc `"fp"`. `tune_incident_thresholds.py`
> ở Bước 3 đọc thẳng field này.

1. Chạy lại để sinh dữ liệu (nếu chưa có sẵn từ lần test trước):
   ```bash
   cd custom_tracking_system
   python main.py --source file \
     --video-path "data/datasets/traffic_intersection/02. Dữ liệu camera giao thông/Phố Huế Trần Khát Chân/12h.10.9.22.mp4" \
     --config config/camera_config_pho_hue.yaml \
     --incident-snapshot-dir docs/assets/incident_review \
     --loop-video
   ```
   (Ưu tiên video CHƯA dùng để train YOLO nếu có sẵn, tránh trùng dữ liệu đã
   biết — nhưng dùng lại video Phố Huế cũng được, vẫn là dữ liệu thật chưa
   được gán nhãn sự cố bao giờ.) Dừng (Ctrl+C) sau vài phút — không cần chạy
   hết cả video, vài trăm sự cố là đủ mẫu.

2. Mở thư mục `docs/assets/incident_review/`. Mỗi sự cố có 2 file cùng tên:
   `<TYPE>_<global_id>_<timestamp>.jpg` (ảnh đã vẽ box+ID mọi đối tượng) và
   `.json` cùng tên (chứa `type`, `severity`, `details`, `label`).

3. Với mỗi ảnh muốn xem: tìm đúng ID được nêu trong `message`/`global_id` của
   file `.json`, đối chiếu hành vi trong ảnh:
   - `SUDDEN_STOP`: xe có thật sự đang phanh/dừng không, hay chỉ đứng yên
     (đèn đỏ — không phải "đột ngột") hoặc nhiễu detection?
   - `SUDDEN_ACCEL`: xe có thật tăng tốc rời đi bất thường, hay chỉ đang di
     chuyển bình thường sau khi tạm chậm (kẹt xe, nhường đường)?
   - `VEHICLE_PROXIMITY`: hai xe (`global_id` và `details.other_id`) có thực
     sự ở gần mức nguy hiểm, hay chỉ là 2 xe đi cạnh nhau bình thường trong
     làn/dừng đèn đỏ?
   - `PREDICTED_COLLISION`: dự đoán va chạm với người đi bộ
     (`details.pedestrian_id`) có hợp lý theo hướng đi quan sát được không?

4. Mở file `.json` tương ứng (Notepad, VSCode...), sửa `"label": null` thành
   `"label": "tp"` (đúng — sự cố thật) hoặc `"label": "fp"` (sai — nhiễu/báo
   giả), lưu lại.

5. Không cần xem hết — lấy mẫu theo từng loại luật (lọc theo tiền tố tên
   file, ví dụ tất cả file bắt đầu `SUDDEN_STOP_`), ~30-50 mẫu mỗi loại đã
   đủ để có thống kê tham khảo (xem Bước 3). File chưa gán nhãn
   (`label: null`) sẽ tự bị `tune_incident_thresholds.py` bỏ qua, không cần
   xoá.

**Ước lượng thời gian:** phụ thuộc số mẫu xem — khoảng 1-2 giờ cho ~150-200
mẫu nếu xem nhanh (vài giây/ảnh).

## Bước 3 — Script phân tích ngưỡng (tự động hoá được, sau khi có nhãn)

**File mới đề xuất:** `custom_tracking_system/scripts/eval/tune_incident_thresholds.py`

1. Đọc toàn bộ JSON ở Bước 1 + CSV nhãn ở Bước 2, join theo `incident_id`.
2. Với mỗi loại luật, tách riêng giá trị đã kích hoạt theo nhãn tp/fp — ví
   dụ `SUDDEN_STOP`: tỉ lệ `speed_after/speed_before`; `VEHICLE_PROXIMITY`:
   `distance_px`.
3. Vẽ histogram/boxplot 2 nhóm tp vs fp cạnh nhau (matplotlib, theo đúng
   style các script eval khác trong project, ví dụ
   `scripts/eval/eval_veri_reid.py`) để xem trực quan có tách biệt rõ không.
4. Nếu đủ mẫu (khuyến nghị ≥ 20 mỗi nhóm), dùng
   `sklearn.metrics.precision_recall_curve` trên giá trị kích hoạt (đảo dấu
   nếu cần để "cao hơn = nghi ngờ hơn") để tìm điểm cân bằng precision/recall
   mới, so với điểm ngưỡng hiện tại (`sudden_stop_ratio=0.3`,
   `min_moving_speed=8`, `proximity_px` hiện tại trong config).
5. In ra bảng so sánh: ngưỡng cũ vs ngưỡng đề xuất, kèm precision/recall ước
   tính trên chính tập đã gán nhãn (ghi rõ đây là in-sample, không phải
   validate độc lập — xem Bước 5).

**Ước lượng thời gian:** 45-60 phút viết script (có thể làm trước, song song
lúc đợi Bước 2, vì không cần dữ liệu thật để viết khung script — chỉ cần khi
chạy mới cần nhãn thật).

## Bước 4 — Thêm yêu cầu "duy trì liên tục" (cải tiến kiến trúc, độc lập với Bước 2-3)

Không cần đợi dữ liệu gán nhãn — có thể làm song song hoặc trước:

- Trong `_check_sudden_stop`/`_check_sudden_accel`
  (`modules/incident_detector.py`), thêm một bộ đếm "số khung hình liên tiếp
  thoả điều kiện" trước khi thật sự phát sinh incident (ví dụ cần đúng điều
  kiện ở ≥ 2-3 khung hình liên tiếp, không chỉ 1 lần kiểm tra).
- Lý do ưu tiên: nhiễu pixel-space thường không tương quan giữa các khung
  hình liên tiếp, còn hành vi thật (phanh, tiến sát) thì có — đây là cách
  giảm false positive rẻ và không cần dữ liệu gán nhãn, có thể áp dụng ngay
  cả khi Bước 2-3 chưa xong.

**Ước lượng thời gian:** 20-30 phút.

## Bước 5 — Áp dụng + validate trên phần dữ liệu chưa dùng để tinh chỉnh

1. Cập nhật ngưỡng mới vào `config/camera_config_pho_hue.yaml` (section
   `incident_detection`, xem `IncidentDetector.__init__` để biết đúng tên
   key — `cooldown_s`, `min_moving_speed`, `sudden_stop_ratio`,
   `sudden_accel_ratio`, `min_track_age_s`...).
2. Chạy lại trên một đoạn video **khác** đoạn dùng để gán nhãn ở Bước 2 (hoặc
   phần cuối video chưa xem) — đếm lại số cảnh báo, so sánh định tính.
3. Cập nhật Chương 5 mục 5.5.4/5.5.6 (báo cáo) với kết quả tinh chỉnh: ngưỡng
   mới, precision/recall trước/sau, để thay thế đoạn "chưa giải quyết triệt
   để" hiện tại bằng kết quả thật.
4. Cập nhật memory `project-status-todo.md` với kết quả.

---

## Tổng thời gian ước tính

| Bước | Ai làm | Thời gian |
|---|---|---|
| 0. Chuẩn bị | tự động | 5 phút |
| 1. Sửa code log snapshot | tự động hoá được | 30-45 phút |
| 2. Gán nhãn tay | **người làm** | 1-2 giờ |
| 3. Script phân tích ngưỡng | tự động hoá được | 45-60 phút (làm song song Bước 2) |
| 4. Thêm "duy trì liên tục" | tự động hoá được | 20-30 phút |
| 5. Áp dụng + validate | bán tự động | 30-45 phút |

**Thứ tự đề xuất nếu làm trong ~2 giờ:** Bước 1 → Bước 4 (không cần nhãn,
làm trước) → bắt đầu Bước 2 (gán nhãn) trong lúc rảnh → Bước 3 khi có đủ
nhãn → Bước 5 nếu còn thời gian, hoặc để lượt sau.
