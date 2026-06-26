# Plan: Lấy ground-truth thật cho ByteTrack (MOTA/IDF1 trên video CCTV thật)

Trạng thái: **đang thực hiện** — tooling đã xong, đang gán nhãn tay phần
còn lại của video. Xem `docs/chuong5_cac_giai_phap_dong_gop_noi_bat.md`
mục 5.5.8 cho bản tóm tắt đưa vào báo cáo.

## Context

Trước khi làm việc này, ByteTrack — thành phần lõi của "theo dõi" — hoàn
toàn chưa có số liệu MOTA/IDF1 nào đo trên dữ liệu thật, chỉ có số đo trên
CARLA (MOTA=-23,6%/-51,6%, không đại diện cho video CCTV thật). Lý do duy
nhất ngăn việc đo trên dữ liệu thật là cần ground-truth track đồng bộ theo
từng khung hình, mà video CCTV thật không tự có.

Hướng đã chọn: **không gán nhãn từ đầu** (tốn công vẽ box) — tái dùng chính
track ByteTrack đã chạy ra (`pred_<CAM_ID>.txt`), chỉ sửa tay cột ID ở những
đoạn ID-switch (xem video có vẽ ID để phát hiện).

## Quy trình 4 bước

**Bước 1 — Sinh `pred_<CAM_ID>.txt` thật** (tự động, `main.py --eval-output`
đã có cơ chế xuất cho mọi nguồn, không chỉ CARLA):
```bash
cd custom_tracking_system
python main.py --source file \
  --video-path "data/datasets/traffic_intersection/02. Dữ liệu camera giao thông/Phố Huế Trần Khát Chân/12h.10.9.22.mp4" \
  --config config/camera_config_pho_hue.yaml --camera-ids CAM_PHO_HUE_REAL \
  --eval-output data/eval_real --max-frames 192
```

**Bước 2 — Render video có vẽ ID để xem lại** — dùng script mới
`scripts/eval/render_pred_review.py` (đọc lại trực tiếp `pred_<CAM_ID>.txt`,
KHÔNG chạy lại model, đảm bảo ID vẽ lên khớp tuyệt đối với ID trong file
pred — khác với `visualize_goal_predictions.py` đã có sẵn, script đó vẽ
`track_id` nội bộ của ByteTrack, không phải `global_id`, nên KHÔNG dùng được
cho việc này):
```bash
python scripts/eval/render_pred_review.py \
  --video "<cùng video ở Bước 1>" \
  --pred data/eval_real/pred_CAM_PHO_HUE_REAL.txt \
  --out docs/assets/eval_real/review.mp4
```
Xem `review.mp4` (số khung hình in sẵn ở góc dưới-trái mỗi khung, tua bằng
VLC + phím `E` để dò chính xác), ghi lại các đoạn ID-switch quan sát được
vào `data/eval_real/id_corrections.csv`:
```
old_id,new_id,frame_start,frame_end
1067,1023,95,130
```
Nghĩa: trong khoảng frame 95–130, mọi dòng có `id=1067` (ID sai, do switch)
sẽ đổi lại thành `1023` (ID đúng, xuyên suốt). Nếu một track biến mất hẳn
(không quay lại, hoặc model bỏ sót từ đó tới hết video) — **không ghi gì**,
đây không phải lỗi đổi ID.

**Bước 3 — Áp sửa, ra `gt_<CAM_ID>.txt`** — script mới
`scripts/eval/apply_id_corrections.py`:
```bash
python scripts/eval/apply_id_corrections.py \
  --pred data/eval_real/pred_CAM_PHO_HUE_REAL.txt \
  --corrections data/eval_real/id_corrections.csv \
  --out data/eval_real/gt_CAM_PHO_HUE_REAL.txt
```
In ra số ID duy nhất trước/sau (phải **giảm** — nếu không giảm, kiểm tra lại
`id_corrections.csv`).

**Bước 4 — Đánh giá** (cơ chế có sẵn, không cần sửa):
```bash
python scripts/eval/evaluate_tracking.py --eval-dir data/eval_real \
  --output docs/assets/eval_real/summary.csv --chart docs/assets/eval_real/mot_metrics.png
```

## Hạn chế cấu trúc đã xác nhận (quan trọng — đọc trước khi báo cáo số liệu)

Vì `gt` là bản copy 1:1 từ `pred` (chỉ sửa cột `id`, giữ nguyên box), **FN
và FP luôn xấp xỉ 0** — không bao giờ có ground-truth nào mà pred "không
phát hiện được", vì gt vốn được tạo ra từ chính các hộp đó. Hệ quả: MOTA
tính theo cách này sẽ luôn bị **thổi phồng giả tạo**, chỉ thực sự phản ánh
tỷ lệ đổi ID (IDSW) trên tổng số dòng hộp — không phản ánh tỷ lệ bỏ sót
phát hiện thật (model miss hoàn toàn 1 đối tượng, không phải tạm mất do che
khuất). Đo thử trên phần đã gán nhãn (frame 4–58/192 tại thời điểm viết)
cho MOTA=99,5%, IDF1=100,8% — IDF1 vượt 100% là bất thường, xác nhận thêm
vấn đề khớp cặp của `py-motmetrics` khi gt/pred gần giống tuyệt đối.

**Kết luận khi báo cáo**: chỉ nêu **IDF1/số lần đổi ID thật** như số liệu
chính, MOTA tổng hợp ghi kèm chú thích rõ giới hạn — không nêu như con số
tuyệt đối đáng tin. Muốn đo đúng tỷ lệ bỏ sót phát hiện (FN) cần một bộ
ground-truth **độc lập** (gán nhãn từ đầu cho ít nhất 1 đoạn ngắn, không tái
dùng track dự đoán) — chưa làm, để ở Chương 6 là hướng phát triển.

## Phát hiện phụ trong lúc làm (đã đưa vào Chương 5 mục 5.5.8)

- **Lỗi `frame_rate` hardcode**: `main.py` luôn dùng `frame_rate=_CARLA_FPS`
  (=10) cho `ByteTrackWrapper`, kể cả nguồn video file thật (3 FPS) — sai
  khoảng thời gian quy đổi `track_buffer` sang giây thật. Đã sửa: thêm
  property `fps` cho `VideoSource`/`FileVideoSource`, `main.py` dùng fps
  thật theo từng camera (nhánh `carla`/`bridge` vẫn giữ `_CARLA_FPS`, đúng
  vì đó là tốc độ mô phỏng cố định).
- **A/B test track_buffer** (proxy: số `global_id` khác nhau/192 khung
  hình, không cần nhãn tay): baseline lỗi=390, sửa fps đúng=383 (↓1,8%),
  sửa fps + tăng buffer 30→90=390 (không cải thiện, có thể do track "ma"
  gây nhiễu ghép cặp trong cảnh đông). Quyết định: giữ `track_buffer=30`,
  chỉ áp dụng fix `frame_rate`. CLI `--track-buffer` giữ lại để thử nghiệm
  tiếp với camera/video khác sau này.

## Các file liên quan

| File | Vai trò |
|---|---|
| `custom_tracking_system/scripts/eval/render_pred_review.py` | Mới — render video có vẽ box+global_id từ pred, dùng để xem lại |
| `custom_tracking_system/scripts/eval/apply_id_corrections.py` | Mới — áp sửa ID theo danh sách tay, ra gt |
| `custom_tracking_system/data/eval_real/pred_CAM_PHO_HUE_REAL.txt` | Sinh tự động ở Bước 1 |
| `custom_tracking_system/data/eval_real/id_corrections.csv` | Người dùng ghi tay khi xem video — **đang điền, chưa xong hết 192 khung hình** |
| `custom_tracking_system/data/eval_real/gt_CAM_PHO_HUE_REAL.txt` | Output Bước 3 — tạo lại mỗi khi corrections cập nhật |
| `custom_tracking_system/docs/assets/eval_real/review.mp4` | Video có vẽ ID, dùng ở Bước 2 |
| `custom_tracking_system/modules/video_source.py` | Sửa — thêm property `fps` |
| `custom_tracking_system/main.py` | Sửa — `cam_frame_rates` thay cho `_CARLA_FPS` hardcode; thêm CLI `--track-buffer` |

## Còn lại cần làm

1. Xem hết `review.mp4` (hiện mới tới ~frame 58/192), điền hết
   `id_corrections.csv`.
2. Chạy lại Bước 3 + 4 với corrections đầy đủ, lấy số liệu IDF1 cuối cùng.
3. Đưa số liệu cuối (IDF1, kèm chú thích giới hạn MOTA) vào Chương 5/6 thay
   cho mô tả "đang thực hiện" hiện tại.
4. (Tuỳ chọn, nếu muốn đo cả FN) Gán nhãn độc lập cho 1 đoạn ngắn (không tái
   dùng pred) để có số FN thật — hiện để ở Chương 6 như hướng phát triển,
   chưa bắt buộc cho báo cáo chính.
