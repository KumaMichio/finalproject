# Ground Truth Evaluation — Bug Investigation Log

Bối cảnh: chạy `main.py --source bridge --eval-output data/eval --max-frames 100`
với CARLA (Town10HD_Opt) rồi đánh giá bằng `evaluate_tracking.py` (MOTA/IDF1 qua
py-motmetrics). Lần chạy đầu tiên cho MOTA âm và Recall = 0% ở mọi camera.

## Đã tìm ra & fix

### 1. Gallery ReID dùng chung không gian embedding giữa person/vehicle
- **Triệu chứng**: mọi object (car + person, nhiều camera) đều bị gán
  `global_id = 1000`.
- **Nguyên nhân**: `DualReIDExtractor` dùng 2 model khác nhau (OSNet
  Market-1501 cho person, OSNet VeRi-776 cho vehicle) → 2 không gian embedding
  khác nhau, nhưng `ReIDExtractor.gallery` là 1 dict chung, và
  `match_with_gallery()` so cosine similarity giữa feature mới với TẤT CẢ
  gallery entry không phân biệt model nào sinh ra → so sánh vô nghĩa, điểm số
  vượt threshold 0.5 một cách giả tạo.
- **Fix**: thêm `group_for_class()` (base trả `'default'`, `DualReIDExtractor`
  trả `'person'`/`'vehicle'` theo `_VEHICLE_CLASSES`). Gallery entry giờ là
  `{'feature':..., 'group':...}`; `match_with_gallery()` / `add_to_gallery()`
  chỉ so sánh trong cùng group.
  Files: `modules/reid.py`, `modules/global_tracking.py`.

### 2. Bias chung ("DC bias") của embedding OSNet quá lớn
- **Triệu chứng**: sau khi sửa (1), vẫn nhiều object khác nhau bị gán cùng
  `global_id`.
- **Nguyên nhân**: feature OSNet (đã L2-normalize) của 2 ảnh HOÀN TOÀN khác
  nhau vẫn có cosine similarity ~0.95–0.97 — do một thành phần chung lớn
  (DC bias) chiếm phần lớn vector, làm threshold 0.5 gần như vô tác dụng.
- **Fix**: thêm **calibration** — lúc init, chạy model qua ~60 ảnh nhiễu ngẫu
  nhiên, lấy mean embedding (`_compute_calibration`), rồi `_calibrate()` trừ
  mean này khỏi mọi feature + renormalize trước khi so sánh. Sau calibrate:
  object khác nhau → similarity ≈ 0, object giống nhau → similarity ≈ 1.
  Mỗi group ('person' / 'vehicle') có calibration mean riêng (model riêng).
  File: `modules/reid.py`.

### 3. Hai object khác nhau trong CÙNG 1 frame/camera vẫn match cùng global_id
- **Triệu chứng**: sau khi sửa (1)+(2), 2 người khác nhau trong cùng 1 frame
  của 1 camera (box rất nhỏ ~11x23px) vẫn có thể có similarity ≥ 0.5 do crop
  quá nhỏ/mờ sau khi resize 256x128 → mất gần hết thông tin nhận dạng.
- **Fix**: thêm ràng buộc logic — trong `process_camera_tracks()`, track theo
  `used_global_ids_this_frame`; nếu match trùng 1 global_id đã được gán cho
  track khác trong CÙNG frame của CÙNG camera → coi như không match, tạo
  global_id mới (vì 2 vật thể xuất hiện đồng thời từ 1 camera không thể là
  cùng 1 object).
  File: `modules/global_tracking.py`.

### 4. Sai bảng class-index (`_CLS_MAP`) trong tracker.py
- **Triệu chứng**: detection class 'car' (class_id=1 theo checkpoint VN-finetune)
  bị ghi thành 'unknown', 'motorcycle' (class_id=2) bị ghi thành 'car', v.v.
  → ảnh hưởng việc DualReIDExtractor chọn model + group cho object.
- **Nguyên nhân**: `modules/tracker.py` có `_CLS_MAP` hard-code theo chỉ số
  **COCO** (`{0:person, 2:car, 3:motorcycle, 5:bus, 7:truck}`), nhưng pipeline
  dùng checkpoint tự fine-tune `weights/yolo11m_vn.pt` với bảng class khác
  (`CLASSES_VN = {0:person, 1:car, 2:motorcycle, 3:bus, 4:truck}`).
  `detector.py` đã gán đúng tên class, nhưng `tracker.py` tính lại tên class
  từ `class_id` số bằng bảng COCO sai, bỏ qua tên đúng từ detector.
- **Fix**: `ByteTrackWrapper` nhận tham số `class_map` (truyền
  `detector.CLASSES` từ `main.py`), dùng đúng bảng tương ứng checkpoint đang
  chạy thay vì `_CLS_MAP` hard-code.
  Files: `modules/tracker.py`, `main.py`.

### Kết quả sau khi fix (1)-(4)
- `pred_CAM_00X.txt` có nhiều `global_id` khác nhau (không còn collapse về 1000).
- MOTA tổng thể cải thiện từ **-11.6% → -2.5%** (lần chạy trước đó).

## Vấn đề CÒN LẠI (chưa fix) — Recall ≈ 0%

Sau khi fix (1)-(4), một lần chạy 100 frame khác cho Recall ≈ 0% ở cả 3
camera. Đã kiểm tra trực tiếp:

- **GT projection (`project_actors_to_cameras` trong `modules/ground_truth.py`)
  có vẻ ĐÚNG**: bbox 'vehicle'/'pedestrian' di chuyển mượt, hợp lý qua từng
  frame, tọa độ nằm trong khung hình hợp lệ → KHÔNG phải lỗi chiếu 3D→2D.

- **Nguyên nhân thật — domain gap của detector (`weights/yolo11m_vn.pt`,
  fine-tune trên VisDrone) khi áp lên scene render CARLA**:
  1. CAM_001 có 431 GT 'vehicle' nhưng pred = 0 detection class 'car' — detector
     không nhận ra xe CARLA trong scene này (miss hoàn toàn).
  2. Toàn bộ pred 'person' là **false positive trên scenery tĩnh**: ví dụ
     CAM_002 detect liên tục cùng 1 box (~87,306)-(98,329) gần như đứng yên
     suốt 80+ frame — rất có thể là một mô hình người trang trí tĩnh trong map
     Town10HD_Opt (không phải actor `walker.pedestrian.*` do TrafficGenerator
     spawn), nên không bao giờ xuất hiện trong GT.
  3. Nhiều object GT rất nhỏ (<5px chiều rộng, vật ở xa) — vốn đã khó/không
     thể detect được bằng bất kỳ model nào ở độ phân giải hiện tại.

### Hướng giải quyết đề xuất (chưa thực hiện)
1. **Fine-tune lại YOLO trên ảnh render từ CARLA** — thu thập frame +
   GT bbox (đã có pipeline `ground_truth.py` sinh label tự động), train tiếp
   từ `yolo11m_vn.pt`. Giải quyết domain gap thật, nhưng tốn thời gian thu
   thập + train.
2. **Tinh chỉnh detector** — giảm `conf_threshold` (đang 0.35), tăng `imgsz`
   (đang 640 → 960/1280) để bắt vật nhỏ hơn. Rẻ, nhanh, nhưng cải thiện hạn
   chế và có thể tăng FP.
