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

---

# FPS thấp — ĐÃ FIX (ghi nhận 2026-06-12, fix xác nhận 2026-06-14)

`_main_loop()` ([server/services/ai_processor.py](../../server/services/ai_processor.py))
hiện đã dùng `detector.detect_batch(frames)` (1 forward pass cho cả 3 camera, line 287)
và throttle ReID qua `REID_INTERVAL`/`reid_track_ids` (line 294-302, chỉ extract feature
cho track mới hoặc mỗi N frame). Cả 2 đề xuất tối ưu chính ở mục dưới đã được áp dụng;
mục này giữ lại để tham khảo lịch sử.

## (lịch sử) Phân tích ban đầu — FPS thấp, vấn đề chưa fix (2026-06-12)

Bối cảnh: chạy server (`server/app.py` + AIProcessor, 3 camera CAM_001..003)
với CARLA, FPS hiển thị trên dashboard (`/ws/stats`) thấp hơn mong đợi.

## Nguyên nhân (phân tích code, chưa profiling thực tế)

`AIProcessor._main_loop()` ([server/services/ai_processor.py](../../server/services/ai_processor.py))
xử lý 3 camera **tuần tự** trong 1 vòng `for camera_id, frame_data in sync_frames.items()`,
mỗi camera:

1. `detector.detect(frame)` — gọi YOLO **riêng từng camera** (3 forward pass/frame).
   `detector.detect_batch()` đã có sẵn (chạy cả 3 frame trong 1 forward pass) nhưng
   **chưa được dùng** trong `_main_loop`.
2. `trackers[camera_id].update(...)` — ByteTrack, rẻ.
3. `global_tracker.process_camera_tracks()` — với MỖI detection, gọi
   `reid_extractor.extract_feature(frame, box, obj_class)` — **1 forward pass OSNet
   riêng cho từng object**, không batch. Đây là chi phí lớn nhất khi nhiều object/frame.
4. Trajectory update (Kalman, rẻ) + incident_detector (rẻ, numpy thuần) + evidence
   buffer + visualize + JPEG encode cho MJPEG.
5. `world.tick()` cuối loop — CARLA synchronous mode, `fixed_delta_seconds=0.1`.

FPS hiển thị = `fps_frame_count / wall_clock_elapsed`, phản ánh đúng tốc độ xử lý
thực — không bị cap bởi tick CARLA. Nghẽn cổ chai nhiều khả năng nhất: (1) 3x YOLO
inference riêng + (3) N x OSNet inference riêng (N = tổng số object detect được
trên 3 camera mỗi frame).

## Hướng giải quyết đề xuất (chưa thực hiện)

1. **Batch YOLO detect 3 camera trong 1 lần inference** — đổi `_main_loop` để gom
   `frame` của 3 camera, gọi `detector.detect_batch(frames)` 1 lần, map kết quả lại
   theo camera_id trước khi vào tracker/reid. Rủi ro thấp, hàm đã viết sẵn.
2. **Throttle/batch ReID** — option a) chỉ chạy `extract_feature` mỗi N frame/track
   (giữ `global_id` đã gán giữa các lần, chỉ cập nhật gallery định kỳ); option b)
   gom toàn bộ crop của 3 camera trong 1 frame thành 1 batch tensor → 1 forward pass
   OSNet. (b) hiệu quả hơn nhưng cần sửa `reid.py` (extract_feature hiện nhận 1
   box/lần).
3. **Đo thực tế trước khi tối ưu** — thêm timing (`time.perf_counter()`) quanh
   detect / reid / tick để biết tỉ lệ đóng góp thật của từng bước, tránh tối ưu sai
   chỗ.
4. Các đòn bẩy phụ (ít ưu tiên hơn): giảm `imgsz` YOLO (640→480), giảm resolution
   camera, giảm tần suất visualize/encode MJPEG.

---

# Fine-tune YOLO11m bổ sung (carla6) — ghi nhận 2026-06-12

## Dataset

- `carla_det3`: 12 camera, 94 vehicle, 39 pedestrian, 1701 ảnh mới thu thập từ
  CARLA (đa dạng hơn carla5: nhiều camera/vehicle/pedestrian hơn).
- Merge vào `visdrone_carla_vn3` = 9033 train / 853 val ảnh.

## Training config

- Fine-tune tiếp từ checkpoint carla5 (`yolo11m_vn.pt`), 15 epoch, batch 8,
  workers 4.
- Bị gián đoạn mất điện 3 lần (epoch 6, epoch 7, giữa epoch 15) → tổng cộng
  4 lần chạy `resume(resume=True)` từ `last.pt` để hoàn thành đủ 15 epoch
  (`resume_carla6.py`).

## Kết quả mAP (epoch 10-15) vs baseline carla5

| Epoch | mAP50 | mAP50-95 |
|-------|-------|----------|
| carla5 (baseline) | 0.5575 | 0.3342 |
| 10 | 0.5595 | 0.3347 |
| 11 | 0.5645 | 0.3391 |
| 12 | 0.5734 | 0.3409 |
| 13 | 0.5666 | 0.3411 |
| 14 | 0.5744 | 0.3483 |
| 15 (final) | 0.5735 | 0.3476 |

Per-class (epoch 15, validation 853 ảnh / 39152 instance):

| Class | P | R | mAP50 | mAP50-95 |
|-------|---|---|-------|----------|
| person | 0.613 | 0.489 | 0.499 | 0.213 |
| car | 0.805 | 0.818 | 0.839 | 0.585 |
| motorcycle | 0.604 | 0.477 | 0.497 | 0.219 |
| bus | 0.827 | 0.511 | 0.612 | 0.462 |
| truck | 0.622 | 0.375 | 0.426 | 0.263 |

## Kết luận

- Cải thiện rõ so với carla5: mAP50 +1.6 điểm (0.5575 → 0.5735), mAP50-95 +1.34
  điểm (0.3342 → 0.3476).
- Epoch 14 nhích cao hơn epoch 15 một chút (0.5744/0.3483 vs 0.5735/0.3476) —
  dao động bình thường ở cuối schedule LR giảm dần, không đáng kể.
- `weights/yolo11m_vn.pt` đã được cập nhật từ `best.pt` của run này
  (`runs/runs/detect/runs/detect/yolo11m_vn_carla6-3`). Plots/metrics lưu tại
  `docs/assets/yolo11m_vn_carla6/`.

---

# TODO — lỗi nhỏ phát hiện khi chạy demo `server/app.py --with-ai` (2026-06-12)

Pipeline chạy ổn định (~12 FPS, RTX 4060), nhưng có 2 vấn đề cần fix sau:

## 1. Vehicle ReID không load được — `No module named 'numpy._core'` — ĐÃ FIX (2026-06-14)

`venv_tracking` hiện có `numpy==2.2.6` (cài qua `requirements_cpu.txt`). Đã verify
trực tiếp: `DualReIDExtractor(vehicle_weights='weights/osnet_veri776.pth')` init OK,
model VeRi-776 load thành công, không còn lỗi `numpy._core`. Giữ phần dưới để tham
khảo lịch sử.

### (lịch sử) Phân tích ban đầu

```
2026-06-12 23:25:10,644 - modules.reid - ERROR - Failed to load vehicle model: No module named 'numpy._core'
```

- **Nguyên nhân**: `weights/osnet_veri776.pth` được lưu (torch.save/pickle) bằng
  numpy>=2.0 (module `numpy._core`), nhưng `venv_tracking` hiện có
  `numpy==1.23.1` (module `numpy.core`, không có `_core`) — unpickle lỗi.
- **Ảnh hưởng**: `DualReIDExtractor` chỉ load được model person (OSNet
  Market-1501), model vehicle (VeRi-776) bị bỏ qua → cross-camera ReID cho xe
  bị giảm chất lượng (rơi về so khớp bằng embedding person-space hoặc không
  match được global_id giữa camera cho vehicle).
- **Hướng fix**: hoặc (a) nâng numpy trong `venv_tracking` lên >=2.0 (kiểm tra
  tương thích torch==2.0.1/torchvision==0.15.2/ultralytics/torchreid với numpy
  2.x trước), hoặc (b) re-save `osnet_veri776.pth` với numpy 1.x (load bằng
  env có numpy 2.x rồi `torch.save` lại state_dict thuần — không qua pickle
  numpy array) trong `venv_tracking` hiện tại.

## 2. Detector dùng `yolov8s.pt` (auto-download COCO) thay vì `weights/yolo11m_vn.pt` — ĐÃ FIX (2026-06-14)

`ai_processor.py` giờ khởi tạo `ObjectDetector(model_type=str(_tracking_dir / "weights" /
"yolo11m_vn.pt"), half=True)` (auto-chọn `CLASSES_VN`), và `ByteTrackWrapper` mỗi camera
nhận `class_map=self.detector.CLASSES` để khớp bảng class VN thay vì `_CLS_MAP` COCO mặc
định. Giữ phần dưới để tham khảo lịch sử.

### (lịch sử) Phân tích ban đầu

```
[Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt ...]
2026-06-12 23:25:09,198 - modules.detector - INFO - Model loaded: yolov8s.pt
```

- **Nguyên nhân**: `server/services/ai_processor.py` (hoặc `ObjectDetector`
  default) không trỏ tới `custom_tracking_system/weights/yolo11m_vn.pt` (model
  đã fine-tune trên carla5/carla6, mAP50=0.5735) — đang dùng default
  `yolov8s.pt` COCO pretrained của ultralytics.
- **Ảnh hưởng**: detection kém chính xác hơn cho các class VN-specific
  (motorcycle, person mật độ cao...) so với model đã fine-tune; class map
  COCO (`_CLS_MAP` trong `modules/tracker.py`) vẫn đúng cho yolov8s nên
  pipeline không lỗi, chỉ kém chính xác.
- **Hướng fix**: set model path = `weights/yolo11m_vn.pt` (và class map VN
  tương ứng, `CLASSES_VN` — xem comment trong `modules/tracker.py`) ở nơi
  `ObjectDetector`/`ai_processor.py` khởi tạo detector cho từng camera.

## 3. ROI/wrong_way của CAM_003 — KIỂM TRA LẠI: KHÔNG CẦN FIX (2026-06-14)

- Lo ngại ban đầu: CAM_003 đổi từ `[25,-30,3]` yaw=45 sang `[25,38,3]` yaw=-90,
  nên polygon `rois.CAM_003.zones[0].polygon` (tính cho vị trí CŨ) có thể sai
  hình học với view mới.
- Đã verify bằng `CameraCalibration.world_to_pixel()`: polygon "8-25m phía
  trước, ±4m hai bên" là toạ độ TƯƠNG ĐỐI so với camera (forward/lateral
  frame của chính camera đó), nên kết quả pixel **giống nhau bất kể vị trí/yaw
  tuyệt đối của camera trong world** — tính lại cho cả vị trí CŨ và MỚI đều ra
  đúng `[[279,421],[681,421],[544,318],[416,318]]`, khớp 100% với config hiện
  tại. `wrong_way.lanes.CAM_003.direction: [0,-1]` cũng đúng vì pitch=0 (comment
  đầu mục `wrong_way` đã giải thích điều này). Không cần thay đổi gì.

## 4. Pedestrian walker AI controller lỗi `set_actor_collisions` (CARLA client/server version mismatch) — ĐÃ FIX (2026-06-14)

- Mọi pedestrian (cả từ `TrafficGenerator` và `ScenarioController.pedestrian_hit`)
  khi gọi `controller.ai.walker.start()` đều bị
  `RuntimeError: rpc::rpc_error during call in function set_actor_collisions`
  — do client `carla==0.9.15` (cài qua pip trong `venv_tracking`) gọi 1 RPC
  function chưa có ở server CARLA 0.9.14. Hậu quả: pedestrian được spawn
  nhưng KHÔNG tự đi bộ (đứng yên tại spawn point).
- **Nguyên nhân thật (không chỉ là RPC thiếu)**: `start()`,
  `go_to_location()`, `set_max_speed()` nằm trong CÙNG 1 try/except —
  `start()` raise `RuntimeError` làm 2 lệnh sau (navigation) bị BỎ QUA HOÀN
  TOÀN, dù `go_to_location()`/`set_max_speed()` hoạt động độc lập với
  `start()` và không phụ thuộc RPC `set_actor_collisions`.
- **Fix**: tách `start()` ra try/except riêng (log debug, bỏ qua lỗi không
  fatal), `go_to_location()`+`set_max_speed()` chạy trong try/except riêng kế
  tiếp bất kể `start()` có lỗi hay không. Áp dụng cho cả
  `TrafficGenerator._start_walker_controllers` (modules/traffic_generator.py)
  và `ScenarioController._spawn_pedestrian` (modules/scenario_controller.py).
- **Verify thực tế** (CARLA server 0.9.14 đang chạy, Town10HD_Opt): spawn 3
  pedestrian qua `TrafficGenerator`, sau 50 tick (5s, fixed_delta=0.1) cả 3 đã
  di chuyển 3.67m / 8.04m / 3.80m từ điểm spawn — đúng như mong đợi dù log vẫn
  có warning `set_actor_collisions` ở mức debug.
- **Hướng fix gốc lâu dài** (vẫn chưa cần ngay): cần CARLA server 0.9.15
  (khớp version với client pip), hoặc cài lại `carla==0.9.14` (cp310 wheel
  nếu có) trong `venv_tracking` để khớp server hiện tại. Việc nâng CARLA
  server lên 0.9.15 ảnh hưởng toàn bộ pipeline (carla_bridge dùng py3.7 egg
  0.9.14) nên cần kiểm tra kỹ trước khi đổi.
- **Giảm thiểu (2026-06-14)**: `collect_carla_cam_data.py`'s main loop từng
  crash hoàn toàn ở `world.tick()` (tick ~3600/5000) với cùng lỗi
  `set_actor_collisions`, khiến `write_yaml()` cuối script không chạy. Đã bọc
  `world.tick()` trong try/except `RuntimeError` — nếu xảy ra, log lỗi và
  `break` khỏi loop sớm nhưng vẫn vào `finally:` (destroy cameras, cleanup
  traffic) và chạy `write_yaml()` bình thường, nên dataset thu thập được tới
  thời điểm đó không bị mất.

## 5. `wrong_way.lanes.CAM_001.direction: [0,-1]` gây false positive cho xe đi đúng chiều — ĐÃ FIX (2026-06-14)

- **Triệu chứng (tiềm ẩn)**: với `direction: [0,-1]` và
  `angle_threshold_deg: 100`, xe đi ĐÚNG chiều (theo `waypoint.next()`, lane
  4/5 của road 20 — lan duong chinh ngay truoc CAM_001) sẽ bị gắn cờ
  `WRONG_WAY` CRITICAL sau `min_frames=10` frame.
- **Verify thực tế**: chạy `get_waypoint().transform.get_forward_vector()`
  cho các lane trong bán kính 25m của cả 3 camera, chiếu qua
  `CameraCalibration.world_to_pixel()` rồi so với `allowed=[0,-1]`:
  - CAM_002: lane đúng chiều → pixel-dir ≈ `[0.98,-0.22]`, góc ≈ 77° — "ok"
    (dưới ngưỡng 100°). Không đổi.
  - CAM_003: các lane (road 20/848/875) → pixel-dir ≈ `[±1,0]`, góc ≈ 90° —
    "ok". Không đổi.
  - CAM_001: lane 4/5 của road 20 (lan chinh truoc camera, xe autopilot đi
    theo `next()`) → pixel-dir ≈ `[0.98,0.2]`/`[0.98,0.22]`, góc ≈ 100-103° so
    với `[0,-1]` — **VƯỢT ngưỡng 100°** → xe đi đúng chiều bị báo sai thành
    WRONG_WAY.
- **Fix**: đổi `wrong_way.lanes.CAM_001.direction` từ `[0,-1]` thành
  `[0.98, 0.2]` (hướng thực tế của lane 4/5 trong khung hình CAM_001, tính từ
  bước verify trên). Sau fix: xe đi đúng chiều → góc ≈ 1° ("ok"); xe đi ngược
  (`~[-0.98,-0.2]`) → góc ≈ 180° → đúng là `WRONG_WAY`.
  File: `config/camera_config.yaml`.
- **Verify thực tế (2026-06-13)**: chạy `IncidentDetector.update()` với track
  giả lập trên CAM_001 — xe di chuyển theo `[0.98,0.2]` (đúng chiều) → 0
  WRONG_WAY sau 25 frame; xe di chuyển theo `[-0.98,-0.2]` (ngược chiều) → 1
  WRONG_WAY CRITICAL đúng tại frame thứ 10 (`min_frames`).

## 6. Kịch bản tai nạn (`ScenarioController`) hiếm khi nằm trong tầm camera nào — ĐÃ FIX (2026-06-13)

- **Triệu chứng**: `_pick_spawn_in_view()` chọn ngẫu nhiên 1 spawn point của
  map nằm trong FOV+bán kính 40m của 1 trong 3 camera. Nhưng map
  Town10HD_Opt có 155 spawn point rải khắp town, chỉ **8/155** nằm trong tầm
  camera (CAM_001: 1, CAM_002: 1, CAM_003: 6) → đa số lần chạy kịch bản
  (`hit_and_run`, `pedestrian_hit`, `red_light_crash`, `rear_end`,
  `sudden_stop`) có `camera_id = None`, tai nạn xảy ra ngoài khung hình, không
  camera nào quan sát/ghi nhận được.
- **Fix**: thêm `ScenarioController._camera_view_spawn_points(cam)` — quét
  vòng tròn quanh mỗi camera (bán kính 8-35m, theo FOV của camera), dùng
  `get_waypoint(project_to_road=True, lane_type=Driving)` để lấy waypoint
  thật trên lane, dedupe theo `(road_id, lane_id, s)`. Kết quả: +9 điểm cho
  CAM_001, +16 cho CAM_002, +79 cho CAM_003 (tổng 257 spawn point, so với 155
  ban đầu). Các điểm này được gộp vào `self.spawn_points` (tính 1 lần trong
  `__init__`, init thêm ~4s) và dùng thay cho
  `world.get_map().get_spawn_points()` ở mọi kịch bản + `_spawn_pedestrian`.
  File: `modules/scenario_controller.py`.
- **Verify thực tế**: chạy `rear_end` 15 lần → 15/15 có `camera_id` khác
  `None` (CAM_002 x3, CAM_003 x11, CAM_001 x1); `pedestrian_hit` 10 lần →
  9/10 có `camera_id` (CAM_003 x9, CAM_001 x1). Trước fix tỉ lệ này ~5%
  (8/155).
- Lưu ý: phân bố nghiêng nhiều về CAM_003 (79/104 điểm hợp lệ) do FOV rộng
  (100°) và gần nhiều lane hơn — CAM_001/CAM_002 vẫn ít hơn nhưng đã có >1
  điểm (9 và 16) thay vì chỉ 1.

## 7. Server `--with-ai` không khởi động được — sai import `boxmot.trackers.bbox.bytetrack` — ĐÃ FIX (2026-06-13)

- **Triệu chứng**: chạy `python app.py --with-ai` báo lỗi
  `ModuleNotFoundError: No module named 'boxmot.trackers.bbox'` ngay tại
  `modules/tracker.py:_init_tracker()`, AI pipeline không init được.
- **Nguyên nhân**: commit "update boxmot version for BE" (b8eca03) nâng
  `boxmot` lên 10.0.16, nhưng `tracker.py` vẫn import theo path cũ
  `boxmot.trackers.bbox.bytetrack.bytetrack.ByteTrack` (đã bị xóa ở version
  mới). Path đúng trong 10.0.16 là
  `boxmot.trackers.bytetrack.byte_tracker.BYTETracker`.
- **Fix**: đổi `_init_tracker()` trong `modules/tracker.py` sang
  `from boxmot.trackers.bytetrack.byte_tracker import BYTETracker` +
  `self.tracker = BYTETracker(**self._init_kwargs)`. Đã kiểm tra
  `BYTETracker.__init__(track_thresh, match_thresh, track_buffer,
  frame_rate)` và `update(dets, frame)` trả về hàng
  `[x1,y1,x2,y2,track_id,score,cls]` tương thích 100% với `_init_kwargs` và
  code parse output hiện có trong `update()` — không cần sửa gì thêm.
- **Verify thực tế**: restart server, log in ra
  `ByteTrackWrapper initialised (frame_rate=10, buffer=30)` x3 (1 per
  camera), pipeline init thành công, sau đó incident detector bắt được
  `PREDICTED_ROI_ENTRY` cho object 1000 ở CAM_001 và CAM_002 — tracking +
  prediction hoạt động bình thường.
