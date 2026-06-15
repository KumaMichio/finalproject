# MUST.md — Thay đổi lớn: Mô phỏng quận nội thành Hà Nội + Dự đoán hướng đi xác suất

> Tài liệu này ghi lại yêu cầu thay đổi lớn (pivot) cho project, kèm phân tích
> khả thi và hướng triển khai đề xuất. Đây là tài liệu ĐỊNH HƯỚNG — chưa code,
> cần thảo luận/duyệt trước khi bắt đầu implement.

---

## 1. Bối cảnh hiện tại

Hệ thống hiện tại (xem `project_context.md`, `development_roadmap.md`):
- Dùng **CARLA 0.9.9.4** (Unreal Engine 4) làm môi trường mô phỏng, với map mặc
  định của CARLA (Town01/Town03...), không phải bản đồ thực tế.
- AI pipeline đầy đủ: detect (YOLO11m fine-tune VN) → track (ByteTrack) → ReID
  (OSNet) → global ID → trajectory prediction (Kalman + GRU Ensemble, horizon
  0.5–3s) → incident detection (14 loại, bao gồm PREDICTED_COLLISION).
- Phần cứng: **i5-9300H, GTX 1050Ti 4GB VRAM, 16GB RAM** — đã là điểm nghẽn cho
  CARLA (UE4) chạy song song với AI pipeline (xem `error.md`, domain gap khi
  fine-tune detector).

---

## 2. Yêu cầu mới (nguyên văn từ người dùng)

1. Với hệ thống hiện tại, cần tạo **1 môi trường mô phỏng** (với các tòa nhà
   hay xe trông đơn giản hơn), nhưng có **công năng như CARLA simulator**, để
   mô phỏng lại **1 quận nội thành trong Hà Nội** — từ đó có thể chuyển đổi từ
   mô hình sang đời thực dễ dàng hơn.
2. Hệ thống phải có khả năng **dự đoán hướng đi của các đối tượng bằng xác
   suất** (chủ yếu xe máy, ô tô, xe bus, xe tải) trong **1 khoảng thời gian
   dài** (dài hơn horizon hiện tại 0.5–3s).
3. Hướng đi dự đoán đó sẽ được **vẽ trên bản đồ**, lấy từ bản đồ thực của quận
   nội thành đó.
4. Hệ thống sẽ **theo dõi đối tượng**, khi có **tai nạn** thì **đánh dấu** và
   **dự đoán hướng đi** của đối tượng đó (và các đối tượng liên quan).

---

## 3. Phân tích khả thi & Hướng triển khai đề xuất

### 3.1 Môi trường mô phỏng — "CARLA-lite" cho quận nội thành Hà Nội

**Vấn đề với việc tự build map mới hoàn toàn trong CARLA/UE4**: cần
RoadRunner + asset 3D + lighting — rất nặng, vượt khả năng GTX 1050Ti khi chạy
song song AI, và tốn thời gian dev rất lớn.

**Hướng đề xuất — tái dùng tối đa pipeline CARLA hiện có, chỉ thay map**:

1. **Lấy dữ liệu bản đồ thực** của 1 khu vực nhỏ (1 ngã tư + vài trăm mét
   đường xung quanh) thuộc quận nội thành Hà Nội qua **OpenStreetMap**
   (Overpass API / OSMnx): road network (centerline, số lane, hướng), building
   footprints (polygon).
2. **Convert OSM → OpenDRIVE (.xodr)**: dùng SUMO's `netconvert` hoặc công cụ
   `osm2xodr` để sinh file `.xodr` mô tả road network thật của khu vực đó.
3. **Load vào CARLA bằng OpenDRIVE Standalone Mode**
   (`client.generate_opendrive_world(xodr_content)`): CARLA tự sinh mesh
   đường đơn giản (không cần asset 3D phức tạp) — đúng yêu cầu "đường/xe đơn
   giản hơn nhưng vẫn có công năng như CARLA" (traffic manager, sensors,
   physics, spawn point, v.v. vẫn hoạt động).
4. **Tòa nhà đơn giản**: spawn các block hộp chữ nhật (CARLA static mesh có
   sẵn hoặc prop đơn giản) theo building footprint từ OSM — đủ để tạo occlusion
   và bối cảnh thị giác giống khu phố thật, không cần model chi tiết.
5. **Xe**: dùng blueprint xe có sẵn trong CARLA, ưu tiên loại phổ biến ở VN
   (xe máy, ô tô con, bus, tải) — đã có sẵn từ `traffic_generator.py`.
6. **Camera đặt tại vị trí tương ứng camera CCTV thực tế** của khu vực đó (nếu
   có ảnh/tọa độ camera thực để đối chiếu) — giúp domain gap giữa sim và đời
   thực giảm ở mức **bố cục không gian** (góc nhìn, layout đường, vị trí giao
   lộ), dù texture/asset đơn giản hơn.

**Lý do "dễ chuyển đổi sang đời thực hơn"**: điểm chuyển giao quan trọng nhất
không phải là độ chi tiết hình ảnh (ảnh nào cũng cần fine-tune detector cho
domain thật), mà là **topology đường + hành vi giao thông + góc camera khớp
với hiện trường thật**. Khi road graph mô phỏng = road graph thật, mô hình dự
đoán hướng đi (trajectory predictor) học được "tại ngã tư X, rẽ trái có xác
suất bao nhiêu" — pattern này transfer được sang dữ liệu tracking thật tại
đúng ngã tư đó, vì nó phụ thuộc vào hình học bản đồ + luật giao thông, không
phụ thuộc texture.

**Phương án fallback nếu OSM→OpenDRIVE quá phức tạp**: vẽ tay road network đơn
giản hóa (vài đoạn thẳng + 1-2 ngã tư) bằng RoadRunner free tier hoặc bằng
cách chỉnh sửa trực tiếp 1 file `.xodr` mẫu của CARLA — vẫn theo tỉ lệ/góc của
khu vực thật, ưu tiên đúng layout giao lộ chính.

---

### 3.2 Dự đoán hướng đi bằng xác suất, horizon dài

**Hiện tại**: `EnsembleTrajectoryPredictor` (Kalman + GRU) dự đoán **1 điểm**
(x,y) tại mỗi horizon 0.5–3s — không có xác suất, không multi-modal, horizon
ngắn.

**Yêu cầu mới**: dự đoán **phân phối xác suất trên nhiều hướng đi khả dĩ**
(multi-modal) trong horizon dài (ví dụ 5–15s, hoặc tới khi xe ra khỏi 1
ngã tư/đoạn đường).

**Hướng đề xuất — Goal-conditioned / route-based prediction (nhẹ, phù hợp
1050Ti)**:

1. **Biểu diễn road graph**: từ OpenDRIVE/OSM, trích xuất graph các
   lane/segment + các "exit" tại mỗi giao lộ (ví dụ ngã tư có 3 lối ra: rẽ
   trái / đi thẳng / rẽ phải).
2. **Goal classifier**: model nhẹ (GRU encoder trên lịch sử vị trí + vận tốc
   world-space → MLP) dự đoán **phân phối xác suất trên các "goal"** (lối ra
   khả dĩ tiếp theo) — output là 1 vector softmax (ví dụ 3-5 giá trị).
3. **Path generator**: với mỗi goal, sinh quỹ đạo dự đoán bằng cách "snap" vào
   lane centerline của route tương ứng (path-following theo road graph) +
   điều chỉnh bằng GRU residual cho tốc độ/độ trễ — tránh phải học toàn bộ
   hình dạng quỹ đạo từ đầu, giảm tải GPU.
4. **Output**: list `[{path: [[x,y],...], probability: 0.0-1.0}]` — top-K
   (K=2-3) hướng đi có xác suất cao nhất, mỗi path dài tới horizon dài
   (5-15s, hoặc đến hết segment đường hiện tại).
5. **Training data**: sinh tự động từ simulator mới (3.1) — chạy traffic với
   autopilot/traffic manager trong thời gian dài, log toạ độ world + lane ID
   thực tế của mỗi xe → dataset (history, future, actual_goal) rất lớn, miễn
   phí (không cần gán nhãn tay).
6. **Per-class**: train riêng theo nhóm object_class (xe máy linh hoạt hơn —
   xác suất đổi lane/luồn lách cao hơn; bus/tải bám lane chặt hơn) — có thể
   dùng 1 model chung + feature "object_class" làm input, hoặc 4 model nhỏ
   riêng (xe máy / ô tô / bus / tải).

Module mới: `modules/probabilistic_trajectory_predictor.py`, có thể kết hợp
(ensemble) với `EnsembleTrajectoryPredictor` hiện có cho horizon ngắn
(0.5-3s, độ chính xác cao) + module mới cho horizon dài (multi-modal, theo
road graph).

---

### 3.3 Vẽ hướng đi dự đoán trên bản đồ thực

1. **Frontend**: thêm `MapView` (Leaflet + OSM tile layer của đúng khu vực
   Hà Nội đã chọn ở 3.1) — đã có sẵn gợi ý trong `development_roadmap.md`
   (Trang Object detail có "TRAJECTORY MAP").
2. **Pixel/world → lat-lon**: pipeline hiện tại đã có `calibration.py`
   (pixel↔world qua CARLA intrinsics/homography). Cần thêm 1 bước
   **world (CARLA coords) → lat/lon thật** — vì map CARLA mới được sinh từ
   OpenDRIVE có gốc tọa độ tương ứng với toạ độ địa lý gốc của khu vực OSM đã
   lấy ở 3.1 (georeference 1 lần khi convert OSM→OpenDRIVE, lưu offset).
3. **API mới**: `GET /api/tracks/{global_id}/predicted_paths` → trả về
   multi-modal paths (3.2) đã convert sang lat/lon.
4. **Hiển thị**: mỗi path là 1 polyline trên Leaflet, độ đậm/màu theo
   `probability` (path xác suất cao = nét đậm/màu nóng hơn).

---

### 3.4 Theo dõi, đánh dấu tai nạn, dự đoán hướng đi liên quan

Hệ thống hiện tại đã có `incident_detector.py` (14 loại, gồm
`PREDICTED_COLLISION`, `SUDDEN_ACCEL`, v.v.) và `evidence_package.py` (ring
buffer 30s).

**Bổ sung cần làm**:
1. Khi 1 incident loại "tai nạn" (va chạm thực sự / collision CARLA event,
   hoặc SUDDEN_STOP + chồng bbox) được phát hiện → đánh dấu global_id liên
   quan với flag `is_accident_marker` (mới trong DB `tracked_objects` hoặc
   `alerts.details`).
2. Với object bị đánh dấu, gọi probabilistic predictor (3.2) để:
   - dự đoán hướng đi của chính đối tượng đó (nếu vẫn di chuyển được — ví dụ
     văng ra sau va chạm),
   - và dự đoán hướng đi điều chỉnh của các đối tượng lân cận (xe phía sau có
     thể đổi lane/dừng) — dùng vị trí accident marker làm "obstacle" bổ sung
     vào path generator.
3. Frontend: marker đỏ tại vị trí tai nạn trên `MapView` + các predicted paths
   liên quan hiển thị nổi bật (ví dụ nét đứt cảnh báo).

---

## 4. Lộ trình triển khai đề xuất (giai đoạn)

```
Giai đoạn 0 — Khảo sát & chọn khu vực
  - Chọn 1 ngã tư/đoạn đường cụ thể trong 1 quận nội thành Hà Nội
    (ưu tiên nơi có ảnh/video CCTV thật để đối chiếu sau này)
  - Lấy dữ liệu OSM (Overpass/OSMnx) cho khu vực đó

Giai đoạn 1 — Map mới trong CARLA (OpenDRIVE standalone)
  - OSM -> .xodr (netconvert / osm2xodr)
  - Load qua generate_opendrive_world(), test spawn camera + traffic
  - Building blocks đơn giản theo footprint OSM
  - Georeference: lưu mapping world(x,y) <-> lat/lon

Giai đoạn 2 — Thu thập dữ liệu trajectory dài hạn
  - Chạy traffic manager lâu dài trên map mới, log (history, future, lane/goal)
    cho 4 class: motorcycle/car/bus/truck

Giai đoạn 3 — Probabilistic trajectory predictor
  - Goal classifier (GRU + MLP) + path generator theo road graph
  - Tích hợp song song với EnsembleTrajectoryPredictor hiện tại

Giai đoạn 4 — Map view + API
  - API /predicted_paths, world->lat/lon
  - Frontend MapView (Leaflet) vẽ multi-path theo xác suất

Giai đoạn 5 — Accident marking + related-object prediction
  - Mở rộng incident_detector + evidence_package
  - Marker trên MapView, predicted paths nổi bật cho object liên quan
```

---

## 5. Kết quả Giai đoạn 1 (đã hoàn thành)

Khu vực chọn: Hà Đông (Mỗ Lao), bbox `map3_smaller.osm` (~830×516m, 677 node,
95 way, 22 buildings: Mulberry Lane, T1/T2, trường học, LK6A/LK11A/LK11B, UBND
phường Mỗ Lao).

1. **OSM → OpenDRIVE → CARLA**: `Map/hanoi_district3.xodr` (541 road segment,
   1307 waypoint @10m) load thành công qua `generate_opendrive_world()`.
   Traffic Manager hoạt động bình thường (11/12 xe spawn, di chuyển 49-110m/15s,
   0 lỗi).
2. **Georeference** (`Map/georeference.py`): công thức 2 chiều CARLA(x,y) ↔
   UTM48N ↔ lat/lon, verify round-trip 7 chữ số thập phân. Offset map3:
   `(581165.61, 2320490.53)`.
3. **Building footprints** (`Map/extract_buildings.py` →
   `Map/buildings_hanoi_district3.json`): 22 building, OBB (center/size/yaw)
   + height (từ `building:levels`, default 3 tầng × 3m).
4. **Render building "khối xám"** (`Map/buildings_render.py`,
   `Map/test_buildings_grey.py`): mỗi building = 1 box wireframe trắng
   (footprint + chiều cao thật) + lưới box nhỏ màu xám phủ footprint (tối đa
   6×6/building) → top-down trông như khối xám đặc, đúng vị trí/hình dạng
   thật. Kết quả: `Map/map3_buildings_grey_topdown.png`.
5. **Traffic light logic-only** (`Map/traffic_light_sim.py`,
   `Map/test_traffic_lights.py`): tìm 26 giao lộ lớn từ topology, mỗi giao lộ
   1 "đèn" chu kỳ Red(10s)/Yellow(2s)/Green(8s) lệch pha random.
   `get_state_near(location, t, radius)` trả về `'Red'/'Yellow'/'Green'/None`
   — đúng format `traffic_light_state` mà
   `incident_detector._check_red_light_violation()` đang cần. Kết quả:
   `Map/map3_traffic_lights_topdown.png` (điểm đỏ/vàng/xanh đúng vị trí giao
   lộ).

**Phát hiện kỹ thuật quan trọng (giới hạn của build CARLA 0.9.14
WindowsNoEditor, ảnh hưởng đến các giai đoạn sau)**:
- Mesh thực (`static.prop.*`, `vehicle.*`) **không hiển thị** khi camera ở
  khoảng cách gần (~30-40m) — đã test với `static.prop.container` và
  `vehicle.tesla.model3`, cả 2 đều không render (chỉ thấy texture sọc
  placeholder của ground). Khả năng do `-quality-level=Low`.
- `debug.draw_line` **không hiển thị** trong `sensor.camera.rgb`.
  `debug.draw_box` hiển thị được, nhưng **chỉ khi camera ở độ cao lớn
  (~500m, top-down)** — đây cũng là lý do cách "khối xám" ở trên dùng toàn
  `draw_box` và mục tiêu cuối là 1 map top-down (khớp với yêu cầu MapView ở
  3.3), không phải view cận cảnh trong CARLA.
- → **Không thể thêm custom 3D model (vd. tự vẽ trong Blender) vào build
  packaged hiện tại**: cần build CARLA từ UE4 source + Editor để import/cook
  asset mới, đòi hỏi ~32GB+ RAM, 100GB+ disk, build nhiều giờ — vượt khả năng
  i5-9300H/16GB RAM/GTX1050Ti 4GB. Ngay cả khi build được, asset tự tạo nhiều
  khả năng vẫn gặp giới hạn render cận cảnh như trên. → Giải pháp "khối xám"
  bằng `debug.draw_box` (top-down) là hướng khả thi duy nhất cho hiển thị
  building trong build hiện tại.

---

## 6. Kế hoạch dữ liệu cho dự đoán hướng đi dài hạn

Mục tiêu cuối: predictor phải **dự đoán được hướng đi dài hạn của phương
tiện ngoài đời thật**, không chỉ trong sim. Vì vậy cần 2 nguồn dữ liệu, cho
2 mục đích khác nhau:

### 6.1 Dữ liệu synthetic (CARLA Traffic Manager) — Giai đoạn 2

- Chạy TM lâu dài trên `hanoi_district3` (road graph = OSM thật của khu vực
  đã chọn), log `(history, future, lane/goal, class)` cho 4 class.
- Ưu điểm: miễn phí, không giới hạn số lượng, nhãn "goal" chính xác 100%
  (route của autopilot biết trước).
- Hạn chế: phân phối hành vi (xác suất rẽ trái/phải/đi thẳng tại từng giao
  lộ, tốc độ, độ "lách" của xe máy VN) là hành vi của TM, không phải hành vi
  thật.

### 6.2 Dữ liệu thật — tự quay video tại khu vực đã chọn

Vì **path generator** (snap theo lane centerline của road graph thật) dùng
cùng road graph với sim nên transfer trực tiếp, không cần data thật. Phần
cần data thật để calibrate là **goal classifier** (phân phối xác suất rẽ).

**Cách lấy dữ liệu nếu không có CCTV sẵn — tự quay**:
- Vị trí: chỗ cao (tầng 2-3 trở lên, ban công/cửa sổ hướng xuống ngã tư) để
  góc nhìn gần giống CCTV, giảm domain gap và che khuất.
- Thời lượng: nhiều session ngắn (30-60 phút) ở các giờ khác nhau (cao điểm
  sáng/chiều, giờ vắng), tổng ~2-3 giờ — ưu tiên đa dạng giờ giấc hơn 1 lần
  quay dài, vì phân phối "goal" phụ thuộc giờ và class hiếm (bus/truck) cần
  đủ thời lượng mới xuất hiện đủ mẫu.

**Quy trình xử lý (phần lớn tự động)**:
1. Detection + Tracking: chạy YOLO11m + ByteTrack hiện có trên video → ra
   track_id + bbox theo thời gian (tự động).
2. Calibration pixel→lat/lon: làm **1 lần thủ công** cho mỗi vị trí quay
   (chọn ~4 điểm mốc đối chiếu video với OSM/Google Maps).
3. Goal label: suy ra ngược từ trajectory đã calibrate — snap vào road
   graph, "goal" = lối ra/junction mà xe thực sự đi tới (tự động, khác với
   sim là route biết trước).
4. **QA thủ công** (phần việc tay chính): kiểm tra/sửa lại class detection —
   detector hiện tại fine-tune trên ảnh render CARLA, có thể nhận nhầm class
   trên ảnh thật (domain gap, xem `error.md`). Mức độ sai quyết định % cần
   spot-check.

**Đề xuất trước khi quay**: chạy thử detector hiện tại trên 1 đoạn video
thật ngắn (vd. clip có sẵn về 1 ngã tư Hà Nội) để ước lượng % sai class,
tránh đầu tư 2-3 giờ quay rồi mới phát hiện domain gap quá lớn.

---

## 7. Rủi ro / hạn chế cần lưu ý

- **Phần cứng (GTX 1050Ti 4GB)**: CARLA + AI pipeline + training model mới
  cùng lúc sẽ rất nặng. Probabilistic predictor cần thiết kế nhẹ (GRU nhỏ +
  rule-based path snapping), tránh model attention/transformer lớn
  (Trajectron++, MTR... — quá nặng cho máy hiện tại, để dành cho UU TIEN 3 nếu
  nâng cấp GPU).
- **OSM → OpenDRIVE**: chất lượng convert phụ thuộc độ chi tiết dữ liệu OSM
  của khu vực chọn (số lane, hướng lane ở VN nhiều khi không khai báo đầy đủ
  trên OSM) — có thể cần chỉnh tay file `.xodr` sau khi convert.
- **Building blocks đơn giản**: cần kiểm tra CARLA OpenDRIVE standalone có hỗ
  trợ spawn static prop tùy ý theo polygon hay phải dùng prop có sẵn (hộp chữ
  nhật scale theo footprint).
- **Domain gap detector**: vẫn tồn tại (đã ghi nhận ở `error.md`), không được
  giải quyết bởi thay đổi này — đây là vấn đề riêng (fine-tune YOLO trên ảnh
  render CARLA, đang chờ GPU NVIDIA mạnh hơn).

---

## 8. Việc cần thảo luận/quyết định tiếp

1. Chọn cụ thể khu vực (quận, ngã tư) trong Hà Nội — cần để lấy dữ liệu OSM.
2. Xác nhận hướng map mới: OpenDRIVE standalone trong CARLA hiện tại (ít thay
   đổi nhất, tái dùng pipeline) vs. chuyển sang engine khác (Godot/Unity —
   nhiều việc hơn nhưng kiểm soát asset đơn giản tốt hơn).
3. Horizon dài cụ thể là bao nhiêu giây / hay theo "đến hết đoạn đường hiện
   tại" (dynamic theo road graph)?
4. Có cần tích hợp ảnh/video CCTV thật của khu vực để đối chiếu domain gap
   ngay từ giai đoạn này, hay để giai đoạn sau?
