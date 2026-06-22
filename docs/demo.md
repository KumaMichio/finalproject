# Kế hoạch Demo sản phẩm

> Cập nhật: 2026-06-19

---

## 1. Kịch bản demo

> **Camera AI phát hiện tai nạn → đánh dấu xe gây tai nạn → cross-camera tracking → dự đoán đường đi dài hạn → vẽ lên bản đồ → hỗ trợ cảnh sát chặn đường.**

Luồng cụ thể:
1. 2–3 camera tại các ngã tư liền tiếp trong khu vực Hà Nội
2. Traffic chạy bình thường
3. Xảy ra va chạm → hệ thống phát hiện (YOLO collision / SUDDEN_STOP event)
4. Xe gây tai nạn bị đánh dấu `accident_marker = True`
5. Xe rời hiện trường → ByteTrack mất dấu ở camera 1
6. Camera 2 pick up xe → cross-camera tracking qua ReID
7. Goal Classifier dự đoán hướng tại ngã tư tiếp theo (left/right/straight + probability)
8. Route predictor chain 2–3 hops → sinh danh sách route với xác suất
9. MapView (Leaflet) vẽ route lên bản đồ Hà Nội thực tế
10. Cảnh sát thấy: "Xe có khả năng 73% đi vào đường X, khuyến nghị chốt tại ngã tư Y"

---

## 2. So sánh phương án demo

| Tiêu chí | CARLA thuần (chọn) | Video CCTV thực |
|---|---|---|
| Kịch bản tai nạn | Script tự động, lặp lại được | Không thể dàn dựng ngoài đường |
| Cross-camera ReID | Dùng GT actor.id (bypass OSNet) | Cần OSNet trained (chưa có) |
| Tọa độ GPS | Tự động từ OpenDRIVE georef | Cần calibrate homography từng camera |
| Map alignment | hanoi_district3.xodr đã có | Cần đo thực tế |
| Tính lặp lại | Cao (script) | Thấp (phụ thuộc camera, thời tiết) |
| Effort | Trung bình | Rất cao |

**Quyết định: CARLA + hanoi_district3.xodr + Leaflet MapView**

Ghi chú cho báo cáo: "Production sẽ dùng OSNet ReID; trong demo simulator, GT vehicle ID được dùng để tránh dependency chưa hoàn thành."

> **Cập nhật 2026-06-22**: đã bổ sung thêm 1 demo THỨ HAI chạy trực tiếp trên video CCTV thật, song song với demo CARLA (không thay thế) — xem mục 8.

---

## 3. Kiến trúc demo

```
CARLA (hanoi_district3.xodr)
  ├─ Camera 1 (ngã tư A) ──→ YOLO11s → ByteTrack → incident_detector
  │                                                        │
  │                                              [accident detected]
  │                                                        │
  ├─ Camera 2 (ngã tư B) ──→ YOLO11s → ByteTrack ←── ReID / GT-ID
  │
  └─ Traffic Manager (autopilot vehicles)

                    ↓ tracks + accident_flag
           route_predictor.py
           ├─ parse hanoi_district3.xodr → junction graph
           ├─ tại mỗi junction: Goal Classifier → P(left/straight/right)
           └─ chain 2–3 hops → [{path: [[lat,lon],...], prob: 0.73}, ...]

                    ↓ JSON via WebSocket/API
           frontend/map_view.html (Leaflet)
           ├─ OSM tile layer (khu vực Hà Nội)
           ├─ Camera icons tại vị trí thực
           ├─ Marker đỏ: vị trí tai nạn
           └─ Polylines: route dự đoán (đậm/nhạt theo probability)
```

---

## 4. Các component — Trạng thái

> Cập nhật 2026-06-19: tất cả 4 component đã hoàn thành.

### 4.1 `scripts/demo_accident_scenario.py` ✅ DONE

**File:** `custom_tracking_system/scripts/demo_accident_scenario.py`

Flow thực tế đã implement:
1. Load `hanoi_district3.xodr` vào CARLA qua `client.generate_opendrive_world()`
2. Spawn 12 xe với TrafficManager autopilot (spawn từ 541 điểm có sẵn)
3. Gắn `sensor.other.collision` vào toàn bộ xe → dict `collision_events`
4. Warmup 80 ticks → disable autopilot xe target → throttle=1.0 → chờ collision event
5. Khi nhận event: re-enable autopilot → bắt đầu GoalClassifier + RoutePredictor loop
6. Ghi `demo/incident_state.json` mỗi tick (đọc bởi `demo/server.py`)

Chạy không cần CARLA (test offline):
```
python scripts/demo_accident_scenario.py --dry-run
```

Chạy với CARLA:
```
python scripts/demo_accident_scenario.py --host localhost --port 2000
```

### 4.2 `route_predictor.py` ✅ DONE

**File:** `custom_tracking_system/route_predictor.py`

Thuật toán đã implement:
- Load `Map/junction_graph.json` (52 junctions, 464 roads — generated bởi `parse_xodr.py`)
- `find_nearest_road(x, y, heading_deg)` — KNN spatial index với penalty góc
- `predict(x, y, heading_deg, direction, n_hops=4)` — traverse junction graph:
  - Tại mỗi junction: pick outgoing road có heading deviation gần nhất với target  
    (straight→0°, left→-90°, right→+90°, u_turn→180°)
  - Return list waypoints `{hop, road_id, x, y, lat, lon, heading_deg, confidence}`
- `predict_probabilistic(direction_probs, min_prob=0.15)` — multi-branch route tree

### 4.3 Georeference ✅ DONE

**File:** `custom_tracking_system/georeference.py`

```python
gr = GeoReference.from_map('hanoi_district3')
lat, lon = gr.carla_to_latlon(x, y)   # pyproj UTM zone 48
x, y     = gr.latlon_to_carla(lat, lon)
graph    = gr.junction_graph()          # load junction_graph.json
```

Cấu hình từ xodr header:
- `+proj=utm +zone=48 +ellps=WGS84 +datum=WGS84 +units=m`
- Offset: `x=581165.61, y=2320490.53`
- CARLA Y-flip: `utm_n = -carla_y + offset_y`
- Round-trip error: 0.000 m (verified)

### 4.4 `demo/map_view.html` + `demo/server.py` ✅ DONE

**Files:** `demo/map_view.html`, `demo/server.py`

Flask server endpoints:
```
GET  /                            → map_view.html
GET  /api/state                   → incident_state.json (current snapshot)
POST /api/incidents/{id}/mark     → manually mark accident vehicle
GET  /api/tracks/{id}/routes      → predicted route for a track
```

Leaflet frontend (polling `/api/state` mỗi 500ms):
- Blue dots: tất cả xe (OSM tile Hà Nội)
- Red pulsing dot: xe tai nạn
- Dashed orange polyline: predicted route (4 hops)
- Sidebar: direction probabilities (bar chart), hop-by-hop lat/lon

Chạy demo:
```bash
# Terminal 1 — chạy kịch bản CARLA (hoặc dry-run)
python scripts/demo_accident_scenario.py --dry-run

# Terminal 2 — web server
python demo/server.py
# → mở http://localhost:5000
```

---

## 5. Thứ tự triển khai — Trạng thái thực tế

```
Foundation
  [x] Phase 4: end-to-end test main.py --source file — DONE (boxmot v19 fix)
  [x] Verify hanoi_district3.xodr load trong CARLA — DONE (541 spawn pts, georef OK)
  [x] Parse .xodr → junction graph (52 junctions, 464 roads) — DONE
  [x] Georeference: world_to_latlon() — DONE (pyproj UTM48, round-trip 0.000m)

Core logic
  [x] route_predictor.py: 4-hop junction graph traverse — DONE
  [x] demo_accident_scenario.py: kịch bản CARLA — DONE (dry-run verified)
  [x] API endpoint (Flask) — DONE

Frontend
  [x] map_view.html: Leaflet + polylines — DONE (HTTP 200, route hops verified)
  [ ] Record demo video từ CARLA output — CẦN CARLA session riêng

Sau demo (báo cáo)
  [ ] OSNet Rank-1 accuracy (GPU cloud — Kaggle T4)
  [ ] Ground truth MOTA/IDF1 cho ByteTrack
  [ ] Viết chương đánh giá + kết luận báo cáo
```

---

## 6. Kết quả kiểm tra CARLA (2026-06-19)

| Metric | Kết quả |
|---|---|
| CARLA server version | 0.9.14 |
| Map name sau generate_opendrive_world | `Carla/Maps/OpenDriveMap` |
| Spawn points | **541** |
| Junctions (CARLA topology) | 38 |
| Simulation tick | OK (vehicle teleport + tick verified) |
| Georef (CARLA 0,0) | lat=20.983237, lon=105.780868 |
| Georef range check | lat trong [20,22]°N ✅ — lon trong [104.5,106.5]°E ✅ |

**Lưu ý:** CARLA 0.9.14 server in warning `cannot parse georeference` cho chuỗi `+proj=utm` trong xodr header — đây là giới hạn của CARLA, không ảnh hưởng vì `georeference.py` dùng pyproj trực tiếp (không phụ thuộc vào CARLA parse georef).

---

## 7. Điểm yếu cần ghi chú trong báo cáo

| Điểm yếu | Lý do chấp nhận được |
|---|---|
| OSNet dùng GT ID trong demo | Component đã thiết kế; pending GPU cloud |
| Goal Classifier 59.7% (không phải 90%+) | Bài toán khó (pixel-space, không có GPS); honest eval; CARLA baseline 70.3% |
| u_turn recall 34.5% | Chỉ 29 mẫu test; class imbalance → đề xuất SMOTE |
| MOTA=-23.6% trên CARLA (domain gap) | Đã eval + thử 4 cấu hình (conf 0.10–0.35, COCO vs VN): conf=0.35 VN là tốt nhất. COCO model tệ hơn VN. Fix thực sự: fine-tune `data/carla_cam_det/` trên GPU cloud |

---

## 8. Demo trên video CCTV thật (Phố Huế – Trần Khát Chân) — bổ sung 2026-06-22

> Lý do: sản phẩm hướng tới ứng dụng CCTV thật, cần chứng minh pipeline chạy được ngoài simulator. `hanoi_district3.xodr` (mục 1-7) chỉ phủ 1 vùng nhỏ ~400×300m, KHÔNG trùng vị trí với bất kỳ video CCTV thật nào trong dataset — đã xác minh bằng toạ độ thật (Phố Huế–Trần Khát Chân, Giảng Võ–Láng Hạ... đều cách map cũ vài km). Giải pháp: dựng map mới đúng tại 1 giao lộ thật, dùng đúng pipeline OSM → SUMO netconvert → parse_xodr đã dùng để tạo hanoi_district3.

### 8.1 Component mới

| File | Vai trò |
|---|---|
| `Map/pho_hue_tkc.osm` / `.xodr` / `_junction_graph.json` | Map giao lộ Phố Huế–Trần Khát Chân thật (101 junctions, 742 roads) |
| `custom_tracking_system/data/calib_pho_hue.json` | Calibration camera cho 1 video cụ thể (best-effort, xem §8.3) |
| `Map/fit_camera_pose.py` | Tool khớp camera bằng overlay road-network lên frame video (tái dùng được cho video/giao lộ khác) |
| `custom_tracking_system/scripts/demo_real_video_scenario.py` | Script demo chính — KHÔNG qua CARLA |

### 8.2 Khác biệt so với demo CARLA (mục 1-7)

- Detect + track THẬT (YOLO11s_vn + ByteTrack), không dùng GT actor ID.
- Goal Classifier dùng weights real-world (`goal_classifier_real.pkl`), không phải bản CARLA.
- KHÔNG có collision sensor → tai nạn phải đánh dấu TAY qua API có sẵn `POST /api/incidents/{id}/mark` (`demo/server.py:55`) trong lúc video đang chạy; script tự poll state file để nhận mark từ bên ngoài và tiếp tục chạy Goal Classifier + Route Predictor cho đúng track đó.
- Map/route dùng `pho_hue_tkc`, không phải `hanoi_district3` (route_predictor.py đã thêm `graph_path` override để 2 map chạy song song, không xung đột).
- `demo/map_view.html` + `demo/server.py` tái dùng nguyên — không sửa (cùng schema JSON `incident_state.json`).

**Chạy:**
```bash
python custom_tracking_system/scripts/demo_real_video_scenario.py --loop
python demo/server.py
# mở http://localhost:5000, gọi POST /api/incidents/{track_id}/mark khi muốn demo tai nạn
```

### 8.3 Hạn chế cần nói rõ khi demo/báo cáo

- **Calibration camera là best-effort**: khớp bằng cách chiếu road network thật (từ OSM) lên 1 frame video và chỉnh tham số camera (vị trí/pitch/yaw/fov) bằng mắt cho tới khi khớp với đường thật trong ảnh — KHÔNG phải đo đạc thực địa hay click 4 điểm chuẩn qua Street View (đã thử nhưng không dò được đúng góc camera). Xem `Map/pho_hue_tkc_calib_overlay.jpg` để soát lại. Sai số ước tính vài mét — đủ cho demo định tính, KHÔNG dùng cho production.
- Calibration hiện chỉ gắn với 1 camera/video cụ thể (`12h.10.9.22.mp4`). Các video KHÁC cùng giao lộ Phố Huế–Trần Khát Chân dùng được calibration này (camera cố định một chỗ); video ở giao lộ KHÁC (Giảng Võ, Cửa Nam...) cần map + calibration riêng theo đúng quy trình ở §8.
- Nhân tiện đã fix 1 bug có sẵn trong `route_predictor.py`: loại các đoạn đường <1.5m (artifact nội bộ junction do netconvert sinh ra) khỏi tìm kiếm nearest-road, tránh route bị "chết" ngay hop đầu khi vehicle ở gần junction phức tạp. Fix áp dụng cho cả map CARLA cũ — đã regression test, vẫn ra đúng 52 junctions/464 roads.
- Đã verify end-to-end (script chạy, world-position hợp lý, mark tay, route 4-hop ra đúng trên map thật) nhưng **chưa rehearsal trực tiếp trước hội đồng** — áp dụng checklist Phase 4 (demo trực tiếp) đã thống nhất cho bản CARLA.

---

## 9. Route Predictor — confidence thật cho hop ≥1 — bổ sung 2026-06-22

> Lý do: confidence hiển thị trên `map_view.html` cho hop 2-4 trước đây luôn là **100% giả** (hardcode `{'straight':1.0,...}` khi Goal Classifier chưa đủ frame, hoặc khi heading change <12° — xem `goal_classifier.py:131-137`), và bản thân Route Predictor luôn giả định "đi thẳng" cho mọi hop sau hop đầu. Đây gây hiểu lầm nghiêm trọng nếu hội đồng hỏi "model chắc chắn đến mức đó dựa vào đâu?".

### 9.1 Bằng chứng đo được

Phân tích `data/trajectories_hanoi_v2/episode_*_goals.csv` (script: `custom_tracking_system/scripts/eval/eval_route_predictor_priors.py`) — 36,997 goal events, 2,263 chuỗi xe CARLA đi qua ≥2 junction liên tiếp trong cùng 1 episode:

| Đại lượng | Giá trị |
|---|---|
| Giả định "luôn đi thẳng" cho hop≥1 — accuracy thật | **37.04%** (n=34,698) |
| Prior thực nghiệm (pooled hop≥1) | straight=0.370, left=0.366, right=0.179, u_turn=0.085 |
| Markov bậc 1 (theo hướng hop trước) — accuracy | 39.01% — chỉ nhích nhẹ, không đáng dùng thay prior đơn giản |

→ "straight" và "left" gần như ngang nhau ở hop≥1 — hệ thống **không có cơ sở** để tự tin hơn ~37% cho các hop sau hop đầu, vì không có tín hiệu hành vi quan sát được (xe chưa tiếp cận junction đó).

### 9.2 Đã sửa trong `custom_tracking_system/route_predictor.py`

- Hop đầu (junction thật đầu tiên gặp): vẫn dùng dự đoán Goal Classifier thật (`direction_probs`).
- Mọi junction thật sau đó: dùng prior thực nghiệm trên (lưu tại `data/route_predictor_priors.json`, có fallback hardcode nếu file thiếu) — chọn hướng có xác suất cao nhất (`straight`, nhưng giờ báo đúng **37%** thay vì 100%.
- **Fix thêm 1 bug liên quan**: hướng rẽ thật từng được gán theo *chỉ số loop* (`hop==0`) thay vì "junction thật đầu tiên gặp" — với map có nhiều đoạn connector ngắn (vd `pho_hue_tkc`), nếu đoạn đường gần nhất chưa tới junction thật, hướng rẽ thật từ Goal Classifier bị bỏ qua hoàn toàn. Đã sửa dùng cờ `direction_applied`, verify lại: hướng thật giờ áp dụng đúng vào junction thật đầu tiên kể cả khi cách vài đoạn connector.
- Đã regression-test cả 2 map (hanoi_district3, pho_hue_tkc) — route vẫn ra hợp lý, không phá pipeline cũ.

### 9.3 Còn lại

- Markov bậc 1 không cải thiện đáng kể — nếu muốn tốt hơn 37-39%, cần thử điều kiện theo loại đường (đường lớn/nhỏ) hoặc road-topology feature, chưa làm.
- Dataset CCTV thật chưa có chuỗi multi-junction liên tục (mỗi clip chỉ quan sát 1 giao lộ) nên chưa tính được prior tương tự cho real-world — đang dùng prior từ CARLA cho cả 2 map.
