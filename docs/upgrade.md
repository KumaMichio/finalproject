# Kế Hoạch Nâng Cấp Hệ Thống — Hướng Thực Tế

> **Trạng thái cập nhật: 2026-06-11**
> Tất cả các hạng mục trong tài liệu này đã hoàn thành (xem bảng "Những gì còn thiếu" bên dưới — đã cập nhật). Xung đột môi trường Python/CARLA (CARLA 0.9.9.4 yêu cầu Python 3.7, nhưng boxmot/ultralytics yêu cầu Python 3.8+) **đã được giải quyết** bằng kiến trúc `carla_bridge` (2 process Python giao tiếp qua TCP — xem `bao_cao_tien_do.md` mục 7 item 2 và `execute.md` mục 2.0/Cách 4). Không còn hạng mục chặn nào trước khi chạy full pipeline.
>
> **Cập nhật môi trường (2026-06-11):** Dự án đã nâng cấp CARLA lên **0.9.14**
> (`WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.14-py3.7-win-amd64.egg`,
> không còn `carla_extracted`). `run.bat`/`start.bat` đã được sửa để trỏ
> `PYTHONPATH` tới file `.egg` này. Đã verify `collect_trajectory_data.py`
> chạy được với CARLA 0.9.14 (Town10HD_Opt, 10Hz, sinh CSV trajectory hợp lệ).
>
> **Mới hoàn thành (2026-06-11):** Giai đoạn "Learned Trajectory Prediction"
> — calibration pixel↔world, Seq2Seq GRU multi-modal + intent classification,
> CARLA trajectory dataset collector, training script, và ensemble với Kalman.
> Xem mục **"Giai Đoạn 6 — Learned Trajectory Prediction"** ở cuối tài liệu.

## Mục Tiêu Nâng Cấp

Biến hệ thống từ prototype AI pipeline + backend API thành một **hệ thống giám sát CCTV hoàn chỉnh sát thực tế**, với các khả năng:

1. Nhận diện đối tượng lạ (chưa biết) ngay khi vào camera — không cần đăng ký trước.
2. Hiển thị nhiều màn hình camera giám sát đồng thời trên Web Dashboard.
3. Phát hiện và cảnh báo ngay lập tức khi xảy ra sự cố (tai nạn, bỏ trốn, đâm người...).
4. Lưu bằng chứng tự động (ảnh crop, video clip) khi có sự cố.
5. Theo dõi đối tượng liên quan xuyên nhiều camera sau sự cố.

---

## Phân Tích Hệ Thống Hiện Tại

### Những gì đã hoạt động đúng hướng

**Nhận diện unknown object đã có cơ chế:**

Khi một đối tượng lạ xuất hiện trong camera, luồng xử lý hiện tại đã đúng:

```
Object vào frame
→ YOLOv8s detect (bounding box + class)
→ OSNet trích xuất appearance feature (512 chiều)
→ So sánh cosine similarity với gallery
→ Không khớp → tạo Global ID mới, thêm vào gallery
→ Sang camera khác → match lại qua ReID
```

Đây là hành vi đúng với thực tế — không cần đăng ký trước, tự build gallery on-the-fly.

### Những gì còn thiếu

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| Web Dashboard | ✅ Hoàn thành | React + Vite + Tailwind; camera grid, incident panel, alert mgmt, object history |
| Incident Detection | ✅ Hoàn thành | 12 loại sự cố (10 reactive + 2 proactive); cooldown 4s |
| Velocity history trong Tracker | ✅ Hoàn thành | ByteTrackWrapper lưu velocity; IncidentDetector tính sudden stop/accel |
| Evidence package | ✅ Hoàn thành | Ring buffer 30s; lưu clip + crop + JSON khi CRITICAL incident |
| CARLA scenario scripting | ✅ Hoàn thành | `scenario_controller.py` — 5 kịch bản tai nạn |
| ByteTrack / Kalman filter | ✅ Hoàn thành | `ByteTrackWrapper` (boxmot.ByteTrack) — Kalman + Hungarian |
| Vehicle ReID | ✅ Hoàn thành | `DualReIDExtractor` — OSNet VeRi-776 fine-tuned (575 classes, 20 checkpoints) |
| Recording & Playback | ⚠️ Một phần | EvidencePackage lưu clip khi sự cố; chưa có continuous recording theo giờ |
| Môi trường Python/CARLA | ✅ Đã nâng cấp | CARLA **0.9.14** (env `carla-sim`, Python 3.7.16); pipeline chính (`boxmot`/`ultralytics`) chạy ở env `veri-train` (Python 3.10) |

---

## CARLA 0.9.14 — Khả Năng Tạo Kịch Bản Tai Nạn

### Kết luận: Hoàn toàn có thể

CARLA cung cấp đầy đủ ba công cụ cần thiết:

---

### Công cụ 1: Collision Sensor

Sensor `sensor.other.collision` gắn vào xe, trigger ngay lập tức khi có va chạm.

```python
collision_bp = blueprint_library.find('sensor.other.collision')
collision_sensor = world.spawn_actor(
    collision_bp,
    carla.Transform(),
    attach_to=vehicle
)

def on_collision(event):
    impulse = event.normal_impulse
    intensity = (impulse.x**2 + impulse.y**2 + impulse.z**2) ** 0.5

    print(f"Va chạm: actor={event.actor.id} → other={event.other_actor.id}")
    print(f"Lực va chạm: {intensity:.1f} (> 500 = tai nạn nghiêm trọng)")

collision_sensor.listen(on_collision)
```

Event object chứa:
- `event.actor` — xe có gắn sensor
- `event.other_actor` — đối tượng bị đâm (xe / người / cột)
- `event.normal_impulse` — vector lực (magnitude = mức độ nghiêm trọng)

---

### Công cụ 2: Traffic Manager — điều khiển hành vi xe

```python
tm = client.get_trafficmanager(8000)

# Bỏ qua xe khác (không nhường đường) → dễ gây va chạm
tm.ignore_vehicles_percentage(reckless_vehicle, 80)

# Vượt đèn đỏ
tm.ignore_lights_percentage(reckless_vehicle, 100)

# Tăng tốc vượt giới hạn (giá trị âm = nhanh hơn giới hạn)
tm.vehicle_percentage_speed_difference(reckless_vehicle, -50)

# Không giữ khoảng cách với xe phía trước → đâm từ phía sau
tm.distance_to_leading_vehicle(reckless_vehicle, 0)

# Bỏ qua người đi bộ
tm.ignore_walkers_percentage(reckless_vehicle, 100)
```

---

### Công cụ 3: Direct Vehicle Control

```python
# Tắt autopilot, lái xe trực tiếp theo script
vehicle.set_autopilot(False)

# Tăng tốc hướng vào xe khác
vehicle.apply_control(carla.VehicleControl(
    throttle=1.0,
    steer=0.3,
    brake=0.0
))

# Dừng xe (giả vờ dừng sau tai nạn, rồi bỏ trốn)
vehicle.apply_control(carla.VehicleControl(
    throttle=0.0,
    brake=1.0
))
```

---

### Bảng kịch bản có thể tạo

| Kịch bản | Khả thi | Cách tạo |
|---|---|---|
| Xe đâm xe bỏ trốn (hit-and-run) | Có | Collision sensor + xe tiếp tục chạy sau va chạm |
| Xe đâm người đi bộ | Có | `ignore_walkers_percentage(100)` |
| Xe vượt đèn đỏ gây tai nạn | Có | `ignore_lights_percentage(100)` |
| Đâm từ phía sau | Có | `distance_to_leading_vehicle(0)` + tốc độ cao |
| Va chạm chuỗi (A → B → C) | Có | Nhiều xe khoảng cách gần, tốc độ cao |
| Xe dừng đột ngột giữa đường | Có | `apply_control(brake=1.0)` đột ngột |

**Không thể làm:** Biển số xe đọc được (CARLA không render biển số thực), vật lý nước / băng ảnh hưởng phanh.

---

## Kiến Trúc Mục Tiêu

```
┌─────────────────────────────────────────────────────────────┐
│              CARLA Simulator / Camera IP Thực Tế            │
│         CAM_001       CAM_002       CAM_003      CAM_N      │
└──────────────┬──────────────┬────────────────────┬──────────┘
               │              │                    │
┌──────────────▼──────────────▼────────────────────▼──────────┐
│                    VideoSource Layer                         │
│          (CARLA / RTSP / File / Webcam — đã có)             │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                     AI Processing Engine                     │
│                                                             │
│  YOLOv8s ✅       ByteTrack + Kalman     OSNet ReID         │
│  Detection    →   (nâng cấp từ IoU)  →  + Vehicle ReID     │
│                   + Velocity History                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Incident Detector (MỚI)                 │   │
│  │  • Proximity detection (hai xe tiến lại gần nhau)   │   │
│  │  • Sudden stop detection (tốc độ giảm đột ngột)     │   │
│  │  • Fleeing detection (tăng tốc sau va chạm)         │   │
│  │  • Cross-camera alert (xe bỏ trốn sang camera khác) │   │
│  │  • Evidence capture (crop ảnh + trigger clip)        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ScenarioController (MỚI) — tạo kịch bản test trong CARLA  │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend (đã có)                   │
│  REST API  +  WebSocket push  +  MJPEG streaming            │
│  Database SQLite 5 bảng                                     │
└──────────────────────────────┬──────────────────────────────┘
                               │  WebSocket real-time
┌──────────────────────────────▼──────────────────────────────┐
│                    Web Dashboard (MỚI)                       │
│                                                             │
│  ┌──────────┬──────────┬──────────┬──────────┐             │
│  │ CAM_001  │ CAM_002  │ CAM_003  │ CAM_004  │  Camera     │
│  │ [live]   │ [live]   │ [live]   │ [live]   │  Grid       │
│  │ bbox+ID  │ bbox+ID  │ bbox+ID  │ bbox+ID  │             │
│  └──────────┴──────────┴──────────┴──────────┘             │
│                                                             │
│  🚨 CRITICAL: Hit-and-run tại CAM_001 — 15:23:07           │
│     Xe #1042 [ảnh crop] đang ở CAM_003                     │
│     [Xem clip sự cố] [Lịch sử di chuyển] [Xác nhận]        │
│                                                             │
│  Timeline: CAM_001 15:23 → CAM_002 15:24 → CAM_003 15:25  │
└─────────────────────────────────────────────────────────────┘
```

---

## Chi Tiết Các Module Cần Xây Dựng

---

### Module 1: `scenario_controller.py` (MỚI)

**Vị trí:** `custom_tracking_system/modules/scenario_controller.py`

**Mục đích:** Script các kịch bản tai nạn có kiểm soát trong CARLA để test hệ thống.

```python
class ScenarioController:
    """
    Tạo và điều phối các kịch bản sự cố trong CARLA.
    Kịch bản được thiết kế để xe xảy ra va chạm trong tầm nhìn camera.
    """

    def __init__(self, world, client, camera_positions):
        self.world = world
        self.tm = client.get_trafficmanager(8000)
        self.camera_positions = camera_positions  # để spawn xe gần camera
        self.active_sensors = []
        self.collision_log = []

    def run_hit_and_run(self, near_camera_id="CAM_001"):
        """
        Kịch bản: Xe A đâm Xe B rồi bỏ trốn.
        Xe A sẽ chạy qua CAM_001 → CAM_002 → CAM_003.
        """
        spawn_points = self._get_spawn_points_near_camera(near_camera_id)

        # Spawn xe nạn nhân — đứng yên hoặc autopilot bình thường
        victim = self._spawn_vehicle(spawn_points[0], autopilot=True)

        # Spawn xe hung thủ — phía sau nạn nhân
        attacker = self._spawn_vehicle(spawn_points[1], autopilot=False)

        # Gắn collision sensor vào xe hung thủ
        self._attach_collision_sensor(attacker, victim)

        # Tăng tốc hướng vào xe nạn nhân
        attacker.set_autopilot(True)
        self.tm.distance_to_leading_vehicle(attacker, 0)
        self.tm.vehicle_percentage_speed_difference(attacker, -80)
        self.tm.ignore_vehicles_percentage(attacker, 100)

        return attacker, victim

    def run_pedestrian_hit(self, near_camera_id="CAM_002"):
        """Kịch bản: Xe đâm người đi bộ."""
        spawn_points = self._get_spawn_points_near_camera(near_camera_id)
        vehicle = self._spawn_vehicle(spawn_points[0], autopilot=True)

        self.tm.ignore_walkers_percentage(vehicle, 100)
        self.tm.vehicle_percentage_speed_difference(vehicle, -30)

        self._attach_collision_sensor(vehicle)
        return vehicle

    def run_red_light_crash(self, intersection_camera="CAM_001"):
        """Kịch bản: Hai xe vượt đèn đỏ từ hai hướng."""
        spawn_points = self._get_spawn_points_near_camera(intersection_camera)

        vehicle_a = self._spawn_vehicle(spawn_points[0], autopilot=True)
        vehicle_b = self._spawn_vehicle(spawn_points[1], autopilot=True)

        # Cả hai xe vượt đèn đỏ
        self.tm.ignore_lights_percentage(vehicle_a, 100)
        self.tm.ignore_lights_percentage(vehicle_b, 100)

        self._attach_collision_sensor(vehicle_a, vehicle_b)
        return vehicle_a, vehicle_b

    def _attach_collision_sensor(self, vehicle, other=None):
        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)

        def on_collision(event):
            impulse = event.normal_impulse
            intensity = (impulse.x**2 + impulse.y**2 + impulse.z**2) ** 0.5
            if intensity > 200:  # lọc va chạm nhẹ
                self.collision_log.append({
                    'timestamp': event.timestamp,
                    'attacker_id': event.actor.id,
                    'victim_id': event.other_actor.id,
                    'intensity': intensity,
                    'victim_type': event.other_actor.type_id
                })

        sensor.listen(on_collision)
        self.active_sensors.append(sensor)

    def get_collision_log(self):
        return self.collision_log.copy()

    def cleanup(self):
        for sensor in self.active_sensors:
            try:
                sensor.destroy()
            except Exception:
                pass
        self.active_sensors.clear()
```

---

### Module 2: `incident_detector.py` (MỚI)

**Vị trí:** `custom_tracking_system/modules/incident_detector.py`

**Mục đích:** Phân tích dữ liệu tracking để phát hiện các pattern sự cố theo thời gian thực.

Cần dữ liệu đầu vào từ tracker: **lịch sử tốc độ** theo thời gian (hiện tại tracker chưa lưu).

#### Các loại sự cố cần phát hiện

| # | Loại | Severity | Logic |
|---|---|---|---|
| 1 | **Collision** | CRITICAL | Hai bounding box overlap + ít nhất 1 xe sudden stop |
| 2 | **Hit and run** | CRITICAL | Collision xảy ra + xe không dừng lại sau đó |
| 3 | **Sudden stop** | WARNING | Tốc độ giảm > 70% trong 3 frame liên tiếp |
| 4 | **Sudden acceleration** | WARNING | Tốc độ tăng > 100% trong 3 frame |
| 5 | **Pedestrian proximity** | CRITICAL | Xe tiến gần người đi bộ với tốc độ cao |
| 6 | **Overspeed** | WARNING | speed > threshold được cấu hình |
| 7 | **Wrong way** | CRITICAL | Hướng di chuyển ngược chiều quy định của lane |
| 8 | **Stopped on road** | WARNING | Xe dừng > N giây ở vị trí không phải bãi đỗ |
| 9 | **Loitering** | WARNING | Người ở cùng khu vực quá lâu |
| 10 | **Camera transition** | INFO | Đối tượng xuất hiện ở camera mới sau khi mất ở camera cũ |

```python
from collections import defaultdict, deque
from datetime import datetime
import numpy as np

class IncidentDetector:
    """
    Phân tích dữ liệu tracking real-time để phát hiện sự cố.
    Nhận input từ GlobalTracker, output là danh sách incident alerts.
    """

    def __init__(self, config=None):
        config = config or {}

        # Ngưỡng phát hiện (có thể cấu hình qua YAML)
        self.speed_limit = config.get('speed_limit', 15)          # pixels/frame
        self.sudden_stop_ratio = config.get('sudden_stop_ratio', 0.3)  # giảm còn 30%
        self.sudden_accel_ratio = config.get('sudden_accel_ratio', 2.0)  # tăng 200%
        self.proximity_threshold = config.get('proximity_threshold', 80)  # pixels
        self.stop_frames = config.get('stop_frames', 50)           # ~5s ở 10fps
        self.loiter_frames = config.get('loiter_frames', 300)      # ~30s
        self.loiter_radius = config.get('loiter_radius', 60)       # pixels

        # Lịch sử tốc độ: {global_id: deque of (timestamp, speed, position)}
        self.speed_history = defaultdict(lambda: deque(maxlen=30))

        # Trạng thái theo dõi per object
        self.stop_counter = defaultdict(int)       # số frame đứng yên
        self.loiter_anchor = {}                    # {global_id: anchor_position}
        self.loiter_counter = defaultdict(int)

        # Đã phát hiện sự cố: tránh duplicate alert
        self.recent_incidents = {}  # {(type, global_id): timestamp}
        self.cooldown_seconds = 5

    def update(self, global_tracks, camera_id, frame_idx):
        """
        Cập nhật trạng thái và kiểm tra sự cố cho tất cả tracks hiện tại.

        Args:
            global_tracks: list of global track dicts từ GlobalTracker
            camera_id: camera đang xử lý
            frame_idx: frame index hiện tại

        Returns:
            list: danh sách incident dicts, empty nếu không có sự cố
        """
        incidents = []
        now = datetime.now()

        # Cập nhật velocity history cho mỗi track
        for track in global_tracks:
            gid = track['global_id']
            pos = self._box_center(track['box'])
            speed = self._compute_speed(gid, pos, now)
            self.speed_history[gid].append((now, speed, pos))

        # Kiểm tra từng loại sự cố
        incidents += self._check_sudden_stop(global_tracks, camera_id, now)
        incidents += self._check_overspeed(global_tracks, camera_id, now)
        incidents += self._check_proximity(global_tracks, camera_id, now)
        incidents += self._check_stopped_on_road(global_tracks, camera_id, now)
        incidents += self._check_loitering(global_tracks, camera_id, now)

        # Lọc duplicate
        incidents = self._deduplicate(incidents, now)

        return incidents

    def _check_sudden_stop(self, tracks, camera_id, now):
        incidents = []
        for track in tracks:
            gid = track['global_id']
            history = list(self.speed_history[gid])
            if len(history) < 5:
                continue

            recent_speeds = [h[1] for h in history[-5:] if h[1] is not None]
            if len(recent_speeds) < 3:
                continue

            speed_before = np.mean(recent_speeds[:2])
            speed_after = recent_speeds[-1]

            # Tốc độ giảm đột ngột: từ đang chạy xuống gần 0
            if speed_before > 5 and speed_after < speed_before * self.sudden_stop_ratio:
                incidents.append(self._make_incident(
                    type='SUDDEN_STOP',
                    severity='WARNING',
                    global_id=gid,
                    camera_id=camera_id,
                    message=f"Xe #{gid} dừng đột ngột: {speed_before:.1f} → {speed_after:.1f} px/frame",
                    details={'speed_before': speed_before, 'speed_after': speed_after}
                ))
        return incidents

    def _check_overspeed(self, tracks, camera_id, now):
        incidents = []
        for track in tracks:
            if track['class'] not in ('car', 'truck', 'bus'):
                continue
            gid = track['global_id']
            history = list(self.speed_history[gid])
            if not history:
                continue
            current_speed = history[-1][1]
            if current_speed and current_speed > self.speed_limit:
                incidents.append(self._make_incident(
                    type='OVERSPEED',
                    severity='WARNING',
                    global_id=gid,
                    camera_id=camera_id,
                    message=f"Xe #{gid} vượt tốc độ: {current_speed:.1f} px/frame",
                    details={'speed': current_speed, 'limit': self.speed_limit}
                ))
        return incidents

    def _check_proximity(self, tracks, camera_id, now):
        """
        Kiểm tra xe tiến lại gần nhau hoặc xe gần người đi bộ với tốc độ cao.
        Đây là dấu hiệu sớm của va chạm sắp xảy ra.
        """
        incidents = []
        vehicles = [t for t in tracks if t['class'] in ('car', 'truck', 'bus')]
        pedestrians = [t for t in tracks if t['class'] == 'person']

        # Xe gần nhau + tốc độ cao
        for i, v1 in enumerate(vehicles):
            for v2 in vehicles[i+1:]:
                dist = self._box_distance(v1['box'], v2['box'])
                if dist < self.proximity_threshold:
                    s1 = self._get_current_speed(v1['global_id'])
                    s2 = self._get_current_speed(v2['global_id'])
                    if (s1 and s1 > 8) or (s2 and s2 > 8):
                        incidents.append(self._make_incident(
                            type='VEHICLE_PROXIMITY',
                            severity='CRITICAL',
                            global_id=v1['global_id'],
                            camera_id=camera_id,
                            message=f"Xe #{v1['global_id']} và #{v2['global_id']} "
                                    f"tiến gần nhau nguy hiểm (dist={dist:.0f}px)",
                            details={
                                'other_id': v2['global_id'],
                                'distance': dist,
                                'speed_1': s1,
                                'speed_2': s2
                            }
                        ))

        # Xe gần người đi bộ
        for v in vehicles:
            for p in pedestrians:
                dist = self._box_distance(v['box'], p['box'])
                speed = self._get_current_speed(v['global_id'])
                if dist < self.proximity_threshold and speed and speed > 5:
                    incidents.append(self._make_incident(
                        type='PEDESTRIAN_DANGER',
                        severity='CRITICAL',
                        global_id=v['global_id'],
                        camera_id=camera_id,
                        message=f"Xe #{v['global_id']} tiến gần người #{p['global_id']} "
                                f"với tốc độ cao (dist={dist:.0f}px, speed={speed:.1f})",
                        details={
                            'pedestrian_id': p['global_id'],
                            'distance': dist,
                            'vehicle_speed': speed
                        }
                    ))
        return incidents

    def _check_stopped_on_road(self, tracks, camera_id, now):
        incidents = []
        for track in tracks:
            if track['class'] not in ('car', 'truck', 'bus'):
                continue
            gid = track['global_id']
            speed = self._get_current_speed(gid)

            if speed is not None and speed < 1.0:
                self.stop_counter[gid] += 1
            else:
                self.stop_counter[gid] = 0

            if self.stop_counter[gid] == self.stop_frames:  # chỉ alert 1 lần
                incidents.append(self._make_incident(
                    type='STOPPED_VEHICLE',
                    severity='WARNING',
                    global_id=gid,
                    camera_id=camera_id,
                    message=f"Xe #{gid} dừng giữa đường trên {self.stop_frames} frame",
                    details={'stop_frames': self.stop_counter[gid]}
                ))
        return incidents

    def _check_loitering(self, tracks, camera_id, now):
        incidents = []
        for track in tracks:
            if track['class'] != 'person':
                continue
            gid = track['global_id']
            pos = self._box_center(track['box'])

            if gid not in self.loiter_anchor:
                self.loiter_anchor[gid] = pos
                self.loiter_counter[gid] = 0
            else:
                anchor = self.loiter_anchor[gid]
                dist = np.linalg.norm(np.array(pos) - np.array(anchor))
                if dist < self.loiter_radius:
                    self.loiter_counter[gid] += 1
                else:
                    self.loiter_anchor[gid] = pos
                    self.loiter_counter[gid] = 0

            if self.loiter_counter[gid] == self.loiter_frames:
                incidents.append(self._make_incident(
                    type='LOITERING',
                    severity='WARNING',
                    global_id=gid,
                    camera_id=camera_id,
                    message=f"Người #{gid} đứng ở cùng khu vực trên {self.loiter_frames} frame",
                    details={'duration_frames': self.loiter_counter[gid]}
                ))
        return incidents

    # --- helpers ---

    def _compute_speed(self, gid, current_pos, now):
        history = list(self.speed_history[gid])
        if not history:
            return None
        last_time, _, last_pos = history[-1]
        dt = (now - last_time).total_seconds()
        if dt < 1e-6:
            return None
        dist = np.linalg.norm(np.array(current_pos) - np.array(last_pos))
        return dist / dt  # pixels/second

    def _get_current_speed(self, gid):
        history = list(self.speed_history[gid])
        return history[-1][1] if history else None

    def _box_center(self, box):
        x1, y1, x2, y2 = box
        return [(x1 + x2) / 2, (y1 + y2) / 2]

    def _box_distance(self, box1, box2):
        c1 = np.array(self._box_center(box1))
        c2 = np.array(self._box_center(box2))
        return np.linalg.norm(c1 - c2)

    def _make_incident(self, type, severity, global_id, camera_id, message, details=None):
        return {
            'type': type,
            'severity': severity,
            'global_id': global_id,
            'camera_id': camera_id,
            'timestamp': datetime.now(),
            'message': message,
            'details': details or {}
        }

    def _deduplicate(self, incidents, now):
        filtered = []
        for inc in incidents:
            key = (inc['type'], inc['global_id'])
            last = self.recent_incidents.get(key)
            if last is None or (now - last).total_seconds() > self.cooldown_seconds:
                self.recent_incidents[key] = now
                filtered.append(inc)
        return filtered
```

---

### Module 3: `evidence_package.py` (MỚI)

**Vị trí:** `custom_tracking_system/modules/evidence_package.py`

**Mục đích:** Khi phát hiện sự cố, tự động đóng gói bằng chứng.

```python
import cv2
import os
import json
from datetime import datetime
from collections import deque

class EvidencePackage:
    """
    Tự động lưu bằng chứng khi xảy ra sự cố:
    - Ảnh crop của các đối tượng liên quan
    - Video clip 30s trước và sau sự cố
    - JSON metadata của sự cố
    """

    def __init__(self, output_dir="evidence"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Ring buffer: lưu 300 frame gần nhất mỗi camera (~30s ở 10fps)
        self.frame_buffers = {}   # {camera_id: deque of (frame, timestamp)}
        self.buffer_size = 300

        # Clip đang ghi tiếp sau sự cố
        self.pending_clips = {}   # {incident_id: {'writer', 'frames_left'}}

    def buffer_frame(self, camera_id, frame, timestamp):
        """Gọi mỗi frame — duy trì ring buffer."""
        if camera_id not in self.frame_buffers:
            self.frame_buffers[camera_id] = deque(maxlen=self.buffer_size)
        self.frame_buffers[camera_id].append((frame.copy(), timestamp))

        # Tiếp tục ghi clip hậu sự cố nếu đang có
        self._write_pending_clips(camera_id, frame)

    def capture(self, incident, frames_dict, global_tracks):
        """
        Tạo evidence package cho 1 sự cố.

        Args:
            incident: incident dict từ IncidentDetector
            frames_dict: {camera_id: frame} — frame hiện tại
            global_tracks: danh sách track hiện tại

        Returns:
            str: đường dẫn thư mục evidence
        """
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        incident_id = f"{incident['type']}_{ts}"
        out_dir = os.path.join(self.output_dir, incident_id)
        os.makedirs(out_dir, exist_ok=True)

        camera_id = incident['camera_id']
        frame = frames_dict.get(camera_id)

        # 1. Ảnh crop đối tượng liên quan
        crops = {}
        if frame is not None:
            involved_ids = [incident['global_id']]
            if 'other_id' in incident.get('details', {}):
                involved_ids.append(incident['details']['other_id'])
            if 'pedestrian_id' in incident.get('details', {}):
                involved_ids.append(incident['details']['pedestrian_id'])

            for track in global_tracks:
                if track['global_id'] in involved_ids:
                    crop = self._crop_object(frame, track['box'])
                    crop_path = os.path.join(out_dir, f"object_{track['global_id']}.jpg")
                    cv2.imwrite(crop_path, crop)
                    crops[track['global_id']] = crop_path

        # 2. Snapshot toàn cảnh
        if frame is not None:
            snapshot_path = os.path.join(out_dir, "snapshot.jpg")
            cv2.imwrite(snapshot_path, frame)

        # 3. Clip trước sự cố (từ ring buffer)
        clip_path = self._save_pre_incident_clip(camera_id, out_dir)

        # 4. Bắt đầu ghi clip sau sự cố (300 frame tiếp theo)
        if frame is not None:
            self._start_post_incident_clip(incident_id, camera_id, frame, out_dir)

        # 5. JSON metadata
        meta = {
            'incident_id': incident_id,
            'type': incident['type'],
            'severity': incident['severity'],
            'timestamp': incident['timestamp'].isoformat(),
            'camera_id': camera_id,
            'global_id': incident['global_id'],
            'message': incident['message'],
            'details': incident['details'],
            'evidence': {
                'crops': crops,
                'snapshot': snapshot_path if frame is not None else None,
                'pre_incident_clip': clip_path,
                'post_incident_clip': os.path.join(out_dir, "clip_post.mp4")
            }
        }
        meta_path = os.path.join(out_dir, "metadata.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return out_dir, meta

    def _crop_object(self, frame, box, padding=20):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)
        return frame[y1:y2, x1:x2]

    def _save_pre_incident_clip(self, camera_id, out_dir):
        """Lưu các frame từ ring buffer thành clip."""
        if camera_id not in self.frame_buffers:
            return None
        frames = list(self.frame_buffers[camera_id])
        if not frames:
            return None

        clip_path = os.path.join(out_dir, "clip_pre.mp4")
        h, w = frames[0][0].shape[:2]
        writer = cv2.VideoWriter(
            clip_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            10, (w, h)
        )
        for frame, _ in frames:
            writer.write(frame)
        writer.release()
        return clip_path

    def _start_post_incident_clip(self, incident_id, camera_id, first_frame, out_dir):
        h, w = first_frame.shape[:2]
        clip_path = os.path.join(out_dir, "clip_post.mp4")
        writer = cv2.VideoWriter(
            clip_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            10, (w, h)
        )
        writer.write(first_frame)
        self.pending_clips[incident_id] = {
            'camera_id': camera_id,
            'writer': writer,
            'frames_left': 299
        }

    def _write_pending_clips(self, camera_id, frame):
        done = []
        for inc_id, clip_info in self.pending_clips.items():
            if clip_info['camera_id'] == camera_id:
                clip_info['writer'].write(frame)
                clip_info['frames_left'] -= 1
                if clip_info['frames_left'] <= 0:
                    clip_info['writer'].release()
                    done.append(inc_id)
        for inc_id in done:
            del self.pending_clips[inc_id]
```

---

### Module 4: Nâng cấp `tracker.py` — Thêm Velocity History

Hiện tại tracker lưu `positions` nhưng không lưu timestamp, nên không tính được tốc độ thực.
Cần thêm `timestamps` vào mỗi track để `IncidentDetector` tính velocity chính xác.

**Thay đổi trong `SimpleTracker.update()`:**

```python
# Trong track dict, thêm trường timestamps:
self.tracks[self.next_id] = {
    'id': self.next_id,
    'box': det['box'],
    'class': det['class'],
    'age': 0,
    'hits': 1,
    'positions': [self._box_center(det['box'])],
    'timestamps': [datetime.now()],   # THÊM MỚI
    'speeds': []                       # THÊM MỚI
}

# Khi update track đã có:
track['positions'].append(self._box_center(det['box']))
track['timestamps'].append(datetime.now())          # THÊM MỚI

# Tính speed và lưu
if len(track['positions']) >= 2:
    dt = (track['timestamps'][-1] - track['timestamps'][-2]).total_seconds()
    if dt > 0:
        d = np.linalg.norm(
            np.array(track['positions'][-1]) - np.array(track['positions'][-2])
        )
        track['speeds'].append(d / dt)
```

---

### Module 5: Web Dashboard

**Vị trí:** `frontend/` (thư mục mới ngang hàng với `custom_tracking_system/` và `server/`)

**Stack:** React 18 + Tailwind CSS + Recharts + WebSocket native

#### Cấu trúc thư mục

```
frontend/
  src/
    components/
      CameraGrid.jsx          — grid nhiều camera live feed
      CameraFeed.jsx          — 1 ô camera: MJPEG img + bbox overlay
      IncidentPanel.jsx       — danh sách sự cố real-time (scroll)
      IncidentCard.jsx        — 1 sự cố: ảnh crop + severity + nút xem clip
      AlertBadge.jsx          — badge số lượng alert chưa xử lý
      StatsBar.jsx            — FPS, active objects, alert count
    pages/
      Dashboard.jsx           — trang chính: camera grid + incident panel
      IncidentDetail.jsx      — chi tiết 1 sự cố: clip + timeline xuyên camera
      ObjectHistory.jsx       — lịch sử di chuyển 1 đối tượng qua các camera
      AlertManagement.jsx     — quản lý + filter toàn bộ alerts
    hooks/
      useWebSocket.js         — kết nối WS, auto-reconnect
      useIncidents.js         — state management cho incidents real-time
    services/
      api.js                  — REST API calls
    App.jsx
    main.jsx
```

#### Layout Dashboard chính

```
┌─────────────────────────────────────────────────────────────────┐
│  CCTV AI  ●ONLINE  FPS:12  Objects:15  Alerts:3  15:23:07      │
├──────────────────────────────────────┬──────────────────────────┤
│                                      │  🚨 CRITICAL  15:23:07  │
│  ┌──────────┬──────────┬──────────┐  │  Hit-and-run tại CAM_001 │
│  │ CAM_001  │ CAM_002  │ CAM_003  │  │  Xe #1042 bỏ trốn       │
│  │ [live]   │ [live]   │ [live]   │  │  → đang ở CAM_003       │
│  │          │          │          │  │  [Xem clip] [Xác nhận]   │
│  └──────────┴──────────┴──────────┘  ├──────────────────────────┤
│                                      │  ⚠ WARNING  15:22:41    │
│                                      │  Xe dừng giữa đường      │
│                                      │  CAM_002 — Xe #1038      │
│                                      │  [Xem clip] [Xác nhận]   │
├──────────────────────────────────────┴──────────────────────────┤
│  Active: 15 objects  |  Today: 7 incidents  |  Cameras: 3/3 OK  │
└─────────────────────────────────────────────────────────────────┘
```

#### Tính năng real-time notification

```javascript
// hooks/useIncidents.js
function useIncidents() {
    const [incidents, setIncidents] = useState([]);
    const { lastMessage } = useWebSocket('/ws/alerts');

    useEffect(() => {
        if (!lastMessage) return;
        const incident = JSON.parse(lastMessage.data);

        // Thêm vào đầu danh sách
        setIncidents(prev => [incident, ...prev.slice(0, 99)]);

        // Âm thanh cảnh báo
        if (incident.severity === 'CRITICAL') {
            new Audio('/sounds/critical.mp3').play();
            // Flash màn hình đỏ 0.5s
            document.body.classList.add('alert-flash');
            setTimeout(() => document.body.classList.remove('alert-flash'), 500);
        }
    }, [lastMessage]);

    return incidents;
}
```

---

## Trạng Thái Triển Khai

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| **Giai đoạn 1** | Lõi phát hiện sự cố | ✅ Hoàn thành |
| **Giai đoạn 2** | Kịch bản test CARLA | ✅ Hoàn thành |
| **Giai đoạn 3** | Web Dashboard | ✅ Hoàn thành |
| **Giai đoạn 4** | Nâng cấp độ chính xác AI | ⏳ Đang làm (11,12,13,14,15,18,19 xong, 16–17 còn lại) |
| **Giai đoạn 5** | Hoàn thiện (recording, evaluation) | ⏳ Chưa làm |
| **Giai đoạn 6** | Learned Trajectory Prediction (Calibration + GRU + Ensemble) | ✅ Hoàn thành (2026-06-11) |

## Thứ Tự Phát Triển

```
GIAI ĐOẠN 1 — Lõi phát hiện sự cố ✅ XONG (2026-05-24)
  1. ✅ Thêm timestamp + speed history vào tracker.py
  2. ✅ Viết incident_detector.py (10 loại sự cố)
  3. ✅ Viết evidence_package.py (ring buffer + clip + crop + JSON)
  4. ✅ Tích hợp vào main.py và ai_processor.py (server)

GIAI ĐOẠN 2 — Kịch bản test CARLA ✅ XONG (2026-05-24)
  5. ✅ Viết scenario_controller.py (5 kịch bản tai nạn)

GIAI ĐOẠN 3 — Web Dashboard ✅ XONG (2026-05-24)
  6. ✅ Khởi tạo project React (Vite + Tailwind)
  7. ✅ Camera grid + MJPEG live feed (CameraGrid, CameraFeed)
  8. ✅ IncidentPanel real-time qua WebSocket (useIncidents hook)
  9. ✅ Âm thanh + flash màn hình đỏ khi CRITICAL
  10. ✅ Trang AlertManagement + ObjectHistory

GIAI ĐOẠN 4 — Nâng cấp độ chính xác AI ⏳ ĐANG LÀM
  11. ✅ YOLOv5s → YOLOv8s (hoàn thành 2026-05-29 — mAP +20%, thêm motorcycle)
  12. ✅ IoU Greedy → ByteTrack (hoàn thành 2026-05-29 — Kalman + Hungarian + dual-threshold, boxmot)
  13. ✅ FPS-normalized speed + proactive incident dùng predicted positions
      (`incident_detector._check_predicted_collision` /
      `_check_predicted_roi_entry`, hoàn thành — xem commit 02af3db)
  14. ✅ Spatio-Temporal Filter cho cross-camera matching
      (`global_tracking.GlobalTracker._is_feasible_match` /
      `is_feasible_transition`, dùng `camera_topology`, hoàn thành — xem commit 02af3db)
  15. ✅ Sửa TrajectoryPredictor dùng timestamp thay frame_idx (hoàn thành 2026-06-01)
  16. ⏳ Fine-tune YOLO11m trên dataset giao thông VN (thêm motorcycle) —
      ĐANG LÀM (2026-06-12): đã đổi `ObjectDetector`/`main.py` sang YOLO11m
      (`modules/detector.py` — `CLASSES_VN` 5 lớp person/car/motorcycle/bus/truck,
      tự dùng `weights/yolo11m_vn.pt` nếu tồn tại, fallback `yolo11m` COCO nếu
      chưa có). Đã thêm `prepare_visdrone_dataset.py` (tải + convert
      VisDrone2019-DET, merge 10 lớp → 5 lớp VN), `collect_carla_detection_data.py`
      (mới — sinh dataset detection synthetic từ CARLA: camera CCTV đặt ở góc
      phố/gắn tường tòa nhà gần các junction, tự động gán nhãn bbox 2D từ
      bounding box 3D của actor + depth camera để loại occlusion),
      `merge_detection_datasets.py` (mới — gộp VisDrone-VN + CARLA thành 1
      dataset YOLO duy nhất `data/visdrone_carla_vn/`), và
      `train_yolo_detector.py` (fine-tune qua `ultralytics`, default trỏ tới
      dataset đã gộp). Chưa chạy training thật — đang chờ máy cloud RTX 4060
      8GB (torch CUDA cần cài riêng, không downgrade torch của pipeline
      chính). `collect_carla_detection_data.py` cũng chưa chạy thử với CARLA
      thật — cần verify nhỏ (`--frames 50`) trước khi thu thập full.
  17. ⏳ Fine-tune OSNet trên VeRi-776 (xe thay người) — model VeRi-776 đã có
      trong `weights/` (`DualReIDExtractor`), nhưng chưa fine-tune thêm trên
      dataset giao thông VN
  18. ✅ Tích hợp VideoSource vào main.py (--source rtsp/file/webcam/bridge, hoàn thành 2026-06-10)
  19. ✅ TrajectoryPredictor: exp-weighted velocity → Kalman Filter constant-acceleration (hoàn thành 2026-06-10)
  → Xem chi tiết: docs/production_model_stack.md

GIAI ĐOẠN 5 — Hoàn thiện ⏳ CHƯA LÀM
  20. Recording liên tục + quản lý storage
  21. Playback clip sự cố từ dashboard (API + UI)
  22. Đánh giá định lượng MOTA / IDF1 / mAP
  23. Settings page: camera management + ROI editor trên UI

GIAI ĐOẠN 6 — Learned Trajectory Prediction ✅ XONG (2026-06-11)
  24. ✅ modules/calibration.py — CameraCalibration (CARLA projective +
      homography cho CCTV thật) + CalibrationStore.load_from_config()
  25. ✅ collect_trajectory_data.py — thu thập trajectory CARLA (world meters)
      ra CSV, kèm scenario ngẫu nhiên
  26. ✅ modules/trajectory_gru.py — Seq2Seq GRU multi-modal (3 hypotheses)
      + auxiliary intent head (straight/turn_left/turn_right/stop)
  27. ✅ train_trajectory_predictor.py — TrajectoryDataset (K-nearest-neighbor
      features, normalization) + training loop → weights/trajectory_gru.pth
  28. ✅ modules/trajectory_predictor.py — LearnedTrajectoryPredictor +
      EnsembleTrajectoryPredictor (giữ nguyên interface TrajectoryPredictor,
      fallback an toàn về Kalman-only nếu thiếu checkpoint/calibration)
  29. ✅ main.py wiring — CalibrationStore + EnsembleTrajectoryPredictor,
      truyền camera_id/obj_class/all_global_tracks vào update_trajectory()
  → Xem chi tiết kiến trúc: phần "Giai Đoạn 6" bên dưới.
```

---

## Phân Tích Chi Tiết Phần Chưa Hoàn Thành (2026-05-26)

### A. AI — Dataset & Model

#### A1. Vehicle ReID — Sai dataset, sai model

**Vấn đề cụ thể:**
`reid.py` dùng OSNet được pretrain trên **Market-1501** — một dataset chứa toàn người đi bộ
quay từ camera an ninh. Khi áp dụng cho xe ô tô, feature vector được học từ đặc trưng
người (quần áo, dáng đi), không phải từ xe (màu sơn, hình dạng cabin, đèn), dẫn đến
hai xe cùng màu xuyên camera gần như không phân biệt được.

**Giải pháp:**
```
Option A (nhanh): Dùng model Vehicle ReID pretrained sẵn
  - Dataset: VeRi-776 (37.778 ảnh / 776 xe / 20 camera)
  - Model: VehicleNet hoặc ResNet50 fine-tuned trên VeRi-776
  - Nguồn: https://vehiclereid.github.io/VeRi/

Option B (chính xác hơn): Dual-branch ReID
  - 1 model cho person  (OSNet / Market-1501) — giữ nguyên
  - 1 model cho vehicle (VehicleNet / VeRi-776) — thêm mới
  - reid.py phân nhánh theo track['class']
```

**File cần sửa:** `custom_tracking_system/modules/reid.py`

```python
class DualReIDExtractor:
    def __init__(self):
        self.person_model  = self._load_osnet()          # Market-1501
        self.vehicle_model = self._load_vehicle_net()    # VeRi-776

    def extract(self, frame, box, object_class):
        if object_class == 'person':
            return self._run(self.person_model, frame, box)
        else:  # car, bus, truck
            return self._run(self.vehicle_model, frame, box)
```

#### A2. Chưa có đánh giá định lượng (MOTA / IDF1 / mAP)

**Vấn đề cụ thể:**
`ground_truth.py` thu thập actor ID + bounding box từ CARLA nhưng chưa kết nối với
`motmetrics` để tính số liệu. Hiện chỉ đánh giá định tính qua quan sát mắt thường.

**Giải pháp:**
```python
# utils/metrics.py — thêm class MOTEvaluator
import motmetrics as mm

class MOTEvaluator:
    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def update(self, gt_objects, pred_tracks):
        # gt_objects: từ CARLA world.get_actors()
        # pred_tracks: output của GlobalTracker
        gt_ids   = [o.id for o in gt_objects]
        pred_ids = [t['global_id'] for t in pred_tracks]
        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        self.acc.update(gt_ids, pred_ids, distances)

    def report(self):
        mh = mm.metrics.create()
        return mh.compute(self.acc,
            metrics=['mota', 'idf1', 'precision', 'recall'])
```

**Dependencies cần thêm:** `motmetrics>=1.4`

#### A3. Speed calibration — px/s chưa ra km/h thực

**Vấn đề cụ thể:**
`tracker.py` tính speed theo pixel/giây. `incident_detector.py` dùng ngưỡng pixel/giây
để phát hiện overspeed. Chưa có ma trận hiệu chỉnh camera (homography) để chuyển
đổi pixel sang mét, nên ngưỡng hiện tại là ước lượng không có cơ sở.

**Giải pháp (với CARLA):** Lấy vị trí thế giới thực từ CARLA actor để tính m/s → km/h thực.
**Giải pháp (với real CCTV):** Tính homography từ điểm tham chiếu thực trên mặt đường,
dùng `cv2.getPerspectiveTransform` để map pixel → mét.

---

### B. AI — Modules Chưa Implement

#### ~~B1. Tracker — ByteTrack + Kalman filter~~ ✅ Hoàn thành 2026-05-29

**Đã triển khai:**
`ByteTrackWrapper` trong `custom_tracking_system/modules/tracker.py` — wraps `boxmot.ByteTrack`.

```
Vòng 1: match detection confidence cao (>0.25) với tracks hiện có (IoU)
Vòng 2: match detection confidence thấp với tracks chưa khớp (dual-threshold)
Kalman: predict vị trí cho tracks không có detection (bị che khuất)
Hungarian: optimal matching thay greedy
```

**Đã làm:**
- `tracker.py` — `ByteTrackWrapper` class (drop-in thay `SimpleTracker`)
- `main.py` — import + init + `update(detections, frame)` call đã cập nhật
- `requirements.txt` — thêm `boxmot>=10.0`

**Dependencies đã thêm:** `boxmot>=10.0` (thay `filterpy>=1.4`)

#### B2. Tích hợp VideoSource vào pipeline — Real CCTV Footage ✅ Hoàn thành 2026-06-10 (cho main.py)

**Đã làm:** `main.py` giờ hỗ trợ `--source {carla, bridge, rtsp, file, webcam}`.
Với `rtsp`/`file`/`webcam`, `initialize_modules()` dựng `MultiVideoSource` từ
`RTSPVideoSource`/`FileVideoSource`/`WebcamVideoSource` (`modules/video_source.py`),
truyền qua `--rtsp-url`/`--video-path`/`--webcam-index` (lặp lại cho nhiều camera)
+ `--camera-ids`. `frame_source.get_synchronized_frames()` trả về cùng format
`{camera_id: {camera_id, frame, timestamp}}` như `CameraController`, nên pipeline
AI (detector → tracker → reid → global tracker → incident detector) không cần sửa.
Xem `docs/execute.md` mục 4 — Cách 5.

**Còn thiếu (server):** `ai_processor.py` (server, dùng cho frontend) **vẫn**
gọi trực tiếp `CameraController` (CARLA-only). Áp dụng cách làm tương tự
`main.py` để server cũng chạy được với real CCTV footage:

```python
# Thay:
self.camera_controller = CameraController(self.carla_client, self.world, config_path)
frames = self.camera_controller.get_synchronized_frames()

# Bằng (ví dụ RTSP):
from modules.video_source import MultiVideoSource, RTSPVideoSource
self.frame_source = MultiVideoSource()
for cam_id, rtsp_url in real_camera_urls.items():
    self.frame_source.add_source(RTSPVideoSource(rtsp_url, cam_id))
frames = self.frame_source.get_synchronized_frames()
```

**Trạng thái test:** Các source `rtsp`/`file`/`webcam` mới wire xong, **chưa
test với camera/luồng thật** — cần kiểm tra thực tế trước khi dùng cho demo.

**Chưa làm — Calibration pixel↔world:** Để dự đoán quỹ đạo (mục B5/Kalman) có
thể hiển thị lên 1 bản đồ chung, cần thêm bước calibration (homography) chuyển
toạ độ pixel của từng camera sang toạ độ world/map — xem `bao_cao_tien_do.md`
mục 7 item (3).

#### B3. Spatio-temporal reasoning cho cross-camera matching

**Vấn đề cụ thể:**
`global_tracking.py` match xuyên camera chỉ bằng cosine similarity của ReID feature.
Không xem xét: thời gian di chuyển giữa 2 camera có hợp lý không, khoảng cách địa lý
giữa các camera. Kết quả: xe ở CAM_001 có thể bị ghép nhầm với xe ở CAM_003 dù
không có đủ thời gian để đi từ CAM_001 đến CAM_003.

**Giải pháp:**
```python
def is_feasible_transition(cam_a, cam_b, time_a, time_b, config):
    """Kiểm tra xem đối tượng có đủ thời gian di chuyển từ cam_a sang cam_b không."""
    travel_time = config['travel_times'][cam_a][cam_b]  # giây, đo trước
    elapsed = (time_b - time_a).total_seconds()
    return travel_time * 0.5 < elapsed < travel_time * 3.0

# Trong GlobalTracker.match():
# Nếu transition không khả thi → set similarity = 0 (không match)
```

---

### C. Backend + Frontend — Tích Hợp Chưa Hoàn Chỉnh

#### C1. Không có API và UI để xem lại clip sự cố

**Vấn đề cụ thể:**
`evidence_package.py` lưu clip MP4 vào `evidence/<incident_id>/clip_pre.mp4` và
`clip_post.mp4`. Tuy nhiên:
- Server không có endpoint nào để stream file clip này về browser.
- FE không có UI để xem clip (trang `IncidentDetail.jsx` chỉ hiện thông tin text đơn giản).

**Giải pháp:**

Backend — thêm endpoint vào `server/routers/alerts.py`:
```python
@router.get("/{alert_id}/clip")
async def stream_clip(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    clip_path = alert.clip_path  # lưu path khi capture evidence
    return FileResponse(clip_path, media_type="video/mp4")
```

Frontend — thêm `<video>` tag vào `IncidentDetail.jsx`:
```jsx
<video controls className="w-full rounded-lg">
  <source src={`/api/alerts/${id}/clip`} type="video/mp4" />
</video>
```

**Cần làm thêm:** Khi `_save_incident()` trong `ai_processor.py` lưu Alert vào DB,
phải đồng thời lưu `clip_path` vào trường `Alert.clip_path`.

#### C2. Settings page — Camera management và ROI editor chưa có

**Vấn đề cụ thể:**
Roadmap mô tả trang Settings với khả năng thêm/xóa camera và vẽ ROI trực tiếp trên
camera feed. Hiện chưa implement. `App.jsx` không có route `/settings`.

**Những gì cần xây dựng:**
```
Trang Settings (/settings):
  - Danh sách camera: thêm (nhập RTSP URL), xóa, bật/tắt
  - ROI Editor: click lên camera feed để vẽ polygon, đặt tên, chọn alert type
  - Alert thresholds: chỉnh ngưỡng overspeed, loiter time, proximity distance
  - Gọi API: POST/DELETE /api/cameras, POST/PUT/DELETE /api/rois
```

#### C3. IncidentDetail page thiếu dữ liệu incident cụ thể

**Vấn đề cụ thể:**
`IncidentDetail.jsx` nhận `id` từ URL (format `TYPE_gid_timestamp`), tự parse `global_id`
rồi gọi `/api/tracks/{global_id}`. Không có endpoint `/api/alerts/{id}` trả về chi tiết
incident bao gồm: loại sự cố, severity, message, details, timestamp, đường dẫn clip.

**Giải pháp:**
```python
# server/routers/alerts.py — endpoint đã có, cần FE gọi đúng
GET /api/alerts/{id}  →  trả về Alert object đầy đủ
```

`IncidentDetail.jsx` cần sửa để:
1. Nhận `alert_id` (integer) thay vì string tổng hợp.
2. Gọi `api.getAlertById(id)` thay vì `api.getTrackById(globalId)`.
3. Hiển thị severity, message, details, clip player.

#### C4. Recording liên tục chưa implement

**Vấn đề cụ thể:**
`EvidencePackage` chỉ lưu clip khi có sự cố (event-triggered). Không có cơ chế ghi
video liên tục từ tất cả camera theo segment 1 giờ như roadmap đề ra.

**Giải pháp — thêm `ContinuousRecorder`:**
```python
# modules/recorder.py (file mới)
class ContinuousRecorder:
    def __init__(self, output_dir, segment_minutes=60, fps=10):
        self.writers = {}     # {camera_id: cv2.VideoWriter}
        self.segment_start = {}

    def write(self, camera_id, frame, timestamp):
        writer = self._get_writer(camera_id, timestamp)
        writer.write(frame)

    def _get_writer(self, camera_id, timestamp):
        # Tạo file mới mỗi segment_minutes phút
        key = timestamp.strftime('%Y%m%d_%H')
        if self.writers.get(camera_id, {}).get('key') != key:
            self._close(camera_id)
            path = f"{self.output_dir}/{camera_id}_{key}.mp4"
            self.writers[camera_id] = {
                'key': key,
                'writer': cv2.VideoWriter(path, ...)
            }
        return self.writers[camera_id]['writer']
```

---

## Giai Đoạn 6 — Learned Trajectory Prediction (Calibration → Seq2Seq GRU → CARLA Dataset → Intent Head → Kalman Ensemble)

**Mục tiêu:** `TrajectoryPredictor` (Kalman constant-acceleration, pixel-space)
chỉ ngoại suy quán tính, không học được hành vi thật (rẽ ở giao lộ, giảm tốc,
tránh nhau). Giai đoạn này thêm một model học (Seq2Seq GRU đa-mode + intent
classification) train trên trajectory thế giới-thực (mét) thu từ CARLA, rồi
ensemble với Kalman hiện có.

### Files mới / sửa

| File | Loại | Việc chính |
|---|---|---|
| `custom_tracking_system/modules/calibration.py` | mới | `CameraCalibration` (projective CARLA + homography CCTV thật), `CalibrationStore.load_from_config()` |
| `custom_tracking_system/collect_trajectory_data.py` | mới | thu thập trajectory CARLA (world meters) → CSV theo episode |
| `custom_tracking_system/modules/trajectory_gru.py` | mới | `TrajectoryGRU` (Seq2Seq, multi-modal 3 hypotheses, intent head), `trajectory_loss`, feature helpers dùng chung |
| `custom_tracking_system/train_trajectory_predictor.py` | mới | `TrajectoryDataset` + training loop → `weights/trajectory_gru.pth` |
| `custom_tracking_system/modules/trajectory_predictor.py` | sửa | thêm `LearnedTrajectoryPredictor`, `EnsembleTrajectoryPredictor` (giữ nguyên `TrajectoryPredictor`/`KalmanFilter2D`); thêm `from __future__ import annotations` (fix `dict \| None` không chạy được trên Python 3.7) |
| `custom_tracking_system/main.py` | sửa | wire `CalibrationStore` + `EnsembleTrajectoryPredictor`, truyền thêm `camera_id`/`obj_class`/`all_global_tracks` vào `update_trajectory()` |
| `run.bat`, `start.bat` | sửa | PYTHONPATH trỏ tới `carla-0.9.14-py3.7-win-amd64.egg` thay vì `carla_extracted` (không còn tồn tại) |

### Kiến trúc

- **Calibration**: pixel ↔ world (mét) qua intrinsics (từ `fov`+`resolution`)
  + extrinsics (`position`/`rotation` trong `camera_config.yaml`), ray-cast
  xuống mặt phẳng `z=ground_z`. Với CCTV thật, dùng
  `from_point_correspondences()` (homography qua `cv2.findHomography`) — cùng
  interface `pixel_to_world`/`world_to_pixel`.
- **TrajectoryGRU**: encoder GRU trên `T_OBS=8` bước quan sát (input =
  `[x,y,vx,vy]` chuẩn hóa + one-hot class + K=3 neighbor gần nhất); decoder
  GRUCell autoregressive cho 3 mode × 5 horizon (0.5/1/1.5/2/3s); thêm head
  mode-probability + head intent (straight/turn_left/turn_right/stop).
  Loss: winner-takes-all displacement + CE(mode) + CE(intent).
- **EnsembleTrajectoryPredictor**: drop-in thay `TrajectoryPredictor` —
  `update_trajectory(global_id, position, timestamp, camera_id, obj_class,
  all_tracks)` luôn cập nhật Kalman (pixel-space); nếu có calibration cho
  `camera_id` và checkpoint GRU tồn tại, convert sang world-space và cập nhật
  GRU. `predict()` trả về format y hệt cũ (`{'t+0.5s':[x,y],...}`, pixel-space)
  — blend Kalman + GRU theo mode-probability cao nhất; method mới
  `predict_modes()` trả 3 mode + intent (pixel-space) cho IncidentDetector
  trong tương lai.
- **Fallback an toàn**: nếu `weights/trajectory_gru.pth` chưa tồn tại hoặc
  không có calibration cho camera → hành vi y hệt `TrajectoryPredictor` cũ
  (Kalman-only), không lỗi.

### Cách chạy

```bash
# 1. Thu thập dataset (CARLA đang chạy, port 2000)
python collect_trajectory_data.py --episodes 10 --steps-per-episode 600 \
    --num-vehicles 15 --num-pedestrians 8 --scenario-ratio 0.3 \
    --out data/trajectories

# 2. Train
python train_trajectory_predictor.py --data data/trajectories --epochs 50 \
    --out weights/trajectory_gru.pth

# 3. Chạy pipeline — tự động dùng GRU nếu checkpoint tồn tại
run.bat
```

### Đã verify (2026-06-11)

- `calibration.py`: roundtrip `world_to_pixel`/`pixel_to_world` đúng cho
  camera_1 (CARLA convention).
- `trajectory_gru.py`: forward pass + loss tính đúng shape/giá trị.
- `collect_trajectory_data.py`: chạy với CARLA 0.9.14 thật (Town10HD_Opt),
  sinh CSV 1350 dòng / 50 ticks hợp lệ.
- `train_trajectory_predictor.py`: train trên dataset tổng hợp nhỏ, loss
  giảm, checkpoint load lại được.
- `EnsembleTrajectoryPredictor`: test cả 2 nhánh — Kalman-only (chưa có
  checkpoint) và Kalman+GRU (có checkpoint, `predict()`/`predict_modes()`
  hoạt động đúng).

### Việc tiếp theo (chưa làm)

- Thu thập dataset CARLA quy mô lớn hơn (nhiều episode, đa dạng kịch bản) và
  train đủ epoch để GRU thực sự hữu ích (hiện mới train smoke-test).
- Tích hợp `predict_modes()` (3 hypotheses + intent) vào
  `incident_detector._check_predicted_collision` để bắt sớm các tình huống
  rẽ làn / đổi hướng mà Kalman tuyến tính bỏ sót.
- Dùng `CameraCalibration` để chuẩn hóa ngưỡng tốc độ trong
  `incident_detector.py` từ px/s sang km/h thực (xem mục A3 bên dưới).

---

## Giai Đoạn 7 — YOLO11m Fine-tune: Dataset VisDrone + CARLA (mới, 2026-06-12)

**Mục tiêu:** Fine-tune `yolo11m` (đang dùng COCO weights mặc định) trên một
dataset detection 5 lớp VN (`person, car, motorcycle, bus, truck`) kết hợp
ảnh thực (VisDrone, góc camera trên cao giống CCTV) và ảnh synthetic từ
CARLA (đa dạng thời tiết/ánh sáng, có nhãn chính xác tuyệt đối vì lấy từ
ground-truth 3D).

### Files mới

| File | Việc chính |
|---|---|
| `custom_tracking_system/collect_carla_detection_data.py` | Sinh dataset detection từ CARLA: spawn traffic + camera RGB/depth đặt theo kiểu CCTV thật (góc phố, gắn tường tòa nhà, cao 5–8m, nhìn xuống ~10–35°, đặt gần các junction trong bản đồ). Mỗi frame, chiếu 3D bounding box của mọi vehicle/walker lên ảnh qua `CameraCalibration.project_point` (tái dùng `modules/calibration.py`), lọc occlusion bằng depth camera, ghi ra YOLO format `data/carla_det/`. |
| `custom_tracking_system/merge_detection_datasets.py` | Gộp nhiều dataset YOLO cùng 5 lớp VN (VisDrone-VN + CARLA, hoặc thêm nguồn khác sau này) thành 1 dataset duy nhất `data/visdrone_carla_vn/` + yaml. |

### Cách chạy

```bash
# 1. Dataset thực (VisDrone) — không cần CARLA
python prepare_visdrone_dataset.py --out data/visdrone_vn

# 2. Dataset synthetic (CARLA) — cần CarlaUE4.exe đang chạy, env carla-sim (Python 3.7)
python collect_carla_detection_data.py --frames 50           # smoke test trước
python collect_carla_detection_data.py --frames 5000 --num-cameras 6 --out data/carla_det

# 3. Gộp 2 dataset
python merge_detection_datasets.py \
    --datasets visdrone=data/visdrone_vn carla=data/carla_det \
    --out data/visdrone_carla_vn

# 4. Fine-tune (env veri-train / cloud RTX 4060, torch CUDA riêng)
python train_yolo_detector.py --data data/visdrone_carla_vn.yaml --epochs 50 \
    --batch 16 --out weights/yolo11m_vn.pt
```

### Camera placement — "góc phố, gắn tường tòa nhà"

Theo yêu cầu thực tế (camera CCTV thường đặt ở góc phố / gắn trên tường nhà,
nhìn xuống giao lộ): `find_junction_centers()` lấy các junction từ
`world.get_map().get_topology()`, sau đó `camera_transform_for_junction()`
đặt camera cách tâm junction 8–15m (≈ vị trí góc tòa nhà), cao 5–8m, pitch
nghiêng xuống 10–35° hướng về tâm giao lộ — mô phỏng đúng góc nhìn camera an
ninh thật, khác hẳn 3 camera ngang tầm mắt trong `camera_config.yaml` (dùng
cho live tracking demo, không phải data collection).

### Trạng thái

- Code đã viết xong, dùng lại `modules/calibration.py` (đã verify ở Giai
  đoạn 6) và `modules/traffic_generator.py` (đã dùng trong
  `collect_trajectory_data.py`).
- **Chưa chạy thử với CARLA thật** — cần `--frames 50` để verify: vị trí
  camera hợp lý, bbox chiếu đúng, occlusion filter không loại bỏ quá nhiều.
- Sau khi verify, chạy thu thập quy mô lớn hơn (nhiều junction/map khác
  nhau, đa dạng `--num-vehicles`/thời tiết qua `world.set_weather()` nếu
  muốn — hiện chưa tham số hóa weather).
- Training thật (`train_yolo_detector.py`) vẫn đang chờ máy GPU 8GB.

---

## Cấu Trúc Thư Mục Sau Khi Nâng Cấp

```
finalproject/
  custom_tracking_system/
    modules/
      camera_controller.py
      traffic_generator.py
      detector.py
      tracker.py               ← NÂNG CẤP: thêm timestamp + speed history
      reid.py
      global_tracking.py
      trajectory_predictor.py
      alert_system.py
      incident_detector.py     ← MỚI
      evidence_package.py      ← MỚI
      scenario_controller.py   ← MỚI
    config/
      camera_config.yaml
      incident_config.yaml     ← MỚI: ngưỡng phát hiện sự cố
    ...
  server/
    services/
      ai_processor.py          ← NÂNG CẤP: tích hợp IncidentDetector
    ...
  frontend/                    ← MỚI: Web Dashboard React
    src/
      components/
      pages/
      hooks/
      services/
    package.json
  docs/
    upgrade.md                 ← FILE NÀY
    ...
```

---

## Dependencies Cần Thêm

```txt
# AI Pipeline
boxmot>=10.0            # ByteTrack — Kalman + Hungarian (đã thêm 2026-05-29)

# Evidence
opencv-python>=4.8      # đã có, dùng VideoWriter

# Evaluation (giai đoạn sau)
motmetrics>=1.4         # MOTA, IDF1

# Frontend (package.json)
react: ^18
react-router-dom: ^6
tailwindcss: ^3
recharts: ^2            # biểu đồ stats
```

---

*Tài liệu cập nhật: 2026-05-29. Thay đổi: Step 11 (YOLOv8s) hoàn thành; Step 12 (ByteTrack) hoàn thành.*
