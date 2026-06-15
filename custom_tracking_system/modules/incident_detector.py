"""
Incident Detector Module
Phát hiện các sự cố giao thông real-time từ dữ liệu tracking.
"""

import numpy as np
from collections import defaultdict, deque
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class IncidentDetector:
    """
    Phân tích dữ liệu tracking để phát hiện các loại sự cố:
      1.  SUDDEN_STOP         — xe dừng đột ngột
      2.  SUDDEN_ACCEL        — xe tăng tốc đột ngột sau khi dừng (dấu hiệu bỏ trốn)
      3.  OVERSPEED           — xe vượt tốc độ cho phép
      4.  VEHICLE_PROXIMITY   — hai xe tiến lại gần nhau nguy hiểm
      5.  PEDESTRIAN_DANGER   — xe áp sát người đi bộ với tốc độ cao
      6.  STOPPED_ON_ROAD     — xe dừng giữa đường quá lâu
      7.  LOITERING           — người đứng ở một khu vực quá lâu
      8.  CROWD_DENSITY       — mật độ người trong vùng quá cao
      9.  WRONG_WAY           — đối tượng di chuyển ngược chiều cấu hình (cấu hình incident_detection.wrong_way)
      10. CAMERA_TRANSITION   — đối tượng chuyển từ camera này sang camera khác
      11. RED_LIGHT_VIOLATION — xe vượt đèn đỏ tại vùng giao lộ (cấu hình incident_detection.red_light)
      12. PREDICTED_COLLISION / PREDICTED_ROI_ENTRY — cảnh báo chủ động (xem dưới)
    """

    def __init__(self, config: dict = None):
        cfg = config or {}

        # --- Ngưỡng cấu hình ---
        self.speed_limit        = cfg.get('speed_limit', 120)       # px/s
        self.sudden_stop_ratio  = cfg.get('sudden_stop_ratio', 0.3) # còn 30% tốc độ cũ
        self.sudden_accel_ratio = cfg.get('sudden_accel_ratio', 2.5)# tăng 250%
        self.min_moving_speed   = cfg.get('min_moving_speed', 8)    # px/s — coi là đang chạy
        self.proximity_px       = cfg.get('proximity_px', 100)      # pixels
        self.ped_proximity_px   = cfg.get('ped_proximity_px', 80)   # pixels
        self.stop_frames        = cfg.get('stop_frames', 50)        # ~5s @ 10fps
        self.loiter_frames      = cfg.get('loiter_frames', 300)     # ~30s
        self.loiter_radius      = cfg.get('loiter_radius', 60)      # pixels
        self.crowd_threshold    = cfg.get('crowd_threshold', 8)     # số người
        self.crowd_area_px      = cfg.get('crowd_area_px', 200)     # px radius
        self.cooldown_s         = cfg.get('cooldown_s', 4)          # giây chống duplicate

        # --- Wrong-way driving ---
        wrong_way_cfg = cfg.get('wrong_way', {}) or {}
        self.wrong_way_lanes        = wrong_way_cfg.get('lanes', {}) or {}
        self.wrong_way_min_speed    = wrong_way_cfg.get('min_speed', 15)      # px/s
        self.wrong_way_min_disp     = wrong_way_cfg.get('min_displacement_px', 15)
        angle_threshold_deg         = wrong_way_cfg.get('angle_threshold_deg', 100)
        self.wrong_way_cos_threshold = float(np.cos(np.radians(angle_threshold_deg)))
        self.wrong_way_window       = wrong_way_cfg.get('window', 8)
        self.wrong_way_frames       = wrong_way_cfg.get('min_frames', 10)

        # --- Red-light violation ---
        red_light_cfg = cfg.get('red_light', {}) or {}
        self.red_light_min_speed = red_light_cfg.get('min_speed', 10)  # px/s

        # --- Loai incident bi tat (qua nhieu, khong can cho demo) ---
        self.disabled_types = set(cfg.get('disabled_types', []) or [])

        # --- State per object ---
        # Lịch sử tốc độ: {global_id: deque[(timestamp, speed)]}
        self.speed_history   = defaultdict(lambda: deque(maxlen=20))
        self.stop_counter    = defaultdict(int)
        self.loiter_anchor   = {}
        self.loiter_counter  = defaultdict(int)
        self.wrong_way_counter = defaultdict(int)

        # Cross-camera: {global_id: camera_id} — lần xuất hiện trước
        self.last_camera     = {}

        # Chống alert trùng: {(type, gid): last_alert_time}
        self._cooldown_map   = {}

        logger.info("IncidentDetector initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, global_tracks: list, camera_id: str,
               predictions: dict = None, rois: dict = None,
               traffic_light_state: str = None) -> list:
        """
        Gọi mỗi frame sau khi có global_tracks từ GlobalTracker.

        Args:
            global_tracks: list dicts — output của GlobalTracker.process_camera_tracks()
                           Mỗi dict cần: global_id, box, class, speeds (list px/s),
                           positions (list [x,y] — lịch sử vị trí, dùng cho WRONG_WAY)
            camera_id:    camera đang xử lý
            predictions:  dict {global_id: {'t+0.5s': [x,y], ...}} từ TrajectoryPredictor
                          — dùng để kích hoạt proactive checks
            rois:         dict {camera_id: [{'name': str, 'polygon': [...], 'alert_types': [...]}]}
                          — dùng cho predicted ROI entry check + RED_LIGHT_VIOLATION
            traffic_light_state: trạng thái đèn giao thông gần camera ("Red"/"Yellow"/
                          "Green"/None) — dùng cho RED_LIGHT_VIOLATION

        Returns:
            list of incident dicts — có thể rỗng
        """
        now = datetime.now()
        incidents = []

        # Cập nhật speed history từ tracker speeds
        for track in global_tracks:
            gid = track['global_id']
            speeds = track.get('speeds', [])
            if speeds:
                self.speed_history[gid].append((now, speeds[-1]))

        # Chạy từng detector phản ứng (reactive)
        incidents += self._check_sudden_stop(global_tracks, camera_id, now)
        incidents += self._check_sudden_accel(global_tracks, camera_id, now)
        incidents += self._check_overspeed(global_tracks, camera_id, now)
        incidents += self._check_vehicle_proximity(global_tracks, camera_id, now)
        incidents += self._check_pedestrian_danger(global_tracks, camera_id, now)
        incidents += self._check_stopped_on_road(global_tracks, camera_id, now)
        incidents += self._check_loitering(global_tracks, camera_id, now)
        incidents += self._check_crowd(global_tracks, camera_id, now)
        incidents += self._check_camera_transition(global_tracks, camera_id, now)
        incidents += self._check_wrong_way(global_tracks, camera_id, now)
        incidents += self._check_red_light_violation(
            global_tracks, camera_id, now, traffic_light_state, rois)

        # Proactive checks dùng predicted positions (chỉ khi có predictions)
        if predictions:
            incidents += self._check_predicted_collision(
                global_tracks, predictions, camera_id, now)
            if rois:
                incidents += self._check_predicted_roi_entry(
                    global_tracks, predictions, rois, camera_id, now)

        # Lọc các loại incident bị tắt (cấu hình incident_detection.disabled_types)
        if self.disabled_types:
            incidents = [inc for inc in incidents if inc['type'] not in self.disabled_types]

        # Lọc duplicate dựa trên cooldown
        incidents = self._deduplicate(incidents, now)

        if incidents:
            for inc in incidents:
                logger.warning(
                    f"[INCIDENT] {inc['type']} | severity={inc['severity']} "
                    f"| obj={inc['global_id']} | cam={inc['camera_id']} "
                    f"| {inc['message']}"
                )

        return incidents

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_sudden_stop(self, tracks, camera_id, now):
        incidents = []
        for track in tracks:
            gid = track['global_id']
            history = list(self.speed_history[gid])
            if len(history) < 5:
                continue

            speeds = [h[1] for h in history]
            speed_before = np.mean(speeds[:3])
            speed_after  = speeds[-1]

            if (speed_before > self.min_moving_speed and
                    speed_after < speed_before * self.sudden_stop_ratio):
                incidents.append(self._make(
                    type='SUDDEN_STOP', severity='WARNING',
                    gid=gid, cam=camera_id,
                    msg=f"Xe #{gid} phanh dừng đột ngột giữa đường",
                    details={'speed_before': round(speed_before, 1),
                             'speed_after': round(speed_after, 1)}
                ))
        return incidents

    def _check_sudden_accel(self, tracks, camera_id, now):
        """Tăng tốc đột ngột sau khi đã gần dừng — dấu hiệu bỏ trốn."""
        incidents = []
        for track in tracks:
            gid = track['global_id']
            history = list(self.speed_history[gid])
            if len(history) < 6:
                continue

            speeds = [h[1] for h in history]
            # Trước đó đang chậm, bây giờ bất ngờ nhanh
            speed_prev = np.mean(speeds[-4:-2])
            speed_now  = speeds[-1]

            if (speed_prev < self.min_moving_speed and
                    speed_now > speed_prev * self.sudden_accel_ratio and
                    speed_now > self.min_moving_speed * 2):
                incidents.append(self._make(
                    type='SUDDEN_ACCEL', severity='CRITICAL',
                    gid=gid, cam=camera_id,
                    msg=f"Xe #{gid} bất ngờ tăng tốc rời khỏi hiện trường, có dấu hiệu bỏ trốn",
                    details={'speed_prev': round(speed_prev, 1),
                             'speed_now': round(speed_now, 1)}
                ))
        return incidents

    def _check_overspeed(self, tracks, camera_id, now):
        incidents = []
        for track in tracks:
            if track['class'] not in ('car', 'truck', 'bus'):
                continue
            gid = track['global_id']
            speed = self._current_speed(gid)
            if speed and speed > self.speed_limit:
                ratio = speed / self.speed_limit
                incidents.append(self._make(
                    type='OVERSPEED', severity='WARNING',
                    gid=gid, cam=camera_id,
                    msg=f"Xe #{gid} đang chạy quá tốc độ cho phép (nhanh hơn khoảng {ratio:.1f} lần mức quy định)",
                    details={'speed': round(speed, 1), 'limit': self.speed_limit}
                ))
        return incidents

    def _check_vehicle_proximity(self, tracks, camera_id, now):
        """Hai xe tiến lại gần nhau nguy hiểm + ít nhất 1 xe đang chạy nhanh."""
        incidents = []
        vehicles = [t for t in tracks if t['class'] in ('car', 'truck', 'bus')]
        for i, v1 in enumerate(vehicles):
            for v2 in vehicles[i+1:]:
                dist = self._box_dist(v1['box'], v2['box'])
                if dist < self.proximity_px:
                    s1 = self._current_speed(v1['global_id']) or 0
                    s2 = self._current_speed(v2['global_id']) or 0
                    if max(s1, s2) > self.min_moving_speed:
                        incidents.append(self._make(
                            type='VEHICLE_PROXIMITY', severity='CRITICAL',
                            gid=v1['global_id'], cam=camera_id,
                            msg=(f"Xe #{v1['global_id']} và xe #{v2['global_id']} "
                                 f"đang tiến rất gần nhau với tốc độ cao, nguy cơ va chạm"),
                            details={
                                'other_id': v2['global_id'],
                                'distance_px': round(dist, 1),
                                'speed_1': round(s1, 1),
                                'speed_2': round(s2, 1),
                            }
                        ))
        return incidents

    def _check_pedestrian_danger(self, tracks, camera_id, now):
        """Xe áp sát người đi bộ với tốc độ cao."""
        incidents = []
        vehicles    = [t for t in tracks if t['class'] in ('car', 'truck', 'bus')]
        pedestrians = [t for t in tracks if t['class'] == 'person']
        for v in vehicles:
            speed = self._current_speed(v['global_id']) or 0
            if speed < self.min_moving_speed:
                continue
            for p in pedestrians:
                dist = self._box_dist(v['box'], p['box'])
                if dist < self.ped_proximity_px:
                    incidents.append(self._make(
                        type='PEDESTRIAN_DANGER', severity='CRITICAL',
                        gid=v['global_id'], cam=camera_id,
                        msg=(f"Xe #{v['global_id']} đang áp sát người đi bộ #{p['global_id']} "
                             f"với tốc độ cao, nguy cơ va chạm cao"),
                        details={
                            'pedestrian_id': p['global_id'],
                            'distance_px': round(dist, 1),
                            'vehicle_speed': round(speed, 1),
                        }
                    ))
        return incidents

    def _check_stopped_on_road(self, tracks, camera_id, now):
        incidents = []
        for track in tracks:
            if track['class'] not in ('car', 'truck', 'bus'):
                continue
            gid = track['global_id']
            speed = self._current_speed(gid) or 0

            if speed < 1.0:
                self.stop_counter[gid] += 1
            else:
                self.stop_counter[gid] = 0

            # Chỉ alert đúng khi vừa đạt ngưỡng (không lặp)
            if self.stop_counter[gid] == self.stop_frames:
                incidents.append(self._make(
                    type='STOPPED_VEHICLE', severity='WARNING',
                    gid=gid, cam=camera_id,
                    msg=f"Xe #{gid} đang dừng bất thường giữa đường trong thời gian dài",
                    details={'stop_frames': self.stop_frames}
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
                dist = np.linalg.norm(np.array(pos) - np.array(self.loiter_anchor[gid]))
                if dist < self.loiter_radius:
                    self.loiter_counter[gid] += 1
                else:
                    self.loiter_anchor[gid] = pos
                    self.loiter_counter[gid] = 0

            if self.loiter_counter[gid] == self.loiter_frames:
                incidents.append(self._make(
                    type='LOITERING', severity='WARNING',
                    gid=gid, cam=camera_id,
                    msg=f"Người #{gid} đứng yên tại một khu vực trong thời gian dài, có thể đáng nghi",
                    details={'duration_frames': self.loiter_frames}
                ))
        return incidents

    def _check_crowd(self, tracks, camera_id, now):
        """Phát hiện vùng tập trung đông người."""
        incidents = []
        persons = [t for t in tracks if t['class'] == 'person']
        if len(persons) < self.crowd_threshold:
            return incidents

        # Kiểm tra từng người có >= crowd_threshold người trong vòng crowd_area_px
        for p in persons:
            center = np.array(self._box_center(p['box']))
            nearby = sum(
                1 for other in persons
                if other['global_id'] != p['global_id']
                and np.linalg.norm(center - np.array(self._box_center(other['box']))) < self.crowd_area_px
            )
            if nearby + 1 >= self.crowd_threshold:
                incidents.append(self._make(
                    type='CROWD_DENSITY', severity='WARNING',
                    gid=p['global_id'], cam=camera_id,
                    msg=f"Phát hiện đông người tập trung tại khu vực camera {camera_id} ({nearby + 1} người)",
                    details={'person_count': nearby + 1, 'area_px': self.crowd_area_px}
                ))
                break  # 1 alert / camera / frame là đủ
        return incidents

    def _check_camera_transition(self, tracks, camera_id, now):
        """Ghi nhận khi đối tượng chuyển sang camera mới."""
        incidents = []
        for track in tracks:
            gid = track['global_id']
            prev_cam = self.last_camera.get(gid)
            if prev_cam is not None and prev_cam != camera_id:
                incidents.append(self._make(
                    type='CAMERA_TRANSITION', severity='INFO',
                    gid=gid, cam=camera_id,
                    msg=f"Đối tượng #{gid} di chuyển từ khu vực {prev_cam} sang khu vực {camera_id}",
                    details={'from_camera': prev_cam, 'to_camera': camera_id}
                ))
            self.last_camera[gid] = camera_id
        return incidents

    def _check_wrong_way(self, tracks, camera_id, now):
        """Đối tượng di chuyển ngược hướng cho phép của camera (cấu hình
        incident_detection.wrong_way.lanes). Yêu cầu di chuyển đủ xa
        (min_displacement_px) + đủ nhanh (min_speed) ngược hướng cho phép
        trong >= min_frames frame liên tiếp để chống nhiễu."""
        incidents = []
        lane = self.wrong_way_lanes.get(camera_id)
        if not lane:
            return incidents

        allowed = np.array(lane.get('direction', [0, 0]), dtype=float)
        norm = np.linalg.norm(allowed)
        if norm == 0:
            return incidents
        allowed = allowed / norm

        for track in tracks:
            if track['class'] not in ('car', 'truck', 'bus', 'motorcycle'):
                continue
            gid = track['global_id']
            positions = track.get('positions', [])

            if len(positions) < self.wrong_way_window:
                self.wrong_way_counter[gid] = 0
                continue

            p0 = np.array(positions[-self.wrong_way_window])
            p1 = np.array(positions[-1])
            disp = p1 - p0
            dist = float(np.linalg.norm(disp))
            speed = self._current_speed(gid) or 0

            if dist < self.wrong_way_min_disp or speed < self.wrong_way_min_speed:
                self.wrong_way_counter[gid] = 0
                continue

            direction = disp / dist
            cos_angle = float(np.dot(direction, allowed))

            if cos_angle < self.wrong_way_cos_threshold:
                self.wrong_way_counter[gid] += 1
            else:
                self.wrong_way_counter[gid] = 0

            if self.wrong_way_counter[gid] == self.wrong_way_frames:
                incidents.append(self._make(
                    type='WRONG_WAY', severity='CRITICAL',
                    gid=gid, cam=camera_id,
                    msg=f"Xe #{gid} đang đi ngược chiều quy định tại {camera_id}",
                    details={
                        'direction':         direction.tolist(),
                        'allowed_direction': allowed.tolist(),
                        'speed':             round(speed, 1),
                    }
                ))
        return incidents

    def _check_red_light_violation(self, tracks, camera_id, now, light_state, rois):
        """Xe vượt đèn đỏ: traffic light gần camera đang Red VA xe (đang chạy,
        speed > red_light.min_speed) nằm trong 1 ROI có alert_types chứa
        'red_light' (vùng giao lộ/stop-line, cấu hình trong camera_config.yaml)."""
        incidents = []
        if light_state != 'Red':
            return incidents

        cam_rois = (rois or {}).get(camera_id, [])
        zones = [r['polygon'] for r in cam_rois
                 if 'red_light' in (r.get('alert_types') or []) and len(r.get('polygon', [])) >= 3]
        if not zones:
            return incidents

        for track in tracks:
            if track['class'] not in ('car', 'truck', 'bus', 'motorcycle'):
                continue
            gid = track['global_id']
            speed = self._current_speed(gid) or 0
            if speed < self.red_light_min_speed:
                continue

            center = self._box_center(track['box'])
            for polygon in zones:
                if self._point_in_polygon(center, polygon):
                    incidents.append(self._make(
                        type='RED_LIGHT_VIOLATION', severity='CRITICAL',
                        gid=gid, cam=camera_id,
                        msg=f"Xe #{gid} vượt đèn đỏ tại {camera_id} với tốc độ cao",
                        details={'speed': round(speed, 1), 'light_state': light_state}
                    ))
                    break
        return incidents

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_speed(self, gid: int):
        history = list(self.speed_history[gid])
        if not history:
            return None
        recent = [h[1] for h in history[-5:] if h[1] is not None]
        if not recent:
            return None
        weights = np.exp(np.linspace(-1, 0, len(recent)))
        return float(np.average(recent, weights=weights))

    def _box_center(self, box):
        x1, y1, x2, y2 = box
        return [(x1 + x2) / 2, (y1 + y2) / 2]

    def _box_dist(self, box1, box2):
        c1 = np.array(self._box_center(box1))
        c2 = np.array(self._box_center(box2))
        return float(np.linalg.norm(c1 - c2))

    # Nhãn tiếng Việt cho các loại đối tượng, dùng trong message tự nhiên
    CLASS_VN = {
        'person':     'người đi bộ',
        'car':        'ô tô',
        'motorcycle': 'xe máy',
        'bus':        'xe buýt',
        'truck':      'xe tải',
    }

    def _label(self, cls: str) -> str:
        return self.CLASS_VN.get(cls, 'đối tượng')

    def _horizon_text(self, horizon: str) -> str:
        """Chuyển 't+0.5s' -> '0.5 giây tới' cho message dễ đọc."""
        try:
            seconds = horizon.lstrip('t+').rstrip('s')
            return f"{seconds} giây tới"
        except Exception:
            return horizon

    def _make(self, type, severity, gid, cam, msg, details=None):
        return {
            'type':      type,
            'severity':  severity,
            'global_id': gid,
            'camera_id': cam,
            'timestamp': datetime.now(),
            'message':   msg,
            'details':   details or {},
        }

    def _deduplicate(self, incidents, now):
        out = []
        for inc in incidents:
            key = (inc['type'], inc['global_id'])
            last = self._cooldown_map.get(key)
            if last is None or (now - last).total_seconds() > self.cooldown_s:
                self._cooldown_map[key] = now
                out.append(inc)
        return out

    # ------------------------------------------------------------------
    # Proactive checks (dùng predicted positions từ TrajectoryPredictor)
    # ------------------------------------------------------------------

    def _check_predicted_collision(self, tracks, predictions, camera_id, now):
        """
        Cảnh báo trước khi va chạm xảy ra dựa trên quỹ đạo dự đoán.
        predictions: {global_id: {'t+0.5s': [x,y], ...}}
        """
        incidents = []
        vehicles    = [t for t in tracks
                       if t['class'] in ('car', 'truck', 'bus', 'motorcycle')]
        pedestrians = [t for t in tracks if t['class'] == 'person']

        for v in vehicles:
            gid = v['global_id']
            v_preds = predictions.get(gid, {})
            if not v_preds:
                continue
            speed = self._current_speed(gid) or 0
            if speed < self.min_moving_speed:
                continue  # xe đứng yên không nguy hiểm

            for ped in pedestrians:
                ped_center = np.array(self._box_center(ped['box']))
                for horizon, pred_pos in v_preds.items():
                    dist = float(np.linalg.norm(np.array(pred_pos) - ped_center))
                    # Ngưỡng rộng hơn cho dự đoán (có sai số Kalman)
                    if dist < self.ped_proximity_px * 1.5:
                        incidents.append(self._make(
                            type='PREDICTED_COLLISION',
                            severity='CRITICAL',
                            gid=gid, cam=camera_id,
                            msg=(f"Cảnh báo: Xe #{gid} có thể va chạm với người đi bộ "
                                 f"#{ped['global_id']} trong {self._horizon_text(horizon)}"),
                            details={
                                'horizon':           horizon,
                                'predicted_pos':     pred_pos,
                                'pedestrian_pos':    ped_center.tolist(),
                                'predicted_distance': round(dist, 1),
                                'vehicle_speed':     round(speed, 1),
                            }
                        ))
                        break  # 1 alert / (xe, người) / frame đủ rồi
        return incidents

    def _check_predicted_roi_entry(self, tracks, predictions, rois, camera_id, now):
        """
        Cảnh báo trước khi đối tượng vào vùng ROI nguy hiểm.
        """
        incidents = []
        cam_rois = rois.get(camera_id, [])
        if not cam_rois:
            return incidents

        for track in tracks:
            gid   = track['global_id']
            preds = predictions.get(gid, {})
            if not preds:
                continue
            for roi in cam_rois:
                polygon = roi.get('polygon', [])
                if len(polygon) < 3:
                    continue
                for horizon, pred_pos in preds.items():
                    if self._point_in_polygon(pred_pos, polygon):
                        incidents.append(self._make(
                            type='PREDICTED_ROI_ENTRY',
                            severity='WARNING',
                            gid=gid, cam=camera_id,
                            msg=(f"Dự đoán {self._label(track['class'])} #{gid} sẽ tiến vào "
                                 f"khu vực giao lộ trong {self._horizon_text(horizon)}"),
                            details={
                                'horizon':      horizon,
                                'roi_name':     roi.get('name', ''),
                                'pred_pos':     pred_pos,
                            }
                        ))
                        break  # 1 alert / (object, roi) / frame
        return incidents

    @staticmethod
    def _point_in_polygon(point, polygon) -> bool:
        """Ray-casting algorithm."""
        x, y = point
        inside = False
        j = len(polygon) - 1
        for i, (xi, yi) in enumerate(polygon):
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y) and
                    x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    # ------------------------------------------------------------------

    def reset_object(self, gid: int):
        """Xoá trạng thái của 1 object (khi object mất khỏi hệ thống)."""
        self.speed_history.pop(gid, None)
        self.stop_counter.pop(gid, None)
        self.loiter_anchor.pop(gid, None)
        self.loiter_counter.pop(gid, None)
        self.last_camera.pop(gid, None)
        self.wrong_way_counter.pop(gid, None)
