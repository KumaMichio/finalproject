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
    Phân tích dữ liệu tracking để phát hiện 10 loại sự cố:
      1.  SUDDEN_STOP        — xe dừng đột ngột
      2.  SUDDEN_ACCEL       — xe tăng tốc đột ngột sau khi dừng (dấu hiệu bỏ trốn)
      3.  OVERSPEED          — xe vượt tốc độ cho phép
      4.  VEHICLE_PROXIMITY  — hai xe tiến lại gần nhau nguy hiểm
      5.  PEDESTRIAN_DANGER  — xe áp sát người đi bộ với tốc độ cao
      6.  STOPPED_ON_ROAD    — xe dừng giữa đường quá lâu
      7.  LOITERING          — người đứng ở một khu vực quá lâu
      8.  CROWD_DENSITY      — mật độ người trong vùng quá cao
      9.  WRONG_WAY          — đối tượng di chuyển ngược chiều (cần cấu hình)
      10. CAMERA_TRANSITION  — đối tượng chuyển từ camera này sang camera khác
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

        # --- State per object ---
        # Lịch sử tốc độ: {global_id: deque[(timestamp, speed)]}
        self.speed_history   = defaultdict(lambda: deque(maxlen=20))
        self.stop_counter    = defaultdict(int)
        self.loiter_anchor   = {}
        self.loiter_counter  = defaultdict(int)

        # Cross-camera: {global_id: camera_id} — lần xuất hiện trước
        self.last_camera     = {}

        # Chống alert trùng: {(type, gid): last_alert_time}
        self._cooldown_map   = {}

        logger.info("IncidentDetector initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, global_tracks: list, camera_id: str) -> list:
        """
        Gọi mỗi frame sau khi có global_tracks từ GlobalTracker.

        Args:
            global_tracks: list dicts — output của GlobalTracker.process_camera_tracks()
                           Mỗi dict cần: global_id, box, class, speeds (list px/s)
            camera_id: camera đang xử lý

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

        # Chạy từng detector
        incidents += self._check_sudden_stop(global_tracks, camera_id, now)
        incidents += self._check_sudden_accel(global_tracks, camera_id, now)
        incidents += self._check_overspeed(global_tracks, camera_id, now)
        incidents += self._check_vehicle_proximity(global_tracks, camera_id, now)
        incidents += self._check_pedestrian_danger(global_tracks, camera_id, now)
        incidents += self._check_stopped_on_road(global_tracks, camera_id, now)
        incidents += self._check_loitering(global_tracks, camera_id, now)
        incidents += self._check_crowd(global_tracks, camera_id, now)
        incidents += self._check_camera_transition(global_tracks, camera_id, now)

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
                    msg=f"Xe #{gid} dừng đột ngột: {speed_before:.0f} → {speed_after:.0f} px/s",
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
                    msg=f"Xe #{gid} tăng tốc đột ngột: {speed_prev:.0f} → {speed_now:.0f} px/s (có thể bỏ trốn)",
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
                incidents.append(self._make(
                    type='OVERSPEED', severity='WARNING',
                    gid=gid, cam=camera_id,
                    msg=f"Xe #{gid} vượt tốc độ: {speed:.0f} px/s (giới hạn {self.speed_limit})",
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
                            msg=(f"Xe #{v1['global_id']} và #{v2['global_id']} "
                                 f"tiến gần nhau nguy hiểm (dist={dist:.0f}px)"),
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
                        msg=(f"Xe #{v['global_id']} áp sát người #{p['global_id']} "
                             f"(dist={dist:.0f}px, speed={speed:.0f}px/s)"),
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
                    msg=f"Xe #{gid} dừng giữa đường hơn {self.stop_frames} frame",
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
                    msg=f"Người #{gid} đứng ở cùng khu vực hơn {self.loiter_frames} frame",
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
                    msg=f"Mật độ đông người tại {camera_id}: {nearby + 1} người trong vùng",
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
                    msg=f"Đối tượng #{gid} chuyển từ {prev_cam} → {camera_id}",
                    details={'from_camera': prev_cam, 'to_camera': camera_id}
                ))
            self.last_camera[gid] = camera_id
        return incidents

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_speed(self, gid: int):
        history = list(self.speed_history[gid])
        return history[-1][1] if history else None

    def _box_center(self, box):
        x1, y1, x2, y2 = box
        return [(x1 + x2) / 2, (y1 + y2) / 2]

    def _box_dist(self, box1, box2):
        c1 = np.array(self._box_center(box1))
        c2 = np.array(self._box_center(box2))
        return float(np.linalg.norm(c1 - c2))

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

    def reset_object(self, gid: int):
        """Xoá trạng thái của 1 object (khi object mất khỏi hệ thống)."""
        self.speed_history.pop(gid, None)
        self.stop_counter.pop(gid, None)
        self.loiter_anchor.pop(gid, None)
        self.loiter_counter.pop(gid, None)
        self.last_camera.pop(gid, None)
