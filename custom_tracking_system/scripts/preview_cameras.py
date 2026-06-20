#!/usr/bin/env python3
"""
Preview camera angles in CARLA.

Spawn 40 vehicles, chờ 150 ticks cho xe phân tán, chụp 1 frame từ mỗi camera
và lưu vào data/camera_preview/ để xem trước khi chạy thu thập dữ liệu.

Usage:
    python scripts/preview_cameras.py
    python scripts/preview_cameras.py --config config/camera_config_6cam.yaml
    python scripts/preview_cameras.py --vehicles 0   # không spawn xe (xem địa hình)
"""

import argparse
import queue
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

_FIXED_DELTA = 0.1
_MOTO_HINTS  = frozenset(['harley-davidson', 'kawasaki', 'yamaha', 'vespa',
                           'ninja', 'low_rider', 'zx125', 'ysf'])
_BICYCLE_HINTS = frozenset(['crossbike', 'omafiets', 'diamondback', 'gazelle', 'bh.'])
VEHICLE_MIX  = {'motorcycle': 0.70, 'car': 0.25, 'truck': 0.03, 'bus': 0.02}


def connect(host, port):
    import carla
    client = carla.Client(host, port)
    client.set_timeout(20.0)
    world  = client.get_world()
    s = world.get_settings()
    s.synchronous_mode      = True
    s.fixed_delta_seconds   = _FIXED_DELTA
    world.apply_settings(s)
    print(f"[CARLA] map={world.get_map().name}  sync@{1/_FIXED_DELTA:.0f}Hz")
    return client, world


def classify(bp_id: str) -> str:
    lo = bp_id.lower()
    if any(h in lo for h in _BICYCLE_HINTS): return 'bicycle'
    if any(h in lo for h in _MOTO_HINTS):    return 'motorcycle'
    if 'truck' in lo or 'firetruck' in lo or 'cybertruck' in lo: return 'truck'
    if 'sprinter' in lo or 'fusorosa' in lo or 'futura' in lo:   return 'bus'
    return 'car'


def spawn_vehicles(client, world, n, tm_port, rng):
    import carla
    if n == 0:
        return []

    pools = {k: [] for k in VEHICLE_MIX}
    for bp in world.get_blueprint_library().filter('vehicle.*'):
        cls = classify(bp.id)
        if cls != 'bicycle' and cls in pools:
            pools[cls].append(bp)

    spawn_pts = world.get_map().get_spawn_points()
    rng.shuffle(spawn_pts)
    vehicles = []
    for sp in spawn_pts[:n * 3]:
        if len(vehicles) >= n:
            break
        avail = {c: bps for c, bps in pools.items() if bps}
        total = sum(VEHICLE_MIX.get(c, 0) for c in avail)
        r, acc = rng.random() * total, 0.0
        cls = list(avail)[-1]
        for c in avail:
            acc += VEHICLE_MIX.get(c, 0)
            if r <= acc:
                cls = c
                break
        bp = rng.choice(avail[cls])
        if bp.has_attribute('color'):
            bp.set_attribute('color', rng.choice(bp.get_attribute('color').recommended_values))
        actor = world.try_spawn_actor(bp, sp)
        if actor:
            # set_autopilot(True) dùng default TM — không gọi get_trafficmanager()
            # vì bị crash với version mismatch 0.9.14 server / 0.9.15 client
            actor.set_autopilot(True)
            vehicles.append(actor)
    print(f"[traffic] Spawned {len(vehicles)}/{n} vehicles")
    return vehicles


def load_configs(path: str) -> list:
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return [
        dict(
            camera_id  = v['camera_id'],
            position   = v['position'],
            rotation   = v['rotation'],
            fov        = v['view_angle'],
            w          = v['resolution'][0],
            h          = v['resolution'][1],
        )
        for v in cfg['cameras'].values()
    ]


def spawn_camera(world, cfg: dict):
    import carla
    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', str(cfg['w']))
    bp.set_attribute('image_size_y', str(cfg['h']))
    bp.set_attribute('fov',          str(cfg['fov']))
    tf = carla.Transform(
        carla.Location(*cfg['position']),
        carla.Rotation(pitch=cfg['rotation'][0],
                       yaw  =cfg['rotation'][1],
                       roll =cfg['rotation'][2]))
    q = queue.Queue()
    sensor = world.spawn_actor(bp, tf)
    sensor.listen(q.put)
    return sensor, q


def draw_info(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Overlay camera info lên ảnh để dễ nhận biết."""
    out = img.copy()
    pos = cfg['position']
    rot = cfg['rotation']
    lines = [
        cfg['camera_id'],
        f"pos: ({pos[0]:.1f}, {pos[1]:.1f}, z={pos[2]})",
        f"pitch={rot[0]}  yaw={rot[1]}  fov={cfg['fov']}",
    ]
    y0 = 28
    for i, txt in enumerate(lines):
        y = y0 + i * 30
        cv2.rectangle(out, (8, y - 22), (8 + len(txt) * 13, y + 6), (0, 0, 0), -1)
        cv2.putText(out, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host',     default='localhost')
    ap.add_argument('--port',     type=int, default=2000)
    ap.add_argument('--tm-port',  type=int, default=8000)
    ap.add_argument('--config',   default='config/camera_config.yaml')
    ap.add_argument('--vehicles', type=int, default=40,
                    help='Số xe spawn để xem preview (0 = chỉ xem địa hình)')
    ap.add_argument('--warmup',   type=int, default=150,
                    help='Ticks chờ xe phân tán trước khi chụp')
    ap.add_argument('--out',      default='data/camera_preview')
    ap.add_argument('--seed',     type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = project_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    client, world = connect(args.host, args.port)
    vehicles = spawn_vehicles(client, world, args.vehicles, args.tm_port, rng)

    cam_cfgs = load_configs(str(project_root / args.config))
    print(f"[config] {len(cam_cfgs)} cameras from {args.config}")

    sensors, queues = [], []
    try:
        for cfg in cam_cfgs:
            s, q = spawn_camera(world, cfg)
            sensors.append(s)
            queues.append(q)
            print(f"  Spawned {cfg['camera_id']}: pos={cfg['position']}  "
                  f"rot={cfg['rotation']}  fov={cfg['fov']}")

        # Warmup
        print(f"\n[warmup] Running {args.warmup} ticks ...")
        for i in range(args.warmup):
            world.tick()
            if (i + 1) % 50 == 0:
                print(f"  tick {i+1}/{args.warmup}")

        # ── Capture ──────────────────────────────────────────────────────────
        # Bước 1: drain toàn bộ frames tich luy tu warmup (non-blocking)
        print("\n[capture] Draining warmup frames ...")
        for q in queues:
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # Bước 2: 1 tick moi, CARLA sync dam bao tat ca sensor callbacks
        # da fire truoc khi tick() return. Nhung callback chay trong thread
        # rieng nen can blocking get() sau do.
        world.tick()
        fid = world.get_snapshot().frame
        print(f"[capture] frame_id={fid}  collecting ...")

        # Bước 3: blocking get cho tung camera
        collected = {}
        for cfg_c, q_c in zip(cam_cfgs, queues):
            cam_id = cfg_c['camera_id']
            img_raw = None
            deadline = time.time() + 4.0
            while time.time() < deadline:
                try:
                    item = q_c.get(timeout=1.0)
                    if item.frame == fid:
                        img_raw = item
                        break
                    # frame cu hon fid -> drain tiep
                except queue.Empty:
                    break

            if img_raw is None:
                print(f"  {cam_id}: TIMEOUT")
                continue

            arr = np.frombuffer(img_raw.raw_data, dtype=np.uint8)
            img = arr.reshape((cfg_c['h'], cfg_c['w'], 4))[:, :, :3]
            collected[cam_id] = (cfg_c, draw_info(img, cfg_c))
            print(f"  {cam_id}: OK")

        print(f"[capture] {len(collected)}/{len(cam_cfgs)} cameras collected")

        saved = []
        for cam_id, (cfg, rgb) in collected.items():
            out_path = out_dir / f"{cam_id}.jpg"
            cv2.imwrite(str(out_path), rgb)
            saved.append(str(out_path))
            print(f"  {cam_id}: saved -> {out_path}")

        # Tạo grid ảnh (2 cột) để so sánh tất cả cùng lúc
        if saved:
            imgs = [cv2.imread(p) for p in saved]
            # Resize về cùng size
            h_tgt, w_tgt = 360, 640
            thumbs = [cv2.resize(im, (w_tgt, h_tgt)) for im in imgs]
            # Pad tới số chẵn
            if len(thumbs) % 2 == 1:
                thumbs.append(np.zeros((h_tgt, w_tgt, 3), dtype=np.uint8))
            rows = []
            for i in range(0, len(thumbs), 2):
                rows.append(np.hstack(thumbs[i:i+2]))
            grid = np.vstack(rows)
            grid_path = out_dir / "grid_all_cameras.jpg"
            cv2.imwrite(str(grid_path), grid)
            print(f"\n[grid] Saved combined preview -> {grid_path}")

    finally:
        print("\n[cleanup] Removing cameras and vehicles ...")
        for s in sensors:
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass
        for v in vehicles:
            try:
                v.destroy()
            except Exception:
                pass
        # Tắt sync mode
        try:
            s2 = world.get_settings()
            s2.synchronous_mode = False
            world.apply_settings(s2)
        except Exception:
            pass

    print(f"\nDone. Preview images at: {out_dir}/")
    print("  Per-camera: <camera_id>.jpg")
    print("  All cameras: grid_all_cameras.jpg")


if __name__ == '__main__':
    main()
