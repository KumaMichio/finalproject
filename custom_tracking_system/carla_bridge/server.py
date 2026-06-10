#!/usr/bin/env python3
"""
CARLA Bridge Server (Python 3.7)

CARLA's Windows PythonAPI (0.9.x) only ships a cp37 build, while the AI
pipeline (torch CUDA for newer GPUs, modern ultralytics/boxmot/torchreid)
needs Python 3.10+. This script bridges the two:

  - Connects to CARLA, enables synchronous mode
  - Spawns cameras (CameraController) and traffic (TrafficGenerator)
  - Each tick, sends the synchronized camera frames to the AI process
    (running main.py --source bridge under Python 3.10) over a local
    TCP socket.

Run with the Python 3.7 venv that has the `carla` package installed:
    venv37\\Scripts\\python.exe carla_bridge\\server.py --config config/camera_config.yaml
"""

import sys
import os
import struct
import socket
import argparse
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import carla  # noqa: E402  (only available in the py3.7 venv)
from modules.camera_controller import CameraController  # noqa: E402
from modules.traffic_generator import TrafficGenerator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# CARLA fixed_delta_seconds = 0.1 -> effective simulation FPS = 10
_CARLA_FPS = 10


def send_frames(conn, sync_frames):
    """Send one batch of synchronized camera frames over the socket.

    Wire format (little-endian):
      uint32  num_cameras
      repeat num_cameras times:
        uint32  camera_id_len
        bytes   camera_id (utf-8)
        double  timestamp
        uint32  height, uint32 width, uint32 channels
        bytes   raw frame data (height*width*channels, uint8)
    """
    parts = [struct.pack('<I', len(sync_frames))]
    for cam_id, data in sync_frames.items():
        frame = data['frame']
        cam_id_bytes = cam_id.encode('utf-8')
        h, w, c = frame.shape
        parts.append(struct.pack('<I', len(cam_id_bytes)))
        parts.append(cam_id_bytes)
        parts.append(struct.pack('<d', float(data['timestamp'])))
        parts.append(struct.pack('<III', h, w, c))
        parts.append(frame.tobytes())
    conn.sendall(b''.join(parts))


def main():
    parser = argparse.ArgumentParser(description='CARLA Bridge Server (Python 3.7)')
    parser.add_argument('--config', default='config/camera_config.yaml',
                        help='Path to camera configuration file')
    parser.add_argument('--host', default='127.0.0.1', help='Bridge listen address')
    parser.add_argument('--port', type=int, default=8765, help='Bridge listen port')
    parser.add_argument('--num-vehicles', type=int, default=10)
    parser.add_argument('--num-pedestrians', type=int, default=5)
    args = parser.parse_args()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    logger.info("CARLA client connected")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / _CARLA_FPS
    world.apply_settings(settings)
    logger.info("Synchronous mode enabled at %d fps", _CARLA_FPS)

    camera_controller = CameraController(client, world, args.config)
    camera_controller.setup_cameras()
    logger.info("Cameras ready: %s", list(camera_controller.cameras.keys()))

    traffic = TrafficGenerator(world, num_vehicles=args.num_vehicles,
                                num_pedestrians=args.num_pedestrians)
    traffic.spawn_actors()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(1)
    logger.info("Waiting for AI client on %s:%d ...", args.host, args.port)
    conn, addr = server_sock.accept()
    logger.info("AI client connected from %s", addr)

    frame_count = 0
    try:
        while True:
            world.tick()
            traffic.update_pedestrians()
            sync_frames = camera_controller.get_synchronized_frames()

            if not sync_frames:
                continue

            try:
                send_frames(conn, sync_frames)
            except (BrokenPipeError, ConnectionResetError, OSError):
                logger.info("AI client disconnected, stopping.")
                break

            frame_count += 1
            if frame_count % 100 == 0:
                logger.info("Sent %d frame batches", frame_count)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        try:
            conn.close()
        except Exception:
            pass
        server_sock.close()

        camera_controller.cleanup()
        traffic.cleanup()

        settings.synchronous_mode = False
        world.apply_settings(settings)
        logger.info("Bridge server cleanup completed")


if __name__ == '__main__':
    main()
