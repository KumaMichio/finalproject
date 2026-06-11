#!/usr/bin/env python3
"""
CARLA Trajectory Dataset Collector

Runs CARLA in synchronous mode, spawns traffic (and occasionally one of the
ScenarioController accident scenarios for diversity), and logs the
world-coordinate trajectory (x, y, vx, vy in meters) of every vehicle and
pedestrian actor every tick.

Output: one CSV per episode under --out, with columns:
    t, actor_id, class, x, y, vx, vy

`class` is one of: car, motorcycle, person — derived from the CARLA
blueprint type_id. World coordinates are CARLA's single global map frame,
shared by all cameras, so the resulting dataset is camera-independent.

Usage:
    python collect_trajectory_data.py --episodes 10 --steps-per-episode 600 \
        --num-vehicles 15 --num-pedestrians 8 --scenario-ratio 0.3 \
        --out data/trajectories
"""

import argparse
import csv
import logging
import random
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from modules.traffic_generator import TrafficGenerator
from modules.scenario_controller import ScenarioController
from modules.ground_truth import GroundTruthCollector

logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_FIXED_DELTA = 0.1  # seconds -> 10 Hz, matches camera_config.yaml system.fixed_delta_seconds

# Substrings of CARLA 0.9.9.4 two-wheeler blueprint type_ids
_MOTORCYCLE_HINTS = (
    'harley-davidson', 'kawasaki', 'yamaha', 'vespa',
    'crossbike', 'omafiets', 'century', 'gazelle', 'diamondback', 'bh',
)

_SCENARIO_METHODS = (
    'hit_and_run', 'pedestrian_hit', 'red_light_crash', 'rear_end', 'sudden_stop',
)


def classify_type_id(type_id: str) -> str:
    if type_id.startswith('walker.'):
        return 'person'
    if type_id.startswith('vehicle.'):
        lowered = type_id.lower()
        if any(hint in lowered for hint in _MOTORCYCLE_HINTS):
            return 'motorcycle'
        return 'car'
    return 'unknown'


def connect_carla():
    import carla
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = _FIXED_DELTA
    world.apply_settings(settings)

    logger.info("Connected to CARLA, synchronous mode @ %.0f Hz", 1.0 / _FIXED_DELTA)
    return client, world


def collect_episode(world, episode_idx: int, args, out_dir: Path):
    traffic = TrafficGenerator(world, num_vehicles=args.num_vehicles,
                                num_pedestrians=args.num_pedestrians)
    traffic.spawn_actors()

    scenario = None
    run_scenario = random.random() < args.scenario_ratio
    if run_scenario:
        # ScenarioController needs its own client handle for the traffic manager
        import carla
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        scenario = ScenarioController(world, client)
        method_name = random.choice(_SCENARIO_METHODS)
        logger.info("Episode %d: running scenario '%s'", episode_idx, method_name)
        getattr(scenario, method_name)()

    gt = GroundTruthCollector(world)
    rows = []

    try:
        for step in range(args.steps_per_episode):
            world.tick()
            traffic.update_pedestrians()

            frame = gt.collect_frame(step)
            t = step * _FIXED_DELTA

            for v in frame['vehicles']:
                cls = classify_type_id(v['type_id'])
                x, y = v['location'][0], v['location'][1]
                vx, vy = v['velocity'][0], v['velocity'][1]
                rows.append((t, v['actor_id'], cls, x, y, vx, vy))

            for p in frame['pedestrians']:
                cls = classify_type_id(p['type_id'])
                x, y = p['location'][0], p['location'][1]
                vx, vy = p['velocity'][0], p['velocity'][1]
                rows.append((t, p['actor_id'], cls, x, y, vx, vy))

    finally:
        if scenario is not None:
            scenario.cleanup()
        traffic.cleanup()

    out_path = out_dir / f"episode_{episode_idx:04d}.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'actor_id', 'class', 'x', 'y', 'vx', 'vy'])
        writer.writerows(rows)

    logger.info("Episode %d: wrote %d rows -> %s", episode_idx, len(rows), out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--steps-per-episode', type=int, default=600)
    parser.add_argument('--num-vehicles', type=int, default=15)
    parser.add_argument('--num-pedestrians', type=int, default=8)
    parser.add_argument('--scenario-ratio', type=float, default=0.3,
                         help='Fraction of episodes that include a random accident scenario')
    parser.add_argument('--out', type=str, default='data/trajectories')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, world = connect_carla()

    for episode_idx in range(args.episodes):
        t0 = time.time()
        collect_episode(world, episode_idx, args, out_dir)
        logger.info("Episode %d done in %.1fs", episode_idx, time.time() - t0)


if __name__ == '__main__':
    main()
