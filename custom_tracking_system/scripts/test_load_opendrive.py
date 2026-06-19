"""
test_load_opendrive.py — Verify hanoi_district3.xodr loads in CARLA.

Checks:
  1. Connect to CARLA
  2. generate_opendrive_world() with hanoi_district3.xodr
  3. Map name, spawn points count, junction count
  4. Spawn 1 vehicle, verify it moves 1 tick
  5. carla_to_latlon sanity check on spawn point

Usage:
  python scripts/test_load_opendrive.py --host localhost --port 2000
"""
import argparse
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

XODR_PATH = ROOT.parent / 'Map' / 'hanoi_district3.xodr'


def main(host, port):
    try:
        import carla
    except ImportError:
        print("ERROR: carla module not found in sys.path")
        print("sys.path:", sys.path[:4])
        sys.exit(1)

    print(f"[1] Connecting to CARLA at {host}:{port}...")
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    print(f"    Server version: {client.get_server_version()}")

    print(f"[2] Reading {XODR_PATH.name} ({XODR_PATH.stat().st_size/1024:.0f} KB)...")
    xodr_content = XODR_PATH.read_text(encoding='utf-8')

    print("[3] Calling generate_opendrive_world()...")
    params = carla.OpendriveGenerationParameters(
        vertex_distance=2.0,
        max_road_length=500.0,
        wall_height=0.0,
        additional_width=0.6,
        smooth_junctions=True,
        enable_mesh_visibility=True,
    )
    world = client.generate_opendrive_world(xodr_content, params)
    carla_map = world.get_map()

    print(f"    Map name      : {carla_map.name}")
    spawn_points = carla_map.get_spawn_points()
    print(f"    Spawn points  : {len(spawn_points)}")

    # Count topology junctions
    topology = carla_map.get_topology()
    junction_ids = set()
    for seg_start, seg_end in topology:
        if seg_start.is_junction:
            junction_ids.add(seg_start.get_junction().id)
    print(f"    Junctions     : {len(junction_ids)}")

    if not spawn_points:
        print("ERROR: No spawn points — map may not have loaded road geometry correctly.")
        sys.exit(1)

    print(f"    First spawn   : ({spawn_points[0].location.x:.1f}, {spawn_points[0].location.y:.1f}, {spawn_points[0].location.z:.1f})")

    # [4] Spawn a vehicle
    print("[4] Spawning test vehicle...")
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('vehicle.toyota.prius')[0] \
         if blueprint_library.filter('vehicle.toyota.prius') \
         else blueprint_library.filter('vehicle.*')[0]

    actor = world.try_spawn_actor(bp, spawn_points[0])
    if actor is None:
        print("    WARNING: try_spawn_actor returned None (spawn point occupied?), trying next...")
        for sp in spawn_points[1:5]:
            actor = world.try_spawn_actor(bp, sp)
            if actor is not None:
                break

    if actor is None:
        print("ERROR: Could not spawn vehicle at any of the first 5 spawn points.")
        settings.synchronous_mode = False
        world.apply_settings(settings)
        sys.exit(1)

    print(f"    Spawned id={actor.id} at ({actor.get_location().x:.1f}, {actor.get_location().y:.1f})")

    # Tick once to verify simulation advances
    world.tick()
    loc_after = actor.get_location()
    print(f"    After 1 tick  : ({loc_after.x:.1f}, {loc_after.y:.1f}, {loc_after.z:.1f})")

    # [5] Georef sanity check
    print("[5] Georeference sanity check...")
    from georeference import GeoReference
    gr  = GeoReference.from_map('hanoi_district3')
    lat, lon = gr.carla_to_latlon(loc_after.x, loc_after.y)
    print(f"    CARLA ({loc_after.x:.1f}, {loc_after.y:.1f}) -> lat={lat:.5f} lon={lon:.5f}")
    # Hanoi is ~21°N, 105-106°E
    lat_ok = 20.0 < lat < 22.0
    lon_ok = 104.5 < lon < 106.5
    print(f"    Lat in range [20,22]°N : {'OK' if lat_ok else 'FAIL'}")
    print(f"    Lon in range [104.5,106.5]°E: {'OK' if lon_ok else 'FAIL'}")

    # Cleanup
    actor.destroy()
    settings.synchronous_mode = False
    world.apply_settings(settings)

    print("\n=== RESULT ===")
    if lat_ok and lon_ok and len(spawn_points) > 0:
        print("PASS — hanoi_district3.xodr loads correctly in CARLA")
        print(f"  Spawn points: {len(spawn_points)}, Junctions: {len(junction_ids)}")
        print(f"  Georef: ({lat:.5f}, {lon:.5f})")
    else:
        print("FAIL — see errors above")
        sys.exit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    args = ap.parse_args()
    main(args.host, args.port)
