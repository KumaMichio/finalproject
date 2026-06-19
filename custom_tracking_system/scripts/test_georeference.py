import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from georeference import GeoReference

gr = GeoReference.from_map('hanoi_district3')

lat, lon = gr.carla_to_latlon(0.0, 0.0)
print(f"CARLA (0,0) -> ({lat:.6f}, {lon:.6f})")

lat2, lon2 = gr.carla_to_latlon(100.0, -50.0)
print(f"CARLA (100,-50) -> ({lat2:.6f}, {lon2:.6f})")

x, y = gr.latlon_to_carla(lat, lon)
print(f"Round-trip -> CARLA ({x:.3f}, {y:.3f})")

g = gr.junction_graph()
if g:
    print(f"Junction graph: {len(g['junctions'])} junctions, {len(g['roads'])} roads")
else:
    print("No junction graph file")
