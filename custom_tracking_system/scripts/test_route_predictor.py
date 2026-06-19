"""Quick smoke test for RoutePredictor."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from georeference import GeoReference
from route_predictor import RoutePredictor

gr = GeoReference.from_map('hanoi_district3')
rp = RoutePredictor.from_map('hanoi_district3', georef=gr)

# Test from junction 3 center (520, 463) heading east
test_cases = [
    dict(x=520.0, y=-463.0, heading_deg=0.0,   direction='straight', label='junction3 -> straight'),
    dict(x=520.0, y=-463.0, heading_deg=0.0,   direction='right',    label='junction3 -> right'),
    dict(x=520.0, y=-463.0, heading_deg=0.0,   direction='left',     label='junction3 -> left'),
    dict(x=32.0,  y=-280.0, heading_deg=270.0, direction='straight', label='junction9 -> straight'),
]

for tc in test_cases:
    label = tc.pop('label')
    hops  = rp.predict(n_hops=3, **tc)
    print(f"\n{label}:")
    if not hops:
        print("  (no route found)")
    for wp in hops:
        lat_str = f"{wp['lat']:.5f},{wp['lon']:.5f}" if wp['lat'] else "no georef"
        print(f"  hop{wp['hop']}: road={wp['road_id']:5s}  "
              f"({wp['x']:7.1f},{wp['y']:7.1f})  hdg={wp['heading_deg']:6.1f}°  "
              f"dir={wp['direction_used']}  latlon={lat_str}")

# Probabilistic test
print("\n--- Probabilistic predict ---")
probs = {'straight': 0.40, 'right': 0.35, 'left': 0.20, 'u_turn': 0.05}
multi = rp.predict_probabilistic(x=520.0, y=-463.0, heading_deg=0.0,
                                  direction_probs=probs, n_hops=2, min_prob=0.15)
for wp in multi:
    print(f"  branch={wp['branch_direction']} p={wp['branch_prob']:.2f}  "
          f"hop{wp['hop']} road={wp['road_id']}  ({wp['x']:.1f},{wp['y']:.1f})")
