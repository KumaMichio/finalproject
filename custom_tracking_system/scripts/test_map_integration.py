"""Quick smoke test for the server map/route-prediction integration --
mirrors the new code path in server/services/ai_processor.py._initialize()
without needing CARLA, a real video, or the FastAPI server running.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from modules.calibration import CalibrationStore
from modules.goal_classifier import GoalClassifier
from georeference import GeoReference
from route_predictor import RoutePredictor

config_path = ROOT / "config" / "camera_config_pho_hue.yaml"
with open(config_path, encoding="utf-8") as f:
    config = yaml.safe_load(f)

calibration = CalibrationStore.load_from_config(str(config_path))
print("calibration cameras:", list(calibration.keys()))

gc = GoalClassifier(
    str(ROOT / "weights" / "goal_classifier_real.pkl"),
    str(ROOT / "weights" / "goal_scaler_real.pkl"),
    str(ROOT / "weights" / "goal_label_encoder_real.pkl"),
    str(ROOT / "weights" / "goal_feature_names_real.json"),
)
print("GoalClassifier classes:", gc.classes)

route_cfg = config["route_prediction"]
map_dir = ROOT.parent / "Map"
graph_path = map_dir / route_cfg["graph_path"]
georef = GeoReference.from_map(route_cfg["map_name"])
rp = RoutePredictor.from_map(route_cfg["map_name"], georef=georef, graph_path=graph_path)

cam_cfg = list(config["cameras"].values())[0]
cx, cy, _ = cam_cfg["position"]
lat, lon = georef.carla_to_latlon(cx, cy)
print(f"Camera lat/lon: {lat:.6f}, {lon:.6f}")

# Fake a vehicle 20m in front of the camera, heading roughly toward the intersection
calib = calibration[cam_cfg["camera_id"]]
route = rp.predict(cx + 5, cy + 15, heading_deg=90.0, direction="straight",
                    n_hops=route_cfg.get("n_hops", 4),
                    direction_probs={"straight": 0.7, "left": 0.2, "right": 0.1, "u_turn": 0.0})
print(f"Predicted route hops: {len(route)}")
for wp in route:
    print(f"  hop={wp['hop']} road={wp['road_id']} lat={wp['lat']} lon={wp['lon']} dir={wp['direction_used']}")

print("SMOKE TEST OK")
