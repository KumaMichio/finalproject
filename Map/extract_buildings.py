"""
Trich xuat building footprint tu map3_smaller.osm, convert sang toa do CARLA
(qua georeference.py), tinh oriented bounding box (OBB) + chieu cao uoc
luong (tu building:levels), luu ra buildings_hanoi_district3.json.

Chay: venv_tracking\\Scripts\\python.exe Map\\extract_buildings.py
"""
import json
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from georeference import GeoReference

OSM_FILE = "map3_smaller.osm"
OUT_FILE = "buildings_hanoi_district3.json"
LEVEL_HEIGHT = 3.0  # met / tang
DEFAULT_LEVELS = 3


def main():
    geo = GeoReference.from_map("hanoi_district3")

    tree = ET.parse(OSM_FILE)
    root = tree.getroot()

    # Index node id -> (lat, lon)
    node_coords = {}
    for node in root.findall("node"):
        node_coords[node.get("id")] = (
            float(node.get("lat")),
            float(node.get("lon")),
        )

    buildings = []
    for way in root.findall("way"):
        tags = {t.get("k"): t.get("v") for t in way.findall("tag")}
        if "building" not in tags:
            continue

        refs = [nd.get("ref") for nd in way.findall("nd")]
        pts = []
        for ref in refs:
            if ref in node_coords:
                lat, lon = node_coords[ref]
                x, y = geo.latlon_to_carla(lat, lon)
                pts.append((x, y))

        if len(pts) < 3:
            continue

        pts_np = np.array(pts, dtype=np.float32)
        # Oriented bounding box (min area rect)
        rect = cv2.minAreaRect(pts_np)  # ((cx,cy),(w,h),angle_deg)
        (cx, cy), (w, h), angle = rect

        levels = tags.get("building:levels")
        try:
            levels = float(levels) if levels else DEFAULT_LEVELS
        except ValueError:
            levels = DEFAULT_LEVELS
        height = max(levels, 1) * LEVEL_HEIGHT

        buildings.append({
            "name": tags.get("name", ""),
            "building_type": tags.get("building", "yes"),
            "center": [cx, cy],
            "size": [w, h],          # CARLA units (met), truoc khi xoay
            "yaw_deg": angle,        # goc xoay (OpenCV convention)
            "height": height,
            "footprint": [[round(x, 2), round(y, 2)] for x, y in pts],
        })

    print(f"Trich xuat {len(buildings)} buildings")
    for b in buildings:
        name = b['name'].encode("ascii", "replace").decode("ascii")
        print(f"  {b['building_type']:8s} center=({b['center'][0]:.1f},{b['center'][1]:.1f}) "
              f"size=({b['size'][0]:.1f}x{b['size'][1]:.1f}) h={b['height']:.1f}m "
              f"yaw={b['yaw_deg']:.1f} name={name}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(buildings, f, ensure_ascii=False, indent=2)
    print(f"Da luu: {OUT_FILE}")


if __name__ == "__main__":
    main()
