"""parse_xodr.py — extract junction graph from a .xodr map.

Usage:
    python parse_xodr.py                                   # default: hanoi_district3
    python parse_xodr.py --xodr Map/pho_hue_tkc.xodr --out Map/pho_hue_tkc_junction_graph.json
"""
import argparse
import xml.etree.ElementTree as ET
import json
from pathlib import Path
import math

_MAP_DIR = Path(__file__).resolve().parents[2] / 'Map'
XODR = _MAP_DIR / 'hanoi_district3.xodr'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xodr', type=Path, default=XODR,
                     help='Input .xodr file (default: Map/hanoi_district3.xodr)')
    ap.add_argument('--out', type=Path, default=_MAP_DIR / 'junction_graph.json',
                     help='Output junction graph .json (default: Map/junction_graph.json)')
    args = ap.parse_args()

    tree = ET.parse(str(args.xodr))
    root = tree.getroot()

    # Header
    header = root.find('header')
    geo    = root.find('.//geoReference')
    offset = root.find('.//offset')
    print(f"Vendor: {header.get('vendor')}")
    print(f"GeoRef: {geo.text.strip() if geo is not None else 'none'}")
    if offset is not None:
        print(f"Offset: x={offset.get('x')} y={offset.get('y')}")

    # Roads
    roads = root.findall('road')
    print(f"\nTotal roads: {len(roads)}")

    road_map = {}  # road_id -> {id, name, length, junction, start_x, start_y, start_hdg, end_x, end_y, predecessor, successor}
    for r in roads:
        rid    = r.get('id')
        length = float(r.get('length', 0))
        junction = r.get('junction', '-1')

        # First geometry element
        geom = r.find('.//planView/geometry')
        if geom is None:
            geom = r.find('.//geometry')
        sx = float(geom.get('x', 0)) if geom is not None else 0.0
        sy = float(geom.get('y', 0)) if geom is not None else 0.0
        hdg = float(geom.get('hdg', 0)) if geom is not None else 0.0

        ex = sx + length * math.cos(hdg)
        ey = sy + length * math.sin(hdg)

        # Predecessor/successor
        pred = r.find('.//link/predecessor')
        succ = r.find('.//link/successor')
        pred_id   = pred.get('elementId') if pred is not None else None
        pred_type = pred.get('elementType') if pred is not None else None
        succ_id   = succ.get('elementId') if succ is not None else None
        succ_type = succ.get('elementType') if succ is not None else None

        road_map[rid] = {
            'id': rid,
            'length': round(length, 3),
            'junction': junction,
            'start': [round(sx,3), round(sy,3)],
            'end':   [round(ex,3), round(ey,3)],
            'heading_rad': round(hdg, 5),
            'predecessor': {'id': pred_id, 'type': pred_type},
            'successor':   {'id': succ_id, 'type': succ_type},
        }

    # Junctions
    junctions = root.findall('junction')
    print(f"Total junctions: {len(junctions)}")

    junction_map = {}
    for j in junctions:
        jid = j.get('id')
        connections = []
        for conn in j.findall('connection'):
            incoming   = conn.get('incomingRoad')
            connecting = conn.get('connectingRoad')
            lane_links = []
            for ll in conn.findall('laneLink'):
                lane_links.append({'from': ll.get('from'), 'to': ll.get('to')})
            connections.append({
                'incoming_road': incoming,
                'connecting_road': connecting,
                'lane_links': lane_links,
            })

        # Estimate junction center from connected road endpoints
        xs, ys = [], []
        for conn in connections:
            r = road_map.get(conn['connecting_road'])
            if r:
                xs.append(r['start'][0]); ys.append(r['start'][1])
                xs.append(r['end'][0]);   ys.append(r['end'][1])
        cx = round(sum(xs)/len(xs), 3) if xs else 0.0
        cy = round(sum(ys)/len(ys), 3) if ys else 0.0

        # Find incoming roads (roads with successor=junction or predecessor=junction)
        incoming_roads = []
        for rid, rd in road_map.items():
            if rd['successor']['id'] == jid and rd['successor']['type'] == 'junction':
                incoming_roads.append(rid)
            if rd['predecessor']['id'] == jid and rd['predecessor']['type'] == 'junction':
                incoming_roads.append(rid)

        junction_map[jid] = {
            'id': jid,
            'center': [cx, cy],
            'connections': connections,
            'incoming_roads': list(set(incoming_roads)),
        }

    # Show first 3 junctions
    for jid in list(junction_map)[:3]:
        j = junction_map[jid]
        print(f"\nJunction {jid}: center={j['center']}, connections={len(j['connections'])}, incoming_roads={j['incoming_roads']}")
        for c in j['connections'][:2]:
            print(f"  incoming={c['incoming_road']} -> connecting={c['connecting_road']} ({len(c['lane_links'])} lane links)")

    # Save
    with open(args.out, 'w') as f:
        json.dump({'junctions': junction_map, 'roads': road_map}, f, indent=2)
    print(f"\nSaved to {args.out}")
    print(f"  {len(junction_map)} junctions, {len(road_map)} roads")


if __name__ == '__main__':
    main()
