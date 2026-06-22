"""Visual camera-pose fitting helper — projects real OSM road polylines onto a
CCTV frame via CameraCalibration.from_carla_params() so the pose (position,
rotation, fov) can be tuned by eye until the projected roads trace the visible
road surface. Used to build custom_tracking_system/data/calib_pho_hue.json.

Usage:
    python fit_camera_pose.py --x 221 --y -190 --z 7 --pitch -12 --yaw -90 --fov 65 --tag mytry
    # inspect pho_hue_tkc_fit_mytry.jpg, adjust args, repeat
"""
import sys, math
sys.path.insert(0, '../custom_tracking_system')
sys.path.insert(0, '../custom_tracking_system/modules')
import xml.etree.ElementTree as ET
import cv2
from calibration import CameraCalibration
from georeference import GeoReference

gr = GeoReference.from_map('pho_hue_tkc')

tree = ET.parse('pho_hue_tkc.osm')
root = tree.getroot()
nodes = {n.get('id'): (float(n.get('lat')), float(n.get('lon'))) for n in root.findall('node')}

def gather(name_filter):
    polylines = []
    for way in root.findall('way'):
        tags = {t.get('k'): t.get('v') for t in way.findall('tag')}
        if tags.get('name') != name_filter:
            continue
        refs = [nd.get('ref') for nd in way.findall('nd')]
        pts = [nodes[r] for r in refs if r in nodes]
        if len(pts) >= 2:
            polylines.append([gr.latlon_to_carla(lat, lon) for lat, lon in pts])
    return polylines

pho_hue = gather('Phố Huế')
# the through road is split into Đại Cồ Việt (west) / Trần Khát Chân (east) names
dai_co_viet = gather('Đại Cồ Việt')
tran_khat_chan = gather('Trần Khát Chân')
overpass = gather('Cầu vượt Trần Khát Chân')

print('Pho Hue segments:', len(pho_hue))
print('Dai Co Viet segments:', len(dai_co_viet))
print('Tran Khat Chan segments:', len(tran_khat_chan))
print('Overpass segments:', len(overpass))

frame = cv2.imread('pho_hue_tkc_sample_frame.jpg')
H, W = frame.shape[:2]

def render(position, rotation, fov, tag):
    calib = CameraCalibration.from_carla_params(
        position=position, rotation=rotation, fov=fov, resolution=[W, H], ground_z=0.0)
    img = frame.copy()

    def proj(x, y, z=0.0):
        uv = calib.world_to_pixel(x, y, z)
        if uv is None:
            return None
        u, v = uv
        return (int(u), int(v))

    def draw_polylines(polylines, color, z=0.0):
        for pl in polylines:
            for (x1, y1), (x2, y2) in zip(pl[:-1], pl[1:]):
                p1, p2 = proj(x1, y1, z), proj(x2, y2, z)
                if p1 and p2:
                    cv2.line(img, p1, p2, color, 3)

    draw_polylines(pho_hue, (0, 0, 255))          # red, ground level
    draw_polylines(dai_co_viet, (255, 0, 0))      # blue, ground level
    draw_polylines(tran_khat_chan, (0, 255, 255)) # yellow, ground level
    draw_polylines(overpass, (0, 165, 255), z=OVERPASS_Z)  # orange, elevated deck

    out = f'pho_hue_tkc_fit_{tag}.jpg'
    cv2.imwrite(out, img)
    print('saved', out)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--x', type=float, required=True)
    ap.add_argument('--y', type=float, required=True)
    ap.add_argument('--z', type=float, required=True)
    ap.add_argument('--pitch', type=float, required=True)
    ap.add_argument('--yaw', type=float, required=True)
    ap.add_argument('--roll', type=float, default=0.0)
    ap.add_argument('--fov', type=float, required=True)
    ap.add_argument('--tag', type=str, default='try')
    ap.add_argument('--overpass-z', type=float, default=6.0)
    args = ap.parse_args()
    OVERPASS_Z = args.overpass_z
    render([args.x, args.y, args.z], [args.pitch, args.yaw, args.roll], args.fov, args.tag)
