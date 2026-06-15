"""
Logic-only traffic light simulation cho hanoi_district3 (khong co
traffic_light blueprint/actor trong CARLA 0.9.14 packaged build nay).

Mooi giao lo lon (junction tu topology co kich thuoc > min_extent) duoc gan
1 "den" voi chu ky Red/Yellow/Green, lech pha random theo seed. Trang thai
tai thoi diem t (giay mo phong) duoc tinh thuan logic - khong can actor CARLA,
chi dung world.debug de ve diem mau lam tin hieu thi giac.

get_state_near() tra ve trang thai cua den gan nhat trong ban kinh radius,
dung dinh dang ('Red'/'Yellow'/'Green'/None) khop voi
incident_detector._check_red_light_violation(..., light_state, ...).
"""
import random

import carla


class TrafficLightSim:
    def __init__(self, locations, red_time=10.0, yellow_time=2.0, green_time=8.0, seed=42):
        self.red_time = red_time
        self.yellow_time = yellow_time
        self.green_time = green_time
        self.cycle = red_time + yellow_time + green_time
        rng = random.Random(seed)
        self.lights = [
            {"location": loc, "phase": rng.uniform(0, self.cycle)}
            for loc in locations
        ]

    def state_at(self, light, t):
        x = (t + light["phase"]) % self.cycle
        if x < self.red_time:
            return "Red"
        if x < self.red_time + self.yellow_time:
            return "Yellow"
        return "Green"

    def get_state_near(self, location, t, radius):
        """Tra ve trang thai cua den gan nhat trong ban kinh radius (met),
        hoac None neu khong co den nao gan."""
        best = None
        best_dist = radius
        for light in self.lights:
            d = location.distance(light["location"])
            if d <= best_dist:
                best_dist = d
                best = light
        if best is None:
            return None
        return self.state_at(best, t)

    _COLORS = {
        "Red": carla.Color(255, 0, 0),
        "Yellow": carla.Color(255, 255, 0),
        "Green": carla.Color(0, 255, 0),
    }

    def draw(self, world, t, life_time=0.5, size=3.0):
        """Ve mark cho moi den bang debug.draw_box (unlit, mau ro net hon
        draw_point vi draw_point bi anh huong boi lighting cua scene)."""
        debug = world.debug
        for light in self.lights:
            state = self.state_at(light, t)
            box = carla.BoundingBox(light["location"], carla.Vector3D(x=size, y=size, z=size))
            debug.draw_box(
                box, carla.Rotation(), thickness=size,
                color=self._COLORS[state], life_time=life_time,
            )


def junction_traffic_lights(carla_map, min_extent=4.0, z=3.0):
    """Tim cac junction du lon (giao lo thuc, khong phai via he/be re nho)
    tu topology, tra ve list carla.Location dat den tai tam giao lo, nang
    len z met de de quan sat tu xa."""
    wps = carla_map.generate_waypoints(distance=5.0)
    seen = {}
    for wp in wps:
        if wp.is_junction:
            j = wp.get_junction()
            if j.id not in seen:
                bb = j.bounding_box
                if bb.extent.x >= min_extent and bb.extent.y >= min_extent:
                    seen[j.id] = carla.Location(x=bb.location.x, y=bb.location.y, z=z)
    return list(seen.values())
