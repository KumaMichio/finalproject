"""
Georeference: chuyen doi giua toa do CARLA world (x, y met, sinh tu
generate_opendrive_world) va toa do dia ly thuc (lat, lon).

Cong thuc (da verify bang 4 goc bbox cua map3_smaller.osm):
    UTM_x = CARLA_x + offset_x
    UTM_y = offset_y - CARLA_y          (CARLA y bi dao dau so voi OpenDRIVE/UTM)
    (lon, lat) = UTM48N_inverse(UTM_x, UTM_y)

offset_x, offset_y lay tu <offset> trong header file .xodr (UTM zone 48N,
EPSG:32648 — dung cho Ha Noi).
"""
from pyproj import Transformer

# offset rieng cho moi map (doc tu <offset x=".." y=".."/> trong file .xodr)
MAP_OFFSETS = {
    "hanoi_district3": (581165.61, 2320490.53),
}

_UTM_EPSG = "EPSG:32648"  # UTM zone 48N (Ha Noi)
_to_utm = Transformer.from_crs("EPSG:4326", _UTM_EPSG, always_xy=True)
_to_latlon = Transformer.from_crs(_UTM_EPSG, "EPSG:4326", always_xy=True)


class GeoReference:
    def __init__(self, offset_x, offset_y):
        self.offset_x = offset_x
        self.offset_y = offset_y

    @classmethod
    def from_map(cls, name):
        return cls(*MAP_OFFSETS[name])

    def carla_to_latlon(self, x, y):
        """CARLA world (x, y) [met] -> (lat, lon)"""
        utm_x = x + self.offset_x
        utm_y = self.offset_y - y
        lon, lat = _to_latlon.transform(utm_x, utm_y)
        return lat, lon

    def latlon_to_carla(self, lat, lon):
        """(lat, lon) -> CARLA world (x, y) [met]"""
        utm_x, utm_y = _to_utm.transform(lon, lat)
        x = utm_x - self.offset_x
        y = self.offset_y - utm_y
        return x, y


if __name__ == "__main__":
    geo = GeoReference.from_map("hanoi_district3")

    # Round-trip test
    test_points = [
        (20.9843810, 105.7816060),
        (20.9875760, 105.7874130),
        (20.9860000, 105.7845000),
    ]
    for lat, lon in test_points:
        x, y = geo.latlon_to_carla(lat, lon)
        lat2, lon2 = geo.carla_to_latlon(x, y)
        print(f"({lat},{lon}) -> CARLA({x:.2f},{y:.2f}) -> ({lat2:.7f},{lon2:.7f})")
