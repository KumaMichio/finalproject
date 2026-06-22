"""
georeference.py — CARLA ↔ lat/lon conversion for hanoi_district3.xodr.

OpenDRIVE header specifies:
    +proj=utm +zone=48 +ellps=WGS84 +datum=WGS84 +units=m
    offset x=581165.61 y=2320490.53

CARLA uses OpenDRIVE local coords but with Y-axis flipped:
    OpenDRIVE x = CARLA x
    OpenDRIVE y = -CARLA y   (CARLA y increases southward)

So:
    UTM_E = CARLA_x + offset_x
    UTM_N = -CARLA_y + offset_y
"""

import json
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

_MAP_DIR = Path(__file__).resolve().parent.parent / 'Map'

_MAP_CONFIGS = {
    'hanoi_district3': {
        'xodr': 'hanoi_district3.xodr',
        'proj': '+proj=utm +zone=48 +ellps=WGS84 +datum=WGS84 +units=m +no_defs',
        'offset_x': 581165.61,
        'offset_y': 2320490.53,
        'flip_y': True,
    },
    'pho_hue_tkc': {
        'xodr': 'pho_hue_tkc.xodr',
        'proj': '+proj=utm +zone=48 +ellps=WGS84 +datum=WGS84 +units=m +no_defs',
        'offset_x': 588263.04,
        'offset_y': 2323135.83,
        'flip_y': True,
    },
}


class GeoReference:
    """
    Converts CARLA world coordinates to geographic lat/lon.

    Attributes:
        offset_x, offset_y  — OpenDRIVE <offset> values (UTM metres)
        flip_y              — True when CARLA Y is negated before adding offset
    """

    def __init__(self, proj_string: str, offset_x: float, offset_y: float,
                 flip_y: bool = True):
        self._offset_x = offset_x
        self._offset_y = offset_y
        self._flip_y   = flip_y

        try:
            from pyproj import Proj, Transformer, CRS
            crs_utm = CRS.from_proj4(proj_string)
            crs_wgs = CRS.from_epsg(4326)
            self._transformer = Transformer.from_crs(crs_utm, crs_wgs,
                                                     always_xy=True)
            self._proj_available = True
            logger.info("GeoReference: pyproj Transformer ready (%s)", proj_string)
        except ImportError:
            logger.warning("pyproj not available — lat/lon conversion will use approximate formula")
            self._proj_available = False
            # Approximate central meridian for UTM zone 48N
            self._approx_lat0 = 2320490.53 / 111320.0  # degrees lat at offset
            self._approx_lon0 = 105.0                   # central meridian zone 48

    @classmethod
    def from_map(cls, map_name: str) -> 'GeoReference':
        cfg = _MAP_CONFIGS.get(map_name)
        if cfg is None:
            raise ValueError(f"GeoReference: unknown map '{map_name}'. "
                             f"Known maps: {list(_MAP_CONFIGS)}")
        return cls(cfg['proj'], cfg['offset_x'], cfg['offset_y'], cfg['flip_y'])

    def carla_to_utm(self, x: float, y: float) -> tuple[float, float]:
        """CARLA world (x, y) → UTM (easting, northing) in metres."""
        utm_e = x + self._offset_x
        utm_n = (-y if self._flip_y else y) + self._offset_y
        return utm_e, utm_n

    def carla_to_latlon(self, x: float, y: float) -> tuple[float, float]:
        """CARLA world (x, y) → (latitude, longitude) in decimal degrees."""
        utm_e, utm_n = self.carla_to_utm(x, y)

        if self._proj_available:
            lon, lat = self._transformer.transform(utm_e, utm_n)
            return lat, lon

        # Rough fallback (±0.001° accuracy, ~100 m)
        lat = utm_n / 111320.0
        lon = self._approx_lon0 + (utm_e - 500000.0) / (111320.0 * math.cos(math.radians(lat)))
        return lat, lon

    def latlon_to_carla(self, lat: float, lon: float) -> tuple[float, float]:
        """(latitude, longitude) → CARLA world (x, y).  Requires pyproj."""
        if not self._proj_available:
            raise RuntimeError("pyproj required for latlon_to_carla")
        from pyproj import Transformer, CRS
        crs_wgs = CRS.from_epsg(4326)
        crs_utm = CRS.from_proj4('+proj=utm +zone=48 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
        t = Transformer.from_crs(crs_wgs, crs_utm, always_xy=True)
        utm_e, utm_n = t.transform(lon, lat)
        x = utm_e - self._offset_x
        y_local = utm_n - self._offset_y
        y = -y_local if self._flip_y else y_local
        return x, y

    def junction_graph(self) -> dict | None:
        """Load pre-parsed junction graph from Map/junction_graph.json (if exists)."""
        p = _MAP_DIR / 'junction_graph.json'
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)
