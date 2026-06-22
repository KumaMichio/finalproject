"""
route_predictor.py — multi-hop route prediction via junction graph.

Algorithm:
  1. Given vehicle position (CARLA x,y) + heading + GoalClassifier direction
  2. Find nearest road segment the vehicle is on
  3. Follow road to its successor junction
  4. At junction, select outgoing road whose heading best matches the turn direction
     (straight=minimal deviation, left≈-90°, right≈+90°, u_turn≈180°)
  5. Repeat for N hops, accumulating predicted waypoints

All coordinates are CARLA local coords (metres).
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MAP_DIR = Path(__file__).resolve().parent.parent / 'Map'

# Target heading deviation (degrees) for each direction relative to incoming heading
_TURN_TARGET = {
    'straight': 0.0,
    'left':    -90.0,
    'right':    90.0,
    'u_turn':  180.0,
}

# Max deviation to accept as a valid match (degrees)
_TURN_TOLERANCE = {
    'straight': 60.0,
    'left':     70.0,
    'right':    70.0,
    'u_turn':   70.0,
}

# Empirical prior for hop>=1 (no GoalClassifier signal exists yet for a
# junction the vehicle hasn't approached), measured from 34,698 real
# multi-junction transitions in CARLA episode data -- see
# scripts/eval/eval_route_predictor_priors.py. An "always straight" rule
# (the previous hardcoded behavior) is only right 37.0% of the time here;
# this is the actual marginal distribution, used both to pick the
# most-likely direction and to report an honest confidence instead of a
# fabricated 1.0.
_DEFAULT_HOP1PLUS_PRIOR = {
    'straight': 0.3704, 'left': 0.3656, 'right': 0.1787, 'u_turn': 0.0854,
}
_HOP1PLUS_PRIOR_PATH = Path(__file__).resolve().parent / 'data' / 'route_predictor_priors.json'


def _load_hop1plus_prior() -> dict:
    try:
        with open(_HOP1PLUS_PRIOR_PATH, encoding='utf-8') as f:
            prior = json.load(f)['pooled_hop1plus_prior']
        return {d: float(prior[d]) for d in _DEFAULT_HOP1PLUS_PRIOR}
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return dict(_DEFAULT_HOP1PLUS_PRIOR)


def _heading_diff(a: float, b: float) -> float:
    """Signed angle difference (a - b) in [-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def _road_heading_deg(road: dict) -> float:
    """Road start heading in degrees."""
    return math.degrees(road['heading_rad'])


class RoutePredictor:
    """
    Predicts multi-hop vehicle routes through the road network.

    Usage:
        rp = RoutePredictor.from_map('hanoi_district3')
        routes = rp.predict(x=120.0, y=-80.0, heading_deg=45.0,
                            direction='right', n_hops=3)
        # returns list of RouteHop namedtuples with carla + latlon coords
    """

    def __init__(self, junction_graph: dict,
                 georef=None):
        self._junctions: dict = junction_graph['junctions']
        self._roads:     dict = junction_graph['roads']
        self._georef     = georef

        # Direction + confidence used for hop>=1, where no GoalClassifier
        # signal exists yet (see _DEFAULT_HOP1PLUS_PRIOR above).
        self._hop1plus_prior = _load_hop1plus_prior()
        self._hop1plus_direction = max(self._hop1plus_prior, key=self._hop1plus_prior.get)
        self._hop1plus_confidence = self._hop1plus_prior[self._hop1plus_direction]

        # Build numpy arrays for fast nearest-road lookup
        self._build_road_index()
        logger.info("RoutePredictor ready: %d junctions, %d roads",
                    len(self._junctions), len(self._roads))

    @classmethod
    def from_map(cls, map_name: str = 'hanoi_district3',
                 georef=None, graph_path: Optional[Path] = None) -> 'RoutePredictor':
        if graph_path is None:
            graph_path = _MAP_DIR / 'junction_graph.json'
        if not graph_path.exists():
            raise FileNotFoundError(
                f"junction_graph.json not found at {graph_path}. "
                "Run scripts/parse_xodr.py first."
            )
        with open(graph_path) as f:
            graph = json.load(f)
        return cls(graph, georef=georef)

    # ------------------------------------------------------------------
    # Spatial index
    # ------------------------------------------------------------------

    def _od_to_carla(self, x: float, y: float) -> tuple[float, float]:
        """OpenDRIVE (x, y_north) → CARLA (x, y_south)."""
        return x, -y

    # Roads shorter than this are netconvert internal-junction connector
    # artifacts (e.g. 0.1m stubs with no predecessor/successor) -- excluded
    # from nearest-road matching so a vehicle near a busy junction doesn't
    # snap onto a dead-end stub and immediately dead-end the route.
    _MIN_INDEXED_ROAD_LEN = 1.5

    def _build_road_index(self):
        road_ids   = []
        midpoints  = []
        headings   = []
        lengths    = []

        for rid, r in self._roads.items():
            if r['length'] < self._MIN_INDEXED_ROAD_LEN:
                continue
            # Store midpoints in CARLA coords (negate y)
            mx = (r['start'][0] + r['end'][0]) / 2.0
            my = -((r['start'][1] + r['end'][1]) / 2.0)
            road_ids.append(rid)
            midpoints.append([mx, my])
            # Heading in OpenDRIVE: angle from east, y-north.
            # In CARLA pixel-space heading stays the same for x-direction,
            # but y is flipped, so heading = -heading_od for y-component.
            # For road-matching purposes we negate the heading.
            headings.append(-math.degrees(r['heading_rad']))
            lengths.append(r['length'])

        self._road_ids  = road_ids
        self._mid_arr   = np.array(midpoints, dtype=np.float32)
        self._hdg_arr   = np.array(headings,  dtype=np.float32)
        self._len_arr   = np.array(lengths,   dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_nearest_road(self, x: float, y: float,
                          heading_deg: Optional[float] = None) -> Optional[str]:
        """
        Find road ID nearest to (x, y).
        If heading_deg provided, prefer roads whose heading aligns (within ±90°).
        """
        pt = np.array([[x, y]], dtype=np.float32)
        dists = np.linalg.norm(self._mid_arr - pt, axis=1)

        if heading_deg is not None:
            # Down-weight roads whose heading is >90° from vehicle heading
            hdiff = np.abs(np.array([_heading_diff(h, heading_deg)
                                     for h in self._hdg_arr]))
            penalty = np.where(hdiff > 90, 3.0, 1.0)
            dists = dists * penalty

        idx = int(np.argmin(dists))
        return self._road_ids[idx]

    def predict(self, x: float, y: float, heading_deg: float,
                direction: str = 'straight',
                n_hops: int = 3,
                prob_threshold: float = 0.0,
                direction_probs: Optional[dict] = None,
                ) -> list[dict]:
        """
        Predict multi-hop route from position (x, y) heading heading_deg.

        Args:
            x, y:            CARLA world coordinates (metres)
            heading_deg:     Current vehicle heading (degrees, 0=east, 90=south)
            direction:       Top-1 GoalClassifier prediction
            n_hops:          Number of junctions to traverse
            prob_threshold:  Minimum probability to include a branch (for multi-path)
            direction_probs: Full probability dict from GoalClassifier (optional,
                             used for probabilistic multi-path)

        Returns:
            List of waypoint dicts:
            {
                hop: int,
                road_id: str,
                x: float, y: float,           # CARLA coords of road endpoint
                lat: float, lon: float,        # lat/lon (None if no georef)
                heading_deg: float,
                direction_used: str,
                confidence: float,
            }
        """
        current_road_id = self.find_nearest_road(x, y, heading_deg)
        current_heading = heading_deg
        waypoints = []

        # The GoalClassifier signal applies to the FIRST real junction
        # (decision point) encountered -- not to loop iteration 0, which may
        # land on a non-junction connector segment first (common with
        # finely-split OSM/netconvert road networks). Track that separately
        # so the real signal isn't silently discarded.
        direction_applied = False

        for hop in range(n_hops):
            road = self._roads.get(current_road_id)
            if road is None:
                break

            # Find successor junction
            succ = road.get('successor', {})
            if succ.get('type') != 'junction':
                # Road ends at another road (no junction) -- nothing to
                # decide here, just follow the chain.
                next_road_id = succ.get('id')
                if next_road_id and next_road_id in self._roads:
                    nr = self._roads[next_road_id]
                    ex, ey = self._od_to_carla(*nr['end'])
                    new_hdg = -math.degrees(nr['heading_rad'])
                    lat, lon = self._to_latlon(ex, ey)
                    waypoints.append({
                        'hop': hop + 1,
                        'road_id': next_road_id,
                        'x': ex, 'y': ey,
                        'lat': lat, 'lon': lon,
                        'heading_deg': new_hdg,
                        'direction_used': direction if not direction_applied else self._hop1plus_direction,
                        'confidence': 1.0,
                    })
                    current_road_id = next_road_id
                    current_heading = new_hdg
                    continue
                break

            junction_id = succ.get('id')
            junction = self._junctions.get(junction_id)
            if junction is None:
                break

            # Decide which direction to apply at this real decision point.
            # First real junction: GoalClassifier prediction (real signal
            # from the observed approach trajectory). Every junction after
            # that: no such signal exists yet (vehicle hasn't approached it),
            # so fall back to the empirical prior measured from real
            # multi-hop sequences (see _DEFAULT_HOP1PLUS_PRIOR) rather than
            # assuming 'straight' outright -- that assumption is only right
            # 37% of the time on the data we measured it against.
            hop_direction = direction if not direction_applied else self._hop1plus_direction
            is_first_junction = not direction_applied
            direction_applied = True

            # Select outgoing road from junction
            outgoing_road_id = self._pick_outgoing_road(
                junction, current_road_id, current_heading, hop_direction
            )
            if outgoing_road_id is None:
                break

            out_road = self._roads[outgoing_road_id]
            ex, ey   = self._od_to_carla(*out_road['end'])
            new_hdg  = -math.degrees(out_road['heading_rad'])
            lat, lon = self._to_latlon(ex, ey)

            waypoints.append({
                'hop': hop + 1,
                'road_id': outgoing_road_id,
                'x': ex, 'y': ey,
                'lat': lat, 'lon': lon,
                'heading_deg': new_hdg,
                'direction_used': hop_direction,
                'confidence': (direction_probs.get(direction, 1.0) if (is_first_junction and direction_probs)
                               else self._hop1plus_confidence),
            })

            current_road_id = outgoing_road_id
            current_heading = new_hdg

        return waypoints

    def predict_probabilistic(self, x: float, y: float, heading_deg: float,
                               direction_probs: dict,
                               n_hops: int = 3,
                               min_prob: float = 0.15) -> list[dict]:
        """
        Run predict() for each direction with prob >= min_prob.
        Returns merged list of waypoints annotated with branch probability.
        """
        results = []
        for direction, prob in direction_probs.items():
            if prob < min_prob:
                continue
            hops = self.predict(x, y, heading_deg, direction=direction,
                                n_hops=n_hops,
                                direction_probs=direction_probs)
            for wp in hops:
                wp = dict(wp)
                wp['branch_prob'] = round(prob, 3)
                wp['branch_direction'] = direction
                results.append(wp)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_outgoing_road(self, junction: dict, incoming_road_id: str,
                             incoming_heading: float, direction: str) -> Optional[str]:
        """
        Select outgoing road from junction based on turn direction.

        Strategy:
          1. Find all connecting roads for this incoming road
          2. Compute each connecting road's heading
          3. Score by how close heading deviation is to the target angle
          4. Return best match within tolerance
        """
        target = _TURN_TARGET[direction]
        tol    = _TURN_TOLERANCE[direction]

        candidates = []
        for conn in junction.get('connections', []):
            if conn['incoming_road'] != incoming_road_id:
                continue
            croadid = conn['connecting_road']
            croad   = self._roads.get(croadid)
            if croad is None:
                continue
            # Use CARLA-convention heading (negated from OpenDRIVE)
            c_heading = -math.degrees(croad['heading_rad'])
            deviation = _heading_diff(c_heading, incoming_heading)
            score     = abs(deviation - target)
            candidates.append((score, croadid, deviation))

        if not candidates:
            # Fallback: no direct connection from this road, use all junction roads
            for conn in junction.get('connections', []):
                croadid = conn['connecting_road']
                croad   = self._roads.get(croadid)
                if croad is None:
                    continue
                c_heading = -math.degrees(croad['heading_rad'])
                deviation = _heading_diff(c_heading, incoming_heading)
                score     = abs(deviation - target)
                candidates.append((score, croadid, deviation))

        if not candidates:
            return None

        # Sort by score (best match first), filter by tolerance
        candidates.sort(key=lambda c: c[0])
        best_score, best_rid, best_dev = candidates[0]

        if best_score > tol:
            logger.debug("No road within tolerance (best dev=%.1f°, target=%.1f°)",
                         best_dev, target)
            # Still return best match rather than None
        return best_rid

    def _to_latlon(self, x: float, y: float) -> tuple[float, float]:
        if self._georef is None:
            return None, None
        try:
            return self._georef.carla_to_latlon(x, y)
        except Exception:
            return None, None
