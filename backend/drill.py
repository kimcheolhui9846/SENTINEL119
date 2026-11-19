"""In-memory dispatch rehearsal for synthetic calls and resources."""

import asyncio
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from time import monotonic
from uuid import uuid4

from .drill_facilities import DrillFacilities
from .drill_model import infer_live


BOUNDARY_PATH = Path(__file__).with_name("data") / "drill-dongs.geojson"
LOCATION_PATH = BOUNDARY_PATH.with_name("drill-location.json")
_OPERATOR = "수보요원"
_CALLER = "신고자"
_INTRO = "119입니다. 무슨 일이신가요?"
_EMS_REQUEST = "성수동에서 사람이 갑자기 쓰러졌어요. 구급차를 보내 주세요."
_ARREST_REPORT = "의식이 없고 숨을 전혀 쉬지 않아요. 심정지가 의심된다고 해서 옆사람이 심폐소생술을 하고 있어요."
_FIRE_REQUEST = "성수동 건물에 불이 났어요. 연기와 불꽃이 보여요. 소방차를 보내 주세요."


@dataclass(frozen=True)
class RoughAddress:
    city: str
    district: str
    dong: str


@dataclass(frozen=True)
class DrillScript:
    address: RoughAddress
    turns: tuple[tuple[str, str], ...]


_SEONGSU_ADDRESS = RoughAddress(city="서울특별시", district="성동구", dong="성수동1가")
_SCRIPTS = {
    "ems": DrillScript(_SEONGSU_ADDRESS, (
        (_OPERATOR, _INTRO), (_CALLER, _EMS_REQUEST), (_CALLER, _ARREST_REPORT),
    )),
    "fire": DrillScript(_SEONGSU_ADDRESS, (
        (_OPERATOR, _INTRO), (_CALLER, _FIRE_REQUEST), (_CALLER, "불길이 번지고 있어요. 건물 밖으로 대피했어요."),
    )),
}
_MIN_CONFIDENCE = .85
_MAX_SAMPLING_ATTEMPTS = 4000
_MAX_SESSIONS = 128
_IDLE_SECONDS = 1800
_EMS_QUALIFICATIONS = {"1급 응급구조사", "간호사"}


class DrillError(Exception):
    def __init__(self, status, code):
        self.status = status
        super().__init__(code)


def _inside_ring(lon, lat, ring, *, include_boundary):
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        if (min(x1, x2) <= lon <= max(x1, x2) and min(y1, y2) <= lat <= max(y1, y2)
                and (lon - x1) * (y2 - y1) == (lat - y1) * (x2 - x1)):
            return include_boundary
        if (y1 > lat) != (y2 > lat) and lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _load_boundary(path, address):
    try:
        features = json.loads(path.read_text(encoding="utf-8"))["features"]
        matches = []
        for feature in features:
            properties = feature.get("properties") if isinstance(feature, dict) else None
            if (isinstance(properties, dict) and properties.get("district") == address.district
                    and properties.get("name") == address.dong):
                matches.append(feature)
        if len(matches) != 1:
            raise ValueError("Expected one boundary for the script address")
        geometry = matches[0]["geometry"]
        coordinates = geometry["coordinates"]
        geometry_type = geometry["type"]
        if geometry_type == "Polygon":
            polygons = [coordinates]
        elif geometry_type == "MultiPolygon":
            polygons = coordinates
        else:
            raise ValueError("Unsupported boundary type")
        if not polygons:
            raise ValueError("Missing boundary")
        for polygon in polygons:
            if not polygon:
                raise ValueError("Missing exterior")
            for ring in polygon:
                if (len(ring) < 4 or ring[0] != ring[-1]
                        or not all(-180 <= lon <= 180 and -90 <= lat <= 90 for lon, lat in ring)):
                    raise ValueError("Invalid ring")
        lons, lats = zip(*(point for polygon in polygons for point in polygon[0]))
        bounds = min(lons), min(lats), max(lons), max(lats)
        if bounds[0] == bounds[2] or bounds[1] == bounds[3]:
            raise ValueError("Empty boundary")
    except (OSError, ValueError, IndexError, KeyError, TypeError):
        raise DrillError(503, "drill_boundary_unavailable") from None
    return address.dong, polygons, bounds


def _caller(boundary, rng):
    name, polygons, (west, south, east, north) = boundary
    for _ in range(_MAX_SAMPLING_ATTEMPTS):
        lon = round(rng.uniform(west, east), 4)
        lat = round(rng.uniform(south, north), 4)
        if any(_inside_ring(lon, lat, exterior, include_boundary=False) and not any(
            _inside_ring(lon, lat, hole, include_boundary=True) for hole in holes
        ) for exterior, *holes in polygons):
            return {"lat": lat, "lon": lon, "dong": name, "source": "synthetic_in_dong"}
    raise DrillError(503, "drill_boundary_unavailable")


def _location_basis(path, address):
    try:
        basis = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(basis, dict)
            or basis.keys() != {"dataset_id", "city", "district", "dong", "count", "level", "zone"}
            or basis["dataset_id"] != "71768" or basis["zone"] != asdict(address)
            or basis["city"] != address.city or type(basis["count"]) is not int or basis["count"] < 5):
        return None
    if basis["level"] == "collection_city" and basis["district"] is None and basis["dong"] is None:
        return basis
    if (basis["level"] == "label_dong" and basis["district"] == address.district
            and basis["dong"] == address.dong):
        return basis
    return None


def _edge_clearance(lon, lat, first, second):
    dx, dy = second[0] - first[0], second[1] - first[1]
    x, y = lon - first[0], lat - first[1]
    length_squared = dx * dx + dy * dy
    ratio = max(0, min(1, (x * dx + y * dy) / length_squared)) if length_squared else 0
    inward_x, inward_y = x - ratio * dx, y - ratio * dy
    return math.hypot(inward_x, inward_y), inward_x, inward_y


def _patrol_disk(lon, lat, polygons):
    edges = [(first, second) for polygon in polygons for ring in polygon
             for first, second in zip(ring, ring[1:])]

    def nearest(x, y):
        return min(_edge_clearance(x, y, first, second) for first, second in edges)

    clearance, inward_x, inward_y = nearest(lon, lat)
    if 0 < clearance < .00065:
        bearing = math.atan2(inward_y, inward_x)
        angles = (0, -math.pi / 12, math.pi / 12, -math.pi / 6, math.pi / 6, -math.pi / 3, math.pi / 3)
        for displacement in (.0015, .001, .00065, .00055, .00035):
            for angle in angles:
                x = lon + displacement * math.cos(bearing + angle)
                y = lat + displacement * math.sin(bearing + angle)
                shifted_clearance, _, _ = nearest(x, y)
                # The ring-free disk must contain the caller as well as the smaller patrol disk.
                if shifted_clearance > max(math.hypot(x - lon, y - lat), clearance):
                    return {"lat": y, "lon": x}, .9 * shifted_clearance
    return {"lat": lat, "lon": lon}, .9 * clearance


def _keep_routes_inside_dong(units, caller, polygons):
    lon, lat = caller["lon"], caller["lat"]
    center, radius = _patrol_disk(lon, lat, polygons)
    for unit in units:
        offsets = [(point["lat"] - lat, point["lon"] - lon) for point in unit["route"]]
        farthest = max((math.hypot(y, x) for y, x in offsets), default=radius)
        scale = min(1, radius / farthest) if farthest else 1
        unit["route"] = [_point(center, y * scale, x * scale) for y, x in offsets]
        unit["route_source"] = "synthetic_in_dong"


def _point(caller, lat_offset=0, lon_offset=0):
    return {"lat": caller["lat"] + lat_offset, "lon": caller["lon"] + lon_offset}


def _resources(caller):
    return [
        {"id": "A1", "name": "구급 1호", "kind": "ambulance", "crew": ["2급 응급구조사"],
         "equipment": ["구급가방"], "status": "patrol",
         "route": [_point(caller, .0006, -.0008), _point(caller, .001, -.0014), _point(caller, .0013, -.0006)]},
        {"id": "A2", "name": "구급 2호", "kind": "ambulance",
         "crew": ["1급 응급구조사", "2급 응급구조사", "간호사"],
         "equipment": ["구급가방", "AED", "산소"], "status": "patrol",
         "route": [_point(caller, -.0011, -.0012), _point(caller, -.0016, -.0003), _point(caller, -.0011, .0004)]},
        {"id": "F1", "name": "펌프 1호", "kind": "fire", "crew": ["소방대원 1", "소방대원 2", "소방대원 3"],
         "equipment": ["펌프", "소방호스"], "status": "patrol",
         "route": [_point(caller, .0013, .0018), _point(caller, .0007, .0021), _point(caller, -.0004, .0018)]},
    ]


def _distance(point, caller):
    lon_scale = math.cos(math.radians(caller["lat"]))
    return (point["lat"] - caller["lat"]) ** 2 + ((point["lon"] - caller["lon"]) * lon_scale) ** 2


def _choose_unit(units, category, caller):
    kind, crew, equipment = {
        "ems": ("ambulance", 2, {"AED", "구급가방"}),
        "fire": ("fire", 3, {"펌프", "소방호스"}),
    }[category]
    eligible = (unit for unit in units if unit["kind"] == kind and unit["status"] == "patrol"
                and len(unit["crew"]) >= crew and equipment <= set(unit["equipment"]) and unit["route"]
                and (category != "ems" or _EMS_QUALIFICATIONS.intersection(unit["crew"])))
    return min(eligible, key=lambda unit: (_distance(unit["route"][0], caller), unit["id"]), default=None)


def _supported(scores, target):
    return scores.get(target, 0) >= _MIN_CONFIDENCE and all(
        score < scores[target] for candidate, score in scores.items() if candidate != target
    )


def _apply_decisions(state, dispatch_scores, subtype_scores):
    caller = state["caller"]
    evidence = {turn["text"] for turn in state["turns"] if turn["speaker"] == _CALLER}
    scenario = state["scenario"]
    request = _EMS_REQUEST if scenario == "ems" else _FIRE_REQUEST
    if state["dispatch"]["state"] == "wait" and request in evidence and _supported(dispatch_scores, scenario):
        unit = _choose_unit(state["units"], scenario, caller)
        if unit:
            unit.update(status="dispatched", route=[unit["route"][0], _point(caller)])
            state["dispatch"] = {"state": "committed", "category": scenario,
                                 "unit_id": unit["id"], "evidence_turn": 2}

    if state["dispatch"]["category"] != "ems" or _ARREST_REPORT not in evidence:
        return
    if _supported(subtype_scores, "cardiac_arrest"):
        state["subtype"] = {"state": "committed", "code": "cardiac_arrest"}
        eligible = (h for h in state["hospitals"] if h["available"] and "cardiac_arrest" in h["capabilities"])
        hospital = min(eligible, key=lambda h: (_distance(h, caller), h["id"]), default=None)
        if hospital:
            state["link"] = {"state": "committed", "hospital_id": hospital["id"]}


class DrillService:
    def __init__(self, boundary_path=BOUNDARY_PATH, rng=None, resources=_resources, *,
                 facilities=None, location_path=LOCATION_PATH):
        self._boundary_path = boundary_path
        self._location_path = location_path
        self._rng = rng if rng is not None else random.Random()
        self._resources = resources
        self._facilities = facilities if facilities is not None else DrillFacilities()
        self._states = {}
        self._locks = {}
        self._last_used = {}
        self._boundaries = {}
        self._pending = 0

    async def create(self, mode, scenario):
        now = monotonic()
        for drill_id, last_used in tuple(self._last_used.items()):
            if now - last_used >= _IDLE_SECONDS and not self._locks[drill_id].locked():
                del self._last_used[drill_id]
                del self._locks[drill_id]
                del self._states[drill_id]
        if len(self._states) + self._pending >= _MAX_SESSIONS:
            raise DrillError(503, "drill_capacity")
        script = _SCRIPTS[scenario]
        if script.address not in self._boundaries:
            self._boundaries[script.address] = _load_boundary(self._boundary_path, script.address)
        boundary = self._boundaries[script.address]
        caller = _caller(boundary, self._rng)
        caller["location_basis"] = _location_basis(self._location_path, script.address)
        units = self._resources(caller)
        _keep_routes_inside_dong(units, caller, boundary[1])
        self._pending += 1
        try:
            hospitals, hospital_source = await self._facilities.for_drill(mode, script.address.district, caller)
        finally:
            self._pending -= 1
        drill_id = uuid4().hex
        state = {
            "id": drill_id, "mode": mode, "scenario": scenario, "turn": 0,
            "total_turns": len(script.turns), "caller": caller, "turns": [],
            "units": units, "hospitals": hospitals, "hospital_source": hospital_source,
            "dispatch": {"state": "wait", "category": None, "unit_id": None, "evidence_turn": None},
            "subtype": {"state": "wait", "code": None},
            "link": {"state": "wait", "hospital_id": None},
            "model_id": None, "model_status": "pending" if mode == "live" else "simulation",
        }
        self._locks[drill_id] = asyncio.Lock()
        self._states[drill_id] = state
        self._last_used[drill_id] = monotonic()
        return state

    def get(self, drill_id):
        if drill_id not in self._states:
            raise DrillError(404, "drill_not_found")
        self._last_used[drill_id] = monotonic()
        return self._states[drill_id]

    async def advance(self, drill_id, expected_turn):
        self.get(drill_id)
        async with self._locks[drill_id]:
            state = self.get(drill_id)
            if state["turn"] != expected_turn:
                raise DrillError(409, "turn_conflict")
            if expected_turn == state["total_turns"]:
                raise DrillError(409, "drill_finished")
            draft = deepcopy(state)
            scenario = draft["scenario"]
            speaker, text = _SCRIPTS[scenario].turns[expected_turn]
            draft["turns"].append({"speaker": speaker, "text": text})
            draft["turn"] += 1
            if draft["mode"] == "live":
                try:
                    model_id, dispatch, subtype = await infer_live(draft["turns"], scenario, state["model_id"])
                finally:
                    self._last_used[drill_id] = monotonic()
                draft.update(model_id=model_id, model_status="live")
            else:
                dispatch, subtype = {scenario: 1}, {"cardiac_arrest": 1} if scenario == "ems" else {}
            _apply_decisions(draft, dispatch, subtype)
            self._states[drill_id] = draft
            return draft
