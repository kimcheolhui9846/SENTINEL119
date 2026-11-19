"""Drill rehearsal HTTP contract with synthetic geometry and model responses."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from importlib.util import find_spec
import json
import math
import os
from pathlib import Path
import random
from threading import Event
from unittest import TestCase
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_team_demo import MODEL_URL, reply_json, stream_response
from tests.test_drill_facilities import directory_result, facility, snapshot_result


OUTER = [[127, 37.5], [127.1, 37.5], [127.1, 37.6], [127, 37.6], [127, 37.5]]
HOLE = [[127.01, 37.51], [127.09, 37.51], [127.09, 37.59], [127.01, 37.59], [127.01, 37.51]]
SECOND = [[127.2, 37.5], [127.21, 37.5], [127.21, 37.51], [127.2, 37.51], [127.2, 37.5]]
DECOY = [[128, 37.5], [128.1, 37.5], [128.1, 37.6], [128, 37.6], [128, 37.5]]
LOCATION_BASIS = {
    "dataset_id": "71768", "city": "서울특별시", "district": None, "dong": None,
    "count": 30, "level": "collection_city",
    "zone": {"city": "서울특별시", "district": "성동구", "dong": "성수동1가"},
}
EMS_TURNS = [
    {"speaker": "수보요원", "text": "119입니다. 무슨 일이신가요?"},
    {"speaker": "신고자", "text": "성수동에서 사람이 갑자기 쓰러졌어요. 구급차를 보내 주세요."},
    {"speaker": "신고자", "text": "의식이 없고 숨을 전혀 쉬지 않아요. 심정지가 의심된다고 해서 옆사람이 심폐소생술을 하고 있어요."},
]


class DrillTests(TestCase):
    def setUp(self):
        self.assertIsNotNone(find_spec("backend.drill_router"), "drill API has not been registered")
        from backend.drill_router import make_drill_router
        self.make_drill_router = make_drill_router
        self.path = Mock(spec=Path)
        self.seongsu = {"type": "Feature", "properties": {"district": "성동구", "name": "성수동1가"},
                        "geometry": {"type": "MultiPolygon", "coordinates": [[OUTER, HOLE], [SECOND]]}}
        self.set_boundaries(self.seongsu)
        self.facilities = Mock(spec=Path)
        self.facilities.read_text.return_value = json.dumps(snapshot_result([
            facility("A201", .001), facility("A205", .002, category="종합병원", emergency="G001"),
        ]))
        self.location = Mock(spec=Path)
        self.location.read_text.return_value = json.dumps(LOCATION_BASIS)
        self.directory = Mock()
        self.directory.locations = AsyncMock(
            return_value=directory_result([], status="unconfigured", fetched_at=None))

    def set_boundaries(self, *features):
        self.path.read_text.return_value = json.dumps({"type": "FeatureCollection", "features": features})

    def connect(self, **kwargs):
        app = FastAPI()
        kwargs.setdefault("boundary_path", self.path)
        kwargs.setdefault("facilities_path", self.facilities)
        kwargs.setdefault("hospital_directory", self.directory)
        kwargs.setdefault("location_path", self.location)
        app.include_router(self.make_drill_router(**kwargs))
        return self.enterContext(TestClient(app))

    def create(self, scenario="ems", mode="simulation"):
        response = self.client.post("/api/drills", json={"scenario": scenario, "mode": mode})
        self.assertEqual(response.status_code, 201, response.text)
        state = response.json()
        self.url = "/api/drills/" + state["id"]
        return state

    def advance(self, expected_turn):
        response = self.client.post(self.url + "/turns", json={"expected_turn": expected_turn})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_caller_stays_inside_dong_and_outside_hole(self):
        self.client = self.connect(rng=random.Random(7))
        points = set()
        for _ in range(24):
            before = self.create()
            caller = before["caller"]
            self.assertEqual(caller["dong"], "성수동1가")
            self.assertEqual(caller["source"], "synthetic_in_dong")
            lat, lon = caller["lat"], caller["lon"]
            self.assertEqual((lat, lon), (round(lat, 4), round(lon, 4)))
            in_outer = 37.5 < lat < 37.6 and 127 < lon < 127.1
            in_hole = 37.51 <= lat <= 37.59 and 127.01 <= lon <= 127.09
            in_second = 37.5 < lat < 37.51 and 127.2 < lon < 127.21
            self.assertTrue((in_outer and not in_hole) or in_second, caller)
            self.assertEqual(self.advance(0)["caller"], caller)
            self.assertEqual(self.client.get(self.url).json()["caller"], caller)
            points.add((lat, lon))
        self.assertEqual(len(points), 24)

    def test_caller_uses_exact_dong_and_district_even_with_decoy_first(self):
        decoy = deepcopy(self.seongsu)
        decoy["properties"]["district"] = "광진구"
        decoy["geometry"] = {"type": "Polygon", "coordinates": [DECOY]}
        self.set_boundaries(decoy, self.seongsu)
        self.client = self.connect(rng=random.Random(7))
        caller = self.create()["caller"]
        self.assertEqual(caller["dong"], "성수동1가")
        self.assertTrue(127 < caller["lon"] < 127.21)

    def test_unsupported_or_ambiguous_rough_address_fails_closed(self):
        other = deepcopy(self.seongsu)
        other["properties"]["name"] = "성수동2가"
        without_district = deepcopy(self.seongsu)
        del without_district["properties"]["district"]
        for features in ((other,), (without_district,), (self.seongsu, self.seongsu)):
            with self.subTest(features=[feature["properties"] for feature in features]):
                self.set_boundaries(*features)
                response = self.connect().post("/api/drills", json={"scenario": "ems", "mode": "simulation"})
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json(), {"detail": "drill_boundary_unavailable"})

    def test_caller_does_not_land_on_the_edge_of_a_hole(self):
        rng = Mock()
        rng.uniform.side_effect = [127.09, 37.55, 127.205, 37.505]
        self.client = self.connect(rng=rng)
        caller = self.create()["caller"]
        self.assertEqual((caller["lon"], caller["lat"]), (127.205, 37.505))

    def test_caller_basis_distinguishes_collection_city_from_simulated_dong(self):
        self.client = self.connect()
        state = self.create()
        self.assertEqual(state["caller"]["location_basis"], LOCATION_BASIS)
        self.assertIsNone(state["caller"]["location_basis"]["dong"])
        self.assertEqual(state["caller"]["dong"], "성수동1가")

    def test_patrol_segments_stay_inside_dong_near_a_hole(self):
        rng = Mock()
        rng.uniform.side_effect = [127.0099, 37.55]
        self.client = self.connect(rng=rng)
        initial = self.create()
        self.advance(0)
        dispatched = self.advance(1)
        for unit in initial["units"] + dispatched["units"]:
            self.assertEqual(unit["route_source"], "synthetic_in_dong")
            for start, end in zip(unit["route"], unit["route"][1:]):
                for index in range(11):
                    ratio = index / 10
                    lat = start["lat"] + ratio * (end["lat"] - start["lat"])
                    lon = start["lon"] + ratio * (end["lon"] - start["lon"])
                    self.assertTrue(127 < lon < 127.1 and 37.5 < lat < 37.6)
                    self.assertFalse(127.01 <= lon <= 127.09 and 37.51 <= lat <= 37.59)

    def test_caller_near_shipped_boundary_still_gets_visible_patrol_routes(self):
        from backend.drill import BOUNDARY_PATH

        self.client = self.connect(rng=random.Random(314), boundary_path=BOUNDARY_PATH)
        state = self.create()
        lon_scale = math.cos(math.radians(state["caller"]["lat"]))
        for unit in state["units"]:
            length = sum(math.hypot(b["lat"] - a["lat"], (b["lon"] - a["lon"]) * lon_scale) * 111_200
                         for a, b in zip(unit["route"], unit["route"][1:]))
            with self.subTest(unit=unit["id"]):
                self.assertGreater(length, 30)

    def test_ems_requires_evidence_and_commits_only_one_resource(self):
        self.client = self.connect()
        initial = self.create()
        self.assertEqual(initial["turn"], 0)
        self.assertEqual(initial["model_status"], "simulation")
        self.assertIsNone(initial["model_id"])
        self.assertEqual(initial["turns"], [])
        self.assertTrue(all(unit["status"] == "patrol" for unit in initial["units"]))
        self.assertTrue({"1급 응급구조사", "2급 응급구조사", "간호사"} <= {
            member for unit in initial["units"] if unit["kind"] == "ambulance" for member in unit["crew"]
        })
        first = self.advance(0)
        self.assertEqual(first["dispatch"]["state"], "wait")
        self.assertEqual(first["subtype"]["state"], "wait")
        self.assertEqual(first["link"]["state"], "wait")
        second = self.advance(1)
        self.assertEqual(second["dispatch"]["category"], "ems")
        self.assertEqual(second["dispatch"]["evidence_turn"], 2)
        self.assertEqual(second["subtype"]["state"], "wait")
        self.assertEqual(second["link"]["state"], "wait")
        dispatched = [unit for unit in second["units"] if unit["status"] == "dispatched"]
        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0]["id"], second["dispatch"]["unit_id"])
        self.assertEqual(dispatched[0]["kind"], "ambulance")
        self.assertGreaterEqual(len(dispatched[0]["crew"]), 2)
        self.assertIn("AED", dispatched[0]["equipment"])
        self.assertEqual(dispatched[0]["route"][-1],
                         {axis: second["caller"][axis] for axis in ("lat", "lon")})
        third = self.advance(2)
        self.assertEqual(third["turns"], EMS_TURNS)
        self.assertEqual(third["dispatch"], second["dispatch"])
        self.assertEqual(third["units"], second["units"])
        self.assertEqual(third["subtype"], {"state": "committed", "code": "cardiac_arrest"})
        linked = next(h for h in third["hospitals"] if h["id"] == third["link"]["hospital_id"])
        self.assertTrue(linked["available"])
        self.assertIn("cardiac_arrest", linked["capabilities"])
        self.assertEqual(linked["id"], "A205")
        self.assertEqual(linked["location_source"], "nmc")
        self.assertEqual(linked["operational_source"], "simulation")

    def test_fire_dispatch_has_no_patient_links(self):
        self.client = self.connect()
        self.create(scenario="fire")
        self.advance(0)
        second = self.advance(1)
        self.assertEqual(second["dispatch"]["category"], "fire")
        selected = next(unit for unit in second["units"] if unit["id"] == second["dispatch"]["unit_id"])
        self.assertEqual(selected["kind"], "fire")
        third = self.advance(2)
        self.assertEqual(third["dispatch"], second["dispatch"])
        self.assertEqual(third["subtype"]["state"], "wait")
        self.assertEqual(third["link"]["state"], "wait")

    def test_no_eligible_unit_remains_waiting(self):
        from backend.drill import _resources
        def unavailable(caller):
            units = _resources(caller)
            for unit in units:
                unit["crew"] = ["운전원", "보조원"]
            return units

        self.client = self.connect(resources=unavailable)
        self.create()
        self.advance(0)
        self.advance(1)
        final = self.advance(2)
        self.assertEqual(final["dispatch"]["state"], "wait")
        self.assertEqual(final["link"]["state"], "wait")
        self.assertTrue(all(unit["status"] == "patrol" for unit in final["units"]))

    def test_no_eligible_hospital_keeps_only_the_link_waiting(self):
        self.facilities.read_text.return_value = json.dumps(snapshot_result([facility("A201", .001)]))
        self.client = self.connect()
        self.create()
        self.advance(0)
        self.advance(1)
        state = self.advance(2)
        self.assertEqual(state["dispatch"]["state"], "committed")
        self.assertEqual(state["subtype"]["state"], "committed")
        self.assertEqual(state["link"], {"state": "wait", "hospital_id": None})

    def test_duplicate_posts_commit_once(self):
        self.client = self.connect()
        self.create()
        self.advance(0)
        def duplicate(_index):
            return self.client.post(self.url + "/turns", json={"expected_turn": 1})

        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(duplicate, range(2)))
        self.assertEqual(sorted(r.status_code for r in responses), [200, 409])
        state = self.client.get(self.url).json()
        self.assertEqual(state["turn"], 2)
        self.assertEqual(len([u for u in state["units"] if u["status"] == "dispatched"]), 1)

    def test_at_capacity_rejects_creation_until_an_idle_session_expires(self):
        import backend.drill as drill
        clock = Mock(return_value=100)
        self.enterContext(patch.multiple(drill, _MAX_SESSIONS=1, _IDLE_SECONDS=10, monotonic=clock))
        self.client = self.connect()
        self.create()
        original_url = self.url
        response = self.client.post("/api/drills", json={"scenario": "ems", "mode": "simulation"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "drill_capacity"})
        clock.return_value = 111
        self.create()
        self.assertEqual(self.client.get(original_url).status_code, 404)

    def test_inflight_turn_blocks_eviction_and_duplicate_inference(self):
        import backend.drill as drill
        started, finish = Event(), Event()
        calls = []
        clock = Mock(return_value=100)

        async def waiting_model(turns, scenario, prior):
            calls.append(len(turns))
            started.set()
            while not finish.is_set():
                await asyncio.sleep(.005)
            return "test-pinned-model", {"ems": 1}, {"cardiac_arrest": 1}

        self.enterContext(patch.multiple(drill, _MAX_SESSIONS=1, _IDLE_SECONDS=10,
                                       monotonic=clock, infer_live=waiting_model))
        self.client = self.connect()
        initial = self.create(mode="live")
        path = self.url + "/turns"
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(self.client.post, path, json={"expected_turn": 0})
            try:
                self.assertTrue(started.wait(2))
                clock.return_value = 111
                capacity = self.client.post("/api/drills", json={"scenario": "ems", "mode": "simulation"})
                self.assertEqual(capacity.json(), {"detail": "drill_capacity"})
                self.assertEqual(self.client.get(self.url).json(), initial)
                duplicate = executor.submit(self.client.post, path, json={"expected_turn": 0})
            finally:
                finish.set()
            self.assertEqual(first.result().status_code, 200)
            self.assertEqual(duplicate.result().status_code, 409)
        self.assertEqual(calls, [1])

    def test_live_failure_and_identity_change_are_atomic_and_retryable(self):
        from backend.team_demo import _model_identity
        body = stream_response()
        sft = body["results"]["sft"]
        pin = _model_identity(sft["model"])
        sft["probabilities"]["dispatch"] = {"fire": .001, "rescue": .001, "ems": .997, "other": .001}
        sft["probabilities"]["ems_subtype"] = {
            "cardiac_arrest": .895, "other_ems": .105, "severe_illness": 0, "illness": 0,
            "injury": 0, "accident": 0, "pregnancy": 0, "poisoning": 0,
        }
        requests = []
        alternate = deepcopy(body)
        alternate["results"]["sft"]["model"]["checkpoint"]["sha256"]["decision_head.pt"] = "e" * 64
        alternate_pin = _model_identity(alternate["results"]["sft"]["model"])
        invalid = deepcopy(body)
        invalid["results"]["sft"]["probabilities"]["ems_subtype"]["other_ems"] = .20
        reply = [httpx.Response(502, content=b"secret-upstream-body"), body, alternate, body, invalid, body]

        def upstream(request):
            requests.append(json.loads(request.content))
            result = reply.pop(0)
            return result if isinstance(result, httpx.Response) else reply_json(result)

        real_client = httpx.AsyncClient
        self.enterContext(patch.object(httpx, "AsyncClient", side_effect=lambda **kwargs:
            real_client(transport=httpx.MockTransport(upstream), **kwargs)))
        env = {"ZORO_INTAKE_BASE_URL": MODEL_URL, "ZORO_INTAKE_MODEL_ID": pin}
        self.enterContext(patch.dict(os.environ, env))
        self.client = self.connect()
        pending = self.create(mode="live")
        self.assertEqual(pending["model_status"], "pending")
        self.assertIsNone(pending["model_id"])
        self.assertEqual(requests, [])
        for turn in range(3):
            previous = self.client.get(self.url).json()
            with patch.dict(os.environ, {"ZORO_INTAKE_MODEL_ID": alternate_pin if turn == 1 else pin}):
                failed = self.client.post(self.url + "/turns", json={"expected_turn": turn})
            self.assertEqual(failed.status_code, 503)
            self.assertEqual(failed.json(), {"detail": "live_model_unavailable"})
            self.assertEqual(self.client.get(self.url).json(), previous)
            state = self.advance(turn)
            self.assertEqual(state["model_id"], pin)
            if turn == 0:
                self.assertEqual(state["dispatch"]["state"], "wait")
        final = self.client.get(self.url).json()
        self.assertEqual(final["model_status"], "live")
        self.assertEqual(final["dispatch"]["state"], "committed")
        self.assertEqual(final["subtype"]["code"], "cardiac_arrest")
        self.assertEqual([len(r["turns"]) for r in requests], [1, 1, 2, 2, 3, 3])
        for request, prefix_size in zip(requests, (1, 1, 2, 2, 3, 3)):
            self.assertEqual(set(request), {"turns", "variants"})
            self.assertEqual(request["variants"], ["sft"])
            self.assertEqual(request["turns"], [
                {"speaker": 0 if turn["speaker"] == "수보요원" else 1, "text": turn["text"]}
                for turn in EMS_TURNS[:prefix_size]
            ])

    def test_bad_ids_and_versions_do_not_change_state(self):
        self.client = self.connect()
        state = self.create()
        self.assertEqual(self.client.get("/api/drills/nonexistent").status_code, 404)
        self.assertEqual(self.client.post(self.url + "/turns", json={"expected_turn": 1}).status_code, 409)
        self.assertEqual(self.client.post(self.url + "/turns", json={"expected_turn": True}).status_code, 422)
        self.assertEqual(self.client.get(self.url).json(), state)

    def test_public_locations_and_provenance_are_fixed_for_the_session(self):
        directory = Mock()
        directory.locations = AsyncMock(return_value=directory_result([
            facility("A201", .001), facility("A205", .002, emergency="G001"),
        ]))
        with patch("backend.drill.infer_live",
                   new=AsyncMock(return_value=("pinned-model", {"ems": 1}, {"cardiac_arrest": 1}))):
            self.client = self.connect(hospital_directory=directory)
            initial = self.create(mode="live")
            self.assertEqual(initial["hospital_source"]["status"], "live")
            self.assertEqual({hospital["id"] for hospital in initial["hospitals"]}, {"A201", "A205"})
            pending = self.advance(0)
            dispatch = self.advance(1)
            final = self.advance(2)
            self.assertEqual(final["link"], {"state": "committed", "hospital_id": "A205"})
            for update in (pending, dispatch, final, self.client.get(self.url).json()):
                self.assertEqual(update["hospital_source"], initial["hospital_source"])
                self.assertEqual(update["hospitals"], initial["hospitals"])
            directory.locations.assert_awaited_once_with("성동구")

    def test_missing_public_locations_never_create_hospital_markers_or_links(self):
        path = Mock(spec=Path)
        path.read_text.side_effect = FileNotFoundError
        self.client = self.connect(facilities_path=path)
        initial = self.create()
        self.assertEqual(initial["hospital_source"]["status"], "unavailable")
        self.assertEqual(initial["hospitals"], [])
        self.advance(0)
        self.advance(1)
        final = self.advance(2)
        self.assertEqual(final["dispatch"]["state"], "committed")
        self.assertEqual(final["subtype"]["state"], "committed")
        self.assertEqual(final["link"], {"state": "wait", "hospital_id": None})
