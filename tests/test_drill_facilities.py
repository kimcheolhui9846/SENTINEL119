"""Provenance and selection for public medical facility locations in a drill."""

import json
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock


FETCHED_AT = "2026-11-19T07:30:00+00:00"
CALLER = {"lat": 37.544, "lon": 127.054}


def facility(facility_id, offset, *, category="병원", emergency="G099"):
    return {
        "id": facility_id, "name": "공개 위치 테스트 " + facility_id,
        "address": "서울특별시 성동구 테스트로 1",
        "lat": CALLER["lat"], "lon": CALLER["lon"] + offset,
        "category": category, "emergency_code": emergency,
    }


def directory_result(items, *, district="성동구", status="ok", fetched_at=FETCHED_AT):
    return {"district": district, "source": "nmc", "status": status,
            "fetched_at": fetched_at, "items": items}


def snapshot_result(items, *, district="성동구", fetched_at=FETCHED_AT):
    return {"district": district, "fetched_at": fetched_at, "items": items}


class FacilityTests(IsolatedAsyncioTestCase):
    def adapter(self, live, snapshot):
        from backend.drill_facilities import DrillFacilities

        self.directory = Mock()
        self.directory.locations = AsyncMock(return_value=live)
        path = Mock(spec=Path)
        if snapshot is None:
            path.read_text.side_effect = FileNotFoundError
        else:
            path.read_text.return_value = json.dumps(snapshot)
        return DrillFacilities(directory=self.directory, snapshot_path=path)

    async def test_live_locations_keep_nmc_metadata_and_include_an_emergency_facility(self):
        emergency = facility("A205", .006, category="종합병원", emergency="G001")
        live = directory_result([
            facility("C101", .0001, category="의원", emergency="G002"),
            facility("A201", .001), facility("A202", .002), facility("A203", .003),
            emergency, {**facility("A206", .00001), "lat": float("nan")},
            facility("A201", .005),
        ])
        adapter = self.adapter(live, None)
        hospitals, source = await adapter.for_drill("live", "성동구", CALLER)
        self.directory.locations.assert_awaited_once_with("성동구")
        self.assertEqual(source, {"provider": "nmc", "status": "live", "fetched_at": FETCHED_AT})
        self.assertEqual([hospital["id"] for hospital in hospitals], ["A201", "A202", "A205"])
        self.assertEqual(hospitals[-1], {
            **emergency, "location_source": "nmc", "operational_source": "simulation",
            "operational_profile": "drill-v1", "capabilities": ["cardiac_arrest"], "available": True,
        })
        self.assertTrue(all("cardiac_arrest" not in hospital["capabilities"] for hospital in hospitals[:-1]))

    async def test_offline_snapshot_supplies_actual_fields_without_using_the_directory(self):
        row = facility("A201", .003)
        adapter = self.adapter(directory_result([]), snapshot_result([row]))
        hospitals, source = await adapter.for_drill("simulation", "성동구", CALLER)
        self.directory.locations.assert_not_awaited()
        self.assertEqual(source, {"provider": "nmc", "status": "snapshot", "fetched_at": FETCHED_AT})
        self.assertEqual(hospitals[0]["id"], "A201")
        self.assertEqual(hospitals[0]["lon"], row["lon"])
        self.assertNotIn("cardiac_arrest", hospitals[0]["capabilities"])

    async def test_stale_or_empty_live_result_falls_back_to_a_verified_snapshot(self):
        row = facility("A205", .006, emergency="G001")
        for live in (directory_result([], status="unconfigured", fetched_at=None),
                     directory_result([facility("A201", .001)], status="stale"),
                     directory_result([])):
            with self.subTest(status=live["status"]):
                adapter = self.adapter(live, snapshot_result([row]))
                hospitals, source = await adapter.for_drill("live", "성동구", CALLER)
                self.assertEqual([h["id"] for h in hospitals], ["A205"])
                self.assertEqual(source["status"], "snapshot")
                self.assertEqual(source["fetched_at"], FETCHED_AT)

    async def test_unverified_snapshot_never_creates_hospital_markers(self):
        row = facility("A201", .003)
        for snapshot in (None, snapshot_result([row], district="광진구"),
                         snapshot_result([row], fetched_at=None), snapshot_result([row], fetched_at="today")):
            with self.subTest(snapshot=snapshot):
                adapter = self.adapter(directory_result([], status="unavailable", fetched_at=None), snapshot)
                hospitals, source = await adapter.for_drill("live", "성동구", CALLER)
                self.assertEqual(hospitals, [])
                self.assertEqual(source, {"provider": "nmc", "status": "unavailable", "fetched_at": None})
