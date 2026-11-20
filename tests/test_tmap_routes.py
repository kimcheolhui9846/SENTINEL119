"""Contract and transport tests with synthetic coordinates and no external calls."""

import http.client
import io
import json
import os
import socket
import subprocess
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from backend.tmap_http import make_tmap_fetcher
from backend.tmap_routes import evaluate_routes


ROOT = Path(__file__).resolve().parents[1]
SCENE = [127.1, 37.5]
BASE = [127.2, 37.6]
HOSPITAL = [127.3, 37.7]


def sample(leg="vehicle_to_incident"):
    return {"leg": leg, "incident": SCENE, "candidates": [
        {"id": "A", "coordinates": BASE, "simulated_eta_min": 3},
        {"id": "B", "coordinates": HOSPITAL},
    ]}


def vendor(distance=1500, seconds=242, *geometries):
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"totalDistance": distance, "totalTime": seconds},
         "geometry": {"type": "Point", "coordinates": BASE}},
        *[{"type": "Feature", "properties": {}, "geometry": geometry} for geometry in geometries],
    ]}


class RouteContractTests(unittest.TestCase):
    def test_each_leg_requests_the_right_direction(self):
        for leg, expected in [
            ("vehicle_to_incident", [(tuple(BASE), tuple(SCENE)), (tuple(HOSPITAL), tuple(SCENE))]),
            ("incident_to_hospital", [(tuple(SCENE), tuple(BASE)), (tuple(SCENE), tuple(HOSPITAL))]),
        ]:
            with self.subTest(leg=leg):
                fetch = Mock(return_value=vendor())
                result = evaluate_routes(sample(leg), fetch_route=fetch)
                self.assertEqual([call.args for call in fetch.call_args_list], expected)
                self.assertEqual(result["leg"], leg)
                self.assertEqual([row["id"] for row in result["routes"]], ["A", "B"])
                for row in result["routes"]:
                    self.assertEqual((row["source"], row["distance_m"], row["duration_s"]), ("tmap", 1500, 242))
                    self.assertIsNone(row["reason"])
                    self.assertEqual(datetime.fromisoformat(row["observed_at"]).utcoffset().total_seconds(), 0)

    def test_geometry_preserves_features_parts_and_invalid_point_gaps(self):
        a, b, c = BASE, SCENE, HOSPITAL
        features = vendor(0, 0,
            {"type": "LineString", "coordinates": [a, b, [999, 0], c, a]},
            {"type": "MultiLineString", "coordinates": [[b, c], [a], [c, b]]})
        row = evaluate_routes(sample(), lambda *_: features)["routes"][0]
        self.assertEqual(row["path_segments"], [[a, b], [c, a], [b, c], [c, b]])
        self.assertEqual((row["distance_m"], row["duration_s"]), (0, 0))

    def test_mixed_failure_preserves_neighbor_and_provenance(self):
        fetch = Mock(side_effect=[TimeoutError("private sentinel"), vendor(800, 70)])
        a, b = evaluate_routes(sample(), fetch)["routes"]
        self.assertEqual((a["id"], a["source"], a["reason"], a["duration_s"]), ("A", "simulation", "timeout", 180))
        self.assertEqual((b["id"], b["source"], b["duration_s"]), ("B", "tmap", 70))
        self.assertIsNone(a["observed_at"])
        self.assertIsNone(a["distance_m"])
        self.assertEqual(a["path_segments"], [])
        self.assertNotIn("private sentinel", str(a))

    def test_offline_zero_is_distinct_from_unknown(self):
        values = sample()
        values["candidates"][0]["simulated_eta_min"] = 0
        with patch.object(socket, "getaddrinfo", side_effect=AssertionError("unexpected network")):
            a, b = evaluate_routes(values)["routes"]
        self.assertEqual((a["duration_s"], a["source"], a["reason"]), (0, "simulation", "offline"))
        self.assertEqual((b["duration_s"], b["source"], b["reason"]), (None, "unknown", "offline"))

    def test_invalid_coordinates_skip_only_affected_queries(self):
        for point in (None, [37, 127], [False, 37], [float("nan"), 37], [10**400, 37]):
            with self.subTest(point=point):
                values = sample()
                values["candidates"][0]["coordinates"] = point
                fetch = Mock(return_value=vendor())
                a, b = evaluate_routes(values, fetch)["routes"]
                self.assertEqual((a["reason"], b["source"]), ("invalid_candidate_coordinates", "tmap"))
                fetch.assert_called_once()
        values = sample()
        values["incident"] = None
        fetch = Mock()
        self.assertTrue(all(r["reason"] == "invalid_incident_coordinates" for r in evaluate_routes(values, fetch)["routes"]))
        fetch.assert_not_called()

    def test_invalid_summary_never_uses_leg_totals(self):
        no_summary = {"type": "FeatureCollection", "features": [{"properties": {"distance": 3, "time": 4}}]}
        wrong_properties = {"type": "FeatureCollection", "features": [{"properties": None}]}
        partial = {"type": "FeatureCollection", "features": [{"properties": {"totalTime": False}}]}
        for value, reason in [(vendor(True, 12), "malformed_response"), (vendor(1, 1.5), "malformed_response"),
                              (vendor(-1, 14), "malformed_response"), (no_summary, "missing_totals"),
                              (wrong_properties, "malformed_response"), (partial, "missing_totals")]:
            with self.subTest(value=value):
                a, b = evaluate_routes(sample(), lambda *_: value)["routes"]
                self.assertEqual(a["reason"], reason)
                self.assertEqual((a["source"], b["source"]), ("simulation", "unknown"))

    def test_rejects_invalid_batches_before_first_lookup(self):
        changes = [
            {"leg": "other"},
            {"candidates": [{"id": "A", "coordinates": BASE}, {"id": "A", "coordinates": SCENE}]},
            {"candidates": [{"id": "A", "coordinates": BASE}, {"id": "B", "simulated_eta_min": True}]},
            {"candidates": [{"id": str(i), "coordinates": BASE} for i in range(17)]},
        ]
        for change in changes:
            with self.subTest(change=change):
                fetch = Mock()
                with self.assertRaises(ValueError):
                    evaluate_routes({**sample(), **change}, fetch)
                fetch.assert_not_called()


class TransportTests(unittest.TestCase):
    def response(self, status, body):
        response = Mock()
        response.status = status
        response.read.side_effect = io.BytesIO(body).read
        return response

    def test_unusable_key_stays_offline_before_connecting(self):
        with patch.object(http.client, "HTTPSConnection") as connection:
            rows = evaluate_routes(sample(), make_tmap_fetcher("invalid\nkey"))["routes"]
            self.assertEqual([r["reason"] for r in rows], ["offline", "offline"])
            connection.assert_not_called()

    def test_connection_failure_preserves_candidates_and_closes_connections(self):
        for error, reason in [(TimeoutError("private_marker"), "timeout"), (OSError("private_marker"), "network_error")]:
            with self.subTest(reason=reason), patch.object(http.client, "HTTPSConnection") as cls:
                cls.return_value.request.side_effect = error
                rows = evaluate_routes(sample(), make_tmap_fetcher("synthetic-marker"))["routes"]
                self.assertEqual([r["reason"] for r in rows], [reason, reason])
                self.assertEqual([r["source"] for r in rows], ["simulation", "unknown"])
                self.assertEqual(cls.return_value.close.call_count, 2)
                self.assertNotIn("private_marker", json.dumps(rows))

    def test_fixed_https_endpoint_and_units(self):
        with patch.object(http.client, "HTTPSConnection") as cls:
            connection = cls.return_value
            connection.getresponse.return_value = self.response(200, json.dumps(vendor()).encode())
            payload = sample()
            payload["candidates"] = payload["candidates"][:1]
            row = evaluate_routes(payload, make_tmap_fetcher("synthetic-marker"))["routes"][0]
            self.assertEqual(row["source"], "tmap")
            positional, keywords = connection.request.call_args
            method, path = positional[:2]
            body = json.loads(keywords["body"] if "body" in keywords else positional[2])
            headers = keywords["headers"] if "headers" in keywords else positional[3]
            self.assertEqual(cls.call_args.args[0], "apis.openapi.sk.com")
            self.assertEqual((method, path), ("POST", "/tmap/routes?version=1"))
            self.assertEqual({k.lower(): v for k, v in headers.items()}["appkey"], "synthetic-marker")
            self.assertEqual([body[k] for k in ("startX", "startY", "endX", "endY")], BASE + SCENE)
            self.assertEqual((body["reqCoordType"], body["resCoordType"]), ("WGS84GEO", "WGS84GEO"))
            self.assertNotIn("synthetic-marker", json.dumps(row))
            connection.close.assert_called_once()

    def test_http_failures_never_forward_redirects_or_expose_key(self):
        cases = [(302, b"synthetic-marker", "http_error"), (429, b"synthetic-marker", "http_error"),
                 (200, b"x" * (2 * 1024 * 1024 + 1), "malformed_response"),
                 (200, b'{"unused":NaN}', "malformed_response"),
                 (200, json.dumps({**vendor(), "nested": None}).encode().replace(
                     b"null", b"[" * 64 + b"0" + b"]" * 64), "malformed_response")]
        for status, payload, reason in cases:
            with self.subTest(status=status, size=len(payload)), patch.object(http.client, "HTTPSConnection") as cls:
                conn = cls.return_value
                conn.getresponse.return_value = self.response(status, payload)
                values = sample()
                values["candidates"] = values["candidates"][:1]
                result = evaluate_routes(values, make_tmap_fetcher("synthetic-marker"))
                self.assertEqual(result["routes"][0]["reason"], reason)
                self.assertNotIn("synthetic-marker", json.dumps(result))
                self.assertEqual(cls.call_count, 1)
                self.assertEqual(conn.request.call_count, 1)
                conn.close.assert_called_once()


class CommandTests(unittest.TestCase):
    def invoke(self, data, *args):
        env = os.environ.copy()
        env.pop("TMAP_APP_KEY", None)
        return subprocess.run([sys.executable, "-B", "-m", "backend.tmap_routes", *args],
                              input=data, capture_output=True, cwd=ROOT, env=env, timeout=8)

    def test_default_and_live_without_key_keep_candidates(self):
        for args in ([], ["--live"]):
            with self.subTest(args=args):
                run = self.invoke(json.dumps(sample()).encode(), *args)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual([r["source"] for r in json.loads(run.stdout)["routes"]], ["simulation", "unknown"])

    def test_invalid_json_and_oversized_input_fail_without_echo(self):
        deep_envelope = json.dumps({**sample(), "nested": None}).encode().replace(
            b"null", b"[" * 64 + b"0" + b"]" * 64)
        for data in (deep_envelope, b'{"private_marker": NaN}', b" " * 65537, b"private_marker", b"\xff",
                     b"[" * 2000 + b"]" * 2000, b"1" * 4500, b'{"a":1e400}'):
            with self.subTest(length=len(data)):
                result = self.invoke(data)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(result.stdout)
                self.assertNotIn(b"private_marker", result.stderr)
                self.assertNotIn(b"Traceback", result.stderr)
