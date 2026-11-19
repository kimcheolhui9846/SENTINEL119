"""Evaluate independent vehicle and hospital route candidates."""

import argparse
import http.client
import json
import os
import sys
from datetime import UTC, datetime

from .tmap_http import RouteFetchError, make_tmap_fetcher, strict_json_loads


MAX_CANDIDATES = 16
MAX_INPUT_BYTES = 64 * 1024
LEGS = ("vehicle_to_incident", "incident_to_hospital")
FETCH_FAILURES = ("offline", "timeout", "network_error", "http_error", "malformed_response")


def _coordinates(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    longitude, latitude = value
    if (
        type(longitude) in (int, float)
        and type(latitude) in (int, float)
        and -180 <= longitude <= 180
        and -90 <= latitude <= 90
    ):
        return longitude, latitude
    return None


def _nonnegative_int(value):
    return type(value) is int and value >= 0


def _validate_envelope(payload):
    if not isinstance(payload, dict) or payload.get("leg") not in LEGS:
        raise ValueError("invalid route input")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) > MAX_CANDIDATES:
        raise ValueError("invalid route input")

    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("invalid route input")
        identifier = candidate.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier != identifier.strip()
            or identifier in seen
        ):
            raise ValueError("invalid route input")
        seen.add(identifier)
        minutes = candidate.get("simulated_eta_min")
        if minutes is None:
            continue
        if not _nonnegative_int(minutes):
            raise ValueError("invalid route input")
        # Multiplication can cross Python's decimal digit limit after JSON decoding.
        try:
            str(minutes * 60)
        except ValueError:
            raise ValueError("invalid route input") from None


def _line_segments(line):
    if not isinstance(line, (list, tuple)):
        return
    segment = []
    for point in line:
        coordinates = _coordinates(point)
        if coordinates is None:
            if len(segment) >= 2:
                yield segment
            segment = []
        else:
            segment.append(list(coordinates))
    if len(segment) >= 2:
        yield segment


def _path_segments(features):
    for feature in features:
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            continue
        coordinates = geometry.get("coordinates")
        kind = geometry.get("type")
        if kind == "LineString":
            yield from _line_segments(coordinates)
        elif kind == "MultiLineString" and isinstance(coordinates, (list, tuple)):
            for line in coordinates:
                yield from _line_segments(line)


def _parse_route(response):
    if not isinstance(response, dict) or response.get("type") != "FeatureCollection":
        return "malformed_response", {}
    features = response.get("features")
    if not isinstance(features, list):
        return "malformed_response", {}
    if not features:
        return "missing_totals", {}
    if not isinstance(features[0], dict):
        return "malformed_response", {}
    properties = features[0].get("properties", {})
    if not isinstance(properties, dict):
        return "malformed_response", {}
    if "totalDistance" not in properties or "totalTime" not in properties:
        return "missing_totals", {}
    distance = properties["totalDistance"]
    duration = properties["totalTime"]
    if (
        not _nonnegative_int(distance)
        or not _nonnegative_int(duration)
        or any(not isinstance(feature, dict) for feature in features)
    ):
        return "malformed_response", {}
    return None, {
        "source": "tmap",
        "distance_m": distance,
        "duration_s": duration,
        "path_segments": list(_path_segments(features)),
        "observed_at": datetime.now(UTC).isoformat(),
    }


def _fetch(fetch_route, start, end):
    try:
        response = fetch_route(start, end)
    except RouteFetchError as error:
        reason = error.reason
        return (reason if reason in FETCH_FAILURES else "malformed_response"), {}
    except TimeoutError:
        return "timeout", {}
    except (OSError, http.client.HTTPException):
        return "network_error", {}
    return _parse_route(response)


def evaluate_routes(payload, fetch_route=None):
    """Return ordered route rows, using an optional (start, end) lookup callable."""
    _validate_envelope(payload)
    leg = payload["leg"]
    incident = _coordinates(payload.get("incident"))
    routes = []

    for candidate in payload["candidates"]:
        point = _coordinates(candidate.get("coordinates"))
        minutes = candidate.get("simulated_eta_min")
        row = {
            "id": candidate["id"],
            "source": "simulation" if minutes is not None else "unknown",
            "distance_m": None,
            "duration_s": minutes * 60 if minutes is not None else None,
            "path_segments": [],
            "observed_at": None,
        }
        if incident is None:
            reason = "invalid_incident_coordinates"
        elif point is None:
            reason = "invalid_candidate_coordinates"
        elif fetch_route is None:
            reason = "offline"
        else:
            start, end = (point, incident) if leg == "vehicle_to_incident" else (incident, point)
            reason, evidence = _fetch(fetch_route, start, end)
            row.update(evidence)
        row["reason"] = reason
        routes.append(row)

    return {"leg": leg, "routes": routes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="enable TMAP using TMAP_APP_KEY")
    args = parser.parse_args()
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    try:
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError("invalid route input")
        payload = strict_json_loads(raw)
        key = os.environ.get("TMAP_APP_KEY", "") if args.live else ""
        fetch_route = make_tmap_fetcher(key) if key else None
        output = json.dumps(evaluate_routes(payload, fetch_route))
    except ValueError:
        print("invalid route input", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
