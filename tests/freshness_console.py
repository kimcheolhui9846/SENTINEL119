"""Serve with python -m uvicorn tests.freshness_console:create_test_app --factory --host 127.0.0.1 --port 8024."""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

from backend.api import create_app
from backend.scenarios import get_frame


def create_test_app():
    app = create_app()
    lifespan = app.router.lifespan_context
    clock = Mock(return_value=datetime(2030, 1, 1, tzinfo=timezone.utc))

    def fault_frame(scenario_id, step, hospital_retry=False):
        if scenario_id == "outage" and step == 2:
            hospital_retry = False  # The synthetic provider stays timed out across retries.
        frame = get_frame(scenario_id, step, hospital_retry)
        for vehicle in frame["vehicles"]:
            if vehicle["id"] == "A":
                vehicle["expires_at"] = (clock() + timedelta(seconds=1)).isoformat()
        return frame

    @asynccontextmanager
    async def fixture_lifespan(application):
        with patch("backend.sessions._now", clock), patch("backend.sessions.get_frame", fault_frame):
            async with lifespan(application):
                yield

    app.router.lifespan_context = fixture_lifespan

    @app.post("/_test/expire-vehicle")
    def expire_vehicle():
        clock.return_value += timedelta(seconds=1)
        return {"now": clock().isoformat()}

    return app
