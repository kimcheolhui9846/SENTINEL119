"""HTTP boundary for the dispatch console."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .hospital_directory import DEFAULT_DISTRICT, SEOUL_DISTRICTS, HospitalDirectory
from .compare import router as compare_router
from .scenarios import list_scenarios
from .sessions import SessionError, SessionService


class CreateSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_id: Annotated[str, Field(min_length=1, max_length=64)]
    mode: Literal["simulation", "live"]


class SessionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["next", "text", "retry", "dispatch"]
    expected_version: Annotated[int, Field(strict=True, ge=0)]
    text: Annotated[str, Field(min_length=1, max_length=2000)] | None = None
    expected_vehicle_id: Annotated[str, Field(min_length=1)] | None = None
    expected_vehicle_revision: Annotated[int, Field(strict=True, ge=0)] | None = None

    @model_validator(mode="after")
    def check_action_fields(self):
        if self.action == "text":
            if self.text is None or not self.text.strip():
                raise ValueError("text is required for a text event")
            self.text = self.text.strip()
        elif self.text is not None:
            raise ValueError("text is only valid for a text event")
        if self.action != "dispatch" and self.model_fields_set & {"expected_vehicle_id", "expected_vehicle_revision"}:
            raise ValueError("vehicle recommendation proof is only valid for a dispatch event")
        return self


def create_app(inference=None) -> FastAPI:
    sessions = SessionService(inference=inference)
    hospitals = HospitalDirectory()

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            sessions.close()

    app = FastAPI(title="SENTINEL119", lifespan=lifespan)

    @app.exception_handler(SessionError)
    async def session_failure(_request: Request, error: SessionError):
        headers = {"Retry-After": "15"} if error.status == 503 else None
        return JSONResponse(status_code=error.status, content={"detail": error.code}, headers=headers)

    app.include_router(compare_router)

    @app.get("/api/scenarios")
    def scenarios():
        return {"scenarios": list_scenarios()}

    @app.get("/api/hospital-locations")
    async def hospital_locations(district: str = DEFAULT_DISTRICT):
        if district not in SEOUL_DISTRICTS:
            raise HTTPException(status_code=422, detail="invalid_district")
        return await hospitals.locations(district)

    @app.post("/api/sessions", status_code=201)
    def create_session(body: CreateSession):
        return sessions.create(body.scenario_id, body.mode)

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        return sessions.get(session_id)

    @app.post("/api/sessions/{session_id}/events", status_code=202)
    def apply_event(session_id: str, body: SessionEvent):
        return sessions.event(
            session_id, body.action, body.expected_version, text=body.text,
            expected_vehicle_id=body.expected_vehicle_id, expected_vehicle_revision=body.expected_vehicle_revision,
        )

    dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    index = dist / "index.html"
    if index.is_file():
        if (dist / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/", include_in_schema=False)
        def console():
            return FileResponse(index)

    return app


app = create_app()
