"""Opt-in HTTP routes for the dispatch rehearsal."""

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .drill import BOUNDARY_PATH, LOCATION_PATH, DrillError, DrillService, _resources
from .drill_facilities import FACILITIES_PATH, DrillFacilities
from .inference import InferenceError


class CreateDrill(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["live", "simulation"]
    scenario: Literal["ems", "fire"]


class AdvanceDrill(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_turn: Annotated[int, Field(strict=True, ge=0)]


def make_drill_router(*, boundary_path=BOUNDARY_PATH, rng=None, resources=_resources,
                      hospital_directory=None, facilities_path=FACILITIES_PATH, location_path=LOCATION_PATH):
    router = APIRouter(prefix="/api/drills")
    facilities = DrillFacilities(hospital_directory, facilities_path)
    drills = DrillService(boundary_path, rng, resources, facilities=facilities, location_path=location_path)

    @router.post("", status_code=201)
    async def create_drill(body: CreateDrill):
        try:
            return await drills.create(body.mode, body.scenario)
        except DrillError as error:
            raise HTTPException(error.status, str(error)) from None

    @router.get("/{drill_id}")
    async def get_drill(drill_id: str):
        try:
            return drills.get(drill_id)
        except DrillError as error:
            raise HTTPException(error.status, str(error)) from None

    @router.post("/{drill_id}/turns")
    async def advance_drill(drill_id: str, body: AdvanceDrill):
        try:
            return await drills.advance(drill_id, body.expected_turn)
        except DrillError as error:
            raise HTTPException(error.status, str(error)) from None
        except InferenceError:
            raise HTTPException(503, "live_model_unavailable") from None

    return router
