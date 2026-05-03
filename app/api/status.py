"""Status endpoint — backend poller calls GET /status/{candidate_id} every 30s."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.logging import get_logger
from app.db.adapter import DatabaseAdapter
from app.db.session import get_db
from app.db import crud

router = APIRouter(prefix="/status", tags=["status"])
log    = get_logger(__name__)


class StatusResponse(BaseModel):
    status:          str
    verdict:         str | None  = None
    composite_score: float | None = None
    error:           str | None  = None


@router.get("/{candidate_id}", response_model=StatusResponse)
async def get_pipeline_status(
    candidate_id: UUID,
    db: DatabaseAdapter = Depends(get_db),
) -> StatusResponse:
    """Return current pipeline status for a candidate."""
    run = await crud.get_run_by_candidate(db, candidate_id)
    if not run:
        raise HTTPException(status_code=404, detail="No pipeline run found for candidate")

    if run.status == "DONE":
        result = await crud.get_result_by_candidate(db, candidate_id)
        if result:
            return StatusResponse(
                status="DONE",
                verdict=result.final_verdict,
                composite_score=result.composite_score,
            )
        return StatusResponse(status="DONE")

    if run.status == "FAILED":
        return StatusResponse(status="FAILED", error=run.error)

    return StatusResponse(status="ML_PROCESSING")
