"""Verdict retrieval routes."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.error_codes import MLErrorCode
from app.core.logging import get_logger
from app.db.adapter import DatabaseAdapter
from app.db.session import get_db
from app.db import crud

router = APIRouter(prefix="/verdict", tags=["verdict"])
log    = get_logger(__name__)


class VerdictResponse(BaseModel):
    candidate_id:      str
    run_id:            str | None
    status:            str
    composite_score:   float | None
    final_verdict:     str | None
    verdict_reason:    str | None
    recommended_action: str | None
    fraud_score:       float | None
    fraud_level:       str | None
    domain_score:      float | None
    communication_score: float | None
    confidence_score:  float | None
    liveness_score:    float | None
    is_duplicate:      bool | None
    workexp_verdict:   str | None
    is_valid_doc:      bool | None


@router.get("/{candidate_id}", response_model=VerdictResponse)
async def get_verdict(
    candidate_id: UUID,
    db: DatabaseAdapter = Depends(get_db),
) -> VerdictResponse:
    run = await crud.get_run_by_candidate(db, candidate_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found for candidate")

    if run.status not in ("DONE", "FAILED"):
        raise HTTPException(
            status_code=202,
            detail=f"Pipeline still running (status={run.status}). Try again later.",
        )

    result = await crud.get_result_by_candidate(db, candidate_id)

    log.info("Verdict fetched", candidate_id=str(candidate_id), status=run.status)

    return VerdictResponse(
        candidate_id=str(candidate_id),
        run_id=str(run.id),
        status=run.status,
        composite_score=result.composite_score if result else None,
        final_verdict=result.final_verdict if result else None,
        verdict_reason=result.verdict_reason if result else None,
        recommended_action=result.recommended_action if result else None,
        fraud_score=result.fraud_score if result else None,
        fraud_level=result.fraud_level if result else None,
        domain_score=result.domain_score if result else None,
        communication_score=result.communication_score if result else None,
        confidence_score=result.confidence_score if result else None,
        liveness_score=result.liveness_score if result else None,
        is_duplicate=result.is_duplicate if result else None,
        workexp_verdict=result.workexp_verdict if result else None,
        is_valid_doc=result.is_valid_doc if result else None,
    )
