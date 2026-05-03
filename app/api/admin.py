"""Admin / ops routes — pipeline run management."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.logging import get_logger
from app.db.adapter import DatabaseAdapter
from app.db.session import get_db
from app.db import crud
from app.db.models import PipelineRun
from app.storage.s3 import build_storage_backend

router = APIRouter(prefix="/admin", tags=["admin"])
log    = get_logger(__name__)


class RunSummary(BaseModel):
    id:           str
    candidate_id: str
    status:       str
    claimed_role: str | None
    retry_count:  int
    error:        str | None
    created_at:   str
    updated_at:   str


def _to_summary(run: PipelineRun) -> RunSummary:
    return RunSummary(
        id=str(run.id),
        candidate_id=str(run.candidate_id),
        status=run.status,
        claimed_role=run.claimed_role,
        retry_count=run.retry_count,
        error=run.error,
        created_at=run.created_at.isoformat(),
        updated_at=run.updated_at.isoformat(),
    )


@router.get("/runs", response_model=list[RunSummary])
async def list_runs(db: DatabaseAdapter = Depends(get_db)) -> list[RunSummary]:
    runs = await crud.list_all_runs(db)
    log.info("Admin: listing runs", count=len(runs))
    return [_to_summary(r) for r in runs]


@router.get("/run/{run_id}", response_model=RunSummary)
async def get_run(
    run_id: UUID,
    db: DatabaseAdapter = Depends(get_db),
) -> RunSummary:
    run = await crud.get_run_by_id(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return _to_summary(run)


@router.delete("/run/{run_id}/cancel")
async def cancel_run(
    run_id: UUID,
    db: DatabaseAdapter = Depends(get_db),
) -> dict:
    run = await crud.get_run_by_id(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status in ("DONE", "FAILED"):
        raise HTTPException(status_code=409, detail=f"Run already in terminal state: {run.status}")
    updated = await crud.update_run_status(db, run, "FAILED", error="Cancelled by admin")
    log.info("Admin: run cancelled", run_id=str(run_id))
    return {"run_id": str(run_id), "status": updated.status}


class TriggerResponse(BaseModel):
    run_id:       str
    candidate_id: str
    status:       str


@router.post("/trigger/{candidate_id}", response_model=TriggerResponse)
async def trigger_pipeline(
    candidate_id: UUID,
    db: DatabaseAdapter = Depends(get_db),
) -> TriggerResponse:
    """Re-trigger the ML pipeline for a candidate using their existing stored media files."""
    cid_str = str(candidate_id)

    # Require an existing run so we have metadata (claimed_role, has_workexp, has_degree)
    existing_run = await crud.get_run_by_candidate(db, candidate_id)
    if not existing_run:
        raise HTTPException(status_code=404, detail="No pipeline run found for candidate")

    storage = build_storage_backend()

    video_key = f"{cid_str}/interview_video.mp4"
    audio_key = f"{cid_str}/interview_audio.wav"

    if not storage.exists(video_key):
        raise HTTPException(status_code=409, detail="interview_video.mp4 not found in storage — cannot re-trigger")
    if not storage.exists(audio_key):
        raise HTTPException(status_code=409, detail="interview_audio.wav not found in storage — cannot re-trigger")

    video_path = storage.get_path(video_key)
    audio_path = storage.get_path(audio_key)
    selfie_path = video_path  # pipeline handles first-frame extraction

    workexp_audio_path: str | None = None
    wv_key = f"{cid_str}/workexp_video.mp4"
    if existing_run.has_workexp and storage.exists(wv_key):
        workexp_audio_path = storage.get_path(wv_key)

    # Create a fresh run record so the status poller sees a new attempt
    new_run = await crud.create_pipeline_run(
        db=db,
        candidate_id=candidate_id,
        application_id=existing_run.application_id,
        session_id=existing_run.session_id,
        claimed_role=existing_run.claimed_role or "unknown",
        has_workexp=existing_run.has_workexp,
        has_degree=existing_run.has_degree,
    )
    await crud.update_run_status(db, new_run, "PROCESSING")
    await crud.increment_retry(db, existing_run)

    from app.tasks.pipeline_task import dispatch_pipeline
    await dispatch_pipeline(
        candidate_id=cid_str,
        run_id=str(new_run.id),
        claimed_role=existing_run.claimed_role or "unknown",
        has_workexp=existing_run.has_workexp,
        has_degree=existing_run.has_degree,
        video_path=video_path,
        audio_path=audio_path,
        selfie_path=selfie_path,
        workexp_audio_path=workexp_audio_path,
        degree_image_path=None,
    )

    log.info("Admin: pipeline re-triggered", candidate_id=cid_str, run_id=str(new_run.id))
    return TriggerResponse(
        run_id=str(new_run.id),
        candidate_id=cid_str,
        status="PROCESSING",
    )
