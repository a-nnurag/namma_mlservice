"""
Celery task — Fraud detection sub-task.

Runs face validation + liveness detection + duplicate check in sequence.
Returns a combined fraud dict for the verdict task chord callback.
"""
from __future__ import annotations

import asyncio
from uuid import UUID

from app.core.logging import get_logger
from app.tasks.celery_app import celery_app

log = get_logger(__name__)


@celery_app.task(
    name="ml.fraud_task",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def run_fraud_task(
    self,
    candidate_id: str,
    video_path: str,
    selfie_path: str,
    audio_path: str | None,
) -> dict:
    """
    Execute fraud analysis pipeline.

    Args:
        candidate_id : UUID string
        video_path   : local path to interview video
        selfie_path  : local path to candidate selfie
        audio_path   : local path to interview audio (optional)

    Returns dict with keys:
        face_result, liveness_result, duplicate_result, fraud_result
    """
    log.bind(candidate_id=candidate_id, task_name="fraud_task")
    log.info("Fraud task started", candidate_id=candidate_id)

    try:
        result = asyncio.run(_run_fraud_async(candidate_id, video_path, selfie_path, audio_path))
        log.info("Fraud task complete", candidate_id=candidate_id)
        return result
    except Exception as exc:
        log.error("Fraud task failed", candidate_id=candidate_id, error=str(exc), exc_info=True)
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


async def _run_fraud_async(
    candidate_id: str,
    video_path: str,
    selfie_path: str,
    audio_path: str | None,
) -> dict:
    from app.db.session import get_db_context
    from app.pipeline.face_validator import validate_face_in_video
    from app.pipeline.liveness_detector import detect_liveness
    from app.pipeline.duplicate_checker import check_duplicate
    from app.pipeline.fraud_aggregator import compute_fraud_score

    face_result      = validate_face_in_video(video_path, candidate_id=candidate_id)
    liveness_result  = detect_liveness(video_path, candidate_id=candidate_id)

    async with get_db_context() as db:
        duplicate_result = await check_duplicate(
            candidate_id=candidate_id,
            db=db,
            image_input=selfie_path,
            audio_path=audio_path,
        )

    fraud_result = compute_fraud_score(
        face_result=face_result,
        liveness_result=liveness_result,
        duplicate_result=duplicate_result,
        candidate_id=candidate_id,
    )

    return {
        "task":             "fraud",
        "face_result":      face_result,
        "liveness_result":  liveness_result,
        "duplicate_result": duplicate_result,
        "fraud_result":     fraud_result,
    }
