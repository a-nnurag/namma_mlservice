"""Celery task — Work experience analysis sub-task."""
from __future__ import annotations

import asyncio

from app.core.logging import get_logger
from app.tasks.celery_app import celery_app

log = get_logger(__name__)


@celery_app.task(
    name="ml.workexp_task",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def run_workexp_task(
    self,
    candidate_id: str,
    audio_path: str,
    claimed_role: str,
    language: str = "kn",
) -> dict:
    """
    Analyze work-experience audio.

    Returns dict with task + workexp_result keys.
    """
    log.bind(candidate_id=candidate_id, task_name="workexp_task")
    log.info("Work experience task started", candidate_id=candidate_id)

    try:
        result = asyncio.run(_run_workexp_async(candidate_id, audio_path, claimed_role, language))
        log.info("Work experience task complete", candidate_id=candidate_id)
        return result
    except Exception as exc:
        log.error("Work experience task failed", candidate_id=candidate_id, error=str(exc))
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


async def _run_workexp_async(
    candidate_id: str,
    audio_path: str,
    claimed_role: str,
    language: str,
) -> dict:
    from app.pipeline.workexp_analyzer import analyze_workexp

    workexp_result = await analyze_workexp(
        audio_path=audio_path,
        claimed_role=claimed_role,
        language=language,
        candidate_id=candidate_id,
    )
    return {"task": "workexp", "workexp_result": workexp_result}
