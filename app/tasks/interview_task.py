"""Celery task — Interview scoring sub-task."""
from __future__ import annotations

import asyncio

from app.core.logging import get_logger
from app.tasks.celery_app import celery_app

log = get_logger(__name__)


@celery_app.task(
    name="ml.interview_task",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def run_interview_task(
    self,
    candidate_id: str,
    audio_path: str,
    claimed_role: str,
    language: str = "kn",
    liveness_score: float = 0.5,
) -> dict:
    """
    Transcribe interview audio and score domain / communication / confidence.

    Returns dict with task + interview_result keys.
    """
    log.bind(candidate_id=candidate_id, task_name="interview_task")
    log.info("Interview task started", candidate_id=candidate_id)

    try:
        result = asyncio.run(
            _run_interview_async(candidate_id, audio_path, claimed_role, language, liveness_score)
        )
        log.info("Interview task complete", candidate_id=candidate_id)
        return result
    except Exception as exc:
        log.error("Interview task failed", candidate_id=candidate_id, error=str(exc))
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


async def _run_interview_async(
    candidate_id: str,
    audio_path: str,
    claimed_role: str,
    language: str,
    liveness_score: float,
) -> dict:
    from app.pipeline.transcriber import transcribe_audio
    from app.pipeline.interview_scorer import build_interview_scorer

    transcription = transcribe_audio(audio_path, language=language, candidate_id=candidate_id)
    transcript    = transcription["text"]

    scorer = build_interview_scorer()
    scores = await scorer.score(
        transcript=transcript,
        claimed_role=claimed_role,
        candidate_id=candidate_id,
    )

    interview_result = {
        "transcript":          transcript,
        "domain_score":        scores["domain_score"],
        "communication_score": scores["communication_score"],
        "confidence_score":    scores["confidence_score"],
        "rationale":           scores.get("rationale", ""),
        "liveness_score":      liveness_score,
    }
    return {"task": "interview", "interview_result": interview_result}
