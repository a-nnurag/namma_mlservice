"""
Pipeline orchestrator.

Dispatches a Celery chord:
  group(fraud_task, workexp_task, interview_task) | verdict_task

The chord fires verdict_task only once all three parallel sub-tasks finish.
"""
from __future__ import annotations

import asyncio
from uuid import UUID

from celery import chord, group

from app.core.error_codes import MLErrorCode
from app.core.exceptions import PipelineError
from app.core.logging import get_logger

log = get_logger(__name__)


async def dispatch_pipeline(
    candidate_id: str,
    run_id: str,
    claimed_role: str,
    has_workexp: bool,
    has_degree: bool,
    video_path: str,
    audio_path: str,
    selfie_path: str,
    workexp_audio_path: str | None,
    degree_image_path: str | None,
    language: str = "kn",
    liveness_score: float = 0.5,
) -> str:
    """
    Build and send the Celery chord.  Returns the chord result ID.
    """
    from app.tasks.fraud_task     import run_fraud_task
    from app.tasks.workexp_task   import run_workexp_task
    from app.tasks.interview_task import run_interview_task
    from app.tasks.verdict_task   import run_verdict_task

    log.info(
        "Dispatching pipeline chord",
        candidate_id=candidate_id,
        run_id=run_id,
        has_workexp=has_workexp,
        has_degree=has_degree,
    )

    sub_tasks = [
        run_fraud_task.s(
            candidate_id=candidate_id,
            video_path=video_path,
            selfie_path=selfie_path,
            audio_path=audio_path,
        ),
        run_interview_task.s(
            candidate_id=candidate_id,
            audio_path=audio_path,
            claimed_role=claimed_role,
            language=language,
            liveness_score=liveness_score,
        ),
    ]

    if has_workexp and workexp_audio_path:
        sub_tasks.append(
            run_workexp_task.s(
                candidate_id=candidate_id,
                audio_path=workexp_audio_path,
                claimed_role=claimed_role,
                language=language,
            )
        )

    callback = run_verdict_task.s(
        candidate_id=candidate_id,
        run_id=run_id,
        claimed_role=claimed_role,
        has_workexp=has_workexp,
        has_degree=has_degree,
        degree_image_path=degree_image_path,
    )

    result = chord(group(sub_tasks))(callback)
    log.info("Pipeline chord dispatched", candidate_id=candidate_id, chord_id=result.id)
    return result.id
