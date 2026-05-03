"""
Celery chord callback — Final Verdict Task.

Called automatically by Celery once all sub-tasks (fraud, workexp, interview)
in the chord finish.  Writes the aggregated result to the DB and marks
the pipeline run as DONE.
"""
from __future__ import annotations

import asyncio
from uuid import UUID

from app.core.logging import get_logger
from app.tasks.celery_app import celery_app

log = get_logger(__name__)


@celery_app.task(name="ml.verdict_task", bind=True)
def run_verdict_task(
    self,
    sub_results: list[dict],
    candidate_id: str,
    run_id: str,
    claimed_role: str,
    has_workexp: bool,
    has_degree: bool,
    degree_image_path: str | None = None,
) -> dict:
    """
    Chord callback.  sub_results contains one dict per sub-task
    (in the order they were added to the chord group).

    Each dict has a "task" key: "fraud" | "workexp" | "interview".
    """
    log.bind(candidate_id=candidate_id, task_name="verdict_task")
    log.info("Verdict task started", candidate_id=candidate_id, sub_count=len(sub_results))

    try:
        result = asyncio.run(
            _run_verdict_async(
                sub_results=sub_results,
                candidate_id=candidate_id,
                run_id=run_id,
                claimed_role=claimed_role,
                has_workexp=has_workexp,
                has_degree=has_degree,
                degree_image_path=degree_image_path,
            )
        )
        log.info("Verdict task complete", candidate_id=candidate_id, verdict=result.get("final_verdict"))
        return result
    except Exception as exc:
        log.error("Verdict task failed", candidate_id=candidate_id, error=str(exc), exc_info=True)
        asyncio.run(_mark_failed(candidate_id, run_id, str(exc)))
        raise


async def _run_verdict_async(
    sub_results: list[dict],
    candidate_id: str,
    run_id: str,
    claimed_role: str,
    has_workexp: bool,
    has_degree: bool,
    degree_image_path: str | None,
) -> dict:
    from app.db.session import get_db_context
    from app.db import crud
    from app.pipeline.verdict_engine import compute_final_verdict
    from app.pipeline.degree_checker import verify_degree

    # Unpack sub-task results
    fraud_result     = {}
    interview_result = {}
    workexp_result   = {}

    for sub in sub_results:
        task_name = sub.get("task", "")
        if task_name == "fraud":
            fraud_result     = sub.get("fraud_result", {})
            liveness         = sub.get("liveness_result", {})
            interview_result.setdefault("liveness_score", liveness.get("liveness_score", 0.5))
        elif task_name == "interview":
            interview_result.update(sub.get("interview_result", {}))
        elif task_name == "workexp":
            workexp_result = sub.get("workexp_result", {})

    # Degree check (synchronous OCR — run in same async context)
    degree_result: dict = {"is_valid_doc": False, "degree_confidence": 0.0}
    if has_degree and degree_image_path:
        try:
            loop = asyncio.get_running_loop()
            degree_result = await loop.run_in_executor(
                None, verify_degree, degree_image_path, candidate_id
            )
        except Exception as exc:
            log.warning("Degree check failed", candidate_id=candidate_id, error=str(exc))

    verdict = compute_final_verdict(
        fraud_result=fraud_result,
        interview_result=interview_result,
        workexp_result=workexp_result,
        degree_result=degree_result,
        has_workexp=has_workexp,
        has_degree=has_degree,
        candidate_id=candidate_id,
    )

    async with get_db_context() as db:
        run = await crud.get_run_by_id(db, UUID(run_id))
        if run:
            await crud.update_run_status(db, run, "DONE")

        fraud_sub      = sub_results[0] if sub_results else {}
        face_r         = fraud_sub.get("face_result", {})
        liveness_r     = fraud_sub.get("liveness_result", {})
        duplicate_r    = fraud_sub.get("duplicate_result", {})

        await crud.upsert_result_fields(
            db=db,
            candidate_id=UUID(candidate_id),
            run_id=UUID(run_id),
            # Face
            face_consistency_score=face_r.get("face_consistency_score"),
            same_person_score=face_r.get("same_person_score"),
            face_missing_seconds=face_r.get("face_missing_seconds"),
            face_verdict=face_r.get("verdict"),
            # Liveness
            liveness_score=liveness_r.get("liveness_score"),
            blink_detected=liveness_r.get("blink_detected"),
            head_moved=liveness_r.get("head_moved"),
            mouth_moved=liveness_r.get("mouth_moved"),
            liveness_verdict=liveness_r.get("verdict"),
            # Duplicate
            is_duplicate=duplicate_r.get("is_duplicate"),
            face_match_score=duplicate_r.get("face_match", {}).get("score"),
            voice_match_score=duplicate_r.get("voice_match", {}).get("score"),
            duplicate_verdict=duplicate_r.get("verdict"),
            matched_candidate_id=duplicate_r.get("matched_candidate_id"),
            # Fraud aggregate
            fraud_score=fraud_result.get("fraud_score"),
            fraud_level=fraud_result.get("fraud_level"),
            fraud_flags=fraud_result.get("flags"),
            # Interview
            domain_score=interview_result.get("domain_score"),
            communication_score=interview_result.get("communication_score"),
            confidence_score=interview_result.get("confidence_score"),
            transcript=interview_result.get("transcript"),
            interview_raw=interview_result,
            # Work exp
            detected_skill=workexp_result.get("detected_skill"),
            skill_confidence=workexp_result.get("skill_confidence"),
            matches_claimed_role=workexp_result.get("matches_claimed_role"),
            workexp_verdict=workexp_result.get("workexp_verdict"),
            # Degree
            is_valid_doc=degree_result.get("is_valid_doc"),
            extracted_degree=degree_result.get("extracted_degree"),
            extracted_institution=degree_result.get("extracted_institution"),
            extracted_year=degree_result.get("extracted_year"),
            degree_confidence=degree_result.get("degree_confidence"),
            # Final
            composite_score=verdict["composite_score"],
            final_verdict=verdict["final_verdict"],
            verdict_reason=verdict["verdict_reason"],
            recommended_action=verdict["recommended_action"],
        )

    # Notify backend SSE subscribers that ML processing is complete.
    # application_id == candidate_id in the meta message — they are the same value
    # passed through from the Kafka meta payload.
    try:
        from app.pubsub import publish_ml_result
        publish_ml_result(
            application_id=candidate_id,
            status=verdict["final_verdict"],
        )
        log.info("Published ML result to Redis pub/sub", candidate_id=candidate_id)
    except Exception as exc:
        # Non-fatal — backend will fall back to polling /application/status
        log.warning("Failed to publish ML result to Redis pub/sub", candidate_id=candidate_id, error=str(exc))

    return verdict


async def _mark_failed(candidate_id: str, run_id: str, error: str) -> None:
    try:
        from app.db.session import get_db_context
        from app.db import crud
        async with get_db_context() as db:
            run = await crud.get_run_by_id(db, UUID(run_id))
            if run:
                await crud.update_run_status(db, run, "FAILED", error=error[:500])
    except Exception as exc:
        log.error("Failed to mark run as FAILED", candidate_id=candidate_id, error=str(exc))
