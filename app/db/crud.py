"""All DB operations for the ML service."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.core.logging import get_logger
from app.db.adapter import DatabaseAdapter
from app.db.models import CandidateEmbedding, PipelineResult, PipelineRun

log = get_logger(__name__)


# ── Pipeline Runs ──────────────────────────────────────────────────────────────

async def get_run_by_candidate(db: DatabaseAdapter, candidate_id: UUID) -> PipelineRun | None:
    return await db.get_by(PipelineRun, candidate_id=candidate_id)


async def get_run_by_id(db: DatabaseAdapter, run_id: UUID) -> PipelineRun | None:
    return await db.get(PipelineRun, run_id)


async def create_pipeline_run(
    db: DatabaseAdapter,
    candidate_id: UUID,
    application_id: UUID | None,
    session_id: UUID | None,
    claimed_role: str,
    has_workexp: bool,
    has_degree: bool,
) -> PipelineRun:
    run = PipelineRun(
        candidate_id=candidate_id,
        application_id=application_id,
        session_id=session_id,
        claimed_role=claimed_role,
        has_workexp=has_workexp,
        has_degree=has_degree,
        status="PENDING",
    )
    created = await db.create(run)
    log.info("Pipeline run created", candidate_id=str(candidate_id), run_id=str(created.id))
    return created


async def update_run_status(
    db: DatabaseAdapter, run: PipelineRun, status: str, error: str | None = None
) -> PipelineRun:
    fields: dict[str, Any] = {"status": status, "updated_at": datetime.now(tz=timezone.utc)}
    if error:
        fields["error"] = error
    return await db.update(run, **fields)


async def increment_retry(db: DatabaseAdapter, run: PipelineRun) -> PipelineRun:
    return await db.update(run, retry_count=run.retry_count + 1)


async def list_all_runs(db: DatabaseAdapter) -> list[PipelineRun]:
    stmt = select(PipelineRun).order_by(PipelineRun.created_at.desc())
    result = await db.execute_query(stmt)
    return list(result.scalars().all())


# ── Pipeline Results ───────────────────────────────────────────────────────────

async def get_result_by_candidate(
    db: DatabaseAdapter, candidate_id: UUID
) -> PipelineResult | None:
    return await db.get_by(PipelineResult, candidate_id=candidate_id)


async def upsert_result_fields(
    db: DatabaseAdapter, candidate_id: UUID, run_id: UUID | None, **fields: Any
) -> PipelineResult:
    existing = await db.get_by(PipelineResult, candidate_id=candidate_id)
    if existing:
        fields["updated_at"] = datetime.now(tz=timezone.utc)
        return await db.update(existing, **fields)
    result = PipelineResult(candidate_id=candidate_id, run_id=run_id, **fields)
    return await db.create(result)


# ── Embeddings ─────────────────────────────────────────────────────────────────

async def store_face_embedding(
    db: DatabaseAdapter, candidate_id: UUID, embedding: list[float]
) -> CandidateEmbedding:
    existing = await db.get_by(
        CandidateEmbedding, candidate_id=candidate_id, embedding_type="FACE"
    )
    if existing:
        return await db.update(existing, embedding_json=embedding)
    emb = CandidateEmbedding(
        candidate_id=candidate_id,
        embedding_type="FACE",
        embedding_json=embedding,
    )
    return await db.create(emb)


async def get_all_face_embeddings(
    db: DatabaseAdapter,
) -> list[CandidateEmbedding]:
    return await db.list_by(CandidateEmbedding, embedding_type="FACE")
