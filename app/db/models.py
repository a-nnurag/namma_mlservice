"""ORM models for the ML service DB (nammakelsa_ml)."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase

try:
    from pgvector.sqlalchemy import Vector  # type: ignore[import]
    _HAS_PGVECTOR = True
except ImportError:
    _HAS_PGVECTOR = False
    Vector = None


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class Base(DeclarativeBase):
    pass


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    application_id = Column(UUID(as_uuid=True), nullable=True)
    session_id = Column(UUID(as_uuid=True), nullable=True)
    claimed_role = Column(String(50), nullable=True)
    has_workexp = Column(Boolean, default=False, nullable=False)
    has_degree = Column(Boolean, default=False, nullable=False)
    status = Column(String(30), nullable=False, default="PENDING", index=True)
    # PENDING | CHUNKS_RECEIVED | PROCESSING | DONE | FAILED
    retry_count = Column(Integer, default=0, nullable=False)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class PipelineResult(Base):
    __tablename__ = "pipeline_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    candidate_id = Column(UUID(as_uuid=True), nullable=False, index=True, unique=True)

    # Face validation
    face_consistency_score = Column(Float, nullable=True)
    same_person_score = Column(Float, nullable=True)
    face_missing_seconds = Column(Integer, nullable=True)
    face_verdict = Column(String(20), nullable=True)

    # Liveness
    liveness_score = Column(Float, nullable=True)
    blink_detected = Column(Boolean, nullable=True)
    head_moved = Column(Boolean, nullable=True)
    mouth_moved = Column(Boolean, nullable=True)
    liveness_verdict = Column(String(20), nullable=True)

    # Duplicate check
    is_duplicate = Column(Boolean, nullable=True)
    face_match_score = Column(Float, nullable=True)
    voice_match_score = Column(Float, nullable=True)
    duplicate_verdict = Column(String(20), nullable=True)
    matched_candidate_id = Column(UUID(as_uuid=True), nullable=True)

    # Fraud aggregate
    fraud_score = Column(Float, nullable=True)
    fraud_level = Column(String(20), nullable=True)  # NONE | LOW | MEDIUM | HIGH | CRITICAL
    fraud_flags = Column(JSONB, nullable=True)

    # Interview scoring
    domain_score = Column(Float, nullable=True)
    communication_score = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    transcript = Column(Text, nullable=True)
    interview_raw = Column(JSONB, nullable=True)

    # Work experience
    detected_skill = Column(String(100), nullable=True)
    skill_confidence = Column(Float, nullable=True)
    matches_claimed_role = Column(Boolean, nullable=True)
    workexp_verdict = Column(String(20), nullable=True)

    # Degree check
    is_valid_doc = Column(Boolean, nullable=True)
    extracted_degree = Column(String(200), nullable=True)
    extracted_institution = Column(String(200), nullable=True)
    extracted_year = Column(String(10), nullable=True)
    degree_confidence = Column(Float, nullable=True)

    # Final verdict
    composite_score = Column(Float, nullable=True)
    final_verdict = Column(String(30), nullable=True)
    verdict_reason = Column(Text, nullable=True)
    recommended_action = Column(String(20), nullable=True)  # PASS | MANUAL_REVIEW | REJECT

    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class CandidateEmbedding(Base):
    """Stores face + voice embeddings for duplicate detection during fraud pipeline."""
    __tablename__ = "candidate_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    embedding_type = Column(String(20), nullable=False)  # FACE | VOICE
    embedding_json = Column(JSONB, nullable=True)  # fallback if pgvector unavailable
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("candidate_id", "embedding_type", name="uq_candidate_embedding_type"),
    )

    if _HAS_PGVECTOR:
        face_embedding = Column(Vector(512), nullable=True)
        voice_embedding = Column(Vector(256), nullable=True)
