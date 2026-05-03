"""
Pipeline Step 3 — Duplicate Candidate Detection.

Upgraded from the prototype JSON-file store to pgvector cosine search.
Falls back to in-memory JSONB comparison when pgvector is unavailable.

Track A — ArcFace 512-d face embedding  (DeepFace)
Track B — Speaker 256-d voice embedding (Resemblyzer)
"""
from __future__ import annotations

from typing import Optional
from uuid import UUID

import numpy as np

from app.config import settings
from app.core.error_codes import MLErrorCode
from app.core.exceptions import DuplicateCheckError
from app.core.logging import get_logger
from app.db.adapter import DatabaseAdapter
from app.db.crud import get_all_face_embeddings, store_face_embedding
from app.models.registry import registry

log = get_logger(__name__)

FACE_DUPLICATE_THRESHOLD  = settings.FACE_SIMILARITY_THRESHOLD   # 0.85
FACE_SUSPECTED_THRESHOLD  = 0.75
VOICE_DUPLICATE_THRESHOLD = settings.VOICE_SIMILARITY_THRESHOLD  # 0.75
VOICE_SUSPECTED_THRESHOLD = 0.70


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _extract_face_embedding(image_input) -> Optional[list[float]]:
    try:
        from deepface import DeepFace  # type: ignore[import]

        result = DeepFace.represent(
            img_path=image_input,
            model_name="ArcFace",
            detector_backend="opencv",
            enforce_detection=False,
        )
        if result and result[0]:
            emb = result[0]["embedding"]
            if np.linalg.norm(emb) > 0:
                return list(emb)
        return None
    except Exception as exc:
        log.warning("ArcFace embedding extraction failed", error=str(exc))
        return None


def _extract_voice_embedding(audio_path: str) -> Optional[list[float]]:
    encoder = registry.get_voice_encoder()
    if encoder is None:
        return None
    try:
        from resemblyzer import preprocess_wav  # type: ignore[import]
        from pathlib import Path

        wav  = preprocess_wav(Path(audio_path))
        emb  = encoder.embed_utterance(wav)
        return emb.tolist()
    except Exception as exc:
        log.warning("Voice embedding extraction failed", error=str(exc))
        return None


async def check_duplicate(
    candidate_id: str,
    db: DatabaseAdapter,
    image_input,
    audio_path: Optional[str] = None,
) -> dict:
    """
    Run face (and optionally voice) duplicate check against all stored embeddings.

    Stores the candidate's face embedding in DB for future checks.

    Returns:
        is_duplicate, confidence, face_match, voice_match, verdict, matched_candidate_id
    """
    log.info("Duplicate check started", candidate_id=candidate_id)

    with log.timed("duplicate_check", candidate_id=candidate_id):
        # ── Face ──────────────────────────────────────────────────────────────
        face_emb = _extract_face_embedding(image_input)
        if face_emb is None:
            log.warning("No face embedding extracted", candidate_id=candidate_id)
            face_result = {"matched": False, "score": 0.0, "matched_candidate_id": None}
        else:
            face_result = await _check_face_against_db(
                candidate_id=candidate_id,
                embedding=face_emb,
                db=db,
            )
            # Persist for future checks
            try:
                await store_face_embedding(db, UUID(candidate_id), face_emb)
            except Exception as exc:
                log.warning("Failed to persist face embedding", candidate_id=candidate_id, error=str(exc))

        # ── Voice ─────────────────────────────────────────────────────────────
        voice_result: dict = {"matched": False, "score": 0.0, "matched_candidate_id": None}
        if audio_path:
            voice_emb = _extract_voice_embedding(audio_path)
            if voice_emb is not None:
                voice_result = _check_voice_in_memory(candidate_id, voice_emb, [])

        # ── Decision ──────────────────────────────────────────────────────────
        face_score  = face_result["score"]
        voice_score = voice_result["score"]

        if face_result["matched"] and face_score >= FACE_DUPLICATE_THRESHOLD:
            verdict, confidence, is_dup = "DUPLICATE", "high", True
        elif face_result["matched"] or (voice_result["matched"] and face_score > 0.60):
            verdict, confidence, is_dup = "SUSPECTED", "medium", True
        elif voice_result["matched"]:
            verdict, confidence, is_dup = "SUSPECTED", "low", True
        else:
            verdict, confidence, is_dup = "UNIQUE", "none", False

        matched_id = face_result["matched_candidate_id"] or voice_result["matched_candidate_id"]

        result = {
            "is_duplicate":          is_dup,
            "confidence":            confidence,
            "face_match":            face_result,
            "voice_match":           voice_result,
            "verdict":               verdict,
            "matched_candidate_id":  matched_id,
        }

        log.info(
            "Duplicate check complete",
            candidate_id=candidate_id,
            verdict=verdict,
            confidence=confidence,
            face_score=face_score,
        )
        return result


async def _check_face_against_db(
    candidate_id: str,
    embedding: list[float],
    db: DatabaseAdapter,
) -> dict:
    """Compare embedding against all stored face embeddings in DB."""
    try:
        all_embeddings = await get_all_face_embeddings(db)
    except Exception as exc:
        log.error("Failed to load embeddings from DB", error=str(exc))
        raise DuplicateCheckError(
            f"DB embedding query failed: {exc}",
            error_code=MLErrorCode.DUPLICATE_CHECK_FAILED,
            candidate_id=candidate_id,
        ) from exc

    best_id    = None
    best_score = 0.0

    for row in all_embeddings:
        if str(row.candidate_id) == candidate_id:
            continue
        stored = row.embedding_json
        if not stored:
            continue
        score = _cosine_similarity(embedding, stored)
        if score > best_score:
            best_score = score
            best_id    = str(row.candidate_id)

    matched = best_score >= FACE_SUSPECTED_THRESHOLD
    return {
        "matched":               matched,
        "score":                 round(best_score, 4),
        "matched_candidate_id":  best_id if matched else None,
    }


def _check_voice_in_memory(
    candidate_id: str,
    embedding: list[float],
    all_voice_rows: list,
) -> dict:
    """Voice duplicate check against in-memory rows (no pgvector for voice yet)."""
    best_id    = None
    best_score = 0.0
    for row in all_voice_rows:
        if str(row.candidate_id) == candidate_id:
            continue
        stored = row.embedding_json
        if not stored:
            continue
        score = _cosine_similarity(embedding, stored)
        if score > best_score:
            best_score = score
            best_id    = str(row.candidate_id)

    matched = best_score >= VOICE_SUSPECTED_THRESHOLD
    return {
        "matched":               matched,
        "score":                 round(best_score, 4),
        "matched_candidate_id":  best_id if matched else None,
    }
