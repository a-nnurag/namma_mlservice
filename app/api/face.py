"""Face uniqueness check — POST /face-check — called by backend during registration."""
from __future__ import annotations

import base64
import os
import tempfile

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.logging import get_logger
from app.db.adapter import DatabaseAdapter
from app.db.session import get_db

router = APIRouter(prefix="/face-check", tags=["face"])
log    = get_logger(__name__)


class FaceCheckRequest(BaseModel):
    candidate_id: str
    image:        str  # base64-encoded JPEG/PNG


class FaceCheckResponse(BaseModel):
    is_unique:            bool
    matched_candidate_id: str | None = None
    confidence:           str        # "high" | "medium" | "low" | "none"


@router.post("", response_model=FaceCheckResponse)
async def face_check(
    body: FaceCheckRequest,
    db: DatabaseAdapter = Depends(get_db),
) -> FaceCheckResponse:
    """Extract ArcFace embedding and check uniqueness against all stored embeddings."""
    try:
        image_bytes = base64.b64decode(body.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        from app.pipeline.duplicate_checker import (
            _extract_face_embedding,
            _check_face_against_db,
            FACE_DUPLICATE_THRESHOLD,
            FACE_SUSPECTED_THRESHOLD,
        )

        embedding = _extract_face_embedding(tmp_path)
        if embedding is None:
            raise HTTPException(status_code=422, detail="No face detected in image")

        result = await _check_face_against_db(
            candidate_id=body.candidate_id,
            embedding=embedding,
            db=db,
        )

        score = result["score"]
        if score >= FACE_DUPLICATE_THRESHOLD:
            confidence = "high"
        elif score >= FACE_SUSPECTED_THRESHOLD:
            confidence = "medium"
        elif score > 0:
            confidence = "low"
        else:
            confidence = "none"

        return FaceCheckResponse(
            is_unique=not result["matched"],
            matched_candidate_id=result["matched_candidate_id"],
            confidence=confidence,
        )

    except HTTPException:
        raise
    except Exception as exc:
        log.error("Face check failed", candidate_id=body.candidate_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
