"""Degree verification endpoint — synchronous / direct call."""
from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.logging import get_logger

router = APIRouter(prefix="/degree", tags=["degree"])
log    = get_logger(__name__)


class DegreeVerifyResponse(BaseModel):
    is_valid_doc:          bool
    extracted_degree:      str
    extracted_institution: str
    extracted_year:        str
    degree_confidence:     float


@router.post("/{candidate_id}", response_model=DegreeVerifyResponse)
async def verify_degree_upload(
    candidate_id: str,
    file: UploadFile = File(...),
) -> DegreeVerifyResponse:
    """Accept a degree certificate image and return OCR-extracted fields."""
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "application/pdf"}:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}",
        )

    suffix = os.path.splitext(file.filename or ".jpg")[1] or ".jpg"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        from app.pipeline.degree_checker import verify_degree
        result = verify_degree(tmp_path, candidate_id=candidate_id)
        log.info("Direct degree verification complete", confidence=result["degree_confidence"])
        return DegreeVerifyResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Degree verification failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
