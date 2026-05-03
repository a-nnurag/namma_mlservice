"""
Pipeline Step 8 — Degree Document Verification.

Extracts degree, institution, and graduation year from a certificate image
using OCR (pytesseract) + optional Groq LLaMA validation.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.config import settings
from app.core.error_codes import MLErrorCode
from app.core.exceptions import DegreeCheckError
from app.core.logging import get_logger

log = get_logger(__name__)

_YEAR_PATTERN   = re.compile(r"\b(19|20)\d{2}\b")
_DEGREE_TOKENS  = [
    "bachelor", "b.e", "b.tech", "b.sc", "b.com", "b.a", "bca",
    "master", "m.e", "m.tech", "m.sc", "m.com", "m.a", "mca", "mba",
    "phd", "diploma", "degree", "graduate",
]
_INSTITUTION_TOKENS = [
    "university", "institute", "college", "school",
    "vidyapeeth", "vishwavidyalaya",
]


def verify_degree(
    image_path: str,
    candidate_id: str = "",
) -> dict:
    """
    OCR-extract degree details from a certificate image.

    Returns:
        is_valid_doc, extracted_degree, extracted_institution,
        extracted_year, degree_confidence
    """
    if not Path(image_path).exists():
        raise DegreeCheckError(
            f"Degree image not found: {image_path}",
            error_code=MLErrorCode.DEGREE_IMAGE_INVALID,
            candidate_id=candidate_id,
        )

    log.info("Degree verification started", candidate_id=candidate_id, image_path=image_path)

    with log.timed("degree_check", candidate_id=candidate_id):
        raw_text = _ocr_image(image_path, candidate_id)
        if not raw_text.strip():
            return _invalid_result("OCR returned empty text")

        lower = raw_text.lower()

        # Detect degree type
        extracted_degree = "unknown"
        for token in _DEGREE_TOKENS:
            if token in lower:
                idx   = lower.index(token)
                chunk = raw_text[max(0, idx - 10): idx + 60]
                extracted_degree = chunk.strip().replace("\n", " ")
                break

        # Detect institution
        extracted_institution = "unknown"
        for token in _INSTITUTION_TOKENS:
            if token in lower:
                idx   = lower.index(token)
                chunk = raw_text[max(0, idx - 20): idx + 80]
                extracted_institution = chunk.strip().replace("\n", " ")
                break

        # Detect year
        years = _YEAR_PATTERN.findall(raw_text)
        extracted_year = years[-1] if years else "unknown"

        # Confidence
        score_parts = [
            extracted_degree    != "unknown",
            extracted_institution != "unknown",
            extracted_year      != "unknown",
        ]
        degree_confidence = round(sum(score_parts) / len(score_parts), 3)
        is_valid_doc      = degree_confidence >= 0.5

        result = {
            "is_valid_doc":           is_valid_doc,
            "extracted_degree":       extracted_degree[:200],
            "extracted_institution":  extracted_institution[:200],
            "extracted_year":         extracted_year,
            "degree_confidence":      degree_confidence,
        }

        log.info(
            "Degree verification complete",
            candidate_id=candidate_id,
            is_valid=is_valid_doc,
            confidence=degree_confidence,
            year=extracted_year,
        )
        return result


def _ocr_image(image_path: str, candidate_id: str) -> str:
    try:
        import pytesseract  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang="eng")
        return text
    except ImportError as exc:
        raise DegreeCheckError(
            "pytesseract / Pillow not installed. pip install pytesseract Pillow",
            error_code=MLErrorCode.DEGREE_OCR_FAILED,
            candidate_id=candidate_id,
        ) from exc
    except Exception as exc:
        log.error("OCR failed", candidate_id=candidate_id, error=str(exc))
        raise DegreeCheckError(
            f"OCR processing failed: {exc}",
            error_code=MLErrorCode.DEGREE_OCR_FAILED,
            candidate_id=candidate_id,
        ) from exc


def _invalid_result(reason: str) -> dict:
    return {
        "is_valid_doc":           False,
        "extracted_degree":       "",
        "extracted_institution":  "",
        "extracted_year":         "",
        "degree_confidence":      0.0,
    }
