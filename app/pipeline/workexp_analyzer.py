"""
Pipeline Step 7 — Work Experience Analyzer.

Analyzes a work experience video clip to detect the role/skill
demonstrated and compare it to the candidate's claimed role.

Uses Groq LLaMA-3.3 for skill extraction from the transcript.
Falls back to keyword matching when no API key is available.
"""
from __future__ import annotations

import re

from app.config import settings
from app.core.error_codes import MLErrorCode
from app.core.exceptions import WorkExpAnalysisError
from app.core.logging import get_logger
from app.pipeline.transcriber import transcribe_audio

log = get_logger(__name__)

_ROLE_SKILL_MAP: dict[str, list[str]] = {
    "software engineer":  ["coding", "programming", "debugging", "git", "api", "python", "java"],
    "data scientist":     ["data", "model", "machine learning", "analysis", "statistics", "python"],
    "devops":             ["deployment", "docker", "kubernetes", "pipeline", "aws", "infrastructure"],
    "frontend developer": ["react", "javascript", "html", "css", "ui", "component", "typescript"],
    "backend developer":  ["api", "rest", "database", "server", "microservice", "sql", "python"],
    "qa engineer":        ["testing", "automation", "bug", "selenium", "test plan", "regression"],
    "project manager":    ["project", "sprint", "stakeholder", "timeline", "agile", "team"],
    "hr":                 ["recruitment", "interview", "hiring", "hr", "onboarding", "payroll"],
    "sales":              ["sales", "client", "revenue", "target", "crm", "customer"],
    "marketing":          ["marketing", "campaign", "seo", "social media", "branding", "analytics"],
}


async def analyze_workexp(
    audio_path: str,
    claimed_role: str,
    language: str = "kn",
    candidate_id: str = "",
) -> dict:
    """
    Transcribe work-experience audio, detect skill, and compare to claimed role.

    Returns:
        detected_skill, skill_confidence, matches_claimed_role, verdict, transcript
    """
    log.info(
        "Work experience analysis started",
        candidate_id=candidate_id,
        claimed_role=claimed_role,
    )

    with log.timed("workexp_analysis", candidate_id=candidate_id):
        try:
            transcription = transcribe_audio(
                audio_path=audio_path,
                language=language,
                candidate_id=candidate_id,
            )
            transcript = transcription["text"]
        except Exception as exc:
            log.warning(
                "Transcription failed — using empty transcript for workexp",
                candidate_id=candidate_id,
                error=str(exc),
            )
            transcript = ""

        if not transcript.strip():
            return {
                "detected_skill":      "unknown",
                "skill_confidence":    0.0,
                "matches_claimed_role": False,
                "workexp_verdict":     "INSUFFICIENT",
                "transcript":         transcript,
            }

        # Try Groq LLaMA for skill extraction
        skill_result = await _extract_skill_with_llm(
            transcript=transcript,
            claimed_role=claimed_role,
            candidate_id=candidate_id,
        )

        if skill_result is None:
            # Keyword fallback
            skill_result = _extract_skill_keywords(transcript, claimed_role)

        detected_skill  = skill_result["detected_skill"]
        confidence      = skill_result["skill_confidence"]
        matches         = skill_result["matches_claimed_role"]

        if matches and confidence >= 0.7:
            verdict = "PASS"
        elif matches and confidence >= 0.4:
            verdict = "WEAK_PASS"
        elif confidence >= 0.3:
            verdict = "MISMATCH"
        else:
            verdict = "FAIL"

        result = {
            "detected_skill":       detected_skill,
            "skill_confidence":     confidence,
            "matches_claimed_role": matches,
            "workexp_verdict":      verdict,
            "transcript":           transcript,
        }

        log.info(
            "Work experience analysis complete",
            candidate_id=candidate_id,
            detected_skill=detected_skill,
            matches=matches,
            verdict=verdict,
        )
        return result


async def _extract_skill_with_llm(
    transcript: str,
    claimed_role: str,
    candidate_id: str,
) -> dict | None:
    """Use Groq LLaMA-3.3 for accurate skill extraction."""
    groq_key = getattr(settings, "GROQ_API_KEY", "")
    if not groq_key:
        return None

    prompt = f"""You are evaluating a job candidate's work experience.

Claimed role: {claimed_role}
Work experience description:
\"\"\"{transcript[:2000]}\"\"\"

Extract:
1. The primary skill/role demonstrated
2. Whether it matches the claimed role (true/false)
3. Confidence score (0.0 to 1.0)

Respond ONLY with valid JSON:
{{"detected_skill": "...", "matches_claimed_role": true, "skill_confidence": 0.0}}"""

    try:
        from groq import Groq  # type: ignore[import]
        import json

        client   = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        raw  = response.choices[0].message.content or ""
        data = json.loads(raw.strip())
        return {
            "detected_skill":       str(data.get("detected_skill", claimed_role)),
            "skill_confidence":     float(data.get("skill_confidence", 0.5)),
            "matches_claimed_role": bool(data.get("matches_claimed_role", False)),
        }
    except Exception as exc:
        log.warning("Groq skill extraction failed", candidate_id=candidate_id, error=str(exc))
        return None


def _extract_skill_keywords(transcript: str, claimed_role: str) -> dict:
    """Keyword-based skill matching fallback."""
    lower = transcript.lower()
    best_role   = "unknown"
    best_hits   = 0
    best_total  = 1

    for role, keywords in _ROLE_SKILL_MAP.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits > best_hits:
            best_hits  = hits
            best_total = len(keywords)
            best_role  = role

    confidence = round(best_hits / best_total, 3)
    matches    = claimed_role.lower() in best_role or best_role in claimed_role.lower()

    return {
        "detected_skill":       best_role,
        "skill_confidence":     confidence,
        "matches_claimed_role": matches,
    }
