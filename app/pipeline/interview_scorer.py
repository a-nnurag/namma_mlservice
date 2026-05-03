"""
Pipeline Step 6 — Interview Scoring — Strategy Pattern.

InterviewScorerBackend     → ABC
RuleBasedScorer            → regex/keyword heuristic (default, no API key)
ClaudeScorer               → Anthropic Claude claude-sonnet-4-6
GeminiScorer               → Google Gemini
OpenAIScorer               → OpenAI GPT-4o

Switch via INTERVIEW_SCORER_BACKEND=rules|claude|gemini|openai
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.config import settings
from app.core.error_codes import MLErrorCode
from app.core.exceptions import InterviewScoringError
from app.core.logging import get_logger

log = get_logger(__name__)

_SCORE_PROMPT = """You are an expert technical interviewer evaluating a candidate's interview transcript.

Role claimed: {role}
Transcript:
\"\"\"
{transcript}
\"\"\"

Score the candidate on three dimensions (0.0 to 1.0 each):
1. domain_score        — technical/domain knowledge relevant to the claimed role
2. communication_score — clarity, structure, grammar, and coherence of answers
3. confidence_score    — assertiveness, completeness, and confidence in responses

Respond ONLY with valid JSON (no markdown), exactly in this format:
{{
  "domain_score": 0.0,
  "communication_score": 0.0,
  "confidence_score": 0.0,
  "rationale": "one short paragraph"
}}"""


# ── Abstract base ─────────────────────────────────────────────────────────────

class InterviewScorerBackend(ABC):
    @abstractmethod
    async def score(
        self,
        transcript: str,
        claimed_role: str,
        candidate_id: str = "",
    ) -> dict:
        """
        Returns:
            domain_score, communication_score, confidence_score, rationale
        """


# ── Rules-based (no API key required) ────────────────────────────────────────

class RuleBasedScorer(InterviewScorerBackend):
    """
    Heuristic scorer based on word count, keyword presence, and sentence structure.
    Good enough for local dev and as a fallback.
    """

    _FILLER = re.compile(r"\b(um+|uh+|like|you know|basically|literally)\b", re.I)
    _SENTENCE_END = re.compile(r"[.!?]")

    async def score(self, transcript: str, claimed_role: str, candidate_id: str = "") -> dict:
        if not transcript or not transcript.strip():
            return _empty_score("Empty transcript")

        words      = transcript.split()
        word_count = len(words)
        sentences  = len(self._SENTENCE_END.findall(transcript)) or 1
        avg_len    = word_count / sentences
        fillers    = len(self._FILLER.findall(transcript))

        # Communication: penalise fillers, reward average sentence length 10-20 words
        comm = min(1.0, max(0.0, (1.0 - fillers / max(word_count, 1)) * (min(avg_len, 20) / 20)))

        # Domain: check for role-relevant keywords
        role_keywords = _role_keywords(claimed_role)
        if role_keywords:
            lower = transcript.lower()
            hits  = sum(1 for kw in role_keywords if kw in lower)
            dom   = round(min(1.0, hits / len(role_keywords)), 3)
        else:
            dom = round(min(1.0, word_count / 300), 3)

        # Confidence: reward longer, more complete answers
        conf = round(min(1.0, word_count / 200), 3)

        return {
            "domain_score":        round(dom, 3),
            "communication_score": round(comm, 3),
            "confidence_score":    round(conf, 3),
            "rationale":           f"Rule-based score. Words={word_count}, fillers={fillers}.",
        }


# ── Claude scorer ─────────────────────────────────────────────────────────────

class ClaudeScorer(InterviewScorerBackend):
    async def score(self, transcript: str, claimed_role: str, candidate_id: str = "") -> dict:
        if not settings.ANTHROPIC_API_KEY:
            raise InterviewScoringError(
                "ANTHROPIC_API_KEY not set",
                error_code=MLErrorCode.SCORER_API_FAILED,
                candidate_id=candidate_id,
            )
        try:
            import anthropic  # type: ignore[import]
            import json

            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            prompt = _SCORE_PROMPT.format(role=claimed_role, transcript=transcript[:6000])
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            return _parse_llm_response(raw, candidate_id)
        except InterviewScoringError:
            raise
        except Exception as exc:
            raise InterviewScoringError(
                f"Claude API call failed: {exc}",
                error_code=MLErrorCode.SCORER_API_FAILED,
                candidate_id=candidate_id,
            ) from exc


# ── Gemini scorer ─────────────────────────────────────────────────────────────

class GeminiScorer(InterviewScorerBackend):
    async def score(self, transcript: str, claimed_role: str, candidate_id: str = "") -> dict:
        if not settings.GOOGLE_API_KEY:
            raise InterviewScoringError(
                "GOOGLE_API_KEY not set",
                error_code=MLErrorCode.SCORER_API_FAILED,
                candidate_id=candidate_id,
            )
        try:
            import google.generativeai as genai  # type: ignore[import]
            import json

            genai.configure(api_key=settings.GOOGLE_API_KEY)
            model  = genai.GenerativeModel("gemini-1.5-flash")
            prompt = _SCORE_PROMPT.format(role=claimed_role, transcript=transcript[:6000])
            resp   = model.generate_content(prompt)
            raw    = resp.text.strip()
            return _parse_llm_response(raw, candidate_id)
        except InterviewScoringError:
            raise
        except Exception as exc:
            raise InterviewScoringError(
                f"Gemini API call failed: {exc}",
                error_code=MLErrorCode.SCORER_API_FAILED,
                candidate_id=candidate_id,
            ) from exc


# ── OpenAI scorer ─────────────────────────────────────────────────────────────

class OpenAIScorer(InterviewScorerBackend):
    async def score(self, transcript: str, claimed_role: str, candidate_id: str = "") -> dict:
        if not settings.OPENAI_API_KEY:
            raise InterviewScoringError(
                "OPENAI_API_KEY not set",
                error_code=MLErrorCode.SCORER_API_FAILED,
                candidate_id=candidate_id,
            )
        try:
            from openai import AsyncOpenAI  # type: ignore[import]
            import json

            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            prompt = _SCORE_PROMPT.format(role=claimed_role, transcript=transcript[:6000])
            resp   = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            return _parse_llm_response(raw, candidate_id)
        except InterviewScoringError:
            raise
        except Exception as exc:
            raise InterviewScoringError(
                f"OpenAI API call failed: {exc}",
                error_code=MLErrorCode.SCORER_API_FAILED,
                candidate_id=candidate_id,
            ) from exc


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_llm_response(raw: str, candidate_id: str) -> dict:
    import json
    try:
        data = json.loads(raw)
        return {
            "domain_score":        float(data.get("domain_score", 0.5)),
            "communication_score": float(data.get("communication_score", 0.5)),
            "confidence_score":    float(data.get("confidence_score", 0.5)),
            "rationale":           str(data.get("rationale", "")),
        }
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise InterviewScoringError(
            f"Invalid LLM response: {exc} — raw={raw[:200]}",
            error_code=MLErrorCode.SCORER_INVALID_RESPONSE,
            candidate_id=candidate_id,
        ) from exc


def _empty_score(reason: str) -> dict:
    return {
        "domain_score":        0.0,
        "communication_score": 0.0,
        "confidence_score":    0.0,
        "rationale":           reason,
    }


def _role_keywords(role: str) -> list[str]:
    role_lower = role.lower()
    _ROLE_MAP: dict[str, list[str]] = {
        "software engineer": ["python", "java", "api", "database", "algorithm", "git", "testing"],
        "data scientist":    ["machine learning", "model", "pandas", "numpy", "statistics", "python"],
        "devops":            ["docker", "kubernetes", "ci/cd", "pipeline", "aws", "linux", "terraform"],
        "frontend":          ["react", "javascript", "html", "css", "typescript", "component"],
        "backend":           ["api", "rest", "database", "server", "python", "node", "sql"],
        "qa":                ["testing", "automation", "bug", "selenium", "test case", "regression"],
        "manager":           ["team", "stakeholder", "project", "deadline", "communication", "leadership"],
    }
    for key, kws in _ROLE_MAP.items():
        if key in role_lower:
            return kws
    return []


# ── Factory ───────────────────────────────────────────────────────────────────

def build_interview_scorer() -> InterviewScorerBackend:
    backend = settings.INTERVIEW_SCORER_BACKEND.lower()
    scorers = {
        "claude":  ClaudeScorer,
        "gemini":  GeminiScorer,
        "openai":  OpenAIScorer,
        "rules":   RuleBasedScorer,
    }
    cls = scorers.get(backend, RuleBasedScorer)
    log.info("Interview scorer selected", backend=backend, scorer=cls.__name__)
    return cls()
