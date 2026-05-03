"""
Pipeline Step 4 — Fraud Aggregation.

Combines face validation, liveness, and duplicate check results
into a single fraud_score and 5-level verdict.
"""
from __future__ import annotations

from app.core.logging import get_logger

log = get_logger(__name__)


def compute_fraud_score(
    face_result:      dict,
    liveness_result:  dict,
    duplicate_result: dict,
    candidate_id:     str = "",
) -> dict:
    """
    Aggregate all fraud signals into a single score + verdict.

    Weights:
      Duplicate detection  40%
      Liveness             30%
      Identity consistency 20%
      Face presence        10%
    """
    log.info("Fraud aggregation started", candidate_id=candidate_id)

    with log.timed("fraud_aggregation", candidate_id=candidate_id):
        flags       = []
        fraud_score = 0.0

        face_consistency = face_result.get("face_consistency_score", 1.0)
        same_person      = face_result.get("same_person_score", 1.0)
        face_verdict_val = face_result.get("verdict", "PASS")
        face_missing_sec = face_result.get("face_missing_seconds", 0)

        liveness_score   = liveness_result.get("liveness_score", 1.0)
        liveness_verdict = liveness_result.get("verdict", "LIVE")
        blink_detected   = liveness_result.get("blink_detected", True)

        is_duplicate     = duplicate_result.get("is_duplicate", False)
        dup_confidence   = duplicate_result.get("confidence", "none")
        dup_verdict      = duplicate_result.get("verdict", "UNIQUE")
        dup_face_score   = duplicate_result.get("face_match", {}).get("score", 0.0)
        dup_voice_score  = duplicate_result.get("voice_match", {}).get("score", 0.0)
        matched_id       = duplicate_result.get("matched_candidate_id")

        # Duplicate (40%)
        if dup_verdict == "DUPLICATE" and dup_confidence == "high":
            fraud_score += 0.40
            flags.append(
                f"HIGH CONFIDENCE DUPLICATE: matches candidate {matched_id} "
                f"(face={dup_face_score:.2f}, voice={dup_voice_score:.2f})"
            )
        elif dup_verdict == "SUSPECTED" and dup_confidence == "medium":
            fraud_score += 0.25
            flags.append(f"SUSPECTED DUPLICATE: possible match with {matched_id} (face={dup_face_score:.2f})")
        elif dup_verdict == "SUSPECTED" and dup_confidence == "low":
            fraud_score += 0.10
            flags.append(f"LOW CONFIDENCE DUPLICATE: voice similarity to {matched_id}")

        # Liveness (30%)
        if liveness_verdict == "SPOOF":
            fraud_score += 0.30
            flags.append("SPOOF DETECTED: liveness check failed — likely photo/video attack")
        elif liveness_verdict == "UNCERTAIN":
            fraud_score += 0.15
            flags.append(f"LIVENESS UNCERTAIN: score={liveness_score:.2f}, blink={blink_detected}")
        elif not blink_detected and liveness_score < 0.6:
            fraud_score += 0.10
            flags.append("No blink detected with low liveness score — review recommended")

        # Identity consistency (20%)
        if same_person < 0.5 and face_verdict_val == "FAIL":
            fraud_score += 0.20
            flags.append(f"IDENTITY MISMATCH: different face mid-interview (same_person={same_person:.2f})")
        elif same_person < 0.7:
            fraud_score += 0.10
            flags.append(f"Face consistency low: same_person_score={same_person:.2f}")

        # Face presence (10%)
        if face_consistency < 0.4:
            fraud_score += 0.10
            flags.append(f"Face absent in {int((1 - face_consistency) * 100)}% of frames")
        elif face_missing_sec > 30:
            fraud_score += 0.05
            flags.append(f"Face missing for {face_missing_sec}s continuously")

        fraud_score = round(min(fraud_score, 1.0), 4)

        if fraud_score >= 0.70:
            fraud_level, recommended_action = "CRITICAL", "REJECT"
        elif fraud_score >= 0.40:
            fraud_level, recommended_action = "HIGH", "REJECT"
        elif fraud_score >= 0.25:
            fraud_level, recommended_action = "MEDIUM", "MANUAL_REVIEW"
        elif fraud_score >= 0.10:
            fraud_level, recommended_action = "LOW", "MANUAL_REVIEW"
        else:
            fraud_level, recommended_action = "NONE", "PASS"

        result = {
            "fraud_score":        fraud_score,
            "fraud_level":        fraud_level,
            "flags":              flags,
            "recommended_action": recommended_action,
        }

        log.info(
            "Fraud aggregation complete",
            candidate_id=candidate_id,
            fraud_score=fraud_score,
            fraud_level=fraud_level,
            recommended_action=recommended_action,
            flag_count=len(flags),
        )
        return result
