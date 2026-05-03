"""
Pipeline Step 9 — Final Verdict Engine.

Combines all sub-scores into a composite_score and issues the final verdict.

Weights:
  Interview  (domain + comm + confidence avg)  35%
  Fraud      (inverted fraud_score)            30%
  Liveness                                     15%
  Work exp   (skill_confidence)                10%
  Degree     (degree_confidence)               10%
"""
from __future__ import annotations

from app.core.logging import get_logger

log = get_logger(__name__)


def compute_final_verdict(
    fraud_result:     dict,
    interview_result: dict,
    workexp_result:   dict,
    degree_result:    dict,
    has_workexp:      bool = True,
    has_degree:       bool = True,
    candidate_id:     str  = "",
) -> dict:
    """
    Compute composite score and final verdict.

    Returns:
        composite_score, final_verdict, verdict_reason, recommended_action
    """
    log.info("Verdict engine started", candidate_id=candidate_id)

    fraud_score  = fraud_result.get("fraud_score", 0.0)
    fraud_action = fraud_result.get("recommended_action", "PASS")
    fraud_level  = fraud_result.get("fraud_level", "NONE")

    # Hard reject — never override fraud
    if fraud_action == "REJECT":
        return {
            "composite_score":    round(1.0 - fraud_score, 3),
            "final_verdict":      "FAIL",
            "verdict_reason":     f"Fraud detected: level={fraud_level}, flags={fraud_result.get('flags', [])}",
            "recommended_action": "REJECT",
        }

    domain_score = interview_result.get("domain_score", 0.0)
    comm_score   = interview_result.get("communication_score", 0.0)
    conf_score   = interview_result.get("confidence_score", 0.0)
    liveness     = interview_result.get("liveness_score", 0.5)
    skill_conf   = workexp_result.get("skill_confidence", 0.5) if has_workexp else 0.5
    degree_conf  = degree_result.get("degree_confidence", 0.5) if has_degree else 0.5

    interview_avg = (domain_score + comm_score + conf_score) / 3.0
    anti_fraud    = 1.0 - fraud_score

    composite = (
        interview_avg * 0.35
        + anti_fraud  * 0.30
        + liveness    * 0.15
        + skill_conf  * 0.10
        + degree_conf * 0.10
    )
    composite = round(composite, 3)

    # Determine verdict
    if composite >= 0.75:
        final_verdict      = "PASS"
        recommended_action = "PASS"
    elif composite >= 0.50 or fraud_action == "MANUAL_REVIEW":
        final_verdict      = "REVIEW"
        recommended_action = "MANUAL_REVIEW"
    else:
        final_verdict      = "FAIL"
        recommended_action = "REJECT"

    reasons = []
    if interview_avg < 0.5:
        reasons.append(f"Low interview score ({interview_avg:.2f})")
    if anti_fraud < 0.75:
        reasons.append(f"Fraud indicators present (level={fraud_level})")
    if has_workexp and skill_conf < 0.4:
        reasons.append("Work experience skill mismatch")
    if has_degree and degree_conf < 0.5:
        reasons.append("Degree document unclear")

    verdict_reason = "; ".join(reasons) if reasons else "All checks passed"

    result = {
        "composite_score":    composite,
        "final_verdict":      final_verdict,
        "verdict_reason":     verdict_reason,
        "recommended_action": recommended_action,
    }

    log.info(
        "Final verdict computed",
        candidate_id=candidate_id,
        composite_score=composite,
        final_verdict=final_verdict,
        recommended_action=recommended_action,
    )
    return result
