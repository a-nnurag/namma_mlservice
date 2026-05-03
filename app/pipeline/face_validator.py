"""
Pipeline Step 1 — Face Validation.

Validates face consistency throughout the interview video using DeepFace Facenet.
Checks for face presence, same-person consistency, and prolonged absence.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.core.error_codes import MLErrorCode
from app.core.exceptions import FaceValidationError
from app.core.logging import get_logger

log = get_logger(__name__)

FRAME_INTERVAL_SEC   = 2
FACE_MISSING_LIMIT   = 10
SIMILARITY_THRESHOLD = 0.65
MODEL_NAME           = "Facenet"


def _extract_frames(video_path: str, interval_sec: int = FRAME_INTERVAL_SEC) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FaceValidationError(
            f"Cannot open video file: {video_path}",
            error_code=MLErrorCode.FACE_VALIDATION_FAILED,
        )
    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_skip = int(fps * interval_sec)
    frames     = []
    frame_idx  = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            frames.append((round(frame_idx / fps, 2), frame))
        frame_idx += 1
    cap.release()
    return frames


def _get_embedding(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    try:
        from deepface import DeepFace  # type: ignore[import]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = DeepFace.represent(
            img_path=frame_rgb,
            model_name=MODEL_NAME,
            enforce_detection=False,
            detector_backend="opencv",
        )
        if result:
            emb = np.array(result[0]["embedding"])
            if np.linalg.norm(emb) > 0:
                return emb
        return None
    except Exception:
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def validate_face_in_video(video_path: str, candidate_id: str = "") -> dict:
    """
    Validate face consistency throughout an interview video.

    Returns dict with face_consistency_score, same_person_score,
    face_missing_seconds, flags, timeline, and verdict.

    Raises FaceValidationError on unrecoverable failures.
    """
    log.info("Face validation started", candidate_id=candidate_id, video_path=video_path)

    with log.timed("face_validation", candidate_id=candidate_id):
        try:
            frames = _extract_frames(video_path)
        except FaceValidationError:
            raise
        except Exception as exc:
            raise FaceValidationError(
                f"Frame extraction failed: {exc}",
                error_code=MLErrorCode.FACE_VALIDATION_FAILED,
                candidate_id=candidate_id,
            ) from exc

        if not frames:
            return _fail_result(["Video has no frames"])

        log.debug("Frames extracted", candidate_id=candidate_id, count=len(frames))

        timeline      = []
        reference_emb = None

        for timestamp, frame in frames:
            emb          = _get_embedding(frame)
            face_present = emb is not None
            entry        = {
                "timestamp_sec": timestamp,
                "face_detected": face_present,
                "same_person":   None,
                "similarity":    None,
            }
            if face_present:
                if reference_emb is None:
                    reference_emb           = emb
                    entry["same_person"]    = True
                    entry["similarity"]     = 1.0
                else:
                    sim                     = _cosine_similarity(reference_emb, emb)
                    entry["similarity"]     = round(sim, 3)
                    entry["same_person"]    = sim >= SIMILARITY_THRESHOLD
            timeline.append(entry)

        total_frames       = len(timeline)
        frames_with_face   = sum(1 for e in timeline if e["face_detected"])
        frames_same_person = sum(
            1 for e in timeline if e["face_detected"] and e["same_person"] is True
        )

        face_consistency_score = round(frames_with_face / total_frames, 3) if total_frames else 0.0
        same_person_score      = round(frames_same_person / frames_with_face, 3) if frames_with_face else 0.0

        max_missing = 0
        current     = 0
        for e in timeline:
            if not e["face_detected"]:
                current += FRAME_INTERVAL_SEC
                max_missing = max(max_missing, current)
            else:
                current = 0

        flags = []
        if face_consistency_score < 0.5:
            flags.append(f"Face detected in only {int(face_consistency_score * 100)}% of frames")
        if max_missing > FACE_MISSING_LIMIT:
            flags.append(f"Face absent for {max_missing}s continuously (limit {FACE_MISSING_LIMIT}s)")
        if same_person_score < 0.7 and frames_with_face > 2:
            flags.append(f"Low same-person score ({same_person_score:.2f}) — possible substitution")
        diff_frames = [e["timestamp_sec"] for e in timeline if e["face_detected"] and e["same_person"] is False]
        if diff_frames:
            flags.append(f"Different face at timestamps: {diff_frames}")

        if same_person_score < 0.5 or max_missing > 30:
            verdict = "FAIL"
        elif flags:
            verdict = "WARNING"
        else:
            verdict = "PASS"

        result = {
            "face_consistency_score": face_consistency_score,
            "same_person_score":      same_person_score,
            "face_missing_seconds":   max_missing,
            "flags":                  flags,
            "timeline":               timeline,
            "verdict":                verdict,
        }

        log.info(
            "Face validation complete",
            candidate_id=candidate_id,
            verdict=verdict,
            consistency=face_consistency_score,
            same_person=same_person_score,
            flags=len(flags),
        )
        return result


def _fail_result(flags: list[str]) -> dict:
    return {
        "face_consistency_score": 0.0,
        "same_person_score":      0.0,
        "face_missing_seconds":   0,
        "flags":                  flags,
        "timeline":               [],
        "verdict":                "FAIL",
    }
