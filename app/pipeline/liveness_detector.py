"""
Pipeline Step 2 — Liveness Detection.

Detects whether the candidate is physically present (live) using MediaPipe
FaceLandmarker.  Checks blink, head movement, mouth movement, and texture.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.core.error_codes import MLErrorCode
from app.core.exceptions import LivenessCheckError
from app.core.logging import get_logger
from app.models.registry import registry

log = get_logger(__name__)

LEFT_EYE     = [362, 385, 387, 263, 373, 380]
RIGHT_EYE    = [33,  160, 158, 133, 153, 144]
NOSE_TIP     = 4
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14

EAR_THRESHOLD        = 0.22
HEAD_MOVE_THRESHOLD  = 0.008
MOUTH_OPEN_THRESHOLD = 0.03


def _get_ear(landmarks: list, eye_indices: list[int]) -> float:
    pts = [(landmarks[i].x, landmarks[i].y) for i in eye_indices]
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    h  = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (v1 + v2) / (2.0 * h) if h != 0 else 0.3


def _texture_score(frame_bgr: np.ndarray) -> float:
    gray      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(1.0, max(0.0, (laplacian - 100) / 400))


def detect_liveness(video_path: str, candidate_id: str = "") -> dict:
    """
    Analyze a video for liveness signals.

    Returns dict with liveness_score, blink_detected, head_moved,
    mouth_moved, is_live, verdict, and details.

    Raises LivenessCheckError on unrecoverable failures.
    """
    log.info("Liveness detection started", candidate_id=candidate_id, video_path=video_path)

    with log.timed("liveness_detection", candidate_id=candidate_id):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise LivenessCheckError(
                f"Cannot open video: {video_path}",
                error_code=MLErrorCode.LIVENESS_CHECK_FAILED,
                candidate_id=candidate_id,
            )

        fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_skip = max(1, int(fps // 10))

        options = registry.get_landmarker_options()
        if options is None:
            cap.release()
            log.warning(
                "MediaPipe landmarker unavailable — returning degraded result",
                candidate_id=candidate_id,
            )
            return _uncertain_result("MediaPipe model not available")

        try:
            import mediapipe as mp  # type: ignore[import]
            from mediapipe.tasks.python import vision as mp_vision  # type: ignore[import]

            detector = mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            cap.release()
            raise LivenessCheckError(
                f"Failed to create FaceLandmarker: {exc}",
                error_code=MLErrorCode.LIVENESS_CHECK_FAILED,
                candidate_id=candidate_id,
            ) from exc

        blink_detected   = False
        head_moved       = False
        mouth_moved      = False
        blink_count      = 0
        ear_was_closed   = False
        texture_scores: list[float] = []
        nose_positions: list[tuple] = []
        mouth_ratios: list[float]   = []
        frames_processed = 0
        frames_with_face = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_processed += 1
                if frames_processed % frame_skip != 0:
                    continue

                texture_scores.append(_texture_score(frame))
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result   = detector.detect(mp_image)

                if not result.face_landmarks:
                    continue

                frames_with_face += 1
                landmarks = result.face_landmarks[0]

                left_ear  = _get_ear(landmarks, LEFT_EYE)
                right_ear = _get_ear(landmarks, RIGHT_EYE)
                avg_ear   = (left_ear + right_ear) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    if not ear_was_closed:
                        blink_count   += 1
                        ear_was_closed = True
                else:
                    ear_was_closed = False
                if blink_count >= 1:
                    blink_detected = True

                nose = landmarks[NOSE_TIP]
                nose_positions.append((nose.x, nose.y))
                if len(nose_positions) > 1:
                    dx = abs(nose_positions[-1][0] - nose_positions[0][0])
                    dy = abs(nose_positions[-1][1] - nose_positions[0][1])
                    if dx > HEAD_MOVE_THRESHOLD or dy > HEAD_MOVE_THRESHOLD:
                        head_moved = True

                mouth_r = abs(landmarks[MOUTH_BOTTOM].y - landmarks[MOUTH_TOP].y)
                mouth_ratios.append(mouth_r)
                if len(mouth_ratios) > 1:
                    if max(mouth_ratios) - min(mouth_ratios) > MOUTH_OPEN_THRESHOLD:
                        mouth_moved = True
        finally:
            cap.release()
            detector.close()

        avg_texture    = round(float(np.mean(texture_scores)), 3) if texture_scores else 0.0
        signals        = [blink_detected, head_moved, mouth_moved, avg_texture > 0.3]
        weights        = [0.35,           0.25,       0.20,        0.20]
        liveness_score = round(sum(w for s, w in zip(signals, weights) if s), 3)
        is_live        = liveness_score >= 0.55

        if liveness_score >= 0.75:
            verdict = "LIVE"
        elif liveness_score >= 0.45:
            verdict = "UNCERTAIN"
        else:
            verdict = "SPOOF"

        result_dict = {
            "liveness_score": liveness_score,
            "blink_detected": blink_detected,
            "head_moved":     head_moved,
            "mouth_moved":    mouth_moved,
            "is_live":        is_live,
            "verdict":        verdict,
            "details": {
                "blink_count":       blink_count,
                "avg_texture_score": avg_texture,
                "frames_processed":  frames_processed,
                "frames_with_face":  frames_with_face,
            },
        }

        log.info(
            "Liveness detection complete",
            candidate_id=candidate_id,
            verdict=verdict,
            liveness_score=liveness_score,
            blinks=blink_count,
        )
        return result_dict


def _uncertain_result(reason: str) -> dict:
    return {
        "liveness_score": 0.5,
        "blink_detected": False,
        "head_moved":     False,
        "mouth_moved":    False,
        "is_live":        False,
        "verdict":        "UNCERTAIN",
        "details":        {"reason": reason},
    }
