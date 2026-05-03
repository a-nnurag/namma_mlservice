"""
Model Registry — Singleton.

Loads all heavy ML models once at startup and exposes them globally.
Pipeline modules call registry.get_*() to retrieve a ready model.

Models loaded:
  - DeepFace (Facenet for face validation, ArcFace for duplicate check)
  - MediaPipe FaceLandmarker (for liveness detection)
  - Resemblyzer VoiceEncoder (for voice duplicate check)
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

log = get_logger(__name__)

# Absolute path to the MediaPipe model file bundled with the service
_FACE_LANDMARKER_PATH = Path(__file__).resolve().parent.parent / "assets" / "face_landmarker.task"


class _ModelRegistry:
    def __init__(self) -> None:
        self._facenet_warmup_done = False
        self._arcface_warmup_done = False
        self._landmarker_options: Any = None
        self._voice_encoder: Any = None

    # ── Warmup ────────────────────────────────────────────────────────────────

    async def warmup(self) -> None:
        """Run model warmup in a thread pool to avoid blocking the event loop."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._warmup_sync)

    def _warmup_sync(self) -> None:
        self._load_deepface_facenet()
        self._load_deepface_arcface()
        self._load_mediapipe_landmarker()
        self._load_voice_encoder()

    def _load_deepface_facenet(self) -> None:
        try:
            import numpy as np
            from deepface import DeepFace  # type: ignore[import]

            dummy = np.zeros((160, 160, 3), dtype=np.uint8)
            DeepFace.represent(
                img_path=dummy,
                model_name="Facenet",
                enforce_detection=False,
                detector_backend="opencv",
            )
            self._facenet_warmup_done = True
            log.info("DeepFace Facenet model warmed up")
        except Exception as exc:
            log.warning("DeepFace Facenet warmup failed", error=str(exc))

    def _load_deepface_arcface(self) -> None:
        try:
            import numpy as np
            from deepface import DeepFace  # type: ignore[import]

            dummy = np.zeros((112, 112, 3), dtype=np.uint8)
            DeepFace.represent(
                img_path=dummy,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="opencv",
            )
            self._arcface_warmup_done = True
            log.info("DeepFace ArcFace model warmed up")
        except Exception as exc:
            log.warning("DeepFace ArcFace warmup failed", error=str(exc))

    def _load_mediapipe_landmarker(self) -> None:
        if not _FACE_LANDMARKER_PATH.exists():
            log.warning(
                "face_landmarker.task not found — liveness detection will be degraded",
                path=str(_FACE_LANDMARKER_PATH),
            )
            return
        try:
            import mediapipe as mp  # type: ignore[import]
            from mediapipe.tasks import python as mp_python  # type: ignore[import]
            from mediapipe.tasks.python import vision as mp_vision  # type: ignore[import]
            from mediapipe.tasks.python.vision import FaceLandmarkerOptions  # type: ignore[import]

            base_options = mp_python.BaseOptions(
                model_asset_path=str(_FACE_LANDMARKER_PATH)
            )
            self._landmarker_options = FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            log.info("MediaPipe FaceLandmarker options ready", path=str(_FACE_LANDMARKER_PATH))
        except Exception as exc:
            log.warning("MediaPipe FaceLandmarker init failed", error=str(exc))

    def _load_voice_encoder(self) -> None:
        try:
            from resemblyzer import VoiceEncoder  # type: ignore[import]

            self._voice_encoder = VoiceEncoder("cpu")
            log.info("Resemblyzer VoiceEncoder loaded")
        except ImportError:
            log.warning("Resemblyzer not installed — voice duplicate check disabled")
        except Exception as exc:
            log.warning("VoiceEncoder load failed", error=str(exc))

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_landmarker_options(self) -> Any | None:
        return self._landmarker_options

    def get_voice_encoder(self) -> Any | None:
        return self._voice_encoder

    def face_landmarker_path(self) -> str:
        return str(_FACE_LANDMARKER_PATH)


registry = _ModelRegistry()
