"""
Pipeline Step 5 — Audio Transcription.

Transcribes Kannada / Hindi / English interview audio to English text.
Uses Google Speech Recognition API via SpeechRecognition library
and deep-translator for translation.
"""
from __future__ import annotations

import os

from app.core.error_codes import MLErrorCode
from app.core.exceptions import TranscriptionError
from app.core.logging import get_logger

log = get_logger(__name__)

LANG_CODES = {
    "kn": "kn-IN",
    "hi": "hi-IN",
    "en": "en-US",
}

DEEP_TRANSLATOR_CODES = {
    "kn": "kn",
    "hi": "hi",
    "en": "en",
}


def _convert_to_wav(audio_path: str) -> str:
    ext      = os.path.splitext(audio_path)[1].lower()
    wav_path = audio_path.replace(ext, "_converted.wav")
    if ext in {".ogg", ".mp3", ".mp4", ".m4a", ".webm", ".flac"}:
        try:
            from pydub import AudioSegment  # type: ignore[import]
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(wav_path, format="wav")
            return wav_path
        except Exception as exc:
            raise TranscriptionError(
                f"Audio conversion failed: {exc}",
                error_code=MLErrorCode.SCORER_API_FAILED,
            ) from exc
    return audio_path


def transcribe_audio(
    audio_path: str,
    language: str = "kn",
    candidate_id: str = "",
) -> dict:
    """
    Transcribe audio and return English text.

    Args:
        audio_path : path to .ogg / .wav / .mp3 / .mp4 / .m4a
        language   : "kn" (Kannada), "hi" (Hindi), "en" (English)

    Returns:
        text, original_text, language, source
    """
    if not os.path.exists(audio_path):
        raise TranscriptionError(
            f"Audio file not found: {audio_path}",
            error_code=MLErrorCode.SCORER_API_FAILED,
            candidate_id=candidate_id,
        )

    log.info("Transcription started", candidate_id=candidate_id, language=language)

    with log.timed("transcription", candidate_id=candidate_id):
        wav_path = _convert_to_wav(audio_path)
        lang_code = LANG_CODES.get(language, "kn-IN")

        try:
            import speech_recognition as sr  # type: ignore[import]
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
            original_text = recognizer.recognize_google(audio_data, language=lang_code)
        except Exception as exc:
            raise TranscriptionError(
                f"Google Speech recognition failed: {exc}",
                error_code=MLErrorCode.SCORER_API_FAILED,
                candidate_id=candidate_id,
            ) from exc
        finally:
            if wav_path != audio_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

        if language != "en":
            try:
                from deep_translator import GoogleTranslator  # type: ignore[import]
                src_code  = DEEP_TRANSLATOR_CODES.get(language, "kn")
                english_text = GoogleTranslator(source=src_code, target="en").translate(original_text)
            except Exception as exc:
                log.warning(
                    "Translation failed — using original text",
                    candidate_id=candidate_id,
                    error=str(exc),
                )
                english_text = original_text
        else:
            english_text = original_text

        result = {
            "text":          english_text,
            "original_text": original_text,
            "language":      language,
            "source":        "google_speech_api",
        }

        log.info(
            "Transcription complete",
            candidate_id=candidate_id,
            language=language,
            text_length=len(english_text),
        )
        return result
