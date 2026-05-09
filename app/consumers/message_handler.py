"""
Kafka message handler.

Receives chunk messages from all topics, stores them in the Redis buffer,
and fires the Celery pipeline once all required media types are complete.
"""
from __future__ import annotations

import json
from typing import Any

from app.buffer.redis_buffer import store_chunk, MediaType
from app.consumers.kafka_consumer import KafkaConsumerAdapter, TOPICS
from app.core.logging import get_logger

log = get_logger(__name__)

_TOPIC_TO_MEDIA: dict[str, MediaType] = {
    "candidate.interview.video": "interview_video",
    "candidate.interview.audio": "interview_audio",
    "candidate.workexp.video":   "workexp_video",
    "candidate.meta":            "meta",
}


def register_handlers(consumer: KafkaConsumerAdapter) -> None:
    for topic in TOPICS:
        consumer.register_handler(topic, _make_handler(topic))
    log.info("Kafka message handlers registered", topics=TOPICS)


def _make_handler(topic: str):
    media_type: MediaType = _TOPIC_TO_MEDIA.get(topic, "meta")  # type: ignore[assignment]

    async def handler(t: str, key: str, payload: dict) -> None:
        candidate_id = payload.get("candidate_id") or key
        if not candidate_id:
            log.warning("Message missing candidate_id", topic=t)
            return

        chunk_index = int(payload.get("chunk_seq", payload.get("chunk_index", 0)))
        is_last     = bool(payload.get("is_last", False))

        # Payload data may be base64-encoded bytes or JSON metadata
        raw_data = payload.get("data", "")
        if isinstance(raw_data, str):
            import base64
            try:
                data = base64.b64decode(raw_data)
            except Exception:
                data = raw_data.encode()
        else:
            data = json.dumps(raw_data).encode()

        log.debug(
            "Chunk received",
            candidate_id=candidate_id,
            topic=t,
            media_type=media_type,
            chunk_index=chunk_index,
            is_last=is_last,
            bytes=len(data),
        )

        ready = await store_chunk(
            candidate_id=candidate_id,
            media_type=media_type,
            chunk_index=chunk_index,
            data=data,
            is_last=is_last,
        )

        if ready:
            await _trigger_pipeline(candidate_id, payload)

    return handler


async def _trigger_pipeline(candidate_id: str, meta_payload: dict) -> None:
    """
    Assemble buffered files, write to local storage, dispatch Celery chord.
    """
    from app.buffer.redis_buffer import assemble_media, cleanup_candidate
    from app.db.session import get_db_context
    from app.db import crud
    from app.storage.s3 import build_storage_backend
    from app.tasks.pipeline_task import dispatch_pipeline

    log.info("Triggering pipeline", candidate_id=candidate_id)

    try:
        storage = build_storage_backend()

        # Assemble media files
        video_bytes = await assemble_media(candidate_id, "interview_video")
        audio_bytes = await assemble_media(candidate_id, "interview_audio")

        video_key = f"{candidate_id}/interview_video.mp4"
        audio_key = f"{candidate_id}/interview_audio.wav"
        video_path = storage.write(video_key, video_bytes)
        audio_path = storage.write(audio_key, audio_bytes)

        # Optional work exp video → audio conversion handled in workexp_task
        workexp_audio_path = None
        try:
            wv_bytes = await assemble_media(candidate_id, "workexp_video")
            wv_key   = f"{candidate_id}/workexp_video.mp4"
            workexp_audio_path = storage.write(wv_key, wv_bytes)
        except Exception:
            pass

        # Selfie: stored earlier at registration (use video first frame as fallback)
        selfie_path = video_path  # pipeline code handles frame extraction

        meta: dict = {}
        try:
            meta_bytes = await assemble_media(candidate_id, "meta")
            meta = json.loads(meta_bytes.decode())
        except Exception:
            pass

        claimed_role = meta.get("claimed_role") or meta_payload.get("claimed_role", "unknown")
        has_workexp  = bool(meta.get("has_workexp", workexp_audio_path is not None))
        has_degree   = bool(meta.get("has_degree", False))
        language     = meta.get("language", "kn")

        # Create DB run record
        async with get_db_context() as db:
            from uuid import UUID
            run = await crud.create_pipeline_run(
                db=db,
                candidate_id=UUID(candidate_id),
                application_id=None,
                session_id=None,
                claimed_role=claimed_role,
                has_workexp=has_workexp,
                has_degree=has_degree,
            )
            run_id = str(run.id)
            await crud.update_run_status(db, run, "PROCESSING")

        await dispatch_pipeline(
            candidate_id=candidate_id,
            run_id=run_id,
            claimed_role=claimed_role,
            has_workexp=has_workexp,
            has_degree=has_degree,
            video_path=video_path,
            audio_path=audio_path,
            selfie_path=selfie_path,
            workexp_audio_path=workexp_audio_path,
            degree_image_path=meta.get("degree_image_path"),
            language=language,
        )

        await cleanup_candidate(candidate_id)
        log.info("Pipeline dispatched", candidate_id=candidate_id, run_id=run_id)

    except Exception as exc:
        log.error("Pipeline trigger failed", candidate_id=candidate_id, error=str(exc), exc_info=True)
