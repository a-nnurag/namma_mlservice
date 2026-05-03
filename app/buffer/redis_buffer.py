"""
Redis Chunk Buffer.

Kafka messages arrive as chunks (chunk_index, is_last, data).
This buffer stores each chunk in Redis and assembles the full
file once is_last=True has been received for every expected
media type.

Redis key layout:
  chunk:{candidate_id}:{media_type}:{chunk_index}  → raw bytes  (TTL=24h)
  meta:{candidate_id}:{media_type}:last_chunk       → int        (TTL=24h)
  meta:{candidate_id}:received_types               → Redis Set  (TTL=24h)
"""
from __future__ import annotations

import asyncio
from typing import Literal

import redis.asyncio as aioredis

from app.config import settings
from app.core.error_codes import MLErrorCode
from app.core.exceptions import ChunkBufferError
from app.core.logging import get_logger

log = get_logger(__name__)

MediaType = Literal["interview_video", "interview_audio", "workexp_video", "meta"]

# Media types that must arrive before the pipeline can start
REQUIRED_MEDIA_TYPES: set[MediaType] = {"interview_video", "interview_audio"}

_redis_client: aioredis.Redis | None = None


def _get_redis() -> aioredis.Redis:
    if _redis_client is None:
        raise RuntimeError("Redis buffer not initialised — call init_buffer() first")
    return _redis_client


async def init_buffer() -> None:
    global _redis_client
    url = settings.REDIS_URL
    db = settings.REDIS_BUFFER_DB
    _redis_client = aioredis.from_url(url, db=db, decode_responses=False)
    await _redis_client.ping()
    log.info("Redis chunk buffer ready", url=url, db=db)


async def close_buffer() -> None:
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None


async def store_chunk(
    candidate_id: str,
    media_type: MediaType,
    chunk_index: int,
    data: bytes,
    is_last: bool,
    ttl: int = settings.CHUNK_BUFFER_TTL_SECONDS,
) -> bool:
    """
    Store one chunk.  Returns True if all required media types are
    complete (pipeline can fire), False otherwise.
    """
    r = _get_redis()
    chunk_key = f"chunk:{candidate_id}:{media_type}:{chunk_index}"
    last_key  = f"meta:{candidate_id}:{media_type}:last_chunk"
    types_key = f"meta:{candidate_id}:received_types"

    try:
        pipe = r.pipeline()
        pipe.setex(chunk_key, ttl, data)
        if is_last:
            pipe.setex(last_key, ttl, str(chunk_index).encode())
            pipe.sadd(types_key, media_type)
            pipe.expire(types_key, ttl)
        await pipe.execute()
    except Exception as exc:
        log.error(
            "Chunk store failed",
            candidate_id=candidate_id,
            media_type=media_type,
            chunk_index=chunk_index,
            error=str(exc),
        )
        raise ChunkBufferError(
            f"Failed to store chunk {chunk_index} for {candidate_id}/{media_type}: {exc}",
            error_code=MLErrorCode.CHUNK_BUFFER_OVERFLOW,
            candidate_id=candidate_id,
        ) from exc

    log.debug(
        "Chunk stored",
        candidate_id=candidate_id,
        media_type=media_type,
        chunk_index=chunk_index,
        is_last=is_last,
    )

    # Check if all required types are complete
    received: set[bytes] = await r.smembers(types_key)
    received_str = {m.decode() for m in received}
    ready = REQUIRED_MEDIA_TYPES.issubset(received_str)

    if ready:
        log.info(
            "All required media chunks received — pipeline ready",
            candidate_id=candidate_id,
            received_types=list(received_str),
        )
    return ready


async def assemble_media(
    candidate_id: str,
    media_type: MediaType,
) -> bytes:
    """
    Read all stored chunks for (candidate_id, media_type) in order
    and return the assembled bytes.  Raises ChunkBufferError if any
    chunk is missing.
    """
    r = _get_redis()
    last_key = f"meta:{candidate_id}:{media_type}:last_chunk"

    raw = await r.get(last_key)
    if raw is None:
        raise ChunkBufferError(
            f"No last-chunk marker found for {candidate_id}/{media_type}",
            error_code=MLErrorCode.ASSEMBLY_FAILED,
            candidate_id=candidate_id,
        )

    last_index = int(raw.decode())
    chunks: list[bytes] = []

    for idx in range(last_index + 1):
        key = f"chunk:{candidate_id}:{media_type}:{idx}"
        data = await r.get(key)
        if data is None:
            raise ChunkBufferError(
                f"Missing chunk {idx} for {candidate_id}/{media_type}",
                error_code=MLErrorCode.CHUNK_GAP_DETECTED,
                candidate_id=candidate_id,
            )
        chunks.append(data)

    log.info(
        "Media assembled",
        candidate_id=candidate_id,
        media_type=media_type,
        chunks=last_index + 1,
        total_bytes=sum(len(c) for c in chunks),
    )
    return b"".join(chunks)


async def cleanup_candidate(candidate_id: str) -> None:
    """Remove all buffer keys for a candidate after pipeline completes."""
    r = _get_redis()
    pattern = f"chunk:{candidate_id}:*"
    keys = await r.keys(pattern)
    meta_pattern = f"meta:{candidate_id}:*"
    keys += await r.keys(meta_pattern)
    if keys:
        await r.delete(*keys)
    log.info("Buffer cleaned", candidate_id=candidate_id, keys_deleted=len(keys))
