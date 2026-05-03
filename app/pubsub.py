"""
Redis pub/sub publisher for the ML service.

Publishes a terminal status event to the channel that the backend SSE
endpoint subscribes to.  Redis pub/sub is global (not DB-scoped), so
the backend can subscribe on DB 2 and the ML service can publish from
any DB connection — they share the same Redis instance.

Called once per candidate at the end of run_verdict_task.
"""
from __future__ import annotations

import json

import redis

from app.config import settings


def publish_ml_result(application_id: str, status: str) -> None:
    """Synchronous publish — safe to call from a Celery worker process."""
    r = redis.Redis.from_url(settings.REDIS_URL, db=0, decode_responses=True)
    try:
        channel = f"ml_result:{application_id}"
        payload = json.dumps({"status": status, "application_id": application_id})
        r.publish(channel, payload)
    finally:
        r.close()
