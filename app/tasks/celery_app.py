"""Celery application factory for the ML service."""
from celery import Celery

from app.config import settings

_broker_url  = f"{settings.REDIS_URL}/{settings.REDIS_BROKER_DB}"
_backend_url = f"{settings.REDIS_URL}/{settings.REDIS_BROKER_DB}"

celery_app = Celery(
    "ml_service",
    broker=_broker_url,
    backend=_backend_url,
    include=[
        "app.tasks.fraud_task",
        "app.tasks.workexp_task",
        "app.tasks.interview_task",
        "app.tasks.verdict_task",
        "app.tasks.pipeline_task",
    ],
)

celery_app.conf.update(
    task_serializer         = "json",
    result_serializer       = "json",
    accept_content          = ["json"],
    timezone                = "UTC",
    enable_utc              = True,
    worker_concurrency      = settings.CELERY_CONCURRENCY,
    task_acks_late          = True,
    task_reject_on_worker_lost = True,
    task_track_started      = True,
    result_expires          = 86400,  # 24h
)
