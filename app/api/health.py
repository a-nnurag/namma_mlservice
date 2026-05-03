"""Health-check routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.logging import get_logger
from app.db.adapter import DatabaseAdapter
from app.db.session import get_db

router = APIRouter(tags=["health"])
log    = get_logger(__name__)


class HealthResponse(BaseModel):
    status: str
    service: str = "ml_service"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/db", response_model=HealthResponse)
async def health_db(db: DatabaseAdapter = Depends(get_db)) -> HealthResponse:
    try:
        from sqlalchemy import text
        await db.execute_query(text("SELECT 1"))
        return HealthResponse(status="ok")
    except Exception as exc:
        log.error("DB health check failed", error=str(exc))
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Database unavailable")


@router.get("/health/redis", response_model=HealthResponse)
async def health_redis() -> HealthResponse:
    try:
        from app.buffer.redis_buffer import _get_redis
        await _get_redis().ping()
        return HealthResponse(status="ok")
    except Exception as exc:
        log.error("Redis health check failed", error=str(exc))
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Redis unavailable")
