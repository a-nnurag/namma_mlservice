"""
ML Service — FastAPI entrypoint.

Startup sequence:
  1. Init structured logging
  2. Verify DB connection + create tables
  3. Init Redis chunk buffer
  4. Warm up ML models (DeepFace, MediaPipe, Resemblyzer) in background
  5. Start Kafka consumer
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.core.error_codes import MLErrorCode
from app.core.exceptions import MLServiceError
from app.core.logging import get_logger, init_logging

init_logging(settings.LOG_LEVEL)
log = get_logger(__name__)

_consumer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _consumer

    log.info("ML service starting", port=settings.ML_SERVICE_PORT)

    # ── DB ────────────────────────────────────────────────────────────────────
    from app.db.session import verify_db_connection, create_tables
    await verify_db_connection()
    await create_tables()

    # ── Redis buffer ──────────────────────────────────────────────────────────
    from app.buffer.redis_buffer import init_buffer
    await init_buffer()

    # ── ML model warmup (non-blocking background task) ────────────────────────
    import asyncio
    from app.models.registry import registry
    asyncio.create_task(registry.warmup())

    # ── Kafka consumer ────────────────────────────────────────────────────────
    from app.consumers.kafka_consumer import build_kafka_consumer
    from app.consumers.message_handler import register_handlers
    _consumer = build_kafka_consumer()
    register_handlers(_consumer)
    await _consumer.start()

    log.info("ML service ready")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    if _consumer:
        await _consumer.stop()

    from app.buffer.redis_buffer import close_buffer
    await close_buffer()

    from app.db.session import close_db
    await close_db()

    log.info("ML service shut down cleanly")


app = FastAPI(
    title="NammaKelsa ML Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(MLServiceError)
async def ml_error_handler(request: Request, exc: MLServiceError) -> JSONResponse:
    log.error(
        "MLServiceError",
        error_code=exc.error_code.value,
        message=exc.message,
        candidate_id=exc.candidate_id,
        detail=exc.detail,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error_code": exc.error_code.value,
            "message":    exc.message,
            "detail":     exc.detail,
        },
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("Unhandled exception", error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error_code": MLErrorCode.INTERNAL_ERROR.value,
            "message":    "Internal server error",
        },
    )


# ── Request context middleware ────────────────────────────────────────────────

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    from app.core.logging import set_ml_context
    request_id = str(uuid.uuid4())
    set_ml_context(candidate_id="", task_name="")
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Routers ───────────────────────────────────────────────────────────────────

from app.api.health  import router as health_router
from app.api.verdict import router as verdict_router
from app.api.degree  import router as degree_router
from app.api.status  import router as status_router
from app.api.face    import router as face_router
from app.api.admin   import router as admin_router

app.include_router(health_router)
app.include_router(verdict_router)
app.include_router(degree_router)
app.include_router(status_router)
app.include_router(face_router)
app.include_router(admin_router)
