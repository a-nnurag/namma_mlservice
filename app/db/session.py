"""Async SQLAlchemy engine and session factory for the ML service."""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.core.exceptions import DBConnectionError
from app.core.logging import get_logger
from app.db.adapter import PostgreSQLAdapter

log = get_logger(__name__)

_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

_SessionFactory = async_sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncGenerator[PostgreSQLAdapter, None]:
    async with _SessionFactory() as session:
        try:
            adapter = PostgreSQLAdapter(session)
            yield adapter
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[PostgreSQLAdapter, None]:
    async with _SessionFactory() as session:
        try:
            adapter = PostgreSQLAdapter(session)
            yield adapter
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def verify_db_connection() -> None:
    from sqlalchemy import text
    try:
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        log.info("ML service database connection verified")
    except Exception as exc:
        log.critical("ML service database connection failed at startup", exc_info=True)
        raise DBConnectionError(str(exc)) from exc


async def create_tables() -> None:
    from app.db.models import Base
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("ML service database tables created/verified")


async def close_db() -> None:
    await _engine.dispose()
    log.info("ML service database pool disposed")
