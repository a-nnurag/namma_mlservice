"""
Database Adapter pattern — same interface as backend, isolated implementation.
All ML service DB operations go through DatabaseAdapter; never raw SQLAlchemy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DBIntegrityError, DBQueryError
from app.core.logging import get_logger

log = get_logger(__name__)
T = TypeVar("T")


class DatabaseAdapter(ABC):
    @abstractmethod
    async def get(self, model: Type[T], id: Any) -> T | None: ...

    @abstractmethod
    async def get_by(self, model: Type[T], **filters: Any) -> T | None: ...

    @abstractmethod
    async def list_by(self, model: Type[T], **filters: Any) -> list[T]: ...

    @abstractmethod
    async def create(self, instance: T) -> T: ...

    @abstractmethod
    async def update(self, instance: T, **fields: Any) -> T: ...

    @abstractmethod
    async def execute_query(self, stmt: Any) -> Any: ...

    @abstractmethod
    async def exists(self, model: Type[T], **filters: Any) -> bool: ...


class PostgreSQLAdapter(DatabaseAdapter):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, model: Type[T], id: Any) -> T | None:
        try:
            return await self._session.get(model, id)
        except SQLAlchemyError as exc:
            log.error("DB get failed", model=model.__name__, exc_info=True)
            raise DBQueryError("get", str(exc)) from exc

    async def get_by(self, model: Type[T], **filters: Any) -> T | None:
        try:
            result = await self._session.execute(select(model).filter_by(**filters))
            return result.scalars().first()
        except SQLAlchemyError as exc:
            raise DBQueryError("get_by", str(exc)) from exc

    async def list_by(self, model: Type[T], **filters: Any) -> list[T]:
        try:
            result = await self._session.execute(select(model).filter_by(**filters))
            return list(result.scalars().all())
        except SQLAlchemyError as exc:
            raise DBQueryError("list_by", str(exc)) from exc

    async def create(self, instance: T) -> T:
        try:
            self._session.add(instance)
            await self._session.flush()
            await self._session.refresh(instance)
            log.debug("DB row created", model=type(instance).__name__,
                      id=str(getattr(instance, "id", "?")))
            return instance
        except IntegrityError as exc:
            await self._session.rollback()
            log.warning("DB integrity error", model=type(instance).__name__, exc_info=True)
            raise DBIntegrityError(str(exc.orig)) from exc
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise DBQueryError("create", str(exc)) from exc

    async def update(self, instance: T, **fields: Any) -> T:
        try:
            for k, v in fields.items():
                setattr(instance, k, v)
            await self._session.flush()
            await self._session.refresh(instance)
            log.debug("DB row updated", model=type(instance).__name__,
                      id=str(getattr(instance, "id", "?")), fields=list(fields.keys()))
            return instance
        except IntegrityError as exc:
            await self._session.rollback()
            raise DBIntegrityError(str(exc.orig)) from exc
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise DBQueryError("update", str(exc)) from exc

    async def execute_query(self, stmt: Any) -> Any:
        try:
            return await self._session.execute(stmt)
        except SQLAlchemyError as exc:
            raise DBQueryError("execute_query", str(exc)) from exc

    async def exists(self, model: Type[T], **filters: Any) -> bool:
        try:
            result = await self._session.execute(
                select(model).filter_by(**filters).limit(1)
            )
            return result.scalars().first() is not None
        except SQLAlchemyError as exc:
            raise DBQueryError("exists", str(exc)) from exc
