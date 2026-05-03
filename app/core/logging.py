"""
Structured JSON logging for the ML service.

Every log line automatically includes: timestamp, level, logger, message,
candidate_id (from context var), task_name, duration_ms (when set).

Usage:
    from app.core.logging import get_logger
    log = get_logger(__name__)
    log.info("Pipeline started", candidate_id=cid, task="fraud_task")
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Generator

from .error_codes import MLErrorCode

_candidate_id_ctx: ContextVar[str] = ContextVar("candidate_id", default="")
_task_name_ctx: ContextVar[str] = ContextVar("task_name", default="")


def set_ml_context(candidate_id: str = "", task_name: str = "") -> None:
    if candidate_id:
        _candidate_id_ctx.set(candidate_id)
    if task_name:
        _task_name_ctx.set(task_name)


class _JSONFormatter(logging.Formatter):
    _SKIP = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        cid = _candidate_id_ctx.get()
        if cid:
            entry["candidate_id"] = cid
        task = _task_name_ctx.get()
        if task:
            entry["task_name"] = task
        for key, value in record.__dict__.items():
            if key not in self._SKIP:
                entry[key] = value
        if record.exc_info:
            entry["traceback"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def _configure_root(level: str) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    h = logging.StreamHandler()
    h.setFormatter(_JSONFormatter())
    root.addHandler(h)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


class _BoundLogger:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, msg: str, **kwargs: Any) -> None:
        exc_info = kwargs.pop("exc_info", False)
        code: MLErrorCode | None = kwargs.pop("error_code", None)
        if code is not None:
            kwargs["error_code"] = code.value if isinstance(code, MLErrorCode) else code
        self._logger.log(level, msg, extra=kwargs, exc_info=exc_info)

    def debug(self, msg: str, **kw: Any) -> None:
        self._log(logging.DEBUG, msg, **kw)

    def info(self, msg: str, **kw: Any) -> None:
        self._log(logging.INFO, msg, **kw)

    def warning(self, msg: str, **kw: Any) -> None:
        self._log(logging.WARNING, msg, **kw)

    def error(self, msg: str, **kw: Any) -> None:
        self._log(logging.ERROR, msg, **kw)

    def critical(self, msg: str, **kw: Any) -> None:
        self._log(logging.CRITICAL, msg, **kw)

    def exception(self, msg: str, **kw: Any) -> None:
        kw["exc_info"] = True
        self._log(logging.ERROR, msg, **kw)

    def bind(self, candidate_id: str = "", task_name: str = "") -> None:
        """Set context variables for subsequent log lines in this thread/task."""
        set_ml_context(candidate_id=candidate_id, task_name=task_name)

    @contextmanager
    def timed(self, operation: str, **kw: Any) -> Generator[None, None, None]:
        """Context manager that logs operation duration in ms."""
        start = time.monotonic()
        self.debug(f"{operation} started", **kw)
        try:
            yield
        finally:
            ms = round((time.monotonic() - start) * 1000, 1)
            self.info(f"{operation} completed", duration_ms=ms, **kw)


def get_logger(name: str) -> _BoundLogger:
    return _BoundLogger(logging.getLogger(name))


def init_logging(level: str = "INFO") -> None:
    _configure_root(level)
