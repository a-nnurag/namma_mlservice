"""LocalStorageBackend — writes files under LOCAL_STORAGE_DIR."""
from __future__ import annotations

import os
from pathlib import Path

from app.config import settings
from app.core.error_codes import MLErrorCode
from app.core.exceptions import StorageError
from app.core.logging import get_logger
from app.storage.base import StorageBackend

log = get_logger(__name__)


class LocalStorageBackend(StorageBackend):
    """Stores files on the local filesystem under LOCAL_STORAGE_DIR."""

    def __init__(self, base_dir: str | None = None) -> None:
        self._base = Path(base_dir or settings.LOCAL_STORAGE_DIR)
        self._base.mkdir(parents=True, exist_ok=True)
        log.info("LocalStorageBackend ready", base_dir=str(self._base))

    def _resolve(self, key: str) -> Path:
        path = (self._base / key).resolve()
        if not str(path).startswith(str(self._base.resolve())):
            raise StorageError(
                f"Key escapes storage root: {key}",
                error_code=MLErrorCode.STORAGE_WRITE_FAILED,
            )
        return path

    def write(self, key: str, data: bytes) -> str:
        path = self._resolve(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            log.debug("Storage write", key=key, bytes=len(data))
            return str(path)
        except OSError as exc:
            log.error("Storage write failed", key=key, error=str(exc))
            raise StorageError(
                f"Write failed for key '{key}': {exc}",
                error_code=MLErrorCode.STORAGE_WRITE_FAILED,
            ) from exc

    def read(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise FileNotFoundError(f"Key not found in storage: {key}")
        try:
            data = path.read_bytes()
            log.debug("Storage read", key=key, bytes=len(data))
            return data
        except OSError as exc:
            log.error("Storage read failed", key=key, error=str(exc))
            raise StorageError(
                f"Read failed for key '{key}': {exc}",
                error_code=MLErrorCode.STORAGE_READ_FAILED,
            ) from exc

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        try:
            path.unlink(missing_ok=True)
            log.debug("Storage delete", key=key)
        except OSError as exc:
            log.warning("Storage delete failed", key=key, error=str(exc))
            raise StorageError(
                f"Delete failed for key '{key}': {exc}",
                error_code=MLErrorCode.STORAGE_DELETE_FAILED,
            ) from exc

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def get_path(self, key: str) -> str:
        return str(self._resolve(key))
