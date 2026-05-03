"""
File Storage — Strategy Pattern.

StorageBackend is the abstract interface. Pipeline tasks only call
write / read / delete / get_path — never reference local paths directly.

Implementations:
  LocalStorageBackend  — /tmp/ml-service/  (hackathon default)
  S3StorageBackend     — AWS S3 (production multi-instance)

Switch via STORAGE_BACKEND=local|s3
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Abstract storage strategy. All pipeline I/O goes through this."""

    @abstractmethod
    def write(self, key: str, data: bytes) -> str:
        """Persist data under key. Returns the resolved file path / URL."""

    @abstractmethod
    def read(self, key: str) -> bytes:
        """Read and return raw bytes for key. Raises FileNotFoundError if missing."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete the stored object. Silent if key doesn't exist."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if key exists in storage."""

    @abstractmethod
    def get_path(self, key: str) -> str:
        """Return local filesystem path (LocalStorage) or presigned URL (S3)."""
