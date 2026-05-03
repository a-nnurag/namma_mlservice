"""S3StorageBackend — stores files in AWS S3 (production)."""
from __future__ import annotations

from app.config import settings
from app.core.error_codes import MLErrorCode
from app.core.exceptions import StorageError
from app.core.logging import get_logger
from app.storage.base import StorageBackend

log = get_logger(__name__)


class S3StorageBackend(StorageBackend):
    """Stores files in AWS S3.  Requires boto3 and valid AWS credentials."""

    def __init__(self) -> None:
        try:
            import boto3  # type: ignore[import]
            self._s3 = boto3.client(
                "s3",
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID or None,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY or None,
            )
            self._bucket = settings.S3_BUCKET_NAME
            log.info("S3StorageBackend ready", bucket=self._bucket)
        except ImportError as exc:
            raise RuntimeError("boto3 is required for S3 storage. pip install boto3") from exc

    def write(self, key: str, data: bytes) -> str:
        try:
            self._s3.put_object(Bucket=self._bucket, Key=key, Body=data)
            log.debug("S3 write", key=key, bytes=len(data))
            return f"s3://{self._bucket}/{key}"
        except Exception as exc:
            log.error("S3 write failed", key=key, error=str(exc))
            raise StorageError(
                f"S3 write failed for key '{key}': {exc}",
                error_code=MLErrorCode.STORAGE_WRITE_FAILED,
            ) from exc

    def read(self, key: str) -> bytes:
        try:
            response = self._s3.get_object(Bucket=self._bucket, Key=key)
            data = response["Body"].read()
            log.debug("S3 read", key=key, bytes=len(data))
            return data
        except self._s3.exceptions.NoSuchKey:
            raise FileNotFoundError(f"S3 key not found: {key}")
        except Exception as exc:
            log.error("S3 read failed", key=key, error=str(exc))
            raise StorageError(
                f"S3 read failed for key '{key}': {exc}",
                error_code=MLErrorCode.STORAGE_READ_FAILED,
            ) from exc

    def delete(self, key: str) -> None:
        try:
            self._s3.delete_object(Bucket=self._bucket, Key=key)
            log.debug("S3 delete", key=key)
        except Exception as exc:
            log.warning("S3 delete failed", key=key, error=str(exc))
            raise StorageError(
                f"S3 delete failed for key '{key}': {exc}",
                error_code=MLErrorCode.STORAGE_DELETE_FAILED,
            ) from exc

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False

    def get_path(self, key: str) -> str:
        """Return a 1-hour presigned URL for the object."""
        try:
            url = self._s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=3600,
            )
            return url
        except Exception as exc:
            raise StorageError(
                f"Failed to generate presigned URL for '{key}': {exc}",
                error_code=MLErrorCode.STORAGE_READ_FAILED,
            ) from exc


def build_storage_backend() -> StorageBackend:
    backend = settings.STORAGE_BACKEND.lower()
    if backend == "s3":
        log.info("Using S3StorageBackend")
        return S3StorageBackend()
    log.info("Using LocalStorageBackend")
    from app.storage.local import LocalStorageBackend
    return LocalStorageBackend()
