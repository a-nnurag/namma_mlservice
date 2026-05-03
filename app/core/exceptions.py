"""Custom exception hierarchy for the ML service."""
from __future__ import annotations
from typing import Any
from .error_codes import MLErrorCode


class MLServiceError(Exception):
    def __init__(
        self,
        message: str,
        error_code: MLErrorCode = MLErrorCode.INTERNAL_ERROR,
        detail: dict[str, Any] | None = None,
        candidate_id: str = "",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.detail = detail or {}
        self.candidate_id = candidate_id

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.error_code}, msg={self.message!r})"


# ── Pipeline ──────────────────────────────────────────────────────────────────

class PipelineError(MLServiceError):
    pass


class PipelineNotFoundError(PipelineError):
    def __init__(self, candidate_id: str) -> None:
        super().__init__(
            f"No pipeline run found for candidate {candidate_id}",
            MLErrorCode.PIPELINE_NOT_FOUND,
            candidate_id=candidate_id,
        )


class PipelineAlreadyRunningError(PipelineError):
    def __init__(self, candidate_id: str) -> None:
        super().__init__(
            f"Pipeline already running for candidate {candidate_id}",
            MLErrorCode.PIPELINE_ALREADY_RUNNING,
            candidate_id=candidate_id,
        )


class PipelineFailedError(PipelineError):
    def __init__(self, candidate_id: str, reason: str = "") -> None:
        super().__init__(
            f"Pipeline failed for candidate {candidate_id}: {reason}",
            MLErrorCode.PIPELINE_FAILED,
            {"reason": reason},
            candidate_id=candidate_id,
        )


class ChunkGapDetectedError(PipelineError):
    def __init__(self, candidate_id: str, media_type: str, missing: list[int]) -> None:
        super().__init__(
            f"Missing chunks for {media_type}: {missing}",
            MLErrorCode.CHUNK_GAP_DETECTED,
            {"media_type": media_type, "missing_chunks": missing},
            candidate_id=candidate_id,
        )


class AssemblyFailedError(PipelineError):
    def __init__(self, candidate_id: str, media_type: str, reason: str = "") -> None:
        super().__init__(
            f"Assembly failed for {media_type}: {reason}",
            MLErrorCode.ASSEMBLY_FAILED,
            {"media_type": media_type, "reason": reason},
            candidate_id=candidate_id,
        )


class VerdictNotReadyError(PipelineError):
    def __init__(self, candidate_id: str, status: str) -> None:
        super().__init__(
            f"Verdict not ready. Current status: {status}",
            MLErrorCode.VERDICT_NOT_READY,
            {"current_status": status},
            candidate_id=candidate_id,
        )


# ── Face ──────────────────────────────────────────────────────────────────────

class FaceError(MLServiceError):
    pass


class FaceNotDetectedError(FaceError):
    def __init__(self, candidate_id: str = "") -> None:
        super().__init__(
            "No face detected in the provided media",
            MLErrorCode.FACE_NOT_DETECTED,
            candidate_id=candidate_id,
        )


class EmbeddingExtractionFailedError(FaceError):
    def __init__(self, candidate_id: str, reason: str = "") -> None:
        super().__init__(
            f"Face embedding extraction failed: {reason}",
            MLErrorCode.EMBEDDING_EXTRACTION_FAILED,
            {"reason": reason},
            candidate_id=candidate_id,
        )


# ── Scoring ───────────────────────────────────────────────────────────────────

class ScoringError(MLServiceError):
    pass


class ScorerAPIFailedError(ScoringError):
    def __init__(self, scorer: str, reason: str = "") -> None:
        super().__init__(
            f"Interview scorer '{scorer}' API call failed: {reason}",
            MLErrorCode.SCORER_API_FAILED,
            {"scorer": scorer, "reason": reason},
        )


class TranscriptEmptyError(ScoringError):
    def __init__(self, candidate_id: str) -> None:
        super().__init__(
            "Transcript is empty — cannot score interview",
            MLErrorCode.TRANSCRIPT_EMPTY,
            candidate_id=candidate_id,
        )


# ── Storage ───────────────────────────────────────────────────────────────────

class StorageError(MLServiceError):
    pass


class StorageWriteFailedError(StorageError):
    def __init__(self, key: str, reason: str = "") -> None:
        super().__init__(
            f"Failed to write '{key}': {reason}",
            MLErrorCode.STORAGE_WRITE_FAILED,
            {"key": key, "reason": reason},
        )


class StorageReadFailedError(StorageError):
    def __init__(self, key: str, reason: str = "") -> None:
        super().__init__(
            f"Failed to read '{key}': {reason}",
            MLErrorCode.STORAGE_READ_FAILED,
            {"key": key, "reason": reason},
        )


class FileNotFoundError(StorageError):
    def __init__(self, key: str) -> None:
        super().__init__(
            f"File not found in storage: {key}",
            MLErrorCode.FILE_NOT_FOUND,
            {"key": key},
        )


# ── Infrastructure ────────────────────────────────────────────────────────────

# ── Face (specific sub-types) ─────────────────────────────────────────────────

class FaceValidationError(FaceError):
    def __init__(self, message: str, error_code: MLErrorCode = MLErrorCode.FACE_VALIDATION_FAILED, candidate_id: str = "") -> None:
        super().__init__(message, error_code, candidate_id=candidate_id)


class LivenessCheckError(FaceError):
    def __init__(self, message: str, error_code: MLErrorCode = MLErrorCode.LIVENESS_CHECK_FAILED, candidate_id: str = "") -> None:
        super().__init__(message, error_code, candidate_id=candidate_id)


class DuplicateCheckError(FaceError):
    def __init__(self, message: str, error_code: MLErrorCode = MLErrorCode.DUPLICATE_CHECK_FAILED, candidate_id: str = "") -> None:
        super().__init__(message, error_code, candidate_id=candidate_id)


# ── Scoring (specific sub-types) ──────────────────────────────────────────────

class TranscriptionError(ScoringError):
    def __init__(self, message: str, error_code: MLErrorCode = MLErrorCode.SCORER_API_FAILED, candidate_id: str = "") -> None:
        super().__init__(message, error_code, candidate_id=candidate_id)


class InterviewScoringError(ScoringError):
    def __init__(self, message: str, error_code: MLErrorCode = MLErrorCode.SCORER_API_FAILED, candidate_id: str = "") -> None:
        super().__init__(message, error_code, candidate_id=candidate_id)


class WorkExpAnalysisError(ScoringError):
    def __init__(self, message: str, error_code: MLErrorCode = MLErrorCode.SCORER_API_FAILED, candidate_id: str = "") -> None:
        super().__init__(message, error_code, candidate_id=candidate_id)


# ── Degree ────────────────────────────────────────────────────────────────────

class DegreeCheckError(MLServiceError):
    def __init__(self, message: str, error_code: MLErrorCode = MLErrorCode.DEGREE_OCR_FAILED, candidate_id: str = "") -> None:
        super().__init__(message, error_code, candidate_id=candidate_id)


# ── Buffer ────────────────────────────────────────────────────────────────────

class ChunkBufferError(PipelineError):
    def __init__(self, message: str, error_code: MLErrorCode = MLErrorCode.CHUNK_BUFFER_OVERFLOW, candidate_id: str = "") -> None:
        super().__init__(message, error_code, candidate_id=candidate_id)


class KafkaConsumerError(MLServiceError):
    def __init__(self, reason: str = "", error_code: MLErrorCode = MLErrorCode.KAFKA_CONSUMER_FAILED, candidate_id: str = "") -> None:
        super().__init__(
            f"Kafka consumer error: {reason}",
            error_code,
            {"reason": reason},
            candidate_id=candidate_id,
        )


class InvalidKafkaMessageError(MLServiceError):
    def __init__(self, topic: str, reason: str = "") -> None:
        super().__init__(
            f"Invalid message on topic '{topic}': {reason}",
            MLErrorCode.KAFKA_MESSAGE_INVALID,
            {"topic": topic, "reason": reason},
        )


class DBConnectionError(MLServiceError):
    def __init__(self, reason: str = "") -> None:
        super().__init__(
            f"Database connection failed: {reason}",
            MLErrorCode.DB_CONNECTION_FAILED,
            {"reason": reason},
        )


class DBQueryError(MLServiceError):
    def __init__(self, operation: str, reason: str = "") -> None:
        super().__init__(
            f"DB query failed in '{operation}': {reason}",
            MLErrorCode.DB_QUERY_FAILED,
            {"operation": operation, "reason": reason},
        )


class DBIntegrityError(MLServiceError):
    def __init__(self, constraint: str = "") -> None:
        super().__init__(
            f"DB integrity constraint violated: {constraint}",
            MLErrorCode.DB_INTEGRITY_ERROR,
            {"constraint": constraint},
        )


class RedisConnectionError(MLServiceError):
    def __init__(self, reason: str = "") -> None:
        super().__init__(
            f"Redis connection failed: {reason}",
            MLErrorCode.REDIS_CONNECTION_FAILED,
            {"reason": reason},
        )
