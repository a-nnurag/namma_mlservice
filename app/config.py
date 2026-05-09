from pydantic_settings import BaseSettings


class MLSettings(BaseSettings):
    # Service
    ML_SERVICE_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # PostgreSQL (ML service has its own separate DB)
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/nammakelsa_ml"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_BROKER_DB: int = 0      # Celery broker + result backend
    REDIS_BUFFER_DB: int = 1      # Kafka chunk buffer
    CHUNK_BUFFER_TTL_SECONDS: int = 86400  # 24h

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_CONSUMER_GROUP: str = "ml-service-group"
    KAFKA_CONSUMER_BACKEND: str = "aiokafka"  # aiokafka | mock
    KAFKA_USE_SSL: bool = False
    KAFKA_SASL_USERNAME: str = ""
    KAFKA_SASL_PASSWORD: str = ""

    # Storage strategy
    STORAGE_BACKEND: str = "local"  # local | s3
    LOCAL_STORAGE_DIR: str = "/tmp/ml-service"
    S3_BUCKET_NAME: str = ""
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "ap-south-1"

    # Interview scorer strategy
    INTERVIEW_SCORER_BACKEND: str = "rules"  # claude | gemini | openai | rules
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # Celery
    CELERY_CONCURRENCY: int = 2
    CELERY_MAX_RETRIES: int = 3
    CELERY_RETRY_BACKOFF: int = 60  # seconds, doubles on each retry

    # Face similarity threshold (same as backend)
    FACE_SIMILARITY_THRESHOLD: float = 0.85
    VOICE_SIMILARITY_THRESHOLD: float = 0.75

    # CORS — comma-separated list of allowed origins
    ALLOWED_ORIGINS: str = "http://localhost:3000"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = MLSettings()
