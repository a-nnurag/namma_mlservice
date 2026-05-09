"""
Kafka Consumer — Adapter Pattern.

KafkaConsumerAdapter       → ABC
AIOKafkaConsumerAdapter    → production (aiokafka)
MockKafkaConsumerAdapter   → in-memory for tests / local dev

Each message on any topic is dispatched to a registered handler:
    handler(topic: str, key: str, payload: dict) -> None
"""
from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Awaitable, Callable

from app.config import settings
from app.core.error_codes import MLErrorCode
from app.core.exceptions import KafkaConsumerError
from app.core.logging import get_logger

log = get_logger(__name__)

Handler = Callable[[str, str, dict], Awaitable[None]]

TOPICS = [
    "candidate.interview.video",
    "candidate.interview.audio",
    "candidate.workexp.video",
    "candidate.meta",
]


# ── Abstract base ─────────────────────────────────────────────────────────────

class KafkaConsumerAdapter(ABC):
    """Abstract Kafka consumer.  Implementations must be async-context-safe."""

    @abstractmethod
    async def start(self) -> None:
        """Connect to broker and begin consuming."""

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully stop the consumer."""

    @abstractmethod
    def register_handler(self, topic: str, handler: Handler) -> None:
        """Register an async handler for a specific topic."""

    @abstractmethod
    def is_running(self) -> bool:
        """Return True if the consumer loop is active."""


# ── AIOKafka production adapter ───────────────────────────────────────────────

class AIOKafkaConsumerAdapter(KafkaConsumerAdapter):
    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        topics: list[str] = TOPICS,
        use_ssl: bool = False,
        sasl_username: str = "",
        sasl_password: str = "",
    ) -> None:
        self._bootstrap = bootstrap_servers
        self._group_id = group_id
        self._topics = topics
        self._use_ssl = use_ssl
        self._sasl_username = sasl_username
        self._sasl_password = sasl_password
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._consumer = None
        self._task: asyncio.Task | None = None
        self._running = False

    def register_handler(self, topic: str, handler: Handler) -> None:
        self._handlers[topic].append(handler)
        log.debug("Handler registered", topic=topic, handler=handler.__name__)

    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        try:
            from aiokafka import AIOKafkaConsumer  # type: ignore[import]
        except ImportError as exc:
            raise KafkaConsumerError(
                "aiokafka not installed. pip install aiokafka",
                error_code=MLErrorCode.KAFKA_CONSUMER_FAILED,
            ) from exc

        kwargs: dict = dict(
            bootstrap_servers=self._bootstrap,
            group_id=self._group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda v: v,
        )
        if self._use_ssl:
            import ssl
            kwargs.update(
                security_protocol="SASL_SSL",
                sasl_mechanism="SCRAM-SHA-256",
                sasl_plain_username=self._sasl_username,
                sasl_plain_password=self._sasl_password,
                ssl_context=ssl.create_default_context(),
            )
        self._consumer = AIOKafkaConsumer(*self._topics, **kwargs)
        try:
            await self._consumer.start()
            self._running = True
            self._task = asyncio.create_task(self._consume_loop())
            log.info(
                "AIOKafka consumer started",
                topics=self._topics,
                group=self._group_id,
            )
        except Exception as exc:
            log.error("Failed to start Kafka consumer", error=str(exc))
            raise KafkaConsumerError(
                f"Kafka consumer start failed: {exc}",
                error_code=MLErrorCode.KAFKA_CONSUMER_FAILED,
            ) from exc

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._consumer:
            await self._consumer.stop()
        log.info("AIOKafka consumer stopped")

    async def _consume_loop(self) -> None:
        assert self._consumer is not None
        try:
            async for msg in self._consumer:
                topic = msg.topic
                key = msg.key.decode() if msg.key else ""
                try:
                    payload = json.loads(msg.value)
                except json.JSONDecodeError:
                    log.warning(
                        "Non-JSON message received",
                        topic=topic,
                        key=key,
                    )
                    continue

                for handler in self._handlers.get(topic, []):
                    try:
                        await handler(topic, key, payload)
                    except Exception as exc:
                        log.error(
                            "Handler raised exception",
                            topic=topic,
                            key=key,
                            handler=handler.__name__,
                            error=str(exc),
                        )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.critical("Kafka consume loop crashed", error=str(exc), exc_info=True)
            self._running = False


# ── Mock adapter (local dev / tests) ─────────────────────────────────────────

class MockKafkaConsumerAdapter(KafkaConsumerAdapter):
    """In-memory consumer.  Push messages via inject_message() in tests."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._running = False
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def register_handler(self, topic: str, handler: Handler) -> None:
        self._handlers[topic].append(handler)

    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        log.info("MockKafkaConsumer started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("MockKafkaConsumer stopped")

    async def inject_message(self, topic: str, key: str, payload: dict) -> None:
        await self._queue.put((topic, key, payload))

    async def _loop(self) -> None:
        while self._running:
            try:
                topic, key, payload = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                for handler in self._handlers.get(topic, []):
                    try:
                        await handler(topic, key, payload)
                    except Exception as exc:
                        log.error(
                            "Mock handler error",
                            topic=topic,
                            error=str(exc),
                        )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break


# ── Factory ───────────────────────────────────────────────────────────────────

def build_kafka_consumer() -> KafkaConsumerAdapter:
    backend = settings.KAFKA_CONSUMER_BACKEND.lower()
    if backend == "aiokafka":
        log.info("Using AIOKafkaConsumerAdapter", servers=settings.KAFKA_BOOTSTRAP_SERVERS, ssl=settings.KAFKA_USE_SSL)
        return AIOKafkaConsumerAdapter(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            group_id=settings.KAFKA_CONSUMER_GROUP,
            use_ssl=settings.KAFKA_USE_SSL,
            sasl_username=settings.KAFKA_SASL_USERNAME,
            sasl_password=settings.KAFKA_SASL_PASSWORD,
        )
    log.info("Using MockKafkaConsumerAdapter")
    return MockKafkaConsumerAdapter()
