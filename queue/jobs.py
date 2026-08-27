"""Redis job queue representation with manual review transitions."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Protocol


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    REVIEW_PENDING = "review_pending"


@dataclass
class Job:
    id: str
    payload: dict[str, Any]
    status: JobStatus = JobStatus.QUEUED
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AsyncRedisLike(Protocol):
    async def set(self, key: str, value: str) -> Any: ...
    async def get(self, key: str) -> Any: ...
    async def lpush(self, key: str, value: str) -> Any: ...
    async def brpop(self, keys: list[str], timeout: int = 0) -> Any: ...


QUEUE_KEY = "generation:queue"
JOB_KEY_PREFIX = "generation:job:"


def _serialize(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "payload": job.payload,
        "status": job.status.value,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def _deserialize(data: dict[str, Any]) -> Job:
    return Job(
        id=data["id"],
        payload=data["payload"],
        status=JobStatus(data["status"]),
        result=data.get("result"),
        error=data.get("error"),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
    )


class RedisJobQueue:
    def __init__(self, redis_client: AsyncRedisLike):
        self._redis = redis_client

    async def enqueue(self, payload: dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        job = Job(id=job_id, payload=payload)
        await self._redis.set(JOB_KEY_PREFIX + job_id, json.dumps(_serialize(job)))
        await self._redis.lpush(QUEUE_KEY, job_id)
        return job_id

    async def dequeue(self, timeout: int = 5) -> Optional[str]:
        res = await self._redis.brpop([QUEUE_KEY], timeout=timeout)
        if not res:
            return None
        _, job_id = res
        return job_id.decode() if isinstance(job_id, bytes) else job_id

    async def get_job(self, job_id: str) -> Optional[Job]:
        raw = await self._redis.get(JOB_KEY_PREFIX + job_id)
        if not raw:
            return None
        decoded = raw.decode() if isinstance(raw, bytes) else raw
        return _deserialize(json.loads(decoded))

    async def update_job(self, job: Job) -> None:
        job.updated_at = datetime.now(timezone.utc).isoformat()
        await self._redis.set(JOB_KEY_PREFIX + job.id, json.dumps(_serialize(job)))

    async def mark_review_result(self, job_id: str, approved: bool) -> None:
        """Updates job status when human review resolves."""
        job = await self.get_job(job_id)
        if not job:
            return

        if approved:
            if job.result is not None:
                # Post-generation: Output was already generated, finalize status
                job.status = JobStatus.DONE
                job.error = None
            else:
                # Pre-generation: Prompt approved, bypass checks and requeue
                job.payload["approved_by_moderator"] = True
                job.status = JobStatus.QUEUED
                job.error = None
                await self.update_job(job)
                await self._redis.lpush(QUEUE_KEY, job_id)
                return
        else:
            job.status = JobStatus.FAILED
            job.error = "Rejected by manual content review"
            job.result = None

        await self.update_job(job)

    async def close(self) -> None:
        close_method = getattr(self._redis, "close", getattr(self._redis, "aclose", None))
        if close_method:
            await close_method()
