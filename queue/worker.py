"""Worker polling logic."""

from __future__ import annotations

import asyncio
import logging
from ..base import GenerationRequest, HumanReviewRequired, Provider, ProviderError
from ..orchestrator import LLMOrchestrator
from ..prompts import default_registry
from .jobs import JobStatus, RedisJobQueue

logger = logging.getLogger("worker")


async def process_one(queue: RedisJobQueue, orchestrator: LLMOrchestrator, job_id: str) -> None:
    job = await queue.get_job(job_id)
    if not job:
        return

    try:
        job.status = JobStatus.RUNNING
        await queue.update_job(job)

        payload = job.payload
        provider = Provider(payload["provider"]) if payload.get("provider") else None

        # Build prompt securely using server template
        template = default_registry.get("creative_writing", "latest")
        system_prompt, user_prompt = template.render(user_prompt=payload["prompt"])

        request = GenerationRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            provider=provider,
            request_id=job_id,
            user_id=payload.get("user_id"),
            metadata={"approved_by_moderator": payload.get("approved_by_moderator", False)},
        )

        response = await orchestrator.generate(request)

    except HumanReviewRequired as exc:
        logger.info("job_held_for_review", extra={"job_id": job_id, "stage": exc.stage})
        job.status = JobStatus.REVIEW_PENDING
        job.error = f"Review pending: {exc.reason}"
        # If blocked post-generation, hold result in metadata (unreadable by user until approved)
        if exc.stage == "post_generate" and 'response' in locals():
            resp = locals()['response']
            job.result = {
                "content": resp.content,
                "provider": resp.provider.value,
                "model": resp.model,
            }
        await queue.update_job(job)

    except ProviderError as exc:
        job.status = JobStatus.FAILED
        job.error = f"LLM generation failed: {exc}"
        await queue.update_job(job)

    except Exception as exc:
        logger.exception("unhandled_worker_processing_failure", extra={"job_id": job_id})
        job.status = JobStatus.FAILED
        job.error = f"Internal system failure: {exc}"
        await queue.update_job(job)

    else:
        job.status = JobStatus.DONE
        job.error = None
        job.result = {
            "content": response.content,
            "provider": response.provider.value,
            "model": response.model,
        }
        await queue.update_job(job)


async def run_worker(
    queue: RedisJobQueue,
    orchestrator: LLMOrchestrator,
    poll_timeout: int = 5,
    stop_event: asyncio.Event | None = None,
) -> None:
    logger.info("worker_started")
    while stop_event is None or not stop_event.is_set():
        try:
            job_id = await queue.dequeue(timeout=poll_timeout)
            if not job_id:
                continue
            await process_one(queue, orchestrator, job_id)
        except Exception:
            logger.exception("worker_loop_iteration_failure")
            await asyncio.sleep(1)
