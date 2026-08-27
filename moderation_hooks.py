"""Moderation interceptor hooks for the LLMOrchestrator pipeline."""

from __future__ import annotations

from typing import Protocol
from .base import GenerationRequest, GenerationResponse, HumanReviewRequired


class ModerationClient(Protocol):
    async def check_text_input(self, text: str, user_id: str | None) -> ModerationResult: ...
    async def check_text_output(self, text: str, user_id: str | None) -> ModerationResult: ...


class ModerationResult:
    def __init__(self, allowed: bool, needs_human_review: bool = False, reason: str = ""):
        self.allowed = allowed
        self.needs_human_review = needs_human_review
        self.reason = reason


def build_pre_generate_hook(moderation_client: ModerationClient, audit_log):
    async def pre_generate_hook(request: GenerationRequest) -> None:
        # Honor bypass metadata if manually resolved/approved by moderator
        if request.metadata.get("approved_by_moderator"):
            return

        result = await moderation_client.check_text_input(request.prompt, request.user_id)
        
        await audit_log.record_moderation_event(
            stage="pre_generate",
            request_id=request.request_id,
            user_id=request.user_id,
            allowed=not result.needs_human_review,
            reason=result.reason,
        )

        if result.needs_human_review:
            await audit_log.enqueue_human_review(request.request_id)
            raise HumanReviewRequired(stage="pre_generate", reason=result.reason)

    return pre_generate_hook


def build_post_generate_hook(moderation_client: ModerationClient, audit_log):
    async def post_generate_hook(request: GenerationRequest, response: GenerationResponse) -> None:
        if request.metadata.get("approved_by_moderator"):
            return

        result = await moderation_client.check_text_output(response.content, request.user_id)
        
        await audit_log.record_moderation_event(
            stage="post_generate",
            request_id=request.request_id,
            user_id=request.user_id,
            allowed=not result.needs_human_review,
            reason=result.reason,
        )

        if result.needs_human_review:
            await audit_log.enqueue_human_review(request.request_id)
            raise HumanReviewRequired(stage="post_generate", reason=result.reason)

    return post_generate_hook
