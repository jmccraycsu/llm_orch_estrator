"""Moderation policy service. Content triggers human review if thresholds are exceeded."""

from __future__ import annotations

import logging
from typing import Optional
from ..moderation_hooks import ModerationResult

logger = logging.getLogger("moderation.service")

# All standard policy violations route to review queue (none auto-block/reject)
FLAG_TEXT_CLASSES = ("child_exploitation", "hate", "violence", "bullying", "self_harm")
REVIEW_THRESHOLD = 1


class ModerationService:
    def __init__(
        self,
        text_client: Optional[HiveTextModerationClient] = None,
        image_client: Optional[SightengineImageModerationClient] = None,
    ):
        self._text_client = text_client
        self._image_client = image_client

    async def check_text_input(self, text: str, user_id: str | None) -> ModerationResult:
        return await self._check_text(text, user_id)

    async def check_text_output(self, text: str, user_id: str | None) -> ModerationResult:
        return await self._check_text(text, user_id)

    async def _check_text(self, text: str, user_id: str | None) -> ModerationResult:
        if self._text_client is None:
            # Fail closed: Route to human review if client is unconfigured
            logger.warning("moderation_unconfigured_flagged", extra={"user_id": user_id})
            return ModerationResult(allowed=True, needs_human_review=True, reason="Moderation unconfigured")

        try:
            raw = await self._text_client.moderate_text(text)
        except Exception:
            logger.exception("moderation_provider_error", extra={"user_id": user_id})
            return ModerationResult(allowed=True, needs_human_review=True, reason="Moderation system error")

        scores = self._text_client.extract_class_scores(raw)
        
        # Check if any violation category is flagged
        flagged_classes = [
            cls for cls in FLAG_TEXT_CLASSES 
            if scores.get(cls, 0) >= REVIEW_THRESHOLD
        ]

        if flagged_classes:
            logger.info("content_flagged_for_review", extra={"classes": flagged_classes, "user_id": user_id})
            return ModerationResult(allowed=True, needs_human_review=True, reason=f"Flagged categories: {', '.join(flagged_classes)}")

        return ModerationResult(allowed=True, needs_human_review=False)

    async def check_image(self, image_url: str) -> ModerationResult:
        if self._image_client is None:
            return ModerationResult(allowed=True, needs_human_review=True, reason="Moderation unconfigured")
        try:
            raw = await self._image_client.moderate_image_url(image_url)
        except Exception:
            return ModerationResult(allowed=True, needs_human_review=True, reason="Moderation system error")

        gore_prob = raw.get("gore", {}).get("prob", 0)
        if gore_prob >= 0.8:
            return ModerationResult(allowed=True, needs_human_review=True, reason="Violence threshold reached")
        return ModerationResult(allowed=True, needs_human_review=False)

    async def close(self) -> None:
        if self._text_client:
            await self._text_client.close()
        if self._image_client:
            await self._image_client.close()
