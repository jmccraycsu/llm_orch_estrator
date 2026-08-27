"""Central LLM execution orchestrator."""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from typing import Awaitable, Callable, Optional
from .base import GenerationRequest, GenerationResponse, LLMAdapter, Provider, ProviderError

logger = logging.getLogger("llm_orchestrator")

PreGenerateHook = Callable[[GenerationRequest], Awaitable[None]]
PostGenerateHook = Callable[[GenerationRequest, GenerationResponse], Awaitable[None]]


class LLMOrchestrator:
    def __init__(
        self,
        adapters: dict[Provider, LLMAdapter],
        default_provider: Provider,
        fallback_chain: Optional[list[Provider]] = None,
        max_retries_per_provider: int = 2,
        retry_backoff_s: float = 1.5,
    ):
        self._adapters = adapters
        self._default_provider = default_provider
        self._fallback_chain = fallback_chain or []
        self._max_retries = max_retries_per_provider
        self._retry_backoff_s = retry_backoff_s
        self._pre_hooks: list[PreGenerateHook] = []
        self._post_hooks: list[PostGenerateHook] = []

    def register_pre_generate_hook(self, hook: PreGenerateHook) -> None:
        self._pre_hooks.append(hook)

    def register_post_generate_hook(self, hook: PostGenerateHook) -> None:
        self._post_hooks.append(hook)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if request.request_id is None:
            request.request_id = str(uuid.uuid4())

        for hook in self._pre_hooks:
            await hook(request)

        order = self._build_attempt_order(request.provider)
        last_error: Optional[Exception] = None
        response: Optional[GenerationResponse] = None

        for provider in order:
            adapter = self._adapters.get(provider)
            if adapter is None:
                continue
            try:
                response = await self._call_with_retries(adapter, request)
                break
            except ProviderError as e:
                logger.warning("provider_attempt_failed", extra={"provider": provider.value, "request_id": request.request_id})
                last_error = e
                continue
        else:
            raise last_error or ProviderError(self._default_provider, "No adapters configured or available", retryable=False)

        for hook in self._post_hooks:
            await hook(request, response)

        return response

    def _build_attempt_order(self, requested: Optional[Provider]) -> list[Provider]:
        order: list[Provider] = []
        if requested:
            order.append(requested)
        if self._default_provider not in order:
            order.append(self._default_provider)
        for p in self._fallback_chain:
            if p not in order:
                order.append(p)
        return order

    async def _call_with_retries(self, adapter: LLMAdapter, request: GenerationRequest) -> GenerationResponse:
        attempt = 0
        while True:
            try:
                return await adapter.generate(request)
            except ProviderError as e:
                attempt += 1
                if not e.retryable or attempt > self._max_retries:
                    raise
                backoff = self._retry_backoff_s * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                await asyncio.sleep(backoff)

    async def close(self) -> None:
        for adapter in self._adapters.values():
            close_method = getattr(adapter, "close", None)
            if close_method:
                await close_method()
