"""Core interfaces and exceptions for the LLM orchestrator."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Provider(str, Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    GENERIC = "generic"


@dataclass
class GenerationRequest:
    prompt: str
    system_prompt: Optional[str] = None
    provider: Optional[Provider] = None
    model: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResponse:
    content: str
    provider: Provider
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    request_id: Optional[str] = None
    raw: Any = None


class ProviderError(Exception):
    """Raised when an LLM provider call fails."""
    def __init__(self, provider: Provider, message: str, retryable: bool = True):
        self.provider = provider
        self.retryable = retryable
        super().__init__(f"[{provider.value}] {message}")


class HumanReviewRequired(Exception):
    """Raised when moderation flags a request, routing it to the human-review queue."""
    def __init__(self, stage: str, reason: str):
        self.stage = stage  # "pre_generate" or "post_generate"
        self.reason = reason
        super().__init__(f"Human review required at {stage}: {reason}")


class LLMAdapter(ABC):
    """Base class every provider adapter implements."""
    provider: Provider
    default_model: str

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise NotImplementedError

    def _timer(self) -> float:
        return time.perf_counter()

    def _elapsed_ms(self, start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 2)
