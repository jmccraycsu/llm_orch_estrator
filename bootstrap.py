"""Wiring configuration file."""

from __future__ import annotations

import logging
from .adapters import ClaudeAdapter, GenericOpenAICompatAdapter, OpenAIAdapter
from .audit.db import create_engine, create_session_factory
from .audit.repository import AuditLogRepository
from .base import Provider
from .config import Settings
from .moderation.clients import HiveTextModerationClient, SightengineImageModerationClient
from .moderation.service import ModerationService
from .moderation_hooks import build_post_generate_hook, build_pre_generate_hook
from .orchestrator import LLMOrchestrator
from .queue.jobs import RedisJobQueue, create_redis_job_queue

logger = logging.getLogger("bootstrap")


def build_job_queue(settings: Settings) -> RedisJobQueue:
    return create_redis_job_queue(settings.redis_url)


def build_audit_repository(settings: Settings) -> AuditLogRepository:
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    return AuditLogRepository(session_factory)


def build_moderation_service(settings: Settings) -> ModerationService:
    text_client = (
        HiveTextModerationClient(api_key=settings.hive_api_key)
        if settings.hive_api_key
        else None
    )
    image_client = (
        SightengineImageModerationClient(
            api_user=settings.sightengine_api_user,
            api_secret=settings.sightengine_api_secret,
        )
        if settings.sightengine_api_user and settings.sightengine_api_secret
        else None
    )
    if text_client is None:
        logger.warning("hive_api_key not configured - text generation checks will default-fail to manual review")
    return ModerationService(text_client=text_client, image_client=image_client)


def build_orchestrator(settings: Settings, audit_repository: AuditLogRepository) -> LLMOrchestrator:
    adapters = {}

    if settings.anthropic_api_key:
        adapters[Provider.CLAUDE] = ClaudeAdapter(api_key=settings.anthropic_api_key)
    if settings.openai_api_key:
        adapters[Provider.OPENAI] = OpenAIAdapter(api_key=settings.openai_api_key)
    if settings.self_hosted_base_url:
        adapters[Provider.GENERIC] = GenericOpenAICompatAdapter(
            base_url=settings.self_hosted_base_url,
            api_key=settings.self_hosted_api_key,
            default_model=settings.self_hosted_model,
        )

    if not adapters:
        raise RuntimeError("No provider credentials found. Provide anthropic_api_key, openai_api_key, or a self-hosted target.")

    # Dynamically select default provider depending on configured APIs
    default_provider = Provider.CLAUDE if Provider.CLAUDE in adapters else next(iter(adapters))
    fallback_chain = [p for p in adapters if p != default_provider]

    orchestrator = LLMOrchestrator(
        adapters=adapters,
        default_provider=default_provider,
        fallback_chain=fallback_chain,
        max_retries_per_provider=settings.max_retries_per_provider,
    )

    moderation_service = build_moderation_service(settings)
    orchestrator.register_pre_generate_hook(build_pre_generate_hook(moderation_service, audit_repository))
    orchestrator.register_post_generate_hook(build_post_generate_hook(moderation_service, audit_repository))
    
    return orchestrator
