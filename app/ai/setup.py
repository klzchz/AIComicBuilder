"""Provider initialization — Python port of src/lib/ai/setup.ts.

Idempotent and safe to call with no API keys configured (it simply registers
nothing). Providers are imported lazily so importing this module has no side
effects and requires no SDK/keys.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.config import settings
from app.ai.provider_factory import set_default_ai_provider, set_default_video_provider

logger = logging.getLogger(__name__)

_initialized = False


def initialize_providers() -> None:
    global _initialized
    if _initialized:
        return

    if settings.OPENAI_API_KEY:
        from app.ai.providers.openai_provider import OpenAIProvider

        set_default_ai_provider(
            OpenAIProvider(),
            lambda upload_dir=None: OpenAIProvider(upload_dir=upload_dir) if upload_dir else OpenAIProvider(),
        )
        logger.info("[AI setup] default text/image provider: OpenAI")
    elif settings.GEMINI_API_KEY:
        from app.ai.providers.gemini import GeminiProvider

        set_default_ai_provider(
            GeminiProvider(),
            lambda upload_dir=None: GeminiProvider(upload_dir=upload_dir) if upload_dir else GeminiProvider(),
        )
        logger.info("[AI setup] default text/image provider: Gemini")

    if settings.SEEDANCE_API_KEY:
        from app.ai.providers.seedance import SeedanceProvider

        set_default_video_provider(
            SeedanceProvider(),
            lambda upload_dir=None: SeedanceProvider(upload_dir=upload_dir) if upload_dir else SeedanceProvider(),
        )
        logger.info("[AI setup] default video provider: Seedance")

    if not (settings.OPENAI_API_KEY or settings.GEMINI_API_KEY):
        logger.info("[AI setup] no text/image provider key configured — providers not registered")

    _initialized = True
