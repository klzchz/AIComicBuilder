"""Provider factory + default-provider registry.

Python port of src/lib/ai/provider-factory.ts merged with the default-provider
registry from src/lib/ai/index.ts (getAIProvider/getVideoProvider/setDefault*).
Keeping both here avoids a circular import with app.ai.__init__.

Providers are imported lazily inside the factory functions so that importing this
module never constructs a client or requires an SDK before it is needed.
"""
from __future__ import annotations

from typing import Callable, Optional

from app.ai.ai_sdk import ProviderConfig
from app.ai.types import AIProvider, VideoProvider

# ── Default provider registry (port of index.ts) ────────────────────

_default_ai_provider: Optional[AIProvider] = None
_default_video_provider: Optional[VideoProvider] = None
_default_ai_provider_factory: Optional[Callable[[Optional[str]], AIProvider]] = None
_default_video_provider_factory: Optional[Callable[[Optional[str]], VideoProvider]] = None


def set_default_ai_provider(
    provider: AIProvider,
    factory: Optional[Callable[[Optional[str]], AIProvider]] = None,
) -> None:
    global _default_ai_provider, _default_ai_provider_factory
    _default_ai_provider = provider
    if factory:
        _default_ai_provider_factory = factory


def set_default_video_provider(
    provider: VideoProvider,
    factory: Optional[Callable[[Optional[str]], VideoProvider]] = None,
) -> None:
    global _default_video_provider, _default_video_provider_factory
    _default_video_provider = provider
    if factory:
        _default_video_provider_factory = factory


def get_ai_provider(upload_dir: Optional[str] = None) -> AIProvider:
    if upload_dir and _default_ai_provider_factory:
        return _default_ai_provider_factory(upload_dir)
    if _default_ai_provider is None:
        raise RuntimeError(
            "No AI provider configured. Call initialize_providers() / set_default_ai_provider() first."
        )
    return _default_ai_provider


def get_video_provider(upload_dir: Optional[str] = None) -> VideoProvider:
    if upload_dir and _default_video_provider_factory:
        return _default_video_provider_factory(upload_dir)
    if _default_video_provider is None:
        raise RuntimeError(
            "No video provider configured. Call initialize_providers() / set_default_video_provider() first."
        )
    return _default_video_provider


def has_ai_provider() -> bool:
    return _default_ai_provider is not None


def has_video_provider() -> bool:
    return _default_video_provider is not None


# ── Factory (port of provider-factory.ts) ───────────────────────────


def create_ai_provider(config: ProviderConfig, upload_dir: Optional[str] = None) -> AIProvider:
    protocol = config.protocol
    if protocol == "openai":
        from app.ai.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model_id,
            upload_dir=upload_dir,
        )
    if protocol == "gemini":
        from app.ai.providers.gemini import GeminiProvider

        return GeminiProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model_id,
            upload_dir=upload_dir,
        )
    if protocol == "kling":
        from app.ai.providers.kling_image import KlingImageProvider

        return KlingImageProvider(
            api_key=config.api_key,
            secret_key=config.secret_key,
            base_url=config.base_url,
            model=config.model_id,
            upload_dir=upload_dir,
        )
    if protocol == "dashscope":
        from app.ai.providers.dashscope_image import DashScopeImageProvider

        return DashScopeImageProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model_id,
            upload_dir=upload_dir,
        )
    raise ValueError(f"Unsupported AI protocol: {protocol}")


def create_video_provider(config: ProviderConfig, upload_dir: Optional[str] = None) -> VideoProvider:
    protocol = config.protocol
    if protocol == "seedance":
        from app.ai.providers.seedance import SeedanceProvider

        return SeedanceProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model_id,
            upload_dir=upload_dir,
        )
    if protocol == "gemini":
        from app.ai.providers.veo import VeoProvider

        return VeoProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model_id,
            upload_dir=upload_dir,
        )
    if protocol == "kling":
        from app.ai.providers.kling_video import KlingVideoProvider

        return KlingVideoProvider(
            api_key=config.api_key,
            secret_key=config.secret_key,
            base_url=config.base_url,
            model=config.model_id,
            upload_dir=upload_dir,
        )
    if protocol == "wan":
        from app.ai.providers.wan_video import WanVideoProvider

        return WanVideoProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model_id,
            upload_dir=upload_dir,
        )
    if protocol == "ucloud-seedance":
        from app.ai.providers.ucloud_seedance import UCloudSeedanceProvider

        return UCloudSeedanceProvider(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model_id,
            upload_dir=upload_dir,
        )
    raise ValueError(f"Unsupported video protocol: {protocol}")


def resolve_ai_provider(text_config: Optional[ProviderConfig] = None) -> AIProvider:
    if text_config:
        return create_ai_provider(text_config)
    return get_ai_provider()


def resolve_image_provider(
    image_config: Optional[ProviderConfig] = None, upload_dir: Optional[str] = None
) -> AIProvider:
    if image_config:
        return create_ai_provider(image_config, upload_dir)
    return get_ai_provider(upload_dir)


def resolve_video_provider(
    video_config: Optional[ProviderConfig] = None, upload_dir: Optional[str] = None
) -> VideoProvider:
    if video_config:
        return create_video_provider(video_config, upload_dir)
    return get_video_provider(upload_dir)
