"""Public AI facade — Python port of src/lib/ai/index.ts.

Exposes the three high-level entry points the pipeline depends on:
  - generate_text(prompt, options=None, *, category=None, project_id=None) -> str
  - generate_image(prompt, options=None, *, category=None, project_id=None) -> str
        (returns a "/uploads/..." web path to the saved image)
  - generate_video(params, *, provider=None) -> VideoGenerateResult

Provider selection ports provider-factory / index.ts: a default provider is chosen
by initialize_providers() from the configured API keys. generate_text additionally
honors per-project Agent bindings (Bailian/Dify/Coze) when a category+project_id is
supplied — mirroring how the app routes categories to external agents.

Import-safe: importing this module never touches the network, DB, or requires keys.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from app.config import settings
from app.ai.types import (
    TextOptions,
    ImageOptions,
    VideoGenerateParams,
    VideoGenerateResult,
    AIProvider,
    VideoProvider,
)
from app.ai.provider_factory import (
    get_ai_provider,
    get_video_provider,
    set_default_ai_provider,
    set_default_video_provider,
    has_ai_provider,
    has_video_provider,
    create_ai_provider,
    create_video_provider,
    resolve_ai_provider,
    resolve_image_provider,
    resolve_video_provider,
)
from app.ai.setup import initialize_providers

logger = logging.getLogger(__name__)

__all__ = [
    "generate_text",
    "generate_image",
    "generate_video",
    "initialize_providers",
    "TextOptions",
    "ImageOptions",
    "VideoGenerateParams",
    "VideoGenerateResult",
    "AIProvider",
    "VideoProvider",
    "get_ai_provider",
    "get_video_provider",
    "set_default_ai_provider",
    "set_default_video_provider",
    "create_ai_provider",
    "create_video_provider",
    "resolve_ai_provider",
    "resolve_image_provider",
    "resolve_video_provider",
]


def _to_uploads_url(fs_path: str) -> str:
    """Convert an absolute filesystem path under UPLOAD_DIR to a /uploads/... web path."""
    try:
        rel = os.path.relpath(os.path.abspath(fs_path), settings.UPLOAD_DIR)
        rel = rel.replace(os.sep, "/")
        if not rel.startswith(".."):
            return "/uploads/" + rel
    except (ValueError, OSError):
        pass
    # PORT NOTE: path is outside UPLOAD_DIR — fall back to the basename under /uploads.
    return "/uploads/" + Path(fs_path).name


def _lookup_agent(category: Optional[str], project_id: Optional[str]):
    """Best-effort lookup of a bound Agent for (project_id, category).

    Returns an AgentConfig or None. Any DB error is swallowed so text generation
    can fall back to the default LLM provider.
    """
    if not category or not project_id:
        return None
    try:
        from sqlalchemy import select

        from app.db.session import SessionLocal
        from app.db.models import Agent, AgentBinding
        from app.ai.agent_caller import AgentConfig

        session = SessionLocal()
        try:
            binding = session.execute(
                select(AgentBinding).where(
                    AgentBinding.project_id == project_id,
                    AgentBinding.category == category,
                )
            ).scalar_one_or_none()
            if not binding or not binding.agent_id:
                return None
            agent = session.get(Agent, binding.agent_id)
            if not agent:
                return None
            return AgentConfig(platform=agent.platform, app_id=agent.app_id, api_key=agent.api_key)
        finally:
            session.close()
    except Exception as e:  # pragma: no cover - routing is best-effort
        logger.warning("[AI] agent lookup failed for category=%s project=%s: %s", category, project_id, e)
        return None


async def generate_text(
    prompt: str,
    options: Optional[TextOptions] = None,
    *,
    category: Optional[str] = None,
    project_id: Optional[str] = None,
) -> str:
    """Generate text with the configured provider, or a bound agent if one exists."""
    initialize_providers()

    agent_config = _lookup_agent(category, project_id)
    if agent_config is not None:
        from app.ai.agent_caller import call_agent

        return await call_agent(agent_config, prompt)

    provider = get_ai_provider()
    return await provider.generate_text(prompt, options)


async def generate_image(
    prompt: str,
    options: Optional[ImageOptions] = None,
    *,
    category: Optional[str] = None,
    project_id: Optional[str] = None,
) -> str:
    """Generate an image and return its "/uploads/..." web path."""
    initialize_providers()
    provider = get_ai_provider()
    fs_path = await provider.generate_image(prompt, options)
    return _to_uploads_url(fs_path)


async def generate_video(
    params: VideoGenerateParams,
    *,
    provider: Optional[object] = None,
) -> VideoGenerateResult:
    """Generate a video with the configured video provider.

    `provider` may be an explicit VideoProvider instance to use instead of the
    default; otherwise the default configured video provider is used.
    """
    initialize_providers()

    if provider is not None and hasattr(provider, "generate_video"):
        video_provider = provider  # type: ignore[assignment]
    else:
        # PORT NOTE: a string protocol override is not plumbed here (no ProviderConfig
        # source in this simple signature); we use the default configured provider.
        video_provider = get_video_provider()

    result = await video_provider.generate_video(params)
    # Return the saved video as a /uploads web path for consistency with generate_image.
    return VideoGenerateResult(
        file_path=_to_uploads_url(result.file_path),
        last_frame_url=result.last_frame_url,
    )
