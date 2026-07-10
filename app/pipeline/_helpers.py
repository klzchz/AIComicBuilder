"""Shared helpers for the generation pipeline.

Ports the small pieces the TS stages relied on that don't already live in
``app.api._common`` (which ports src/lib/shot-asset-utils.ts, staleness.ts and
import-utils.ts):

    - ``get_active_asset`` / ``patch_asset``  (shot-asset-utils.ts)
    - ``resolve_prompt`` / ``resolve_slot_contents`` async-tolerant wrappers
      (src/lib/ai/prompts/resolver.ts — built in parallel)
    - ``get_model_max_duration``               (src/lib/ai/model-limits.ts)
    - ``payload_of``                           parse a Task's JSON payload

Cross-subsystem dependencies (app.ai, app.ai.prompts, app.task_queue) are
imported LAZILY inside functions so importing this package never triggers an
import cycle and stays safe while sibling subsystems are built in parallel.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ShotAsset, Task


# ---------------------------------------------------------------------------
# Task payload
# ---------------------------------------------------------------------------


def payload_of(task: Any) -> dict[str, Any]:
    """Return a task's payload as a dict.

    The worker (built in parallel) hands handlers a ``Task`` row whose
    ``payload`` column is a JSON string (mirroring Drizzle's json mode). Be
    liberal: accept a Task, a raw dict, or a JSON string.
    """
    if isinstance(task, Task) or hasattr(task, "payload"):
        raw = getattr(task, "payload", None)
    else:
        raw = task
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


# ---------------------------------------------------------------------------
# Shot assets — the two helpers not present in app.api._common
# (get_active_asset + patch_asset from src/lib/shot-asset-utils.ts)
# ---------------------------------------------------------------------------


def get_active_asset(
    db: Session, shot_id: str, type_: str, sequence_in_type: int = 0
) -> Optional[ShotAsset]:
    """Get the single currently-active asset for a (shot, type, seq) slot.

    Port of getActiveAsset.
    """
    return db.execute(
        select(ShotAsset)
        .where(
            ShotAsset.shot_id == shot_id,
            ShotAsset.type == type_,
            ShotAsset.sequence_in_type == sequence_in_type,
            ShotAsset.is_active == 1,
        )
        .limit(1)
    ).scalar_one_or_none()


def patch_asset(db: Session, asset_id: str, **patch: Any) -> None:
    """Update an existing asset row in place. Port of patchAsset.

    Accepts snake_case column names: file_url, status, prompt, model_provider,
    model_id, meta. ``meta``/``characters`` are JSON-encoded when given as
    dict/list.
    """
    asset = db.get(ShotAsset, asset_id)
    if asset is None:
        return
    for key, value in patch.items():
        if key in ("meta", "characters") and value is not None and not isinstance(value, str):
            value = json.dumps(value)
        setattr(asset, key, value)


# ---------------------------------------------------------------------------
# Prompt resolution (src/lib/ai/prompts/resolver.ts — built in parallel)
# ---------------------------------------------------------------------------


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


async def resolve_prompt(prompt_key: str, user_id: str = "", project_id: str = "") -> Optional[str]:
    """Resolve a prompt's system content (project > global > code default).

    Port of resolvePrompt. The resolver lives in app.ai.prompts (built in
    parallel); import lazily and tolerate a sync or async implementation. If it
    isn't available yet, return None so the AI layer falls back to its own
    default system prompt.
    """
    try:
        from app.ai.prompts.resolver import resolve_prompt as _resolve  # lazy
    except Exception:  # pragma: no cover — resolver built in parallel
        return None
    try:
        return await _maybe_await(_resolve(prompt_key, user_id=user_id, project_id=project_id))
    except Exception:  # pragma: no cover
        return None


async def resolve_slot_contents(
    prompt_key: str, user_id: str = "", project_id: str = ""
) -> dict[str, str]:
    """Resolve slot contents without building the full prompt.

    Port of resolveSlotContents. Returns {} when the resolver isn't available.
    """
    try:
        from app.ai.prompts.resolver import resolve_slot_contents as _resolve  # lazy
    except Exception:  # pragma: no cover
        return {}
    try:
        result = await _maybe_await(_resolve(prompt_key, user_id=user_id, project_id=project_id))
        return result or {}
    except Exception:  # pragma: no cover
        return {}


# ---------------------------------------------------------------------------
# Model duration limits (src/lib/ai/model-limits.ts)
# ---------------------------------------------------------------------------

MODEL_MAX_DURATIONS: dict[str, int] = {
    "veo-2.0-generate-001": 8,
    "veo-3.0-generate-001": 8,
    "veo-3.0-fast-generate-001": 8,
    "veo-3.1-generate-001": 8,
    "veo-3.1-fast-generate-001": 8,
    "kling-v1": 10,
    "kling-v1-5": 10,
    "kling-v2.5-turbo": 10,
    "kling-v3": 15,
    "doubao-seedance-1-5-pro-250528": 12,
    "doubao-seedance-1-5-pro-251215": 12,
    "doubao-seedance-1-0-lite-250528": 5,
    "wan2.7-t2v": 15,
    "wan2.7-r2v": 15,
    "wan2.6-t2v": 15,
    "wan2.6-i2v-flash": 15,
    "wan2.6-i2v": 10,
    "wan2.6-r2v": 10,
    "wan2.6-r2v-flash": 10,
}

# Family-level fallback: if modelId contains this substring, use this duration.
_FAMILY_MAX_DURATIONS: list[tuple[str, int]] = [
    ("veo", 8),
    ("kling-v3", 15),
    ("kling", 10),
    ("seedance-1-0", 5),
    ("seedance", 12),
    ("wan2.7", 15),
    ("wan2.6", 15),
    ("wan", 15),
]

DEFAULT_MAX_DURATION = 12


def get_model_max_duration(model_id: Optional[str]) -> int:
    """Max supported video duration (seconds) for a model. Unknown -> 12.

    Port of getModelMaxDuration.
    """
    if not model_id:
        return DEFAULT_MAX_DURATION
    lower = model_id.lower()
    if lower in MODEL_MAX_DURATIONS:
        return MODEL_MAX_DURATIONS[lower]
    # Prefix match, longest key first.
    for key in sorted(MODEL_MAX_DURATIONS.keys(), key=len, reverse=True):
        if lower.startswith(key) or key.startswith(lower):
            return MODEL_MAX_DURATIONS[key]
    # Family substring match (order matters — more specific first).
    for family, duration in _FAMILY_MAX_DURATIONS:
        if family in lower:
            return duration
    return DEFAULT_MAX_DURATION


# ---------------------------------------------------------------------------
# AI layer thin wrappers (app.ai — built in parallel, import lazily)
# ---------------------------------------------------------------------------


async def ai_generate_text(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    images: Optional[list[str]] = None,
) -> str:
    """Call the text model via app.ai.generate_text (lazy import)."""
    from app.ai import generate_text  # lazy
    from app.ai.types import TextOptions

    return await generate_text(
        prompt,
        TextOptions(
            system_prompt=system_prompt,
            temperature=temperature,
            images=images or [],
        ),
    )


async def ai_generate_image(
    prompt: str,
    *,
    size: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    quality: Optional[str] = None,
    reference_images: Optional[list[str]] = None,
) -> str:
    """Call the image model via app.ai.generate_image (lazy import). Returns path."""
    from app.ai import generate_image  # lazy
    from app.ai.types import ImageOptions

    return await generate_image(
        prompt,
        ImageOptions(
            size=size,
            aspect_ratio=aspect_ratio,
            quality=quality,
            reference_images=reference_images or [],
        ),
    )
