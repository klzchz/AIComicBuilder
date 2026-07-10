"""Script outline stage — Python port of src/lib/pipeline/script-outline.ts.

Generates a story outline from an idea and saves it on the episode (if
``episode_id`` given) or the project.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from sqlalchemy import update

from app.db.models import Episode, Project
from app.db.session import db_session
from app.pipeline._helpers import ai_generate_text, payload_of, resolve_prompt


async def run_script_outline(
    project_id: str,
    idea: str,
    *,
    episode_id: Optional[str] = None,
    user_id: str = "",
    model_config: Optional[dict] = None,
) -> dict[str, Any]:
    """Generate and persist a story outline. Port of handleScriptOutline body."""
    system_prompt = await resolve_prompt("script_outline", user_id=user_id, project_id=project_id)

    # PORT NOTE: the TS route builds a per-request model from `model_config`
    # (resolveAIProvider). The Python AI layer resolves providers globally, so
    # model_config is accepted for API parity but provider selection is
    # delegated to app.ai.
    result = await ai_generate_text(
        f"Creative concept: {idea}",
        system_prompt=system_prompt,
        temperature=0.7,
    )
    outline = result.strip()

    now = int(time.time())
    with db_session() as s:
        if episode_id:
            s.execute(
                update(Episode).where(Episode.id == episode_id).values(outline=outline, updated_at=now)
            )
        else:
            s.execute(
                update(Project).where(Project.id == project_id).values(outline=outline, updated_at=now)
            )

    return {"outline": outline}


async def handle_script_outline(task: Any) -> dict[str, Any]:
    """Task handler: unwrap payload and run the stage."""
    p = payload_of(task)
    return await run_script_outline(
        p["projectId"],
        p.get("idea", ""),
        episode_id=p.get("episodeId"),
        user_id=p.get("userId") or "",
        model_config=p.get("modelConfig"),
    )
