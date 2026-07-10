"""Script parse stage — Python port of src/lib/pipeline/script-parse.ts.

Parses a project's raw script into a structured screenplay (JSON) via the LLM,
touches the project's ``updated_at``, and auto-enqueues character extraction.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from sqlalchemy import update

from app.db.models import Project
from app.db.session import db_session
from app.pipeline._helpers import ai_generate_text, payload_of, resolve_prompt


async def run_script_parse(
    project_id: str,
    *,
    user_id: str = "",
    model_config: Optional[dict] = None,
) -> Any:
    """Parse the project's script into a screenplay JSON. Port of handleScriptParse."""
    with db_session() as s:
        project = s.get(Project, project_id)
        if not project or not project.script:
            raise ValueError("Project or script not found")
        script = project.script

    system_prompt = await resolve_prompt("script_parse", user_id=user_id, project_id=project_id)

    # Prompt builders live in app.ai.prompts (built in parallel) — import lazily.
    from app.ai.prompts.script_parse import build_script_parse_prompt

    result = await ai_generate_text(
        build_script_parse_prompt(script),
        system_prompt=system_prompt,
        temperature=0.7,
    )
    screenplay = json.loads(result)

    now = int(time.time())
    with db_session() as s:
        s.execute(update(Project).where(Project.id == project_id).values(updated_at=now))

    # Auto-enqueue character extraction now that the screenplay is parsed.
    from app.task_queue import enqueue  # lazy

    enqueue(
        "character_extract",
        {
            "projectId": project_id,
            "screenplay": result,
            "modelConfig": model_config,
            "userId": user_id,
        },
        project_id=project_id,
    )

    return screenplay


async def handle_script_parse(task: Any) -> Any:
    p = payload_of(task)
    return await run_script_parse(
        p["projectId"],
        user_id=p.get("userId") or "",
        model_config=p.get("modelConfig"),
    )
