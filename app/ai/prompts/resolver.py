"""DB-backed prompt override resolver.

Python port of src/lib/ai/prompts/resolver.ts. Sync (not async), using
SessionLocal. Merge precedence:
  project-level full override > global full override
  > per-slot project override > per-slot global override > code default

PORT NOTE: A DB session is opened only when a resolve function is called, never
at import time (import-safety).
"""
from typing import Optional

from sqlalchemy import select

from app.db.session import SessionLocal
from app.db.models import PromptTemplate
from app.ai.prompts.registry import get_prompt_definition, get_default_slot_contents


def resolve_prompt(
    prompt_key: str,
    user_id: str,
    project_id: Optional[str] = None,
) -> str:
    """Resolve a prompt's system content by merging project-level overrides >
    global overrides > code defaults."""
    definition = get_prompt_definition(prompt_key)
    if not definition:
        raise ValueError(f"Unknown prompt key: {prompt_key}")

    slot_contents = dict(get_default_slot_contents(prompt_key) or {})

    session = SessionLocal()
    try:
        # Check for full-prompt override first (advanced mode, slot_key = None)
        full_overrides = session.execute(
            select(PromptTemplate).where(
                PromptTemplate.user_id == user_id,
                PromptTemplate.prompt_key == prompt_key,
                PromptTemplate.slot_key.is_(None),
            )
        ).scalars().all()

        project_full = next(
            (o for o in full_overrides if o.scope == "project" and o.project_id == project_id),
            None,
        )
        global_full = next((o for o in full_overrides if o.scope == "global"), None)

        if project_id and project_full:
            return project_full.content
        if global_full:
            return global_full.content

        # No full override — resolve slot by slot
        slot_overrides = session.execute(
            select(PromptTemplate).where(
                PromptTemplate.user_id == user_id,
                PromptTemplate.prompt_key == prompt_key,
            )
        ).scalars().all()

        for slot_key in list(slot_contents.keys()):
            # Project-level slot override
            if project_id:
                project_slot = next(
                    (
                        o
                        for o in slot_overrides
                        if o.slot_key == slot_key
                        and o.scope == "project"
                        and o.project_id == project_id
                    ),
                    None,
                )
                if project_slot:
                    slot_contents[slot_key] = project_slot.content
                    continue
            # Global slot override
            global_slot = next(
                (o for o in slot_overrides if o.slot_key == slot_key and o.scope == "global"),
                None,
            )
            if global_slot:
                slot_contents[slot_key] = global_slot.content
    finally:
        session.close()

    return definition.build_full_prompt(slot_contents)


def resolve_slot_contents(
    prompt_key: str,
    user_id: str,
    project_id: Optional[str] = None,
) -> dict:
    """Resolve slot contents without building the full prompt. Used for prompts
    that need dynamic parameters (frame, video, etc.)."""
    definition = get_prompt_definition(prompt_key)
    if not definition:
        raise ValueError(f"Unknown prompt key: {prompt_key}")

    slot_contents = dict(get_default_slot_contents(prompt_key) or {})

    session = SessionLocal()
    try:
        overrides = session.execute(
            select(PromptTemplate).where(
                PromptTemplate.user_id == user_id,
                PromptTemplate.prompt_key == prompt_key,
            )
        ).scalars().all()

        for slot_key in list(slot_contents.keys()):
            if project_id:
                project_slot = next(
                    (
                        o
                        for o in overrides
                        if o.slot_key == slot_key
                        and o.scope == "project"
                        and o.project_id == project_id
                    ),
                    None,
                )
                if project_slot:
                    slot_contents[slot_key] = project_slot.content
                    continue
            global_slot = next(
                (o for o in overrides if o.slot_key == slot_key and o.scope == "global"),
                None,
            )
            if global_slot:
                slot_contents[slot_key] = global_slot.content
    finally:
        session.close()

    return slot_contents
