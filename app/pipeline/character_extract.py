"""Character extraction stage — Python port of src/lib/pipeline/character-extract.ts.

Extracts characters (and relationships) from a screenplay via the LLM,
optionally de-duplicating against existing main characters when extracting for
an episode, then persists new Character rows and CharacterRelation rows.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy import and_, select

from app.api._common import serialize_many
from app.core.ids import new_id
from app.db.models import Character, CharacterRelation
from app.db.session import db_session
from app.pipeline._helpers import ai_generate_text, payload_of, resolve_prompt


async def run_character_extract(
    project_id: str,
    screenplay: str,
    *,
    episode_id: Optional[str] = None,
    user_id: str = "",
    model_config: Optional[dict] = None,
) -> dict[str, Any]:
    """Extract & persist characters/relationships. Port of handleCharacterExtract."""
    system_prompt = await resolve_prompt("character_extract", user_id=user_id, project_id=project_id)

    from app.ai.prompts.character_extract import build_character_extract_prompt

    result = await ai_generate_text(
        build_character_extract_prompt(screenplay),
        system_prompt=system_prompt,
        temperature=0.5,
    )
    parsed = json.loads(result)

    # Support both formats: new {characters, relationships} and legacy array.
    relationships: list[dict] = []
    if isinstance(parsed, list):
        extracted = parsed
    else:
        extracted = parsed.get("characters") or []
        relationships = parsed.get("relationships") or []

    new_characters = extracted

    # AI deduplication when extracting for an episode with existing main chars.
    if episode_id:
        with db_session() as s:
            existing_chars = list(
                s.execute(
                    select(Character).where(
                        and_(Character.project_id == project_id, Character.scope == "main")
                    )
                ).scalars()
            )
            existing_names = [c.name for c in existing_chars]

        if existing_names:
            try:
                dedupe_result = await ai_generate_text(
                    "Existing characters: "
                    f"{json.dumps(existing_names)}\n\nNewly extracted characters: "
                    f"{json.dumps([c.get('name') for c in extracted])}\n\n"
                    "Return a JSON array of ONLY the truly new character names that are "
                    "NOT variants or aliases of existing characters. Consider nicknames, "
                    "shortened names, and honorific variations as the same character.",
                    system_prompt=(
                        "You are a character deduplication assistant. "
                        "Return only a JSON array of strings."
                    ),
                    temperature=0,
                )
                new_names = set(json.loads(dedupe_result))
                new_characters = [c for c in extracted if c.get("name") in new_names]
            except Exception as dedupe_err:  # noqa: BLE001
                print(f"[CharacterExtract] Deduplication failed, inserting all: {dedupe_err}")

    scope = "guest" if episode_id else "main"
    created_ids: list[str] = []
    with db_session() as s:
        for char in new_characters:
            char_id = new_id()
            s.add(
                Character(
                    id=char_id,
                    project_id=project_id,
                    name=char.get("name"),
                    description=char.get("description") or "",
                    visual_hint=char.get("visualHint") or "",
                    height_cm=char.get("heightCm") or 0,
                    body_type=char.get("bodyType") or "average",
                    performance_style=char.get("performanceStyle") or "",
                    scope=scope,
                    episode_id=episode_id,
                )
            )
            created_ids.append(char_id)

    # Auto-create character relationships from AI extraction.
    if relationships:
        with db_session() as s:
            all_chars = list(
                s.execute(select(Character).where(Character.project_id == project_id)).scalars()
            )
            name_to_id = {c.name: c.id for c in all_chars}
            for rel in relationships:
                a_id = name_to_id.get(rel.get("characterA"))
                b_id = name_to_id.get(rel.get("characterB"))
                if a_id and b_id and a_id != b_id:
                    try:
                        s.add(
                            CharacterRelation(
                                id=new_id(),
                                project_id=project_id,
                                character_a_id=a_id,
                                character_b_id=b_id,
                                relation_type=rel.get("relationType") or "neutral",
                                description=rel.get("description") or "",
                            )
                        )
                        s.flush()
                    except Exception as e:  # noqa: BLE001 — skip duplicates silently
                        s.rollback()
                        print(
                            f"[CharacterExtract] Skipped relation "
                            f"{rel.get('characterA')} <-> {rel.get('characterB')}: {e}"
                        )

    # Return the created character rows (freshly loaded, serialized like the API).
    with db_session() as s:
        created = list(
            s.execute(select(Character).where(Character.id.in_(created_ids))).scalars()
        ) if created_ids else []
        return {"characters": serialize_many(created)}


async def handle_character_extract(task: Any) -> dict[str, Any]:
    p = payload_of(task)
    return await run_character_extract(
        p["projectId"],
        p.get("screenplay", ""),
        episode_id=p.get("episodeId"),
        user_id=p.get("userId") or "",
        model_config=p.get("modelConfig"),
    )
