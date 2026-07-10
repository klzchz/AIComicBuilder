"""Character reference image stage — Python port of src/lib/pipeline/character-image.ts.

Generates a 4-view character turnaround sheet and stores it as the character's
reference image.
"""
from __future__ import annotations

from typing import Any, Optional

from app.db.models import Character
from app.db.session import db_session
from app.pipeline._helpers import ai_generate_image, payload_of


async def run_character_image(
    character_id: str,
    *,
    model_config: Optional[dict] = None,
) -> dict[str, Any]:
    """Generate & store a character turnaround sheet. Port of handleCharacterImage."""
    with db_session() as s:
        character = s.get(Character, character_id)
        if not character:
            raise ValueError("Character not found")
        name = character.name
        description = character.description or character.name

    from app.ai.prompts.character_image import build_character_turnaround_prompt

    prompt = build_character_turnaround_prompt(description, name)

    image_path = await ai_generate_image(
        prompt,
        size="2560x1440",
        aspect_ratio="16:9",
        quality="hd",
    )

    with db_session() as s:
        character = s.get(Character, character_id)
        if character:
            character.reference_image = image_path

    return {"imagePath": image_path}


async def handle_character_image(task: Any) -> dict[str, Any]:
    p = payload_of(task)
    return await run_character_image(
        p["characterId"],
        model_config=p.get("modelConfig"),
    )
