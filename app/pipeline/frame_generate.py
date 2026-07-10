"""Frame generation stage — Python port of src/lib/pipeline/frame-generate.ts.

For one shot, generates the first (opening) and last (closing) keyframes as
images, attaching character reference sheets as visual anchors and threading the
previous shot's last frame for continuity. Reads/writes first_frame / last_frame
asset rows in the unified shot_assets table and flips shot status.
"""
from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import and_, desc, select

from app.core.ids import new_id
from app.db.models import Character, CharacterCostume, Episode, Project, Shot, ShotAsset
from app.db.session import db_session
from app.pipeline._helpers import (
    ai_generate_image,
    get_active_asset,
    payload_of,
    resolve_slot_contents,
)


async def run_frame_generate(
    shot_id: str,
    project_id: str,
    *,
    user_id: str = "",
    model_config: Optional[dict] = None,
) -> dict[str, Any]:
    """Generate first + last keyframes for a shot. Port of handleFrameGenerate."""
    # --- Load shot, characters, costume overrides, palette, prev shot --------
    with db_session() as s:
        shot = s.get(Shot, shot_id)
        if not shot:
            raise ValueError("Shot not found")

        shot_sequence = shot.sequence
        shot_prompt = shot.prompt or ""
        shot_episode_id = shot.episode_id
        composition_guide = shot.composition_guide
        focal_point = shot.focal_point
        depth_of_field = shot.depth_of_field
        raw_costume_overrides = shot.costume_overrides

        char_rows = list(
            s.execute(select(Character).where(Character.project_id == project_id)).scalars()
        )

        # Parse costume overrides from the shot.
        costume_overrides: dict[str, str] = (
            __import_json(raw_costume_overrides) if (raw_costume_overrides or "").strip() else {}
        )

        # Build character descriptions, applying costume overrides when present.
        character_desc_parts: list[str] = []
        # Snapshot character fields we need downstream (outside the session).
        project_characters: list[dict] = []
        for c in char_rows:
            description = c.description
            costume_id = costume_overrides.get(c.id)
            if costume_id:
                costume = s.get(CharacterCostume, costume_id)
                if costume and costume.description:
                    description = f"{c.description}. Current outfit: {costume.description}"
            desc = f"{c.name}: {description}"
            if c.performance_style:
                desc += f" [Performance: {c.performance_style}]"
            character_desc_parts.append(desc)
            project_characters.append(
                {
                    "id": c.id,
                    "name": c.name,
                    "reference_image": c.reference_image,
                    "height_cm": c.height_cm,
                    "body_type": c.body_type,
                }
            )
        character_descriptions = "\n".join(character_desc_parts)

        # Previous shot in the project (highest sequence below this shot).
        previous_shot = s.execute(
            select(Shot)
            .where(and_(Shot.project_id == project_id, Shot.sequence < shot_sequence))
            .order_by(desc(Shot.sequence))
            .limit(1)
        ).scalar_one_or_none()
        previous_shot_id = previous_shot.id if previous_shot else None

        # Color palette: episode overrides project.
        color_palette = ""
        if shot_episode_id:
            episode = s.get(Episode, shot_episode_id)
            if episode and episode.color_palette:
                color_palette = episode.color_palette
        if not color_palette:
            project = s.get(Project, project_id)
            if project and project.color_palette:
                color_palette = project.color_palette

    # --- Resolve prompt slot contents ----------------------------------------
    frame_first_slots = await resolve_slot_contents(
        "frame_generate_first", user_id=user_id, project_id=project_id
    )
    frame_last_slots = await resolve_slot_contents(
        "frame_generate_last", user_id=user_id, project_id=project_id
    )

    # --- Build composition suffix --------------------------------------------
    composition_suffix = ""
    if composition_guide:
        composition_suffix += f", {composition_guide.replace('_', ' ')} composition"
    if focal_point:
        composition_suffix += f", focus on {focal_point}"
    if depth_of_field == "shallow":
        composition_suffix += ", shallow depth of field, bokeh background"
    elif depth_of_field == "deep":
        composition_suffix += ", deep focus, everything sharp"
    if color_palette:
        composition_suffix += (
            f"\n\nGLOBAL COLOR PALETTE (mandatory): {color_palette}. "
            "All frames must adhere to this color scheme."
        )

    # Character height context for multi-character shots.
    chars_in_prompt = [c for c in project_characters if c["name"] in shot_prompt]
    if len(chars_in_prompt) > 1:
        height_chars = sorted(
            (c for c in chars_in_prompt if c["height_cm"] and c["height_cm"] > 0),
            key=lambda c: c["height_cm"] or 170,
            reverse=True,
        )
        height_info = ", ".join(
            f"{c['name']}: {c['height_cm']}cm ({c['body_type'] or 'average'})" for c in height_chars
        )
        if height_info:
            composition_suffix += (
                f". Character heights: {height_info}. Maintain correct relative proportions"
            )

    # --- Mark shot generating, read/mark keyframe asset rows ------------------
    with db_session() as s:
        shot = s.get(Shot, shot_id)
        if shot:
            shot.status = "generating"

        first_frame_asset = get_active_asset(s, shot_id, "first_frame", 0)
        last_frame_asset = get_active_asset(s, shot_id, "last_frame", 0)

        first_frame_asset_id = first_frame_asset.id if first_frame_asset else None
        last_frame_asset_id = last_frame_asset.id if last_frame_asset else None
        # Fall back to the shot prompt when no asset prompt exists (back-compat).
        start_frame_desc_text = (first_frame_asset.prompt if first_frame_asset else None) or shot_prompt
        end_frame_desc_text = (last_frame_asset.prompt if last_frame_asset else None) or shot_prompt
        stored_char_names = list(first_frame_asset.characters and _json_load(first_frame_asset.characters) or []) if first_frame_asset else []

        # Mark assets as generating.
        if first_frame_asset:
            first_frame_asset.status = "generating"
        if last_frame_asset:
            last_frame_asset.status = "generating"

        # Previous shot's last_frame url for visual continuity.
        prev_last_frame_url = None
        if previous_shot_id:
            prev_asset = get_active_asset(s, previous_shot_id, "last_frame", 0)
            prev_last_frame_url = prev_asset.file_url if prev_asset else None

    # Pick character refs to attach as visual anchors. Prefer characters listed
    # on the asset row; fall back to the first 3 chars with reference images.
    chars_with_refs = [c for c in project_characters if c["reference_image"]]
    if stored_char_names:
        relevant_chars = [c for c in chars_with_refs if c["name"] in stored_char_names]
    else:
        relevant_chars = chars_with_refs[:3]
    char_ref_images = [c["reference_image"] for c in relevant_chars]

    print(
        f"[FrameGenerate] Shot {shot_sequence}: using {len(relevant_chars)} chars: "
        f"{', '.join(c['name'] for c in relevant_chars) or 'fallback'}"
    )

    # --- Generate first frame ------------------------------------------------
    from app.ai.prompts.frame_generate import build_first_frame_prompt, build_last_frame_prompt

    first_frame_prompt = build_first_frame_prompt(
        scene_description=shot_prompt,
        start_frame_desc=start_frame_desc_text,
        character_descriptions=character_descriptions,
        previous_last_frame=prev_last_frame_url,
        slot_contents=frame_first_slots,
    )
    if composition_suffix:
        first_frame_prompt += composition_suffix
    first_frame_path = await ai_generate_image(
        first_frame_prompt, quality="hd", reference_images=char_ref_images
    )

    # --- Generate last frame -------------------------------------------------
    last_frame_prompt = build_last_frame_prompt(
        scene_description=shot_prompt,
        end_frame_desc=end_frame_desc_text,
        character_descriptions=character_descriptions,
        first_frame_path=first_frame_path,
        slot_contents=frame_last_slots,
    )
    if composition_suffix:
        last_frame_prompt += composition_suffix
    last_frame_path = await ai_generate_image(
        last_frame_prompt, quality="hd", reference_images=[first_frame_path, *char_ref_images]
    )

    # --- Persist results: patch existing asset rows or insert new ones -------
    from app.api._common import insert_asset_version
    import json

    relevant_names = [c["name"] for c in relevant_chars]
    with db_session() as s:
        if first_frame_asset_id:
            asset = s.get(ShotAsset, first_frame_asset_id)
            if asset:
                asset.file_url = first_frame_path
                asset.status = "completed"
        else:
            insert_asset_version(
                s,
                shot_id,
                "first_frame",
                0,
                prompt=start_frame_desc_text,
                file_url=first_frame_path,
                status="completed",
                characters=json.dumps(relevant_names),
            )

        if last_frame_asset_id:
            asset = s.get(ShotAsset, last_frame_asset_id)
            if asset:
                asset.file_url = last_frame_path
                asset.status = "completed"
        else:
            insert_asset_version(
                s,
                shot_id,
                "last_frame",
                0,
                prompt=end_frame_desc_text,
                file_url=last_frame_path,
                status="completed",
                characters=json.dumps(relevant_names),
            )

        shot = s.get(Shot, shot_id)
        if shot:
            shot.status = "completed"

    return {"firstFrame": first_frame_path, "lastFrame": last_frame_path}


def __import_json(text: str) -> dict:
    import json

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


def _json_load(text: str):
    import json

    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return []


async def handle_frame_generate(task: Any) -> dict[str, Any]:
    p = payload_of(task)
    return await run_frame_generate(
        p["shotId"],
        p["projectId"],
        user_id=p.get("userId") or "",
        model_config=p.get("modelConfig"),
    )
