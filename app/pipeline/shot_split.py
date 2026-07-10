"""Shot split stage — Python port of src/lib/pipeline/shot-split.ts.

Splits a screenplay into a professional shot list via the LLM. Handles both the
scene-grouped output (array of scenes, each with a ``shots`` array) and the flat
shot-array format (backwards compat). Persists Scene, Shot and Dialogue rows.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy import and_, or_, select

from app.api._common import serialize_many
from app.core.ids import new_id
from app.db.models import (
    Character,
    CharacterRelation,
    Dialogue,
    Episode,
    Project,
    Scene,
    Shot,
)
from app.db.session import db_session
from app.pipeline._helpers import ai_generate_text, payload_of, resolve_prompt


async def run_shot_split(
    project_id: str,
    screenplay: str,
    *,
    episode_id: Optional[str] = None,
    user_id: str = "",
    model_config: Optional[dict] = None,
) -> dict[str, Any]:
    """Split a screenplay into shots and persist them. Port of handleShotSplit."""
    # --- Gather context (characters, relations, project/episode data) --------
    with db_session() as s:
        if episode_id:
            char_rows = list(
                s.execute(
                    select(Character).where(
                        and_(
                            Character.project_id == project_id,
                            or_(
                                Character.episode_id.is_(None),
                                Character.episode_id == episode_id,
                            ),
                        )
                    )
                ).scalars()
            )
        else:
            char_rows = list(
                s.execute(
                    select(Character).where(Character.project_id == project_id)
                ).scalars()
            )
        # Snapshot the character fields we need into plain tuples.
        project_characters = [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "performance_style": c.performance_style,
            }
            for c in char_rows
        ]

        relations = list(
            s.execute(
                select(CharacterRelation).where(CharacterRelation.project_id == project_id)
            ).scalars()
        )
        relations_snapshot = [
            {
                "a": r.character_a_id,
                "b": r.character_b_id,
                "type": r.relation_type,
                "description": r.description,
            }
            for r in relations
        ]

        project = s.get(Project, project_id)
        world_setting = project.world_setting if project else ""
        target_duration = (project.target_duration or 0) if project else 0
        project_color_palette = project.color_palette if project else ""

        episode_color_palette = ""
        if episode_id:
            episode = s.get(Episode, episode_id)
            if episode and episode.color_palette:
                episode_color_palette = episode.color_palette
            if episode and episode.target_duration and episode.target_duration > 0:
                target_duration = episode.target_duration

    char_by_id = {c["id"]: c for c in project_characters}
    character_descriptions = "\n".join(
        f"{c['name']}: {c['description']}" for c in project_characters
    )

    # Character relationships text block.
    relations_text = ""
    if relations_snapshot:
        relations_text = "\n\n## CHARACTER RELATIONSHIPS\n"
        for r in relations_snapshot:
            char_a = char_by_id.get(r["a"])
            char_b = char_by_id.get(r["b"])
            if char_a and char_b:
                extra = f" ({r['description']})" if r["description"] else ""
                relations_text += f"- {char_a['name']} ↔ {char_b['name']}: {r['type']}{extra}\n"
        relations_text += (
            "\nUse these relationships to inform framing, character proximity, "
            "and eye direction in compositions.\n"
        )

    color_palette = episode_color_palette or project_color_palette or ""

    system_prompt = await resolve_prompt("shot_split", user_id=user_id, project_id=project_id)

    from app.ai.prompts.shot_split import build_shot_split_prompt

    performance_styles = [
        {"name": c["name"], "performanceStyle": c["performance_style"]}
        for c in project_characters
        if c["performance_style"]
    ]

    user_prompt = build_shot_split_prompt(
        screenplay,
        character_descriptions,
        None,
        color_palette or None,
        performance_styles or None,
    ) + relations_text

    # Inject world setting.
    if world_setting:
        user_prompt = (
            f"【World Setting】\n{world_setting}\n\n"
            "All shots must be consistent with this world setting.\n\n"
        ) + user_prompt

    # Inject target duration.
    if target_duration and target_duration > 0:
        mins = target_duration // 60
        secs = target_duration % 60
        user_prompt += (
            f"\n\nTarget total duration: {target_duration} seconds "
            f"({mins}m{secs}s). Ensure all shot durations sum to approximately "
            "this target.\n"
        )

    result = await ai_generate_text(user_prompt, system_prompt=system_prompt, temperature=0.5)
    parsed = json.loads(result)

    # Scene-grouped array vs flat shot array (backwards compat).
    is_scene_grouped = bool(parsed) and isinstance(parsed[0], dict) and isinstance(
        parsed[0].get("shots"), list
    )

    created_ids: list[str] = []

    def insert_shot(s, shot_data: dict, scene_id: Optional[str]) -> str:
        # Metadata-only insert. Image/video assets live in the shot_assets table
        # and are produced by independent downstream pipelines.
        shot_id = new_id()
        s.add(
            Shot(
                id=shot_id,
                project_id=project_id,
                sequence=shot_data.get("sequence") or 0,
                prompt=shot_data.get("prompt") or "",
                motion_script=shot_data.get("motionScript") or "",
                video_script=shot_data.get("videoScript") or "",
                camera_direction=shot_data.get("cameraDirection") or "static",
                duration=shot_data.get("duration") or 10,
                transition_in=shot_data.get("transitionIn") or "cut",
                transition_out=shot_data.get("transitionOut") or "cut",
                composition_guide=shot_data.get("compositionGuide") or "",
                focal_point=shot_data.get("focalPoint") or "",
                depth_of_field=shot_data.get("depthOfField") or "medium",
                sound_design=shot_data.get("soundDesign") or "",
                music_cue=shot_data.get("musicCue") or "",
                episode_id=episode_id,
                scene_id=scene_id,
            )
        )
        # Create dialogues for this shot.
        shot_dialogues = shot_data.get("dialogues") or []
        for i, dialogue in enumerate(shot_dialogues):
            matched = next(
                (c for c in project_characters if c["name"] == dialogue.get("character")),
                None,
            )
            if matched:
                s.add(
                    Dialogue(
                        id=new_id(),
                        shot_id=shot_id,
                        character_id=matched["id"],
                        text=dialogue.get("text"),
                        sequence=i,
                    )
                )
        return shot_id

    with db_session() as s:
        if is_scene_grouped:
            global_sequence = 1
            for scene_idx, scene in enumerate(parsed):
                scene_id = new_id()
                s.add(
                    Scene(
                        id=scene_id,
                        episode_id=episode_id or "",
                        project_id=project_id,
                        title=scene.get("sceneTitle") or "",
                        description=scene.get("sceneDescription") or "",
                        lighting=scene.get("lighting") or "",
                        color_palette=scene.get("colorPalette") or "",
                        sequence=scene_idx + 1,
                    )
                )
                for shot_data in scene.get("shots") or []:
                    shot_data["sequence"] = global_sequence
                    global_sequence += 1
                    created_ids.append(insert_shot(s, shot_data, scene_id))
        else:
            for shot_data in parsed:
                created_ids.append(insert_shot(s, shot_data, None))

    with db_session() as s:
        created = list(
            s.execute(select(Shot).where(Shot.id.in_(created_ids))).scalars()
        ) if created_ids else []
        # Preserve insertion order.
        by_id = {row.id: row for row in created}
        ordered = [by_id[i] for i in created_ids if i in by_id]
        return {"shots": serialize_many(ordered)}


async def handle_shot_split(task: Any) -> dict[str, Any]:
    p = payload_of(task)
    return await run_shot_split(
        p["projectId"],
        p.get("screenplay", ""),
        episode_id=p.get("episodeId"),
        user_id=p.get("userId") or "",
        model_config=p.get("modelConfig"),
    )
