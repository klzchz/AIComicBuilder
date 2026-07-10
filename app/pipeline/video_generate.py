"""Video generation stage — Python port of src/lib/pipeline/video-generate.ts.

Turns a shot's first/last keyframes into a video clip via the video provider,
clamping the requested duration to the model's max, persisting the result as a
new versioned ``keyframe_video`` asset, and running a best-effort quality check.
"""
from __future__ import annotations

from typing import Any, Optional

from app.db.models import Character, Shot
from app.db.session import db_session
from app.pipeline._helpers import (
    get_active_asset,
    get_model_max_duration,
    payload_of,
    resolve_slot_contents,
)


async def run_video_generate(
    shot_id: str,
    *,
    project_id: Optional[str] = None,
    user_id: str = "",
    ratio: str = "16:9",
    model_config: Optional[dict] = None,
) -> dict[str, Any]:
    """Generate a keyframe-interpolated video for a shot. Port of handleVideoGenerate."""
    with db_session() as s:
        shot = s.get(Shot, shot_id)
        if not shot:
            raise ValueError("Shot not found")

        shot_project_id = shot.project_id
        shot_duration = shot.duration
        shot_camera_direction = shot.camera_direction or "static"
        shot_video_script = shot.video_script
        shot_motion_script = shot.motion_script
        shot_prompt = shot.prompt

        # Read first/last frame from shot_assets.
        first_frame_asset = get_active_asset(s, shot_id, "first_frame", 0)
        last_frame_asset = get_active_asset(s, shot_id, "last_frame", 0)
        first_frame_url = first_frame_asset.file_url if first_frame_asset else None
        last_frame_url = last_frame_asset.file_url if last_frame_asset else None
        first_frame_desc = first_frame_asset.prompt if first_frame_asset else None
        last_frame_desc = last_frame_asset.prompt if last_frame_asset else None

        if not first_frame_url or not last_frame_url:
            raise ValueError("Shot frames not generated yet")

        char_rows = list(
            s.execute(
                # project characters used as CharacterRef for the video prompt
                __select_characters(shot_project_id)
            ).scalars()
        )
        project_characters = [
            {
                "name": c.name,
                "description": c.description,
                "heightCm": c.height_cm,
                "bodyType": c.body_type,
                "performanceStyle": c.performance_style,
            }
            for c in char_rows
        ]

    effective_project_id = project_id or shot_project_id

    # PORT NOTE: the TS resolves a versioned upload dir from the shot's
    # storyboard version and hands it to the video provider factory. The Python
    # video provider is resolved globally (app.ai), so output-dir routing is
    # delegated there; model_config is accepted for parity only.
    video_model_id = None
    if model_config and isinstance(model_config.get("video"), dict):
        video_model_id = model_config["video"].get("modelId")
    model_max_duration = get_model_max_duration(video_model_id)
    effective_duration = min(shot_duration if shot_duration is not None else 10, model_max_duration)

    video_slots = await resolve_slot_contents(
        "video_generate", user_id=user_id, project_id=effective_project_id
    )

    with db_session() as s:
        shot = s.get(Shot, shot_id)
        if shot:
            shot.status = "generating"

    video_script = shot_video_script or shot_motion_script or shot_prompt or ""

    from app.ai.prompts.video_generate import build_video_prompt

    prompt = build_video_prompt(
        video_script=video_script,
        camera_direction=shot_camera_direction,
        start_frame_desc=first_frame_desc,
        end_frame_desc=last_frame_desc,
        duration=effective_duration,
        characters=project_characters,
        slot_contents=video_slots,
    )

    from app.ai import generate_video  # lazy
    from app.ai.types import VideoGenerateParams

    result = await generate_video(
        VideoGenerateParams(
            prompt=prompt,
            duration=effective_duration,
            ratio=ratio,
            first_frame=first_frame_url,
            last_frame=last_frame_url,
        )
    )

    # Persist the keyframe video output as a new versioned asset row.
    from app.api._common import insert_asset_version

    with db_session() as s:
        insert_asset_version(
            s,
            shot_id,
            "keyframe_video",
            0,
            prompt=prompt,
            file_url=result.file_path,
            status="completed",
        )
        shot = s.get(Shot, shot_id)
        if shot:
            shot.status = "completed"

    # Best-effort quality check — never blocks or fails generation.
    try:
        from app.pipeline.video_quality_check import check_video_quality

        quality_result = await check_video_quality(result.file_path, first_frame_url)
        print(
            f"[VideoQuality] Shot {shot_id}: score={quality_result['score']}, "
            f"pass={quality_result['pass']}"
        )
        if not quality_result["pass"]:
            print(f"[VideoQuality] Issues: {', '.join(quality_result['issues'])}")
        return {
            "videoPath": result.file_path,
            "qualityScore": quality_result["score"],
            "qualityIssues": quality_result["issues"],
        }
    except Exception as e:  # noqa: BLE001
        print(f"[VideoQuality] Quality check skipped: {e}")

    return {"videoPath": result.file_path}


def __select_characters(project_id: str):
    from sqlalchemy import select

    return select(Character).where(Character.project_id == project_id)


async def handle_video_generate(task: Any) -> dict[str, Any]:
    p = payload_of(task)
    return await run_video_generate(
        p["shotId"],
        project_id=p.get("projectId"),
        user_id=p.get("userId") or "",
        ratio=p.get("ratio") or "16:9",
        model_config=p.get("modelConfig"),
    )
