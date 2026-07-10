"""Generation dispatcher router — port of projects/[id]/generate/route.ts.

POST /projects/{id}/generate is the single entry point for every AI generation
action (script/outline/shot-split/frame/video/reference pipelines). The TS route
is ~3900 lines: it handles a large set of "inline" actions directly (many of
them streaming text/JSON responses) and enqueues the remaining image/video jobs
onto the background task queue.

This port faithfully reproduces:
  * project ownership check (404 on miss),
  * request-body shape { action, payload?, modelConfig?, episodeId? },
  * the inline-vs-queue dispatch decision.

PORT NOTE: The actual inline handlers (streaming LLM calls, image/video
provider orchestration, shot-split persistence, etc.) live in the AI/pipeline
layer, which is being ported in parallel under app.pipeline. Each inline action
is delegated to ``app.pipeline.generate_actions.handle_action`` when available;
until that module exists the endpoint returns 501 for inline actions. Queue
actions are fully functional and enqueue a Task exactly like the TS fallback.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._common import get_user_id, json_error, not_found, serialize
from app.db.models import Project, Task
from app.db.session import get_db

router = APIRouter()

# Actions the TS route handles directly (everything else is enqueued).
INLINE_ACTIONS = {
    "script_outline",
    "script_generate",
    "script_parse",
    "character_extract",
    "single_character_image",
    "batch_character_image",
    "shot_split",
    "generate_keyframe_prompts",
    "single_shot_rewrite",
    "batch_frame_generate",
    "single_frame_generate",
    "single_video_generate",
    "batch_video_generate",
    "single_scene_frame",
    "batch_scene_frame",
    "single_reference_video",
    "batch_reference_video",
    "single_video_prompt",
    "batch_video_prompt",
    "ai_optimize_text",
    "video_assemble",
    "batch_ref_image_generate",
    "single_ref_image_generate",
    "generate_ref_prompts",
    "single_ref_image_generate_all",
}


@router.post("/projects/{id}/generate")
async def generate(id: str, request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)

    # Verify project ownership
    owner = db.execute(
        select(Project.id).where(Project.id == id, Project.user_id == user_id)
    ).scalar_one_or_none()
    if not owner:
        return not_found()

    body = await request.json()
    action = body.get("action")
    payload = body.get("payload")
    model_config = body.get("modelConfig")
    episode_id = body.get("episodeId")

    print(f"[Generate] action={action}, projectId={id}, episodeId={episode_id or 'none'}")

    if action in INLINE_ACTIONS:
        # PORT NOTE: delegate to the inline AI/pipeline handlers (built in
        # parallel). These return streaming or JSON responses in the TS source.
        try:
            from app.pipeline.generate_actions import handle_action  # lazy
        except Exception:
            return json_error(
                501,
                f"Inline generation action '{action}' not available "
                "(AI/pipeline layer built in parallel)",
            )
        return await handle_action(
            db,
            action=action,
            project_id=id,
            user_id=user_id,
            payload=payload,
            model_config=model_config,
            episode_id=episode_id,
        )

    # Fallback: image/video generation jobs go through the task queue.
    from app.task_queue import enqueue  # lazy — import-safe queue package

    task_payload = {
        "projectId": id,
        **(payload or {}),
        "modelConfig": model_config,
        "episodeId": episode_id,
        "userId": user_id,
    }
    task_id = enqueue(
        action,
        task_payload,
        project_id=id,
        episode_id=episode_id or None,
    )

    task = db.get(Task, task_id)
    return JSONResponse(serialize(task), status_code=201)
