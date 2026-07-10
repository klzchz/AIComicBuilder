"""Inline generation dispatcher — glue for POST /projects/{id}/generate.

The TS ``generate/route.ts`` (~3900 lines) handled a large set of "inline"
actions directly. This module maps each inline action from
``app/api/generate.py``'s INLINE_ACTIONS set onto the already-ported pipeline
``run_*`` stages (which encapsulate the real orchestration), plus a few
text-only helpers implemented directly via the AI layer.

Design:
- Frame/video stages already handle keyframe-vs-reference mode internally, so
  the reference/scene variants route to the same run_* stages.
- Each branch is defensively wrapped: a failure returns a JSON error response
  rather than raising, so the endpoint never 500s the whole app.

PORT NOTE: response bodies here are pragmatic ({"ok": true, "action", "result"})
rather than byte-identical to the TS streaming responses. The web UI refetches
project/shot state after each action, so DB mutations surface correctly.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Character, Project, Shot


def _ok(action: str, result: Any = None, status: int = 200) -> JSONResponse:
    return JSONResponse({"ok": True, "action": action, "result": result}, status_code=status)


def _err(action: str, message: str, status: int = 500) -> JSONResponse:
    return JSONResponse({"ok": False, "action": action, "error": message}, status_code=status)


def _screenplay(db: Session, project_id: str, payload: dict, episode_id: Optional[str]) -> str:
    """Resolve the working script text: explicit payload wins, else episode/project."""
    for key in ("screenplay", "script", "text"):
        if payload.get(key):
            return payload[key]
    if episode_id:
        from app.db.models import Episode

        ep = db.get(Episode, episode_id)
        if ep and ep.script:
            return ep.script
    proj = db.get(Project, project_id)
    return (proj.script if proj else "") or ""


async def handle_action(
    db: Session,
    *,
    action: str,
    project_id: str,
    user_id: str,
    payload: Optional[dict],
    model_config: Optional[dict],
    episode_id: Optional[str],
) -> JSONResponse:
    payload = payload or {}
    # Lazy imports: keep this module import-safe and avoid pipeline import cycles.
    from app import pipeline as P

    common = dict(user_id=user_id, model_config=model_config)

    try:
        # ---- Script / text stages -------------------------------------------
        if action == "script_outline":
            idea = payload.get("idea") or (db.get(Project, project_id).idea if db.get(Project, project_id) else "")
            res = await P.run_script_outline(project_id, idea, episode_id=episode_id, **common)
            return _ok(action, res)

        if action == "script_generate":
            from app.ai import generate_text
            from app.ai.prompts.script_generate import build_script_generate_prompt

            idea = payload.get("idea", "")
            prompt = build_script_generate_prompt(idea) if _accepts(build_script_generate_prompt, 1) else build_script_generate_prompt(idea=idea)
            text = await generate_text(prompt, category="script_generate", project_id=project_id)
            return _ok(action, {"script": text})

        if action == "script_parse":
            res = await P.run_script_parse(project_id, **common)
            return _ok(action, res)

        if action == "ai_optimize_text":
            from app.ai import generate_text

            text = payload.get("text", "")
            instruction = payload.get("instruction", "Improve and polish the following text.")
            out = await generate_text(f"{instruction}\n\n{text}", project_id=project_id)
            return _ok(action, {"text": out})

        # ---- Characters ------------------------------------------------------
        if action == "character_extract":
            screenplay = _screenplay(db, project_id, payload, episode_id)
            res = await P.run_character_extract(project_id, screenplay, episode_id=episode_id, **common)
            return _ok(action, res)

        if action == "single_character_image":
            cid = payload.get("characterId") or payload.get("character_id")
            if not cid:
                return _err(action, "characterId is required", 400)
            res = await P.run_character_image(cid, model_config=model_config)
            return _ok(action, res)

        if action == "batch_character_image":
            chars = db.execute(select(Character.id).where(Character.project_id == project_id)).scalars().all()
            done = []
            for cid in chars:
                try:
                    await P.run_character_image(cid, model_config=model_config)
                    done.append(cid)
                except Exception as e:  # keep going on per-item failure
                    done.append({"id": cid, "error": str(e)})
            return _ok(action, {"processed": done})

        # ---- Shots -----------------------------------------------------------
        if action == "shot_split":
            screenplay = _screenplay(db, project_id, payload, episode_id)
            res = await P.run_shot_split(project_id, screenplay, episode_id=episode_id, **common)
            return _ok(action, res)

        # Frame generation (keyframe + reference/scene variants share the stage).
        if action in ("single_frame_generate", "single_scene_frame",
                      "single_ref_image_generate", "single_ref_image_generate_all"):
            shot_id = payload.get("shotId") or payload.get("shot_id")
            if not shot_id:
                return _err(action, "shotId is required", 400)
            res = await P.run_frame_generate(shot_id, project_id, **common)
            return _ok(action, res)

        if action in ("batch_frame_generate", "batch_scene_frame", "batch_ref_image_generate"):
            return await _batch_over_shots(db, project_id, episode_id, action,
                                           lambda sid: P.run_frame_generate(sid, project_id, **common))

        # Video generation (keyframe + reference variants share the stage).
        if action in ("single_video_generate", "single_reference_video"):
            shot_id = payload.get("shotId") or payload.get("shot_id")
            if not shot_id:
                return _err(action, "shotId is required", 400)
            ratio = payload.get("ratio", "16:9")
            res = await P.run_video_generate(shot_id, project_id=project_id, ratio=ratio, **common)
            return _ok(action, res)

        if action in ("batch_video_generate", "batch_reference_video"):
            ratio = payload.get("ratio", "16:9")
            return await _batch_over_shots(db, project_id, episode_id, action,
                                           lambda sid: P.run_video_generate(sid, project_id=project_id, ratio=ratio, **common))

        # ---- Video prompts (text) -------------------------------------------
        if action in ("single_video_prompt", "generate_keyframe_prompts",
                      "generate_ref_prompts", "single_shot_rewrite"):
            shot_id = payload.get("shotId") or payload.get("shot_id")
            if not shot_id:
                return _err(action, "shotId is required", 400)
            res = await _make_video_prompt(db, shot_id, project_id)
            return _ok(action, res)

        if action == "batch_video_prompt":
            return await _batch_over_shots(db, project_id, episode_id, action,
                                           lambda sid: _make_video_prompt(db, sid, project_id))

        # ---- Assemble --------------------------------------------------------
        if action == "video_assemble":
            res = await P.run_video_assemble(project_id)
            return _ok(action, res)

        return _err(action, f"Unknown inline action '{action}'", 400)

    except Exception as e:  # never bubble a 500 out of the dispatcher
        return _err(action, f"{type(e).__name__}: {e}")


async def _batch_over_shots(db: Session, project_id: str, episode_id: Optional[str], action: str, fn) -> JSONResponse:
    q = select(Shot.id).where(Shot.project_id == project_id)
    if episode_id:
        q = q.where(Shot.episode_id == episode_id)
    q = q.order_by(Shot.sequence)
    shot_ids = db.execute(q).scalars().all()
    done = []
    for sid in shot_ids:
        try:
            await fn(sid)
            done.append(sid)
        except Exception as e:
            done.append({"id": sid, "error": str(e)})
    return _ok(action, {"processed": done})


async def _make_video_prompt(db: Session, shot_id: str, project_id: str) -> dict:
    """Build and persist a shot's video prompt via the ported builder."""
    from app.ai import generate_text  # noqa: F401 (parity import; builder is deterministic)
    from app.ai.prompts.video_generate import build_video_prompt

    shot = db.get(Shot, shot_id)
    if not shot:
        raise ValueError(f"shot {shot_id} not found")
    prompt = build_video_prompt(
        video_script=shot.video_script or shot.prompt or "",
        camera_direction=shot.camera_direction or "static",
        scene_description=shot.prompt or "",
        duration=shot.duration,
    )
    shot.video_prompt = prompt
    db.add(shot)
    db.commit()
    return {"shotId": shot_id, "videoPrompt": prompt}


def _accepts(fn, positional: int) -> bool:
    """True if fn can be called with `positional` positional args."""
    import inspect

    try:
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        return len(params) >= positional
    except (ValueError, TypeError):
        return True
