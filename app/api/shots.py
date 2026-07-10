"""Shots router — port of:
    projects/[id]/shots/route.ts
    projects/[id]/shots/[shotId]/route.ts
    projects/[id]/shots/[shotId]/upload/route.ts
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Request, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._common import (
    assert_project_ownership,
    insert_asset_version,
    json_error,
    load_shot_dialogues,
    not_found,
    serialize,
)
from app.config import settings
from app.core.ids import new_id
from app.db.models import Shot
from app.db.session import get_db

router = APIRouter()

# PATCH whitelist (camelCase body key -> model attribute).
_ALLOWED_KEYS = {
    "prompt": "prompt",
    "duration": "duration",
    "sequence": "sequence",
    "motionScript": "motion_script",
    "videoScript": "video_script",
    "videoPrompt": "video_prompt",
    "cameraDirection": "camera_direction",
    "transitionIn": "transition_in",
    "transitionOut": "transition_out",
    "compositionGuide": "composition_guide",
    "focalPoint": "focal_point",
    "depthOfField": "depth_of_field",
    "soundDesign": "sound_design",
    "musicCue": "music_cue",
    "costumeOverrides": "costume_overrides",
}

_UPLOAD_FIELDS = ("firstFrame", "lastFrame", "sceneRefFrame", "reference_image")

# Upload field -> shot_assets slot (type, sequence_in_type).
_FIELD_TO_SLOT = {
    "firstFrame": ("first_frame", 0),
    "lastFrame": ("last_frame", 0),
    "sceneRefFrame": ("reference", 0),
}


def _get_shot(db: Session, shot_id: str, project_id: str) -> Shot | None:
    return db.execute(
        select(Shot).where(Shot.id == shot_id, Shot.project_id == project_id)
    ).scalar_one_or_none()


@router.get("/projects/{id}/shots")
def list_shots(id: str, request: Request, db: Session = Depends(get_db)):
    if not assert_project_ownership(db, request, id):
        return not_found()
    project_shots = (
        db.execute(
            select(Shot).where(Shot.project_id == id).order_by(Shot.sequence.asc())
        )
        .scalars()
        .all()
    )
    enriched = []
    for shot in project_shots:
        row = serialize(shot)
        row["dialogues"] = load_shot_dialogues(db, shot.id)
        enriched.append(row)
    return enriched


@router.patch("/projects/{id}/shots/{shot_id}")
async def patch_shot(id: str, shot_id: str, request: Request, db: Session = Depends(get_db)):
    """Updates only metadata fields on the shots table. Image/video assets live
    in the shot_assets table and must be patched via /shots/{shotId}/assets."""
    if not assert_project_ownership(db, request, id):
        return not_found()
    shot = _get_shot(db, shot_id, id)
    if not shot:
        return not_found()

    body = await request.json()
    touched = False
    for key, attr in _ALLOWED_KEYS.items():
        if key in body:
            setattr(shot, attr, body[key])
            touched = True

    if touched:
        db.flush()
    return serialize(shot)


@router.delete("/projects/{id}/shots/{shot_id}")
def delete_shot(id: str, shot_id: str, request: Request, db: Session = Depends(get_db)):
    if not assert_project_ownership(db, request, id):
        return not_found()
    shot = _get_shot(db, shot_id, id)
    if not shot:
        return not_found()
    db.delete(shot)
    return Response(status_code=204)


@router.post("/projects/{id}/shots/{shot_id}/upload")
async def upload_shot_frame(
    id: str, shot_id: str, request: Request, db: Session = Depends(get_db)
):
    if not assert_project_ownership(db, request, id):
        return not_found()
    shot = _get_shot(db, shot_id, id)
    if not shot:
        return not_found()

    form = await request.form()
    file = form.get("file")
    field = form.get("field")
    if not isinstance(file, UploadFile) or not field:
        return json_error(400, "Missing file or field")
    if field not in _UPLOAD_FIELDS:
        return json_error(400, "Invalid field")

    data = await file.read()
    ext = (file.filename or "").rsplit(".", 1)[-1] if "." in (file.filename or "") else "png"
    filename = f"{new_id()}.{ext or 'png'}"
    directory = os.path.join(settings.UPLOAD_DIR, "frames")
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, "wb") as fh:
        fh.write(data)

    # For reference_image uploads, just return the file path without touching the DB.
    if field == "reference_image":
        return {"url": filepath}

    # PORT NOTE: The TS route still writes to legacy shots.{firstFrame,lastFrame,
    # sceneRefFrame} columns, which no longer exist in the current schema
    # (assets moved to the unified shot_assets table). The port stores the
    # upload as a new active shot_assets version in the corresponding slot.
    type_, seq = _FIELD_TO_SLOT[field]
    insert_asset_version(
        db,
        shot_id,
        type_,
        seq,
        prompt="",
        file_url=filepath,
        status="completed",
    )
    db.flush()
    return serialize(shot)
