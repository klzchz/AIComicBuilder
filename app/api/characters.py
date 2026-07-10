"""Characters router — port of:
    projects/[id]/characters/route.ts
    projects/[id]/characters/[characterId]/route.ts
    projects/[id]/characters/[characterId]/costumes/route.ts
    projects/[id]/characters/[characterId]/upload/route.ts
"""
from __future__ import annotations

import json
import os

from fastapi import APIRouter, Depends, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._common import (
    assert_project_ownership,
    json_error,
    not_found,
    serialize,
    serialize_many,
)
from app.config import settings
from app.core.ids import new_id
from app.db.models import Character, CharacterCostume
from app.db.session import get_db

router = APIRouter()


def _get_character(db: Session, character_id: str, project_id: str) -> Character | None:
    return db.execute(
        select(Character).where(
            Character.id == character_id, Character.project_id == project_id
        )
    ).scalar_one_or_none()


@router.get("/projects/{id}/characters")
def list_characters(id: str, request: Request, db: Session = Depends(get_db)):
    if not assert_project_ownership(db, request, id):
        return not_found()
    rows = (
        db.execute(select(Character).where(Character.project_id == id)).scalars().all()
    )
    return serialize_many(rows)


@router.patch("/projects/{id}/characters/{character_id}")
async def patch_character(
    id: str, character_id: str, request: Request, db: Session = Depends(get_db)
):
    if not assert_project_ownership(db, request, id):
        return not_found()
    existing = _get_character(db, character_id, id)
    if not existing:
        return not_found()

    body = await request.json()
    if "name" in body:
        existing.name = body["name"]
    if "description" in body:
        existing.description = body["description"]
    if "visualHint" in body:
        existing.visual_hint = body["visualHint"]
    if "referenceImage" in body:
        existing.reference_image = body["referenceImage"]
    if "scope" in body:
        existing.scope = body["scope"]
        # When promoting to main, auto-clear episodeId
        if body["scope"] == "main":
            existing.episode_id = None
    if "episodeId" in body and body.get("scope") != "main":
        existing.episode_id = body["episodeId"]

    db.flush()
    return serialize(existing)


@router.delete("/projects/{id}/characters/{character_id}")
def delete_character(
    id: str, character_id: str, request: Request, db: Session = Depends(get_db)
):
    if not assert_project_ownership(db, request, id):
        return not_found()
    existing = _get_character(db, character_id, id)
    if existing:
        db.delete(existing)
    return Response(status_code=204)


@router.get("/projects/{id}/characters/{character_id}/costumes")
def list_costumes(id: str, character_id: str, request: Request, db: Session = Depends(get_db)):
    if not assert_project_ownership(db, request, id):
        return not_found()
    if not _get_character(db, character_id, id):
        return not_found()
    rows = (
        db.execute(
            select(CharacterCostume).where(CharacterCostume.character_id == character_id)
        )
        .scalars()
        .all()
    )
    return serialize_many(rows)


@router.post("/projects/{id}/characters/{character_id}/costumes")
async def create_costume(
    id: str, character_id: str, request: Request, db: Session = Depends(get_db)
):
    if not assert_project_ownership(db, request, id):
        return not_found()
    if not _get_character(db, character_id, id):
        return not_found()
    body = await request.json()
    costume = CharacterCostume(
        id=new_id(),
        character_id=character_id,
        name=body.get("name") or "default",
        description=body.get("description") or "",
        reference_image=body.get("referenceImage") or None,
    )
    db.add(costume)
    db.flush()
    # TS responds with the literal insert payload (camelCase, no createdAt).
    return JSONResponse(
        {
            "id": costume.id,
            "characterId": costume.character_id,
            "name": costume.name,
            "description": costume.description,
            "referenceImage": costume.reference_image,
        },
        status_code=201,
    )


@router.post("/projects/{id}/characters/{character_id}/upload")
async def upload_character_image(
    id: str, character_id: str, request: Request, db: Session = Depends(get_db)
):
    if not assert_project_ownership(db, request, id):
        return not_found()
    character = _get_character(db, character_id, id)
    if not character:
        return not_found()

    form = await request.form()
    file = form.get("file")
    if not isinstance(file, UploadFile):
        return json_error(400, "Missing file")

    data = await file.read()
    ext = (file.filename or "").rsplit(".", 1)[-1] if "." in (file.filename or "") else "png"
    filename = f"{new_id()}.{ext or 'png'}"
    directory = os.path.join(settings.UPLOAD_DIR, "characters")
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, "wb") as fh:
        fh.write(data)

    # Append to history
    try:
        history = json.loads(character.reference_image_history or "[]")
        if not isinstance(history, list):
            history = []
    except (ValueError, TypeError):
        history = []
    if character.reference_image and character.reference_image not in history:
        history.append(character.reference_image)
    if filepath not in history:
        history.append(filepath)

    character.reference_image = filepath
    character.reference_image_history = json.dumps(history)
    db.flush()
    return serialize(character)
