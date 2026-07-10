"""Character relations router — port of:
    projects/[id]/character-relations/route.ts
    projects/[id]/character-relations/[relationId]/route.ts
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._common import assert_project_ownership, json_error, not_found, serialize_many
from app.core.ids import new_id
from app.db.models import Character, CharacterRelation
from app.db.session import get_db

router = APIRouter()


@router.get("/projects/{id}/character-relations")
def list_relations(id: str, request: Request, db: Session = Depends(get_db)):
    if not assert_project_ownership(db, request, id):
        return not_found()
    rows = (
        db.execute(select(CharacterRelation).where(CharacterRelation.project_id == id))
        .scalars()
        .all()
    )
    return serialize_many(rows)


@router.post("/projects/{id}/character-relations")
async def create_relation(id: str, request: Request, db: Session = Depends(get_db)):
    if not assert_project_ownership(db, request, id):
        return not_found()
    body = await request.json()

    char_a = body.get("characterAId")
    char_b = body.get("characterBId")
    if not char_a or not char_b:
        return json_error(400, "Missing character ids")

    # Ensure both characters belong to this project
    found = (
        db.execute(
            select(Character.id).where(
                Character.project_id == id, Character.id.in_([char_a, char_b])
            )
        )
        .scalars()
        .all()
    )
    if len(found) != 2:
        return json_error(400, "Invalid character ids")

    relation = CharacterRelation(
        id=new_id(),
        project_id=id,
        character_a_id=char_a,
        character_b_id=char_b,
        relation_type=body.get("relationType") or "neutral",
        description=body.get("description") or "",
    )
    db.add(relation)
    db.flush()
    # TS returns the literal insert payload (no createdAt).
    return JSONResponse(
        {
            "id": relation.id,
            "projectId": relation.project_id,
            "characterAId": relation.character_a_id,
            "characterBId": relation.character_b_id,
            "relationType": relation.relation_type,
            "description": relation.description,
        },
        status_code=201,
    )


@router.delete("/projects/{id}/character-relations/{relation_id}")
def delete_relation(id: str, relation_id: str, request: Request, db: Session = Depends(get_db)):
    if not assert_project_ownership(db, request, id):
        return not_found()
    row = db.execute(
        select(CharacterRelation).where(
            CharacterRelation.id == relation_id, CharacterRelation.project_id == id
        )
    ).scalar_one_or_none()
    if row:
        db.delete(row)
    return {"ok": True}
