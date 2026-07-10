"""Mood board router — port of:
    projects/[id]/mood-board/route.ts
    projects/[id]/mood-board/[imageId]/route.ts
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._common import assert_project_ownership, not_found, serialize_many
from app.core.ids import new_id
from app.db.models import MoodBoardImage
from app.db.session import get_db

router = APIRouter()


@router.get("/projects/{id}/mood-board")
def list_mood_board(id: str, request: Request, db: Session = Depends(get_db)):
    if not assert_project_ownership(db, request, id):
        return not_found()
    rows = (
        db.execute(select(MoodBoardImage).where(MoodBoardImage.project_id == id))
        .scalars()
        .all()
    )
    return serialize_many(rows)


@router.post("/projects/{id}/mood-board")
async def create_mood_board(id: str, request: Request, db: Session = Depends(get_db)):
    if not assert_project_ownership(db, request, id):
        return not_found()
    body = await request.json()
    image = MoodBoardImage(
        id=new_id(),
        project_id=id,
        image_url=body.get("imageUrl"),
        annotation=body.get("annotation") or "",
        extracted_style=body.get("extractedStyle") or "",
    )
    db.add(image)
    db.flush()
    # TS returns the literal insert payload (no createdAt).
    return JSONResponse(
        {
            "id": image.id,
            "projectId": image.project_id,
            "imageUrl": image.image_url,
            "annotation": image.annotation,
            "extractedStyle": image.extracted_style,
        },
        status_code=201,
    )


@router.delete("/projects/{id}/mood-board/{image_id}")
def delete_mood_board(id: str, image_id: str, request: Request, db: Session = Depends(get_db)):
    if not assert_project_ownership(db, request, id):
        return not_found()
    row = db.execute(
        select(MoodBoardImage).where(
            MoodBoardImage.id == image_id, MoodBoardImage.project_id == id
        )
    ).scalar_one_or_none()
    if row:
        db.delete(row)
    return {"ok": True}
