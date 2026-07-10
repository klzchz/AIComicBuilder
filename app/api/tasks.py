"""Tasks router — port of tasks/[id]/route.ts."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._common import get_user_id, not_found, serialize
from app.db.models import Project, Task
from app.db.session import get_db

router = APIRouter()


@router.get("/tasks/{id}")
def get_task(id: str, request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    if not user_id:
        return not_found()

    row = db.execute(
        select(Task)
        .join(Project, Task.project_id == Project.id, isouter=True)
        .where(Task.id == id, Project.user_id == user_id)
    ).scalar_one_or_none()

    if not row:
        return not_found()

    return serialize(row)
