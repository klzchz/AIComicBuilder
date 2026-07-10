"""Projects router — port of src/app/api/projects/route.ts and projects/[id]/route.ts."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._common import (
    get_user_id,
    load_assets_by_shot,
    load_shot_dialogues,
    mark_downstream_stale,
    not_found,
    resolve_project,
    serialize,
    serialize_many,
)
from app.core.ids import new_id
from app.db.models import Character, Episode, Project, Shot, StoryboardVersion
from app.db.session import get_db

router = APIRouter()

# Fields the PATCH handler accepts (camelCase body key -> model attribute).
_PATCHABLE = {
    "title": "title",
    "idea": "idea",
    "script": "script",
    "outline": "outline",
    "status": "status",
    "generationMode": "generation_mode",
    "useProjectPrompts": "use_project_prompts",
    "colorPalette": "color_palette",
    "worldSetting": "world_setting",
    "targetDuration": "target_duration",
    "bgmUrl": "bgm_url",
}


@router.get("/projects")
def list_projects(request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    rows = (
        db.execute(
            select(Project)
            .where(Project.user_id == user_id)
            .order_by(Project.created_at.desc())
        )
        .scalars()
        .all()
    )
    return serialize_many(rows)


@router.post("/projects")
async def create_project(request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    body = await request.json()
    project = Project(
        id=new_id(),
        user_id=user_id,
        title=body.get("title"),
        script=body.get("script") or "",
    )
    db.add(project)
    db.flush()
    return JSONResponse(serialize(project), status_code=201)


@router.get("/projects/{id}")
def get_project(id: str, request: Request, db: Session = Depends(get_db)):
    project = resolve_project(db, id, get_user_id(request))
    if not project:
        return not_found()

    version_id = request.query_params.get("versionId")

    # All storyboard versions for this project (newest first)
    all_versions = (
        db.execute(
            select(StoryboardVersion)
            .where(StoryboardVersion.project_id == id)
            .order_by(StoryboardVersion.version_num.desc())
        )
        .scalars()
        .all()
    )
    resolved_version_id = version_id or (all_versions[0].id if all_versions else None)

    project_characters = (
        db.execute(select(Character).where(Character.project_id == id)).scalars().all()
    )

    project_shots = []
    if resolved_version_id:
        project_shots = (
            db.execute(
                select(Shot)
                .where(Shot.project_id == id, Shot.version_id == resolved_version_id)
                .order_by(Shot.sequence.asc())
            )
            .scalars()
            .all()
        )

    assets_by_shot = load_assets_by_shot(db, [s.id for s in project_shots])

    enriched_shots = []
    for shot in project_shots:
        row = serialize(shot)
        row["dialogues"] = load_shot_dialogues(db, shot.id)
        row["assets"] = assets_by_shot.get(shot.id, [])
        enriched_shots.append(row)

    project_episodes = (
        db.execute(
            select(Episode).where(Episode.project_id == id).order_by(Episode.sequence.asc())
        )
        .scalars()
        .all()
    )

    return {
        **serialize(project),
        "episodes": serialize_many(project_episodes),
        "characters": serialize_many(project_characters),
        "shots": enriched_shots,
        "versions": [
            {
                "id": v.id,
                "label": v.label,
                "versionNum": v.version_num,
                "createdAt": v.created_at,
            }
            for v in all_versions
        ],
    }


@router.patch("/projects/{id}")
async def patch_project(id: str, request: Request, db: Session = Depends(get_db)):
    project = resolve_project(db, id, get_user_id(request))
    if not project:
        return not_found()

    body = await request.json()
    for key, attr in _PATCHABLE.items():
        if key in body:
            setattr(project, attr, body[key])

    if "script" in body:
        mark_downstream_stale(db, "project", id)

    db.flush()
    return serialize(project)


@router.delete("/projects/{id}")
def delete_project(id: str, request: Request, db: Session = Depends(get_db)):
    project = resolve_project(db, id, get_user_id(request))
    if not project:
        return not_found()
    db.delete(project)
    return Response(status_code=204)
