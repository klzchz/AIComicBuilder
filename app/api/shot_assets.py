"""Shot assets router — port of:
    projects/[id]/shots/[shotId]/assets/route.ts (PUT sync)
    projects/[id]/shots/[shotId]/assets/[assetId]/activate/route.ts (POST)

PUT body: { items: [{ id?, type, sequenceInType, prompt?, characters?,
                       fileUrl?, status? }] }

Logic per type group present in `items`:
    - PATCH existing active rows by id (prompt / characters / fileUrl / status)
    - INSERT new rows (when no matching id)
    - DELETE active rows of that type whose id is no longer in the list
The op is scoped per `type`, so submitting only `reference` items does not
touch the shot's first_frame / last_frame rows.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._common import (
    activate_asset_version,
    assert_project_ownership,
    json_error,
    not_found,
)
from app.core.ids import new_id
from app.db.models import Shot, ShotAsset
from app.db.session import get_db

router = APIRouter()


def _shot_in_project(db: Session, shot_id: str, project_id: str) -> bool:
    return (
        db.execute(
            select(Shot.id).where(Shot.id == shot_id, Shot.project_id == project_id)
        ).scalar_one_or_none()
        is not None
    )


@router.put("/projects/{id}/shots/{shot_id}/assets")
async def sync_shot_assets(
    id: str, shot_id: str, request: Request, db: Session = Depends(get_db)
):
    if not assert_project_ownership(db, request, id):
        return not_found()
    if not _shot_in_project(db, shot_id, id):
        return not_found()

    body = await request.json()
    items = body.get("items")
    if not isinstance(items, list):
        return json_error(400, "items must be an array")

    # Group submitted items by type
    by_type: dict[str, list[dict]] = {}
    for item in items:
        type_ = item.get("type")
        if not type_:
            continue
        by_type.setdefault(type_, []).append(item)

    for type_, group in by_type.items():
        # Pull all active rows of this type for this shot
        existing = (
            db.execute(
                select(ShotAsset).where(
                    ShotAsset.shot_id == shot_id,
                    ShotAsset.type == type_,
                    ShotAsset.is_active == 1,
                )
            )
            .scalars()
            .all()
        )
        existing_by_id = {r.id: r for r in existing}
        submitted_ids = {item["id"] for item in group if item.get("id")}

        # 1) Delete active rows that are no longer in the submitted list
        for row in existing:
            if row.id not in submitted_ids:
                db.delete(row)

        # 2) Patch + insert
        for item in group:
            existing_row = existing_by_id.get(item.get("id")) if item.get("id") else None
            if existing_row is not None:
                if "prompt" in item:
                    existing_row.prompt = item["prompt"]
                if "characters" in item:
                    existing_row.characters = json.dumps(item["characters"])
                if "fileUrl" in item:
                    existing_row.file_url = item["fileUrl"]
                if "status" in item:
                    existing_row.status = item["status"]
                if item.get("sequenceInType") != existing_row.sequence_in_type:
                    existing_row.sequence_in_type = item.get("sequenceInType")
            else:
                db.add(
                    ShotAsset(
                        id=item.get("id") or new_id(),
                        shot_id=shot_id,
                        type=type_,
                        sequence_in_type=item.get("sequenceInType"),
                        asset_version=1,
                        is_active=1,
                        prompt=item.get("prompt") or "",
                        file_url=item.get("fileUrl"),
                        status=item.get("status") or "pending",
                        characters=(
                            json.dumps(item["characters"])
                            if item.get("characters") is not None
                            else None
                        ),
                    )
                )
        db.flush()

    return {"ok": True}


@router.post("/projects/{id}/shots/{shot_id}/assets/{asset_id}/activate")
def activate_shot_asset(
    id: str, shot_id: str, asset_id: str, request: Request, db: Session = Depends(get_db)
):
    """Switch the "current" version of a (shot, type, sequenceInType) slot to
    the specified asset row. Used by the UI version-history arrows."""
    if not assert_project_ownership(db, request, id):
        return not_found()
    if not _shot_in_project(db, shot_id, id):
        return not_found()

    target = db.get(ShotAsset, asset_id)
    if not target:
        return json_error(404, "Asset not found")
    if target.shot_id != shot_id:
        return json_error(400, "Asset does not belong to this shot")

    activate_asset_version(
        db, target.shot_id, target.type, target.sequence_in_type, target.asset_version
    )
    db.flush()
    return {"ok": True}
