"""Prompt presets router — port of:
    prompt-presets/route.ts
    prompt-presets/[presetId]/route.ts
    prompt-presets/[presetId]/apply/route.ts
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._common import get_user_id, json_error, serialize
from app.core.ids import new_id
from app.db.models import PromptPreset, PromptTemplate, PromptVersion
from app.db.session import get_db

router = APIRouter()

# Port of src/lib/ai/prompts/presets.ts — empty for now, preset content will
# be authored later. Entries: {id, name, nameKey, descriptionKey, promptKey, slots}.
BUILT_IN_PRESETS: list[dict[str, Any]] = []


@router.get("/prompt-presets")
def list_presets(request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    user_presets = (
        db.execute(select(PromptPreset).where(PromptPreset.user_id == user_id))
        .scalars()
        .all()
    )
    built_in = [{**p, "isBuiltIn": True} for p in BUILT_IN_PRESETS]
    user = [{**serialize(p), "isBuiltIn": False} for p in user_presets]
    return built_in + user


@router.post("/prompt-presets")
async def create_preset(request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    body = await request.json()
    if not body.get("name") or not body.get("promptKey") or not body.get("slots"):
        return json_error(400, "name, promptKey, and slots are required")

    preset = PromptPreset(
        id=new_id(),
        name=body["name"],
        user_id=user_id,
        prompt_key=body["promptKey"],
        slots=json.dumps(body["slots"]),
    )
    db.add(preset)
    db.flush()
    return JSONResponse(serialize(preset), status_code=201)


@router.delete("/prompt-presets/{preset_id}")
def delete_preset(preset_id: str, request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    existing = db.execute(
        select(PromptPreset).where(
            PromptPreset.id == preset_id, PromptPreset.user_id == user_id
        )
    ).scalar_one_or_none()
    if not existing:
        return json_error(404, "Preset not found or not owned by user")
    db.delete(existing)
    return Response(status_code=204)


@router.post("/prompt-presets/{preset_id}/apply")
async def apply_preset(preset_id: str, request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    body = await request.json()
    scope = body.get("scope") or "global"
    project_id = body.get("projectId")

    # Find the preset (from BUILT_IN_PRESETS or DB)
    built_in = next((p for p in BUILT_IN_PRESETS if p["id"] == preset_id), None)
    if built_in:
        preset_prompt_key = built_in["promptKey"]
        preset_slots = built_in["slots"]
    else:
        db_preset = db.get(PromptPreset, preset_id)
        if not db_preset:
            return json_error(404, "Preset not found")
        # Verify ownership for user presets
        if db_preset.user_id and db_preset.user_id != user_id:
            return json_error(403, "Forbidden")
        preset_prompt_key = db_preset.prompt_key
        try:
            preset_slots = json.loads(db_preset.slots)
        except (ValueError, TypeError):
            preset_slots = {}

    # For each slot in the preset, upsert into prompt_templates
    results: dict[str, Any] = {}
    for slot_key, content in preset_slots.items():
        query = select(PromptTemplate).where(
            PromptTemplate.user_id == user_id,
            PromptTemplate.prompt_key == preset_prompt_key,
            PromptTemplate.slot_key == slot_key,
        )
        if scope == "project" and project_id:
            query = query.where(
                PromptTemplate.scope == "project",
                PromptTemplate.project_id == project_id,
            )
        else:
            query = query.where(
                PromptTemplate.scope == "global",
                PromptTemplate.project_id.is_(None),
            )

        existing = db.execute(query).scalar_one_or_none()
        if existing:
            # Save version history before update
            db.add(
                PromptVersion(
                    id=new_id(), template_id=existing.id, content=existing.content
                )
            )
            existing.content = content
            db.flush()
            results[slot_key] = serialize(existing)
        else:
            inserted = PromptTemplate(
                id=new_id(),
                user_id=user_id,
                prompt_key=preset_prompt_key,
                slot_key=slot_key,
                scope=scope,
                project_id=project_id if scope == "project" else None,
                content=content,
            )
            db.add(inserted)
            db.flush()
            results[slot_key] = serialize(inserted)

    return {"success": True, "results": results}
