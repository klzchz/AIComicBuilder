"""Prompt templates router — port of:
    prompt-templates/route.ts                       (GET global overrides)
    prompt-templates/[promptKey]/route.ts           (PUT/DELETE global)
    prompt-templates/[promptKey]/versions/route.ts  (GET version history)
    prompt-templates/[promptKey]/versions/[vid]/restore/route.ts (POST)
    prompt-templates/preview/route.ts               (POST preview)
    prompt-templates/registry/route.ts              (GET registry)
    prompt-templates/validate/route.ts              (POST validate)
    projects/[id]/prompt-templates/route.ts         (GET project overrides)
    projects/[id]/prompt-templates/[promptKey]/route.ts (PUT/DELETE project)
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._common import (
    get_user_id,
    json_error,
    not_found,
    resolve_project,
    serialize,
    serialize_many,
)
from app.core.ids import new_id
from app.db.models import PromptTemplate, PromptVersion
from app.db.session import get_db

router = APIRouter()


def _registry():
    """Lazy import of the prompt registry (ported in parallel under app.ai).

    PORT NOTE: expects app.ai.prompts.registry to expose PROMPT_REGISTRY,
    get_prompt_definition(key) and get_default_slot_contents(key), mirroring
    src/lib/ai/prompts/registry.ts. Returns None while unavailable.
    """
    try:
        from app.ai.prompts import registry  # noqa: PLC0415

        return registry
    except Exception:  # pragma: no cover — module built in parallel
        return None


def _scope_filter(query, user_id: str, prompt_key: str | None, scope: str, project_id: str | None):
    query = query.where(
        PromptTemplate.user_id == user_id,
        PromptTemplate.scope == scope,
    )
    if prompt_key is not None:
        query = query.where(PromptTemplate.prompt_key == prompt_key)
    if scope == "project":
        query = query.where(PromptTemplate.project_id == project_id)
    else:
        query = query.where(PromptTemplate.project_id.is_(None))
    return query


def _upsert_templates(
    db: Session,
    user_id: str,
    prompt_key: str,
    body: dict[str, Any],
    scope: str,
    project_id: str | None,
):
    """Shared PUT body handling for global and project scopes (slots / full)."""
    mode = body.get("mode")

    if mode == "slots":
        slots = body.get("slots")
        if not isinstance(slots, dict):
            return json_error(400, "slots is required in slots mode")

        results: dict[str, Any] = {}
        for slot_key, content in slots.items():
            existing = db.execute(
                _scope_filter(
                    select(PromptTemplate), user_id, prompt_key, scope, project_id
                ).where(PromptTemplate.slot_key == slot_key)
            ).scalar_one_or_none()

            if existing:
                # Save current content as a version before updating
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
                    prompt_key=prompt_key,
                    slot_key=slot_key,
                    scope=scope,
                    project_id=project_id,
                    content=content,
                )
                db.add(inserted)
                db.flush()
                results[slot_key] = serialize(inserted)
        return results

    if mode == "full":
        content = body.get("content")
        if not isinstance(content, str):
            return json_error(400, "content is required in full mode")

        existing = db.execute(
            _scope_filter(
                select(PromptTemplate), user_id, prompt_key, scope, project_id
            ).where(PromptTemplate.slot_key.is_(None))
        ).scalar_one_or_none()

        if existing:
            db.add(
                PromptVersion(
                    id=new_id(), template_id=existing.id, content=existing.content
                )
            )
            existing.content = content
            db.flush()
            return serialize(existing)

        inserted = PromptTemplate(
            id=new_id(),
            user_id=user_id,
            prompt_key=prompt_key,
            slot_key=None,
            scope=scope,
            project_id=project_id,
            content=content,
        )
        db.add(inserted)
        db.flush()
        return JSONResponse(serialize(inserted), status_code=201)

    return json_error(400, "Invalid mode")


# ---------------------------------------------------------------------------
# Registry-backed endpoints (registered before /{prompt_key} routes)
# ---------------------------------------------------------------------------


@router.get("/prompt-templates/registry")
def get_registry():
    registry = _registry()
    if registry is None:
        # PORT NOTE: prompt registry not ported yet (built in parallel).
        return json_error(501, "Prompt registry not available")
    return [
        {
            "key": d.key,
            "nameKey": d.name_key,
            "descriptionKey": d.description_key,
            "category": d.category,
            "slots": [
                {
                    "key": s.key,
                    "nameKey": s.name_key,
                    "descriptionKey": s.description_key,
                    "defaultContent": s.default_content,
                    "editable": s.editable,
                }
                for s in d.slots
            ],
        }
        for d in registry.PROMPT_REGISTRY
    ]


@router.post("/prompt-templates/preview")
async def preview_prompt(request: Request):
    body = await request.json()
    prompt_key = body.get("promptKey")
    slots = body.get("slots") or {}

    if not prompt_key:
        return json_error(400, "promptKey is required")

    registry = _registry()
    if registry is None:
        return json_error(501, "Prompt registry not available")

    definition = registry.get_prompt_definition(prompt_key)
    if not definition:
        return json_error(404, f"Unknown prompt key: {prompt_key}")

    # Merge provided slots with defaults
    default_contents = registry.get_default_slot_contents(prompt_key) or {}
    merged_slots = {**default_contents, **slots}
    full_prompt = definition.build_full_prompt(merged_slots)

    # Compute highlights: which slots were overridden vs default
    highlights: dict[str, str] = {}
    for slot in definition.slots:
        if slot.key in slots and slots[slot.key] != slot.default_content:
            highlights[slot.key] = "overridden"
        else:
            highlights[slot.key] = "default"

    return {"fullPrompt": full_prompt, "highlights": highlights}


@router.post("/prompt-templates/validate")
async def validate_prompt(request: Request):
    body = await request.json()
    prompt_key = body.get("promptKey")
    content = body.get("content")

    if not prompt_key:
        return json_error(400, "promptKey is required")
    if not isinstance(content, str):
        return json_error(400, "content is required")

    registry = _registry()
    if registry is None:
        return json_error(501, "Prompt registry not available")

    definition = registry.get_prompt_definition(prompt_key)
    if not definition:
        return json_error(404, f"Unknown prompt key: {prompt_key}")

    warnings: list[str] = []
    # Check that locked slots' default content is still present in the edit
    for slot in definition.slots:
        if not slot.editable and slot.default_content:
            if slot.default_content not in content:
                warnings.append(
                    f'Locked slot "{slot.key}" content has been modified or removed. '
                    "This may cause unexpected behavior."
                )

    return {"valid": len(warnings) == 0, "warnings": warnings}


# ---------------------------------------------------------------------------
# Global overrides
# ---------------------------------------------------------------------------


@router.get("/prompt-templates")
def list_global_templates(request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    rows = (
        db.execute(_scope_filter(select(PromptTemplate), user_id, None, "global", None))
        .scalars()
        .all()
    )
    return serialize_many(rows)


@router.put("/prompt-templates/{prompt_key}")
async def put_global_template(prompt_key: str, request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    body = await request.json()
    return _upsert_templates(db, user_id, prompt_key, body, "global", None)


@router.delete("/prompt-templates/{prompt_key}")
def delete_global_template(prompt_key: str, request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    rows = (
        db.execute(
            _scope_filter(select(PromptTemplate), user_id, prompt_key, "global", None)
        )
        .scalars()
        .all()
    )
    for row in rows:
        db.delete(row)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Version history
# ---------------------------------------------------------------------------


@router.get("/prompt-templates/{prompt_key}/versions")
def list_versions(prompt_key: str, request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    templates = (
        db.execute(
            select(PromptTemplate).where(
                PromptTemplate.user_id == user_id,
                PromptTemplate.prompt_key == prompt_key,
            )
        )
        .scalars()
        .all()
    )
    if not templates:
        return []

    all_versions = []
    for template in templates:
        versions = (
            db.execute(
                select(PromptVersion)
                .where(PromptVersion.template_id == template.id)
                .order_by(PromptVersion.created_at.desc())
            )
            .scalars()
            .all()
        )
        for version in versions:
            all_versions.append(
                {
                    "id": version.id,
                    "templateId": version.template_id,
                    "slotKey": template.slot_key,
                    "scope": template.scope,
                    "projectId": template.project_id,
                    "content": version.content,
                    "createdAt": version.created_at,
                }
            )

    all_versions.sort(key=lambda v: v["createdAt"] or 0, reverse=True)
    return all_versions


@router.post("/prompt-templates/{prompt_key}/versions/{vid}/restore")
def restore_version(prompt_key: str, vid: str, request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)

    version = db.get(PromptVersion, vid)
    if not version:
        return json_error(404, "Version not found")

    template = db.get(PromptTemplate, version.template_id)
    if not template:
        return json_error(404, "Parent template not found")

    if template.user_id != user_id:
        return json_error(403, "Forbidden")

    # Save current content as a new version (for undo)
    db.add(PromptVersion(id=new_id(), template_id=template.id, content=template.content))
    template.content = version.content
    db.flush()
    return {"success": True, "template": serialize(template)}


# ---------------------------------------------------------------------------
# Project-level overrides
# ---------------------------------------------------------------------------


@router.get("/projects/{id}/prompt-templates")
def list_project_templates(id: str, request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    if not resolve_project(db, id, user_id):
        return not_found()
    rows = (
        db.execute(_scope_filter(select(PromptTemplate), user_id, None, "project", id))
        .scalars()
        .all()
    )
    return serialize_many(rows)


@router.put("/projects/{id}/prompt-templates/{prompt_key}")
async def put_project_template(
    id: str, prompt_key: str, request: Request, db: Session = Depends(get_db)
):
    user_id = get_user_id(request)
    if not resolve_project(db, id, user_id):
        return not_found()
    body = await request.json()
    return _upsert_templates(db, user_id, prompt_key, body, "project", id)


@router.delete("/projects/{id}/prompt-templates/{prompt_key}")
def delete_project_template(
    id: str, prompt_key: str, request: Request, db: Session = Depends(get_db)
):
    user_id = get_user_id(request)
    if not resolve_project(db, id, user_id):
        return not_found()
    rows = (
        db.execute(
            _scope_filter(select(PromptTemplate), user_id, prompt_key, "project", id)
        )
        .scalars()
        .all()
    )
    for row in rows:
        db.delete(row)
    return Response(status_code=204)
