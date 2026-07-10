"""Agents router — port of:
    agents/route.ts
    agents/[id]/route.ts
    projects/[id]/agent-bindings/route.ts
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._common import get_user_id, json_error, not_found, serialize, serialize_many
from app.core.ids import new_id
from app.db.models import Agent, AgentBinding
from app.db.session import get_db

router = APIRouter()

VALID_CATEGORIES = [
    "script_outline",
    "script_generate",
    "script_parse",
    "character_extract",
    "shot_split",
    "keyframe_prompts",
    "video_prompts",
    "ref_image_prompts",
    "ref_video_prompts",
]


@router.get("/agents")
def list_agents(request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    rows = db.execute(select(Agent).where(Agent.user_id == user_id)).scalars().all()
    return serialize_many(rows)


@router.post("/agents")
async def create_agent(request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    body = await request.json()

    if not body.get("name") or not body.get("category") or not body.get("appId") or not body.get("apiKey"):
        return json_error(400, "Missing required fields")
    if body["category"] not in VALID_CATEGORIES:
        return json_error(400, "Invalid category")

    agent = Agent(
        id=new_id(),
        user_id=user_id,
        name=body["name"],
        platform=body.get("platform") or "bailian",
        category=body["category"],
        app_id=body["appId"],
        api_key=body["apiKey"],
        description=body.get("description") or "",
    )
    db.add(agent)
    db.flush()
    return JSONResponse(serialize(agent), status_code=201)


@router.patch("/agents/{id}")
async def patch_agent(id: str, request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    existing = db.execute(
        select(Agent).where(Agent.id == id, Agent.user_id == user_id)
    ).scalar_one_or_none()
    if not existing:
        return not_found()

    body = await request.json()
    if "name" in body:
        existing.name = body["name"]
    if "category" in body:
        existing.category = body["category"]
    if "appId" in body:
        existing.app_id = body["appId"]
    if "apiKey" in body:
        existing.api_key = body["apiKey"]
    if "description" in body:
        existing.description = body["description"]
    db.flush()
    return serialize(existing)


@router.delete("/agents/{id}")
def delete_agent(id: str, request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    existing = db.execute(
        select(Agent).where(Agent.id == id, Agent.user_id == user_id)
    ).scalar_one_or_none()
    if not existing:
        return not_found()
    db.delete(existing)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Agent bindings (per-project category -> agent)
# NOTE: like the TS route, these endpoints do not verify project ownership.
# ---------------------------------------------------------------------------


@router.get("/projects/{id}/agent-bindings")
def list_agent_bindings(id: str, db: Session = Depends(get_db)):
    rows = db.execute(
        select(AgentBinding, Agent.name)
        .outerjoin(Agent, AgentBinding.agent_id == Agent.id)
        .where(AgentBinding.project_id == id)
    ).all()
    return [
        {
            "id": b.id,
            "projectId": b.project_id,
            "category": b.category,
            "agentId": b.agent_id,
            "agentName": name,
        }
        for b, name in rows
    ]


@router.put("/projects/{id}/agent-bindings")
async def put_agent_binding(id: str, request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    category = body.get("category")
    agent_id = body.get("agentId")

    if category not in VALID_CATEGORIES:
        return json_error(400, "Invalid category")

    if not agent_id:
        rows = (
            db.execute(
                select(AgentBinding).where(
                    AgentBinding.project_id == id, AgentBinding.category == category
                )
            )
            .scalars()
            .all()
        )
        for row in rows:
            db.delete(row)
        return {"ok": True}

    existing = db.execute(
        select(AgentBinding).where(
            AgentBinding.project_id == id, AgentBinding.category == category
        )
    ).scalar_one_or_none()

    if existing:
        existing.agent_id = agent_id
    else:
        db.add(
            AgentBinding(
                id=new_id(), project_id=id, category=category, agent_id=agent_id
            )
        )
    db.flush()
    return {"ok": True}
