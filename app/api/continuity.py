"""Continuity + emotion analysis router — port of:
    projects/[id]/continuity-check/route.ts
    projects/[id]/emotion-analysis/route.ts

Both are AI-backed. The AI provider factory / continuity checker live under
app.ai and app.pipeline (built in parallel) and are imported lazily.
"""
from __future__ import annotations

import json
import re

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._common import (
    assert_project_ownership,
    json_error,
    load_shot_legacy_views_batch,
    not_found,
)
from app.db.models import Shot
from app.db.session import get_db

router = APIRouter()

EMOTION_PROMPT = """Analyze these shot descriptions from a screenplay and rate each for tension and emotional intensity on a 0-100 scale.

Shots:
{shots}

Output ONLY valid JSON array (no markdown):
[{"shotSequence": 1, "tension": 50, "emotion": 60}, ...]

Guidelines:
- tension: 0=calm/peaceful, 50=mild conflict, 100=peak crisis
- emotion: 0=neutral, 50=moderate feeling, 100=overwhelming emotion"""


def _resolve_ai_provider(model_config):
    """Lazy port of resolveAIProvider(modelConfig) from ai/provider-factory."""
    from app.ai.provider_factory import resolve_ai_provider  # lazy

    return resolve_ai_provider(model_config)


@router.post("/projects/{id}/continuity-check")
async def continuity_check(id: str, request: Request, db: Session = Depends(get_db)):
    if not assert_project_ownership(db, request, id):
        return not_found()
    body = await request.json()

    all_shots = (
        db.execute(
            select(Shot).where(Shot.project_id == id).order_by(Shot.sequence.asc())
        )
        .scalars()
        .all()
    )
    legacy = load_shot_legacy_views_batch(db, [s.id for s in all_shots])

    shots_with_frames = []
    for s in all_shots:
        view = legacy.get(s.id)
        first = view.first_frame if view else None
        last = view.last_frame if view else None
        if first and last:
            shots_with_frames.append(
                {"sequence": s.sequence, "firstFrame": first, "lastFrame": last}
            )

    if len(shots_with_frames) < 2:
        return {"results": [], "message": "Need at least 2 shots with frames"}

    # PORT NOTE: resolveAIProvider + checkContinuity live in app.ai / app.pipeline
    # (built in parallel). Return 501 while the AI vision layer is unavailable.
    try:
        provider = _resolve_ai_provider(body.get("modelConfig"))
        from app.pipeline.continuity_check import check_continuity  # lazy
    except Exception:
        return json_error(501, "Continuity check AI provider not available")

    results = []
    for i in range(len(shots_with_frames) - 1):
        current = shots_with_frames[i]
        nxt = shots_with_frames[i + 1]
        if current["lastFrame"] and nxt["firstFrame"]:
            result = await check_continuity(
                provider, current["lastFrame"], nxt["firstFrame"]
            )
            results.append(
                {
                    "shotASequence": current["sequence"],
                    "shotBSequence": nxt["sequence"],
                    "pass": result["pass"],
                    "issues": result["issues"],
                }
            )

    return {"results": results}


@router.post("/projects/{id}/emotion-analysis")
async def emotion_analysis(id: str, request: Request, db: Session = Depends(get_db)):
    if not assert_project_ownership(db, request, id):
        return not_found()
    body = await request.json()

    all_shots = (
        db.execute(
            select(Shot).where(Shot.project_id == id).order_by(Shot.sequence.asc())
        )
        .scalars()
        .all()
    )
    if not all_shots:
        return {"scores": []}

    shots_text = "\n".join(
        f"Shot {s.sequence}: {s.prompt or s.motion_script or ''}" for s in all_shots
    )

    # PORT NOTE: uses resolveAIProvider(...).generate_text — app.ai built in parallel.
    try:
        provider = _resolve_ai_provider(body.get("modelConfig"))
        result = await provider.generate_text(EMOTION_PROMPT.replace("{shots}", shots_text))
        match = re.search(r"\[[\s\S]*\]", result)
        if not match:
            return {"scores": []}
        return {"scores": json.loads(match.group(0))}
    except Exception:  # noqa: BLE001 — TS returns empty scores on any failure
        return {"scores": []}
