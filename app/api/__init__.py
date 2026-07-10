"""API package — assembles every resource router into a single ``api_router``.

``api_router`` is mounted at prefix ``/api`` by app.main, so the individual
routers declare paths WITHOUT the ``/api`` prefix (e.g. ``/projects``). The
resulting URLs match the original Next.js App Router routes exactly.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.api import (
    agents,
    character_relations,
    characters,
    continuity,
    download,
    episodes,
    generate,
    imports,
    local_gpu,
    models,
    mood_board,
    projects,
    prompt_presets,
    prompt_templates,
    shot_assets,
    shots,
    storyboard,
    tasks,
    uploads,
)

api_router = APIRouter()

# Order matters: more specific static paths (e.g. /prompt-templates/registry)
# are registered by their own routers before dynamic /{param} catch-alls within
# the same router. Across routers, FastAPI matching is registration-ordered, so
# resource routers with fixed prefixes are safe in any order here.
for _module in (
    projects,
    episodes,
    characters,
    character_relations,
    shots,
    shot_assets,
    storyboard,
    mood_board,
    continuity,
    generate,
    download,
    imports,
    uploads,
    prompt_templates,
    prompt_presets,
    agents,
    models,
    tasks,
    local_gpu,
):
    api_router.include_router(_module.router)

__all__ = ["api_router"]
