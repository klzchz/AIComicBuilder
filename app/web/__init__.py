"""Server-rendered web UI — functional port of the Next.js pages (src/app/[locale]).

Jinja2 + HTMX + Tailwind (Play CDN). Pages server-render their initial state
straight from the DB, then use HTMX partials for status polling and a small
fetch() helper (static/app.js) for JSON mutations against /api.

UI is English-only by design (no locale switcher) — labels reuse messages/en.json copy.
"""
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

_WEB_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(_WEB_DIR / "templates"))

web_router = APIRouter()


def mount_static(app) -> None:
    """Mount app/web/static at /static."""
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR / "static")), name="static")


def _render(request: Request, template: str, **ctx) -> HTMLResponse:
    ctx.setdefault("active_tab", None)
    return templates.TemplateResponse(request, template, ctx)


# ── data helpers (server-render initial state from the DB) ─────────────────

def _serialize_shot(shot, assets, dialogues) -> dict:
    def active(type_, seq=0):
        for a in assets:
            if a.type == type_ and a.sequence_in_type == seq and a.is_active == 1:
                return a
        return None

    refs = sorted(
        [a for a in assets if a.type == "reference" and a.is_active == 1],
        key=lambda a: a.sequence_in_type,
    )
    ff, lf = active("first_frame"), active("last_frame")
    kv, rv = active("keyframe_video"), active("reference_video")

    def img(a):
        return {"id": a.id, "url": a.file_url, "status": a.status, "prompt": a.prompt or ""} if a else None

    return {
        "id": shot.id,
        "sequence": shot.sequence,
        "prompt": shot.prompt or "",
        "video_script": shot.video_script or "",
        "motion_script": shot.motion_script or "",
        "camera_direction": shot.camera_direction or "static",
        "duration": shot.duration,
        "video_prompt": shot.video_prompt or "",
        "status": shot.status,
        "is_stale": bool(shot.is_stale),
        "first_frame": img(ff),
        "last_frame": img(lf),
        "references": [img(a) for a in refs],
        "keyframe_video": img(kv),
        "reference_video": img(rv),
        "dialogues": dialogues,
    }


def _load_shots(session, project_id: str, version_id: str | None):
    from sqlalchemy import select

    from app.db.models import Character, Dialogue, Shot, ShotAsset, StoryboardVersion

    versions = (
        session.execute(
            select(StoryboardVersion)
            .where(StoryboardVersion.project_id == project_id)
            .order_by(StoryboardVersion.version_num)
        )
        .scalars()
        .all()
    )
    if version_id is None and versions:
        version_id = versions[-1].id  # newest version by default (mirrors GET /api/projects/{id})

    q = select(Shot).where(Shot.project_id == project_id).order_by(Shot.sequence)
    if version_id:
        q = q.where(Shot.version_id == version_id)
    else:
        q = q.where(Shot.version_id.is_(None))
    shots = session.execute(q).scalars().all()

    shot_ids = [s.id for s in shots]
    assets_by_shot: dict[str, list] = {sid: [] for sid in shot_ids}
    dialogues_by_shot: dict[str, list] = {sid: [] for sid in shot_ids}
    if shot_ids:
        rows = session.execute(select(ShotAsset).where(ShotAsset.shot_id.in_(shot_ids))).scalars().all()
        for a in rows:
            assets_by_shot.setdefault(a.shot_id, []).append(a)
        drows = session.execute(
            select(Dialogue, Character.name)
            .join(Character, Dialogue.character_id == Character.id)
            .where(Dialogue.shot_id.in_(shot_ids))
            .order_by(Dialogue.sequence)
        ).all()
        for d, char_name in drows:
            dialogues_by_shot.setdefault(d.shot_id, []).append({"character": char_name, "text": d.text})

    serialized = [
        _serialize_shot(s, assets_by_shot.get(s.id, []), dialogues_by_shot.get(s.id, []))
        for s in shots
    ]
    return serialized, versions, version_id


def _shot_step_state(shot: dict, mode: str) -> dict:
    """Per-step pipeline status for the vertical shot card (task_plan.md):
    text → frames → video prompt → video; each done/todo/running/failed."""

    def asset_state(a):
        if not a:
            return "todo"
        if a["status"] == "generating":
            return "running"
        if a["status"] == "failed":
            return "failed"
        return "done" if a["url"] else "todo"

    text = "done" if (shot["prompt"] or shot["video_script"]) else "todo"
    if mode == "reference":
        frame_assets = shot["references"]
        video = shot["reference_video"]
    else:
        frame_assets = [a for a in (shot["first_frame"], shot["last_frame"]) if a]
        video = shot["keyframe_video"]

    states = [asset_state(a) for a in frame_assets]
    if not states:
        frames = "todo"
    elif "running" in states:
        frames = "running"
    elif "failed" in states:
        frames = "failed"
    elif all(s == "done" for s in states) and (mode == "reference" or len(states) == 2):
        frames = "done"
    else:
        frames = "todo"

    steps = {
        "text": text,
        "frames": frames,
        "video_prompt": "done" if shot["video_prompt"] else "todo",
        "video": asset_state(video),
    }
    order = ["text", "frames", "video_prompt", "video"]
    steps["next"] = next((s for s in order if steps[s] in ("todo", "failed")), None)
    steps["any_running"] = any(steps[s] == "running" for s in order)
    steps["any_failed"] = any(steps[s] == "failed" for s in order)
    return steps


def _storyboard_ctx(session, project, version_id: str | None) -> dict:
    shots, versions, active_version = _load_shots(session, project.id, version_id)
    mode = project.generation_mode or "keyframe"
    cards = [{"shot": s, "steps": _shot_step_state(s, mode)} for s in shots]
    # Kanban lanes mirror the source (needs frames / needs prompt / needs video / done).
    kanban = {"needs_frames": [], "needs_prompt": [], "needs_video": [], "done": []}
    for c in cards:
        nxt = c["steps"]["next"]
        if nxt in ("text", "frames"):
            kanban["needs_frames"].append(c)
        elif nxt == "video_prompt":
            kanban["needs_prompt"].append(c)
        elif nxt == "video":
            kanban["needs_video"].append(c)
        else:
            kanban["done"].append(c)
    return {
        "cards": cards,
        "kanban": kanban,
        "versions": [{"id": v.id, "label": v.label, "version_num": v.version_num} for v in versions],
        "active_version": active_version,
        "mode": mode,
        "any_running": any(c["steps"]["any_running"] for c in cards),
    }


def _project_dict(project) -> dict:
    return {
        "id": project.id,
        "title": project.title,
        "idea": project.idea or "",
        "script": project.script or "",
        "outline": project.outline or "",
        "status": project.status,
        "generation_mode": project.generation_mode,
        "final_video_url": project.final_video_url,
    }


def _editor_page(request: Request, project_id: str, template: str, active_tab: str, extra=None):
    from app.db.session import db_session
    from app.db.models import Project

    with db_session() as session:
        project = session.get(Project, project_id)
        if not project:
            return _render(request, "not_found.html")
        ctx = {"project": _project_dict(project), "active_tab": active_tab}
        if extra:
            ctx.update(extra(session, project))
    return _render(request, template, **ctx)


def _characters_ctx(session, project) -> dict:
    from sqlalchemy import select

    from app.db.models import Character

    chars = (
        session.execute(select(Character).where(Character.project_id == project.id))
        .scalars()
        .all()
    )
    rows = [
        {
            "id": c.id,
            "name": c.name,
            "description": c.description or "",
            "visual_hint": c.visual_hint or "",
            "reference_image": c.reference_image,
            "scope": c.scope,
        }
        for c in chars
    ]
    return {
        "characters_main": [c for c in rows if c["scope"] == "main"],
        "characters_guest": [c for c in rows if c["scope"] != "main"],
    }


# ── pages ───────────────────────────────────────────────────────────────────

@web_router.get("/", response_class=HTMLResponse)
def projects_page(request: Request):
    from sqlalchemy import select

    from app.db.session import db_session
    from app.db.models import Project

    with db_session() as session:
        projects = session.execute(select(Project).order_by(Project.created_at.desc())).scalars().all()
        rows = [
            {
                "id": p.id,
                "title": p.title,
                "idea": p.idea or "",
                "status": p.status,
                "generation_mode": p.generation_mode,
                "created_at": p.created_at,
            }
            for p in projects
        ]
    return _render(request, "index.html", projects=rows)


@web_router.get("/project/{project_id}")
def project_root(project_id: str):
    return RedirectResponse(f"/project/{project_id}/script")


@web_router.get("/project/{project_id}/script", response_class=HTMLResponse)
def script_page(request: Request, project_id: str):
    return _editor_page(request, project_id, "script.html", "script")


@web_router.get("/project/{project_id}/characters", response_class=HTMLResponse)
def characters_page(request: Request, project_id: str):
    return _editor_page(request, project_id, "characters.html", "characters", _characters_ctx)


@web_router.get("/project/{project_id}/characters/partial", response_class=HTMLResponse)
def characters_partial(request: Request, project_id: str):
    """HTMX polling target — re-renders the character grid from the DB."""
    from app.db.session import db_session
    from app.db.models import Project

    with db_session() as session:
        project = session.get(Project, project_id)
        if not project:
            return HTMLResponse("", status_code=404)
        ctx = _characters_ctx(session, project)
        ctx["project"] = _project_dict(project)
    return templates.TemplateResponse(request, "_characters_grid.html", ctx)


@web_router.get("/project/{project_id}/storyboard", response_class=HTMLResponse)
def storyboard_page(
    request: Request, project_id: str, versionId: str | None = None, view: str = "list"
):
    def extra(session, project):
        ctx = _storyboard_ctx(session, project, versionId)
        ctx["view"] = view if view in ("list", "kanban") else "list"
        return ctx

    return _editor_page(request, project_id, "storyboard.html", "storyboard", extra)


@web_router.get("/project/{project_id}/storyboard/partial", response_class=HTMLResponse)
def storyboard_partial(
    request: Request, project_id: str, versionId: str | None = None, view: str = "list"
):
    """HTMX polling target — re-renders the shot list / kanban from the DB."""
    from app.db.session import db_session
    from app.db.models import Project

    with db_session() as session:
        project = session.get(Project, project_id)
        if not project:
            return HTMLResponse("", status_code=404)
        ctx = _storyboard_ctx(session, project, versionId)
        ctx["view"] = view if view in ("list", "kanban") else "list"
        ctx["project"] = _project_dict(project)
    return templates.TemplateResponse(request, "_shots.html", ctx)


@web_router.get("/project/{project_id}/preview", response_class=HTMLResponse)
def preview_page(request: Request, project_id: str):
    def extra(session, project):
        ctx = _storyboard_ctx(session, project, None)
        total = sum((c["shot"]["duration"] or 0) for c in ctx["cards"])
        done = sum(1 for c in ctx["cards"] if c["steps"]["video"] == "done")
        return {
            "cards": ctx["cards"],
            "mode": ctx["mode"],
            "total_duration": total,
            "videos_done": done,
        }

    return _editor_page(request, project_id, "preview.html", "preview", extra)


AGENT_CATEGORIES = [
    ("script_outline", "Script Outline"),
    ("script_generate", "Script Generation"),
    ("script_parse", "Script Parsing"),
    ("character_extract", "Character Extraction"),
    ("shot_split", "Shot Splitting"),
    ("keyframe_prompts", "Keyframe Prompts"),
    ("video_prompts", "Video Prompts"),
    ("ref_image_prompts", "Reference Image Prompts"),
    ("ref_video_prompts", "Reference Video Prompts"),
]


@web_router.get("/project/{project_id}/settings", response_class=HTMLResponse)
def project_settings_page(request: Request, project_id: str):
    """Per-project settings — agent bindings (port of agent-picker)."""
    from sqlalchemy import select

    def extra(session, project):
        from app.db.models import Agent, AgentBinding

        agents = session.execute(select(Agent).order_by(Agent.name)).scalars().all()
        bindings = (
            session.execute(select(AgentBinding).where(AgentBinding.project_id == project.id))
            .scalars()
            .all()
        )
        return {
            "agents": [{"id": a.id, "name": a.name, "category": a.category} for a in agents],
            "bindings": {b.category: b.agent_id for b in bindings},
            "agent_categories": AGENT_CATEGORIES,
        }

    return _editor_page(request, project_id, "project_settings.html", "settings", extra)


@web_router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    from sqlalchemy import select

    from app.db.session import db_session
    from app.db.models import Agent, PromptTemplate

    with db_session() as session:
        agents = session.execute(select(Agent).order_by(Agent.created_at.desc())).scalars().all()
        agent_rows = [
            {
                "id": a.id,
                "name": a.name,
                "category": a.category,
                "platform": a.platform,
                "app_id": a.app_id,
                "description": a.description or "",
            }
            for a in agents
        ]
        tpls = (
            session.execute(select(PromptTemplate).where(PromptTemplate.scope == "global"))
            .scalars()
            .all()
        )
        tpl_rows = [
            {"id": t.id, "prompt_key": t.prompt_key, "slot_key": t.slot_key, "content": t.content}
            for t in tpls
        ]
    return _render(
        request,
        "settings.html",
        agents=agent_rows,
        prompt_templates=tpl_rows,
        agent_categories=AGENT_CATEGORIES,
    )
