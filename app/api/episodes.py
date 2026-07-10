"""Episodes router — port of:
    projects/[id]/episodes/route.ts
    projects/[id]/episodes/[episodeId]/route.ts
    projects/[id]/episodes/reorder/route.ts
    projects/[id]/merge-episodes/route.ts
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api._common import (
    get_user_id,
    json_error,
    load_assets_by_shot,
    load_shot_dialogues,
    load_shot_legacy_views_batch,
    mark_downstream_stale,
    not_found,
    resolve_project,
    serialize,
    serialize_many,
)
from app.core.ids import new_id
from app.db.models import (
    Character,
    Episode,
    EpisodeCharacter,
    Project,
    Shot,
    StoryboardVersion,
)
from app.db.session import get_db

router = APIRouter()

_PATCHABLE = {
    "title": "title",
    "description": "description",
    "keywords": "keywords",
    "idea": "idea",
    "script": "script",
    "outline": "outline",
    "status": "status",
    "generationMode": "generation_mode",
    "targetDuration": "target_duration",
}


def _resolve_episode(db: Session, project_id: str, episode_id: str) -> Episode | None:
    return db.execute(
        select(Episode).where(Episode.id == episode_id, Episode.project_id == project_id)
    ).scalar_one_or_none()


@router.get("/projects/{id}/episodes")
def list_episodes(id: str, request: Request, db: Session = Depends(get_db)):
    project = resolve_project(db, id, get_user_id(request))
    if not project:
        return not_found()

    all_episodes = (
        db.execute(
            select(Episode).where(Episode.project_id == id).order_by(Episode.sequence.asc())
        )
        .scalars()
        .all()
    )

    enriched = []
    for ep in all_episodes:
        data = serialize(ep)
        if ep.final_video_url:
            data["previewImages"] = []
            enriched.append(data)
            continue

        # 1) Collect frame images from shot_assets, deduplicated
        ep_shot_ids = [
            row
            for row in db.execute(select(Shot.id).where(Shot.episode_id == ep.id)).scalars()
        ]
        legacy = load_shot_legacy_views_batch(db, ep_shot_ids)
        frame_set: list[str] = []
        is_reference = ep.generation_mode == "reference"
        for sid in ep_shot_ids:
            view = legacy.get(sid)
            if not view:
                continue
            if is_reference:
                if view.scene_ref_frame and view.scene_ref_frame not in frame_set:
                    frame_set.append(view.scene_ref_frame)
            else:
                if view.first_frame and view.first_frame not in frame_set:
                    frame_set.append(view.first_frame)
                if view.last_frame and view.last_frame not in frame_set:
                    frame_set.append(view.last_frame)

        if frame_set:
            data["previewImages"] = frame_set
            enriched.append(data)
            continue

        # 2) Fall back to character reference images linked to this episode
        linked_char_ids = (
            db.execute(
                select(EpisodeCharacter.character_id).where(
                    EpisodeCharacter.episode_id == ep.id
                )
            )
            .scalars()
            .all()
        )
        char_urls: list[str] = []
        if linked_char_ids:
            char_urls = [
                url
                for url in db.execute(
                    select(Character.reference_image).where(
                        Character.id.in_(linked_char_ids),
                        Character.reference_image.is_not(None),
                    )
                ).scalars()
                if url
            ]
        data["previewImages"] = char_urls
        enriched.append(data)

    return enriched


@router.post("/projects/{id}/episodes")
async def create_episode(id: str, request: Request, db: Session = Depends(get_db)):
    project = resolve_project(db, id, get_user_id(request))
    if not project:
        return not_found()

    body = await request.json()
    max_seq = db.execute(
        select(func.max(Episode.sequence)).where(Episode.project_id == id)
    ).scalar()
    episode = Episode(
        id=new_id(),
        project_id=id,
        title=body.get("title"),
        description=body.get("description") or "",
        keywords=body.get("keywords") or "",
        sequence=(max_seq or 0) + 1,
    )
    db.add(episode)
    db.flush()
    return JSONResponse(serialize(episode), status_code=201)


@router.put("/projects/{id}/episodes/reorder")
async def reorder_episodes(id: str, request: Request, db: Session = Depends(get_db)):
    project = resolve_project(db, id, get_user_id(request))
    if not project:
        return not_found()

    body = await request.json()
    ordered_ids = body.get("orderedIds")
    if not isinstance(ordered_ids, list) or not ordered_ids:
        return json_error(400, "orderedIds must be a non-empty array")

    for index, episode_id in enumerate(ordered_ids):
        ep = _resolve_episode(db, id, episode_id)
        if ep:
            ep.sequence = index + 1
    db.flush()
    return {"success": True}


@router.get("/projects/{id}/episodes/{episode_id}")
def get_episode(id: str, episode_id: str, request: Request, db: Session = Depends(get_db)):
    project = resolve_project(db, id, get_user_id(request))
    episode = _resolve_episode(db, id, episode_id) if project else None
    if not project or not episode:
        return not_found()

    version_id = request.query_params.get("versionId")

    all_versions = (
        db.execute(
            select(StoryboardVersion)
            .where(
                StoryboardVersion.project_id == id,
                StoryboardVersion.episode_id == episode_id,
            )
            .order_by(StoryboardVersion.version_num.desc())
        )
        .scalars()
        .all()
    )
    resolved_version_id = version_id or (all_versions[0].id if all_versions else None)

    # Characters linked to this episode via episode_characters
    linked_char_ids = (
        db.execute(
            select(EpisodeCharacter.character_id).where(
                EpisodeCharacter.episode_id == episode_id
            )
        )
        .scalars()
        .all()
    )
    ep_characters = []
    if linked_char_ids:
        ep_characters = (
            db.execute(select(Character).where(Character.id.in_(linked_char_ids)))
            .scalars()
            .all()
        )
    # No links = no characters for this episode (user needs to run character extraction)

    episode_shots = []
    if resolved_version_id:
        episode_shots = (
            db.execute(
                select(Shot)
                .where(
                    Shot.project_id == id,
                    Shot.episode_id == episode_id,
                    Shot.version_id == resolved_version_id,
                )
                .order_by(Shot.sequence.asc())
            )
            .scalars()
            .all()
        )

    assets_by_shot = load_assets_by_shot(db, [s.id for s in episode_shots])
    enriched_shots = []
    for shot in episode_shots:
        row = serialize(shot)
        row["dialogues"] = load_shot_dialogues(db, shot.id)
        row["assets"] = assets_by_shot.get(shot.id, [])
        enriched_shots.append(row)

    # Response merges episode fields but presents the project id/title at the
    # top level, with the episode id under `episodeId` (matches the TS route).
    return {
        **serialize(episode),
        "id": project.id,
        "episodeId": episode.id,
        "title": project.title,
        "idea": episode.idea,
        "script": episode.script,
        "status": episode.status,
        "finalVideoUrl": episode.final_video_url,
        "generationMode": episode.generation_mode,
        "characters": serialize_many(ep_characters),
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


@router.patch("/projects/{id}/episodes/{episode_id}")
async def patch_episode(
    id: str, episode_id: str, request: Request, db: Session = Depends(get_db)
):
    project = resolve_project(db, id, get_user_id(request))
    episode = _resolve_episode(db, id, episode_id) if project else None
    if not project or not episode:
        return not_found()

    body = await request.json()
    for key, attr in _PATCHABLE.items():
        if key in body:
            setattr(episode, attr, body[key])

    if "script" in body:
        mark_downstream_stale(db, "episode", episode_id)

    db.flush()
    return serialize(episode)


@router.delete("/projects/{id}/episodes/{episode_id}")
def delete_episode(id: str, episode_id: str, request: Request, db: Session = Depends(get_db)):
    project = resolve_project(db, id, get_user_id(request))
    episode = _resolve_episode(db, id, episode_id) if project else None
    if not project or not episode:
        return not_found()

    # Refuse to delete the last episode
    count = db.execute(
        select(func.count()).select_from(Episode).where(Episode.project_id == id)
    ).scalar()
    if (count or 0) <= 1:
        return json_error(400, "Cannot delete the last episode")

    db.delete(episode)
    return Response(status_code=204)


@router.post("/projects/{id}/merge-episodes")
async def merge_episodes(id: str, request: Request, db: Session = Depends(get_db)):
    project = resolve_project(db, id, get_user_id(request))
    if not project:
        return json_error(404, "Project not found")

    body = await request.json()
    episode_ids = body.get("episodeIds")
    if not isinstance(episode_ids, list) or len(episode_ids) < 2:
        return json_error(400, "At least 2 episodes required")

    selected = (
        db.execute(
            select(Episode)
            .where(Episode.project_id == id, Episode.id.in_(episode_ids))
            .order_by(Episode.sequence.asc())
        )
        .scalars()
        .all()
    )
    if len(selected) != len(episode_ids):
        return json_error(400, "Some episodes not found")

    missing = next((e for e in selected if not e.final_video_url), None)
    if missing:
        return json_error(400, f'Episode "{missing.title}" has no video')

    try:
        # Lazy import: the video subsystem is built in parallel.
        from app.video.ffmpeg import assemble_video

        video_paths = [e.final_video_url for e in selected]
        result = assemble_video(
            video_paths=video_paths,
            subtitles=[],
            project_id=id,
            shot_durations=[],
        )
        return {"videoUrl": result.get("video_path"), "status": "ok"}
    except Exception as err:  # noqa: BLE001 — mirror TS catch-all
        print(f"[MergeEpisodes] Error: {err}")
        return json_error(500, str(err) or "Merge failed")
