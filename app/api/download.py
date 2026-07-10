"""Project download router — port of projects/[id]/download/route.ts.

Streams a ZIP of all project assets (character refs, shot frames/videos, final
video). The TS route used `archiver`; this uses the stdlib ``zipfile``.
Non-JSON error bodies (plain text + status) are preserved.
"""
from __future__ import annotations

import io
import os
import re
import zipfile

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api._common import get_user_id, load_shot_legacy_views_batch, resolve_project
from app.db.models import Character, Shot
from app.db.session import get_db

router = APIRouter()

# Keep alphanumerics, CJK, underscore, hyphen; replace everything else with "_".
_SAFE_RE = re.compile(r"[^a-zA-Z0-9一-鿿_-]")


def _ext(path: str, default: str) -> str:
    return os.path.splitext(path)[1] or default


@router.get("/projects/{id}/download")
def download_project(id: str, request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    project = resolve_project(db, id, user_id) if user_id else None
    if not project:
        return Response("Project not found", status_code=404)

    all_shots = (
        db.execute(
            select(Shot).where(Shot.project_id == id).order_by(Shot.sequence.asc())
        )
        .scalars()
        .all()
    )
    if not all_shots:
        return Response("No shots to download", status_code=400)

    project_chars = (
        db.execute(select(Character).where(Character.project_id == id)).scalars().all()
    )

    buffer = io.BytesIO()
    archive = zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=5)

    def add_file(src_path: str | None, archive_name: str) -> None:
        if not src_path:
            return
        abs_path = os.path.abspath(src_path)
        if os.path.isfile(abs_path):
            archive.write(abs_path, arcname=archive_name)

    # 1. Character reference images
    for char in project_chars:
        if char.reference_image:
            ext = _ext(char.reference_image, ".png")
            safe = _SAFE_RE.sub("_", char.name)
            add_file(char.reference_image, f"characters/{safe}{ext}")

    # 2. Shot assets (all types) — read from unified shot_assets table
    legacy_map = load_shot_legacy_views_batch(db, [s.id for s in all_shots])
    for shot in all_shots:
        prefix = f"shot-{str(shot.sequence).zfill(2)}"
        view = legacy_map.get(shot.id)
        if not view:
            continue
        if view.first_frame:
            add_file(view.first_frame, f"{prefix}/first-frame{_ext(view.first_frame, '.png')}")
        if view.last_frame:
            add_file(view.last_frame, f"{prefix}/last-frame{_ext(view.last_frame, '.png')}")
        if view.video_url:
            add_file(view.video_url, f"{prefix}/video{_ext(view.video_url, '.mp4')}")
        if view.scene_ref_frame:
            add_file(view.scene_ref_frame, f"{prefix}/scene-frame{_ext(view.scene_ref_frame, '.png')}")
        if view.reference_video_url:
            add_file(view.reference_video_url, f"{prefix}/ref-video{_ext(view.reference_video_url, '.mp4')}")
        ref_idx = 1
        for ref in view.reference_images:
            if ref.file_url:
                add_file(ref.file_url, f"{prefix}/ref-{str(ref_idx).zfill(2)}{_ext(ref.file_url, '.png')}")
                ref_idx += 1

    # 3. Final assembled video
    if project.final_video_url:
        add_file(project.final_video_url, f"final-video{_ext(project.final_video_url, '.mp4')}")

    archive.close()
    data = buffer.getvalue()

    safe_name = _SAFE_RE.sub("_", project.title or "project")
    from urllib.parse import quote

    return Response(
        content=data,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{quote(safe_name)}-storyboard.zip"'
        },
    )
