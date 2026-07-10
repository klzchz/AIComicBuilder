"""Uploads router — port of:
    uploads/[...path]/route.ts             (serve uploaded files)
    projects/[id]/upload-script/route.ts   (upload a script file -> episodes)
"""
from __future__ import annotations

import asyncio
import json
import os

from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api._common import (
    chunk_text,
    extract_text_from_file,
    get_user_id,
    json_error,
    not_found,
    resolve_project,
    serialize_many,
)
from app.config import settings
from app.core.ids import new_id
from app.db.models import Episode
from app.db.session import get_db

router = APIRouter()

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}


@router.get("/uploads/{path:path}")
def serve_upload(path: str):
    upload_dir = os.path.realpath(settings.UPLOAD_DIR)
    resolved = os.path.realpath(os.path.join(upload_dir, path))

    # Prevent directory traversal
    if not (resolved == upload_dir or resolved.startswith(upload_dir + os.sep)):
        return json_error(403, "Forbidden")

    if not os.path.isfile(resolved):
        return not_found()

    ext = os.path.splitext(resolved)[1].lower()
    content_type = MIME_TYPES.get(ext, "application/octet-stream")
    return FileResponse(resolved, media_type=content_type)


@router.post("/projects/{id}/upload-script")
async def upload_script(id: str, request: Request, db: Session = Depends(get_db)):
    """Upload a script file (txt/docx/pdf), split it into episodes with the
    LLM, and create the episode rows."""
    user_id = get_user_id(request)
    if not resolve_project(db, id, user_id):
        return not_found()

    form = await request.form()
    file = form.get("file")
    model_config_raw = form.get("modelConfig")

    if not isinstance(file, UploadFile):
        return json_error(400, "No file uploaded")
    if not model_config_raw:
        return json_error(400, "No model config provided")

    model_config = json.loads(model_config_raw)
    if not model_config.get("text"):
        return json_error(400, "No text model configured")

    # Extract text from the file
    data = await file.read()
    try:
        full_text = extract_text_from_file(data, file.filename or "")
    except Exception as err:  # noqa: BLE001 — mirror TS catch-all
        return json_error(400, str(err) or "Failed to parse file")

    if not full_text.strip():
        return json_error(400, "File contains no text")

    chunks = chunk_text(full_text)

    try:
        # Lazy imports: AI layer and prompt builders are built in parallel.
        from app.ai import generate_text
        from app.ai.prompts.script_split import build_script_split_prompt
    except Exception:
        # PORT NOTE: needs app.ai.generate_text + app.ai.prompts.script_split.
        return json_error(501, "Script split AI pipeline not available")

    system = None
    try:
        from app.ai.prompts.resolver import resolve_prompt

        system = resolve_prompt("script_split", user_id=user_id, project_id=id)
        if asyncio.iscoroutine(system):
            system = await system
    except Exception:  # pragma: no cover — resolver built in parallel
        system = None

    from app.api.imports import extract_json

    async def process_chunk(idx: int, chunk: str) -> list:
        prompt = build_script_split_prompt(
            chunk,
            chunk_index=idx,
            total_chunks=len(chunks),
            episode_offset=0,  # approximate — exact offset tricky with concurrency
        )
        try:
            from app.ai.types import TextOptions

            text = await generate_text(prompt, TextOptions(system_prompt=system))
        except TypeError:
            text = await generate_text(prompt)
        return json.loads(extract_json(text))

    chunk_results = await asyncio.gather(
        *(process_chunk(i, c) for i, c in enumerate(chunks))
    )
    all_episodes = [ep for chunk in chunk_results for ep in chunk]

    if not all_episodes:
        return json_error(422, "AI could not split the script into episodes")

    # Get current max sequence
    max_seq = db.execute(
        select(func.max(Episode.sequence)).where(Episode.project_id == id)
    ).scalar()
    seq = (max_seq or 0) + 1

    created = []
    for ep in all_episodes:
        row = Episode(
            id=new_id(),
            project_id=id,
            title=ep.get("title"),
            description=ep.get("description") or "",
            keywords=ep.get("keywords") or "",
            idea=ep.get("idea") or "",
            sequence=seq,
        )
        seq += 1
        db.add(row)
        created.append(row)
    db.flush()

    print(f"[UploadScript] Created {len(created)} episodes from {len(chunks)} chunks")

    return JSONResponse(
        {"episodes": serialize_many(created), "count": len(created)},
        status_code=201,
    )
