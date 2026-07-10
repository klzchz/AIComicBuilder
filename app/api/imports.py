"""Script import router — port of:
    projects/[id]/import/parse/route.ts
    projects/[id]/import/split/route.ts
    projects/[id]/import/characters/route.ts
    projects/[id]/import/generate/route.ts
    projects/[id]/import/logs/route.ts

All user-facing import log messages were translated from Chinese to English.
"""
from __future__ import annotations

import asyncio
import json
import re

from fastapi import APIRouter, Depends, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api._common import (
    add_import_log,
    chunk_text,
    extract_text_from_file,
    get_user_id,
    json_error,
    not_found,
    resolve_project,
    serialize_many,
)
from app.core.ids import new_id
from app.db.models import (
    Character,
    CharacterRelation,
    Episode,
    EpisodeCharacter,
    ImportLog,
)
from app.db.session import db_session, get_db

router = APIRouter()


def extract_json(text: str) -> str:
    """Port of extractJSON (ai-sdk.ts): pull the first JSON object/array out
    of an LLM response that may be wrapped in markdown fences or prose."""
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1)
    text = text.strip()
    start_candidates = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not start_candidates:
        return text
    start = min(start_candidates)
    end = max(text.rfind("]"), text.rfind("}"))
    return text[start : end + 1] if end > start else text[start:]


async def _ai_generate_text(prompt: str, system: str | None = None) -> str:
    """Call the AI layer (built in parallel) lazily.

    PORT NOTE: The TS routes build a per-request language model from the
    client-supplied modelConfig (createLanguageModel). The Python AI layer
    resolves providers globally (app.ai.setup.initialize_providers); the
    request's modelConfig is validated for presence but provider selection is
    delegated to app.ai.
    """
    from app.ai import generate_text  # lazy — built in parallel

    try:
        from app.ai.types import TextOptions

        return await generate_text(prompt, TextOptions(system_prompt=system))
    except TypeError:
        return await generate_text(prompt)


def _resolve_system_prompt(prompt_key: str, user_id: str, project_id: str) -> str | None:
    """Port of resolvePrompt(promptKey, {userId, projectId}) — lazy import."""
    try:
        from app.ai.prompts.resolver import resolve_prompt  # lazy

        result = resolve_prompt(prompt_key, user_id=user_id, project_id=project_id)
        if asyncio.iscoroutine(result):  # tolerate async resolver
            return asyncio.get_event_loop().run_until_complete(result)
        return result
    except Exception:  # pragma: no cover — resolver built in parallel
        return None


# ---------------------------------------------------------------------------
# Step 1: parse the uploaded file into raw text
# ---------------------------------------------------------------------------


@router.post("/projects/{id}/import/parse")
async def import_parse(id: str, request: Request, db: Session = Depends(get_db)):
    if not resolve_project(db, id, get_user_id(request)):
        return not_found()

    form = await request.form()
    file = form.get("file")
    if not isinstance(file, UploadFile):
        return json_error(400, "No file")

    add_import_log(db, id, 1, "running", f"Started parsing file: {file.filename}")

    try:
        data = await file.read()
        text = extract_text_from_file(data, file.filename or "")
        if not text.strip():
            add_import_log(db, id, 1, "error", "File content is empty")
            return json_error(400, "Empty file")

        add_import_log(
            db, id, 1, "done",
            f"Parsing complete, {len(text)} characters total",
            {"charCount": len(text)},
        )
        return {"text": text, "charCount": len(text)}
    except Exception as err:  # noqa: BLE001 — mirror TS catch-all
        msg = str(err) or "Parse failed"
        add_import_log(db, id, 1, "error", msg)
        return json_error(400, msg)


# ---------------------------------------------------------------------------
# Step 2: extract characters + relationships with the LLM
# ---------------------------------------------------------------------------


@router.post("/projects/{id}/import/characters")
async def import_characters(id: str, request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    if not resolve_project(db, id, user_id):
        return not_found()

    body = await request.json()
    if not (body.get("modelConfig") or {}).get("text"):
        return json_error(400, "No text model")

    text = body.get("text") or ""
    chunks = chunk_text(text)
    system = _resolve_system_prompt("import_character_extract", user_id, id)

    add_import_log(
        db, id, 2, "running",
        f"Started character extraction, {len(chunks)} chunks total",
    )
    db.commit()  # make progress logs visible to concurrent readers

    try:
        from app.ai.prompts.import_character_extract import (  # lazy
            build_import_character_extract_prompt,
        )
    except Exception:
        # PORT NOTE: prompt builders live in app.ai.prompts (built in parallel).
        return json_error(501, "Import character extraction prompts not available")

    async def process_chunk(idx: int, chunk: str) -> dict:
        with db_session() as log_db:
            add_import_log(
                log_db, id, 2, "running",
                f"Processing chunk {idx + 1}/{len(chunks)}...",
            )
        prompt = build_import_character_extract_prompt(chunk)
        result_text = await _ai_generate_text(prompt, system)
        try:
            parsed = json.loads(extract_json(result_text))
        except (ValueError, TypeError):
            print(
                f"[ImportChars] Chunk {idx + 1} JSON parse failed. Raw:\n{result_text[:500]}..."
            )
            with db_session() as log_db:
                add_import_log(
                    log_db, id, 2, "running",
                    f"Chunk {idx + 1} JSON parse failed, retrying...",
                )
            retry_text = await _ai_generate_text(
                prompt + "\n\nIMPORTANT: Return COMPLETE, VALID JSON.", system
            )
            parsed = json.loads(extract_json(retry_text))
        # Support both {characters, relationships} and legacy array format
        if isinstance(parsed, list):
            return {"chars": parsed, "rels": []}
        return {
            "chars": parsed.get("characters") or [],
            "rels": parsed.get("relationships") or [],
        }

    try:
        chunk_results = await asyncio.gather(
            *(process_chunk(i, c) for i, c in enumerate(chunks))
        )
    except Exception as err:  # noqa: BLE001
        msg = str(err) or "Unknown error"
        add_import_log(db, id, 2, "error", f"Character extraction failed: {msg}")
        return json_error(500, msg)

    # Merge & deduplicate characters by name, sum frequencies
    char_map: dict[str, dict] = {}
    all_relations: list[dict] = []
    for result in chunk_results:
        for c in result["chars"]:
            key = (c.get("name") or "").lower().strip()
            existing = char_map.get(key)
            if existing:
                existing["frequency"] += c.get("frequency") or 0
                if len(c.get("description") or "") > len(existing.get("description") or ""):
                    existing["description"] = c.get("description")
            else:
                char_map[key] = dict(c)
        all_relations.extend(result["rels"])

    merged = sorted(char_map.values(), key=lambda c: c.get("frequency") or 0, reverse=True)

    # Classify: frequency >= 2 = main, else guest
    characters_out = [
        {**c, "scope": "main" if (c.get("frequency") or 0) >= 2 else "guest"}
        for c in merged
    ]

    # Deduplicate relationships
    rel_set: set[str] = set()
    unique_relations = []
    for r in all_relations:
        key = "<->".join(sorted([r.get("characterA") or "", r.get("characterB") or ""]))
        if key in rel_set:
            continue
        rel_set.add(key)
        unique_relations.append(r)

    main_count = sum(1 for c in characters_out if c["scope"] == "main")
    guest_count = sum(1 for c in characters_out if c["scope"] == "guest")
    add_import_log(
        db, id, 2, "done",
        f"Extraction complete: {len(characters_out)} characters "
        f"({main_count} main, {guest_count} guest), {len(unique_relations)} relationships",
        {"characters": characters_out, "relationships": unique_relations},
    )

    return {"characters": characters_out, "relationships": unique_relations}


# ---------------------------------------------------------------------------
# Step 3: split the script into episodes with the LLM
# ---------------------------------------------------------------------------


@router.post("/projects/{id}/import/split")
async def import_split(id: str, request: Request, db: Session = Depends(get_db)):
    user_id = get_user_id(request)
    if not resolve_project(db, id, user_id):
        return not_found()

    body = await request.json()
    if not (body.get("modelConfig") or {}).get("text"):
        return json_error(400, "No text model")

    chunks = chunk_text(body.get("text") or "")
    system = _resolve_system_prompt("script_split", user_id, id)

    add_import_log(
        db, id, 3, "running",
        f"Started automatic episode splitting, {len(chunks)} chunks total",
    )
    db.commit()

    try:
        from app.ai.prompts.script_split import build_script_split_prompt  # lazy
    except Exception:
        # PORT NOTE: prompt builders live in app.ai.prompts (built in parallel).
        return json_error(501, "Script split prompts not available")

    # Build character context for the prompt
    all_names = [c.get("name") for c in (body.get("allCharacters") or []) if c.get("name")]
    char_context = (
        "\n\nAll extracted characters (assign each to ONLY the episodes where "
        f"they actually appear): {', '.join(all_names)}"
        if all_names
        else ""
    )

    async def process_chunk(idx: int, chunk: str) -> list:
        with db_session() as log_db:
            add_import_log(
                log_db, id, 3, "running",
                f"Processing chunk {idx + 1}/{len(chunks)}...",
            )
        prompt = build_script_split_prompt(
            chunk + char_context,
            chunk_index=idx,
            total_chunks=len(chunks),
            episode_offset=0,
        )
        result_text = await _ai_generate_text(prompt, system)
        try:
            return json.loads(extract_json(result_text))
        except (ValueError, TypeError):
            print(
                f"[ImportSplit] Chunk {idx + 1} JSON parse failed. "
                f"Raw output:\n{result_text[:500]}..."
            )
            with db_session() as log_db:
                add_import_log(
                    log_db, id, 3, "running",
                    f"Chunk {idx + 1} JSON parse failed, retrying...",
                )
            retry_text = await _ai_generate_text(
                prompt
                + "\n\nIMPORTANT: Return COMPLETE, VALID JSON. Fewer episodes "
                "is better than broken JSON.",
                system,
            )
            return json.loads(extract_json(retry_text))

    try:
        chunk_results = await asyncio.gather(
            *(process_chunk(i, c) for i, c in enumerate(chunks))
        )
        all_episodes = [ep for chunk in chunk_results for ep in chunk]
    except Exception as err:  # noqa: BLE001
        msg = str(err) or "Unknown error"
        add_import_log(db, id, 3, "error", f"Episode split failed: {msg}")
        return json_error(500, msg)

    add_import_log(
        db, id, 3, "done",
        f"Episode split complete, {len(all_episodes)} episodes total",
        {"episodes": all_episodes},
    )
    return {"episodes": all_episodes}


# ---------------------------------------------------------------------------
# Step 4: persist episodes + characters + relations
# ---------------------------------------------------------------------------


@router.post("/projects/{id}/import/generate")
async def import_generate(id: str, request: Request, db: Session = Depends(get_db)):
    if not resolve_project(db, id, get_user_id(request)):
        return not_found()

    body = await request.json()
    episodes_data = body.get("episodes") or []
    characters_data = body.get("characters") or []
    relationships = body.get("relationships") or []

    add_import_log(
        db, id, 4, "running",
        f"Started creating {len(episodes_data)} episodes and "
        f"{len(characters_data)} characters",
    )

    # 1. Create all characters (main + guest), build name -> id map
    char_id_by_name: dict[str, str] = {}
    for char in characters_data:
        char_id = new_id()
        db.add(
            Character(
                id=char_id,
                project_id=id,
                name=char.get("name"),
                description=char.get("description") or "",
                visual_hint=char.get("visualHint") or "",
                scope=char.get("scope"),
                episode_id=None,  # all characters are project-level now
            )
        )
        char_id_by_name[(char.get("name") or "").lower().strip()] = char_id

    # 1b. Create character relationships
    if relationships:
        for rel in relationships:
            a_id = char_id_by_name.get((rel.get("characterA") or "").lower().strip())
            b_id = char_id_by_name.get((rel.get("characterB") or "").lower().strip())
            if a_id and b_id and a_id != b_id:
                try:
                    db.add(
                        CharacterRelation(
                            id=new_id(),
                            project_id=id,
                            character_a_id=a_id,
                            character_b_id=b_id,
                            relation_type=rel.get("relationType") or "neutral",
                            description=rel.get("description") or "",
                        )
                    )
                    db.flush()
                except Exception:  # noqa: BLE001 — skip duplicates
                    db.rollback()

    rel_suffix = f" and {len(relationships)} relationships" if relationships else ""
    add_import_log(
        db, id, 4, "running",
        f"Created {len(characters_data)} characters{rel_suffix}",
    )

    # 2. Create episodes
    max_seq = db.execute(
        select(func.max(Episode.sequence)).where(Episode.project_id == id)
    ).scalar()
    seq = (max_seq or 0) + 1

    created: list[Episode] = []
    for ep in episodes_data:
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

    # 3. Create episode_characters relations
    relation_count = 0
    for i, ep_data in enumerate(episodes_data):
        episode_id = created[i].id if i < len(created) else None
        if not episode_id or not ep_data.get("characters"):
            continue
        for char_name in ep_data["characters"]:
            char_id = char_id_by_name.get((char_name or "").lower().strip())
            if not char_id:
                continue
            db.add(
                EpisodeCharacter(
                    id=new_id(), episode_id=episode_id, character_id=char_id
                )
            )
            relation_count += 1

    add_import_log(
        db, id, 4, "done",
        f"Import complete! Created {len(characters_data)} characters and "
        f"{len(created)} episodes ({relation_count} character assignments)",
        {"episodeCount": len(created), "characterCount": len(characters_data)},
    )
    db.flush()

    return JSONResponse(
        {
            "episodes": serialize_many(created),
            "characterCount": len(characters_data),
        },
        status_code=201,
    )


# ---------------------------------------------------------------------------
# Import logs
# ---------------------------------------------------------------------------


@router.get("/projects/{id}/import/logs")
def list_import_logs(id: str, request: Request, db: Session = Depends(get_db)):
    if not resolve_project(db, id, get_user_id(request)):
        return not_found()
    logs = (
        db.execute(
            select(ImportLog)
            .where(ImportLog.project_id == id)
            .order_by(ImportLog.created_at.asc())
        )
        .scalars()
        .all()
    )
    return serialize_many(logs)


@router.delete("/projects/{id}/import/logs")
def delete_import_logs(id: str, request: Request, db: Session = Depends(get_db)):
    if not resolve_project(db, id, get_user_id(request)):
        return not_found()
    rows = (
        db.execute(select(ImportLog).where(ImportLog.project_id == id)).scalars().all()
    )
    for row in rows:
        db.delete(row)
    return Response(status_code=204)
