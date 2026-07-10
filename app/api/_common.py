"""Shared helpers for the API routers.

Ports of:
    - src/lib/get-user-id.ts            -> get_user_id
    - src/lib/assert-project-ownership.ts -> resolve_project / assert_project_ownership
    - src/lib/staleness.ts              -> mark_downstream_stale
    - src/lib/shot-asset-utils.ts       -> asset serializers + legacy views + activate
    - src/lib/import-utils.ts           -> add_import_log / chunk_text / extract_text_from_file
    - src/lib/utils/upload-url.ts       -> upload_url

Response shape: the TS API returns Drizzle rows whose field names are
camelCase while the SQLite columns are snake_case. ``serialize`` converts a
SQLAlchemy model instance into the camelCase dict the frontend expects.
JSON TEXT columns (characters/meta/slots/payload/result/metadata) are parsed
back into objects to match the Drizzle ``mode: "json"`` behaviour.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable

from fastapi import Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.ids import new_id
from app.db.models import (
    Character,
    ImportLog,
    Project,
    Shot,
    ShotAsset,
)

# ---------------------------------------------------------------------------
# User identity + ownership (get-user-id.ts / assert-project-ownership.ts)
# ---------------------------------------------------------------------------


def get_user_id(request: Request) -> str:
    """Port of getUserIdFromRequest: the client sends its id in x-user-id."""
    return request.headers.get("x-user-id") or ""


def resolve_project(db: Session, project_id: str, user_id: str) -> Project | None:
    """Return the project row if owned by user_id, else None.

    The web UI is single-tenant (no auth/fingerprint), so requests carry an
    empty user id and projects are created with user_id="". An empty id is a
    valid owner key here and matches the empty-owner rows; it is NOT treated as
    "deny all" (which would 404 every mutation the browser makes).
    """
    return db.execute(
        select(Project).where(Project.id == project_id, Project.user_id == user_id)
    ).scalar_one_or_none()


def assert_project_ownership(db: Session, request: Request, project_id: str) -> Project | None:
    """Port of assertProjectOwnership(request, projectId)."""
    return resolve_project(db, project_id, get_user_id(request))


def json_error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def not_found() -> JSONResponse:
    return json_error(404, "Not found")


# ---------------------------------------------------------------------------
# camelCase serialization
# ---------------------------------------------------------------------------

# Columns stored as TEXT holding JSON — parsed on the way out, matching the
# Drizzle `mode: "json"` columns in schema.ts.
_JSON_FIELDS = {"characters", "meta", "slots", "payload", "result", "metadata"}


def _camel(key: str) -> str:
    key = key.rstrip("_")  # e.g. ImportLog.metadata_ maps to column "metadata"
    parts = key.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def serialize(obj: Any, json_fields: Iterable[str] | None = None) -> dict[str, Any]:
    """Serialize a SQLAlchemy model instance to a camelCase dict."""
    fields = set(json_fields) if json_fields is not None else _JSON_FIELDS
    out: dict[str, Any] = {}
    for attr in obj.__mapper__.column_attrs:
        key = _camel(attr.key)
        value = getattr(obj, attr.key)
        if attr.key.rstrip("_") in fields:
            value = _maybe_json(value) if value is not None else None
        out[key] = value
    return out


def serialize_many(rows: Iterable[Any], json_fields: Iterable[str] | None = None) -> list[dict[str, Any]]:
    return [serialize(r, json_fields) for r in rows]


def serialize_shot_asset(a: ShotAsset) -> dict[str, Any]:
    """Asset shape returned by project/episode GET (route.ts inline mapping)."""
    return {
        "id": a.id,
        "shotId": a.shot_id,
        "type": a.type,
        "sequenceInType": a.sequence_in_type,
        "assetVersion": a.asset_version,
        "isActive": a.is_active,
        "prompt": a.prompt,
        "fileUrl": a.file_url,
        "status": a.status,
        "characters": json.loads(a.characters) if a.characters else None,
        "modelProvider": a.model_provider,
        "modelId": a.model_id,
        "meta": json.loads(a.meta) if a.meta else None,
    }


# ---------------------------------------------------------------------------
# Staleness (src/lib/staleness.ts)
# ---------------------------------------------------------------------------


def mark_downstream_stale(db: Session, entity_type: str, entity_id: str) -> None:
    """Mark downstream assets as stale when the script changes."""
    if entity_type == "episode":
        shot_col, char_col = Shot.episode_id, Character.episode_id
    else:
        shot_col, char_col = Shot.project_id, Character.project_id
    for row in db.execute(select(Shot).where(shot_col == entity_id)).scalars():
        row.is_stale = 1
    for row in db.execute(select(Character).where(char_col == entity_id)).scalars():
        row.is_stale = 1


# ---------------------------------------------------------------------------
# Shot asset legacy views (src/lib/shot-asset-utils.ts)
# ---------------------------------------------------------------------------


class ShotLegacyView:
    """Legacy-shaped view of a shot's currently-active assets."""

    __slots__ = (
        "first_frame",
        "last_frame",
        "start_frame_desc",
        "end_frame_desc",
        "video_url",
        "reference_video_url",
        "scene_ref_frame",
        "reference_images",
    )

    def __init__(self, assets: list[ShotAsset]):
        def find(type_: str) -> ShotAsset | None:
            for a in assets:
                if a.type == type_ and a.sequence_in_type == 0:
                    return a
            return None

        first = find("first_frame")
        last = find("last_frame")
        keyframe_video = find("keyframe_video")
        reference_video = find("reference_video")
        refs = sorted(
            (a for a in assets if a.type == "reference"),
            key=lambda a: a.sequence_in_type,
        )
        # The "scene ref frame" historically was a single primary reference
        # anchor — map it to the first reference asset (sequence_in_type=0).
        scene_ref = refs[0] if refs else None

        self.first_frame = first.file_url if first else None
        self.last_frame = last.file_url if last else None
        self.start_frame_desc = first.prompt if first else None
        self.end_frame_desc = last.prompt if last else None
        self.video_url = keyframe_video.file_url if keyframe_video else None
        self.reference_video_url = reference_video.file_url if reference_video else None
        self.scene_ref_frame = scene_ref.file_url if scene_ref else None
        self.reference_images = refs


def load_shot_legacy_views_batch(db: Session, shot_ids: list[str]) -> dict[str, ShotLegacyView]:
    """Port of loadShotLegacyViewsBatch: one query for many shots."""
    if not shot_ids:
        return {}
    rows = (
        db.execute(
            select(ShotAsset).where(
                ShotAsset.shot_id.in_(shot_ids), ShotAsset.is_active == 1
            )
        )
        .scalars()
        .all()
    )
    by_shot: dict[str, list[ShotAsset]] = {}
    for row in rows:
        by_shot.setdefault(row.shot_id, []).append(row)
    return {sid: ShotLegacyView(by_shot.get(sid, [])) for sid in shot_ids}


def activate_asset_version(
    db: Session, shot_id: str, type_: str, sequence_in_type: int, asset_version: int
) -> None:
    """Port of activateAssetVersion: flip is_active within a slot."""
    slot_rows = (
        db.execute(
            select(ShotAsset).where(
                ShotAsset.shot_id == shot_id,
                ShotAsset.type == type_,
                ShotAsset.sequence_in_type == sequence_in_type,
            )
        )
        .scalars()
        .all()
    )
    for row in slot_rows:
        row.is_active = 1 if row.asset_version == asset_version else 0


def insert_asset_version(
    db: Session,
    shot_id: str,
    type_: str,
    sequence_in_type: int,
    **fields: Any,
) -> ShotAsset:
    """Insert a new active version for a slot, deactivating older rows.

    Port of insertAssetVersion (shot-asset-utils.ts): the new row gets
    max(asset_version)+1 and is_active=1; siblings are flipped to 0.
    """
    siblings = (
        db.execute(
            select(ShotAsset).where(
                ShotAsset.shot_id == shot_id,
                ShotAsset.type == type_,
                ShotAsset.sequence_in_type == sequence_in_type,
            )
        )
        .scalars()
        .all()
    )
    next_version = max((s.asset_version for s in siblings), default=0) + 1
    for s in siblings:
        s.is_active = 0
    asset = ShotAsset(
        id=new_id(),
        shot_id=shot_id,
        type=type_,
        sequence_in_type=sequence_in_type,
        asset_version=next_version,
        is_active=1,
        **fields,
    )
    db.add(asset)
    return asset


# ---------------------------------------------------------------------------
# Import utils (src/lib/import-utils.ts)
# ---------------------------------------------------------------------------

CHUNK_SIZE = 10000  # ~10000 chars per chunk


def add_import_log(
    db: Session,
    project_id: str,
    step: int,
    status: str,
    message: str,
    metadata: Any = None,
) -> None:
    db.add(
        ImportLog(
            id=new_id(),
            project_id=project_id,
            step=step,
            status=status,
            message=message,
            metadata_=json.dumps(metadata if metadata is not None else {}),
        )
    )
    db.flush()


def chunk_text(text: str) -> list[str]:
    """Split text at paragraph boundaries, each chunk <= CHUNK_SIZE chars."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > CHUNK_SIZE and current:
            chunks.append(current.strip())
            current = ""
        current += ("\n\n" if current else "") + para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def extract_text_from_file(data: bytes, filename: str) -> str:
    """Port of extractTextFromFile: txt / docx / pdf (lazy imports)."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "txt":
        return data.decode("utf-8", errors="replace")
    if ext == "docx":
        import io

        from docx import Document  # python-docx (lazy)

        doc = Document(io.BytesIO(data))
        return "\n\n".join(p.text for p in doc.paragraphs)
    if ext == "pdf":
        import io

        from pypdf import PdfReader  # lazy

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"Unsupported file type: .{ext}")


# ---------------------------------------------------------------------------
# Upload URL (src/lib/utils/upload-url.ts)
# ---------------------------------------------------------------------------


def upload_url(file_path: str) -> str:
    """Convert a local file path (e.g. "./uploads/frames/a.png") to an API URL."""
    normalized = file_path.replace("\\", "/")
    if normalized.startswith("/api/uploads/"):
        return normalized
    stripped = re.sub(r"^.*?uploads/", "", normalized)
    return f"/api/uploads/{stripped}"


# ---------------------------------------------------------------------------
# Dialogue enrichment shared by project/episode/shot list endpoints
# ---------------------------------------------------------------------------


def load_shot_dialogues(db: Session, shot_id: str) -> list[dict[str, Any]]:
    from app.db.models import Dialogue

    rows = db.execute(
        select(Dialogue, Character.name)
        .join(Character, Dialogue.character_id == Character.id)
        .where(Dialogue.shot_id == shot_id)
        .order_by(Dialogue.sequence.asc())
    ).all()
    return [
        {
            "id": d.id,
            "text": d.text,
            "characterId": d.character_id,
            "characterName": name,
            "sequence": d.sequence,
        }
        for d, name in rows
    ]


def load_assets_by_shot(db: Session, shot_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Bulk-load ALL shot assets (all versions) grouped by shot id, ordered by
    (type, sequence_in_type, asset_version desc) — used by project/episode GET."""
    if not shot_ids:
        return {}
    rows = (
        db.execute(
            select(ShotAsset)
            .where(ShotAsset.shot_id.in_(shot_ids))
            .order_by(
                ShotAsset.type,
                ShotAsset.sequence_in_type,
                ShotAsset.asset_version.desc(),
            )
        )
        .scalars()
        .all()
    )
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(row.shot_id, []).append(serialize_shot_asset(row))
    return out
