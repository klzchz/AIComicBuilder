"""Video assembly stage — Python port of src/lib/pipeline/video-assemble.ts.

Concatenates every shot's active video clip (keyframe_video / reference_video)
in sequence order, applies transitions, burns dialogue subtitles, mixes BGM, and
adds title/credits cards via the ffmpeg subsystem. Marks the project completed.
"""
from __future__ import annotations

import time
from typing import Any

from sqlalchemy import asc, select

from app.api._common import load_shot_legacy_views_batch
from app.db.models import Character, Dialogue, Episode, Project, Shot
from app.db.session import db_session
from app.pipeline._helpers import payload_of


async def run_video_assemble(project_id: str) -> dict[str, Any]:
    """Assemble the final video for a project. Port of handleVideoAssemble."""
    with db_session() as s:
        project_shots = list(
            s.execute(
                select(Shot).where(Shot.project_id == project_id).order_by(asc(Shot.sequence))
            ).scalars()
        )

        legacy = load_shot_legacy_views_batch(s, [sh.id for sh in project_shots])

        # Keep shots that have an active video (keyframe or reference).
        completed_shots: list[dict] = []
        video_paths: list[str] = []
        for sh in project_shots:
            view = legacy.get(sh.id)
            video_url = None
            if view:
                video_url = view.video_url or view.reference_video_url
            if not video_url:
                continue
            completed_shots.append(
                {
                    "id": sh.id,
                    "sequence": sh.sequence,
                    "duration": sh.duration,
                    "transition_in": sh.transition_in,
                    "transition_out": sh.transition_out,
                    "episode_id": sh.episode_id,
                }
            )
            video_paths.append(video_url)

        if not video_paths:
            raise ValueError("No video clips to assemble")

        # Build transitions from shot transition_out / transition_in.
        transitions: list[str] = []
        for i in range(len(completed_shots) - 1):
            shot = completed_shots[i]
            next_shot = completed_shots[i + 1]
            if shot["transition_out"] and shot["transition_out"] != "cut":
                transitions.append(shot["transition_out"])
            else:
                transitions.append(next_shot["transition_in"] or "cut")

        # Gather dialogues for subtitles.
        subtitles: list[dict[str, Any]] = []
        for shot in completed_shots:
            rows = s.execute(
                select(
                    Dialogue.text,
                    Character.name,
                    Dialogue.sequence,
                    Shot.sequence,
                    Dialogue.start_ratio,
                    Dialogue.end_ratio,
                )
                .join(Character, Dialogue.character_id == Character.id)
                .join(Shot, Dialogue.shot_id == Shot.id)
                .where(Dialogue.shot_id == shot["id"])
                .order_by(asc(Dialogue.sequence))
            ).all()

            count = len(rows)
            for idx, (text, char_name, _dseq, shot_sequence, start_ratio, end_ratio) in enumerate(rows):
                sr = float(start_ratio) if start_ratio not in (None, "") else None
                er = float(end_ratio) if end_ratio not in (None, "") else None
                subtitles.append(
                    {
                        "text": f"{char_name}: {text}",
                        "shot_sequence": shot_sequence,
                        "dialogue_sequence": idx,
                        "dialogue_count": count,
                        "start_ratio": sr,
                        "end_ratio": er,
                    }
                )

        # Load project for title card and BGM.
        project = s.get(Project, project_id)
        project_title = project.title if project else None
        bgm_path = project.bgm_url if (project and project.bgm_url) else None

        # Episode-level BGM overrides project-level.
        episode_id = completed_shots[0]["episode_id"] if completed_shots else None
        if episode_id:
            episode = s.get(Episode, episode_id)
            if episode and episode.bgm_url:
                bgm_path = episode.bgm_url

        shot_durations = [sh["duration"] if sh["duration"] is not None else 10 for sh in completed_shots]

    # Title and credits cards.
    title_card = {"text": project_title, "duration": 3} if project_title else None
    credits_card = {"text": "Made with AIComicBuilder", "duration": 2}

    # ffmpeg subsystem (already written) — import lazily.
    from app.video.ffmpeg import assemble_video

    result = assemble_video(
        video_paths=video_paths,
        subtitles=subtitles,
        project_id=project_id,
        shot_durations=shot_durations,
        transitions=transitions,
        bgm_path=bgm_path,
        title_card=title_card,
        credits_card=credits_card,
    )

    now = int(time.time())
    with db_session() as s:
        project = s.get(Project, project_id)
        if project:
            project.status = "completed"
            project.updated_at = now

    return {"outputPath": result["video_path"], "srtPath": result["srt_path"]}


async def handle_video_assemble(task: Any) -> dict[str, Any]:
    p = payload_of(task)
    return await run_video_assemble(p["projectId"])
