"""Video assembly via ffmpeg — Python port of src/lib/video/ffmpeg.ts.

The TS original used fluent-ffmpeg; this port shells out to the system
``ffmpeg``/``ffprobe`` binaries with subprocess. Function names and semantics
mirror the source (snake_case). Import-safe: no ffmpeg/DB access at import
time — a missing ffmpeg binary raises a clear RuntimeError only when a
function is actually called.

Extra utilities not present in the TS file but part of the video subsystem
contract: ``probe_duration`` (ffprobe) and ``extract_frames`` (frame export).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

from app.config import settings
from app.core.ids import new_id

# Transition types supported between shots (mirrors TransitionType in TS):
# "cut" | "dissolve" | "fade_in" | "fade_out" | "wipeleft" | "slideright" | "circleopen"
DEFAULT_XFADE_DURATION = 0.5


# ── Binary resolution ─────────────────────────────────────────────────────

def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(
            f"System '{name}' binary not found on PATH. Install ffmpeg "
            f"(e.g. 'apt install ffmpeg' or 'brew install ffmpeg') to use the video subsystem."
        )
    return path


def _run(cmd: list[str], error_prefix: str) -> None:
    """Run an ffmpeg/ffprobe command, raising a descriptive error on failure."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        # Last stderr lines carry the actual ffmpeg error message.
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        raise RuntimeError(f"{error_prefix}: {tail}")


# ── Title / credits cards ─────────────────────────────────────────────────

def generate_title_card(
    text: str,
    duration: float,
    output_dir: str,
    options: dict[str, Any] | None = None,
) -> str:
    """Render a full-screen text card (1920x1080) to an mp4. Port of generateTitleCard."""
    opts = options or {}
    font_size = opts.get("font_size", opts.get("fontSize", 48))
    bg_color = opts.get("bg_color", opts.get("bgColor", "black"))
    text_color = opts.get("text_color", opts.get("textColor", "white"))

    card_path = str(Path(output_dir).resolve() / f"title-{new_id()}.mp4")
    escaped_text = text.replace("'", "'\\''")

    ffmpeg_bin = _require_binary("ffmpeg")
    cmd = [
        ffmpeg_bin, "-y",
        "-f", "lavfi",
        "-i", f"color=c={bg_color}:s=1920x1080:d={duration}",
        "-vf",
        f"drawtext=text='{escaped_text}':fontsize={font_size}:fontcolor={text_color}:"
        "x=(w-text_w)/2:y=(h-text_h)/2",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-t", str(duration),
        "-pix_fmt", "yuv420p",
        card_path,
    ]
    _run(cmd, "Title card generation failed")
    return card_path


# ── SRT subtitle generation ───────────────────────────────────────────────

def format_srt_time(seconds: float) -> str:
    """Format seconds as an SRT timestamp HH:MM:SS,mmm. Port of formatSrtTime."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _sub_field(sub: dict[str, Any], snake: str, camel: str, default: Any = None) -> Any:
    """Read a subtitle field accepting both snake_case and camelCase keys."""
    if snake in sub:
        return sub[snake]
    return sub.get(camel, default)


def generate_srt_file(
    subtitles: Sequence[dict[str, Any]],
    shot_durations: Sequence[float],
    output_path: str,
) -> str:
    """Write an .srt next to output_path from per-shot dialogue timing. Port of generateSrtFile.

    Each subtitle entry: text, shot_sequence (1-based), dialogue_sequence
    (0-based within shot), dialogue_count, and optional start_ratio/end_ratio
    (0-1, relative to the shot duration). Without explicit ratios, the shot
    duration is divided equally among its dialogues.
    """
    srt_path = str(Path(output_path).with_suffix(".srt")) if output_path.endswith(".mp4") else output_path

    shot_start_times: list[float] = []
    cumulative = 0.0
    for duration in shot_durations:
        shot_start_times.append(cumulative)
        cumulative += duration

    srt_entries: list[str] = []
    index = 1

    for sub in subtitles:
        shot_idx = int(_sub_field(sub, "shot_sequence", "shotSequence", 0)) - 1
        if shot_idx < 0 or shot_idx >= len(shot_durations):
            continue

        shot_start = shot_start_times[shot_idx]
        shot_dur = shot_durations[shot_idx]

        start_ratio = _sub_field(sub, "start_ratio", "startRatio")
        end_ratio = _sub_field(sub, "end_ratio", "endRatio")

        if start_ratio is not None and end_ratio is not None:
            # Use explicit timing ratios from the DB
            start_time = shot_start + shot_dur * float(start_ratio)
            end_time = shot_start + shot_dur * float(end_ratio)
        else:
            # Auto-distribute: divide shot duration equally among dialogues
            dialogue_count = int(_sub_field(sub, "dialogue_count", "dialogueCount", 1)) or 1
            dialogue_sequence = int(_sub_field(sub, "dialogue_sequence", "dialogueSequence", 0))
            segment_dur = shot_dur / dialogue_count
            start_time = shot_start + segment_dur * dialogue_sequence
            end_time = start_time + segment_dur

        srt_entries.append(
            f"{index}\n{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n{sub['text']}\n"
        )
        index += 1

    Path(srt_path).write_text("\n".join(srt_entries), encoding="utf-8")
    return srt_path


def escape_subtitle_path(p: str) -> str:
    """Escape a path for the ffmpeg subtitles filter (colon, backslash, single quote)."""
    return p.replace("\\", "/").replace(":", "\\:").replace("'", "'\\''")


def map_transition_name(t: str) -> str:
    """Map our transition type to the ffmpeg xfade transition name."""
    if t in ("fade_in", "fade_out"):
        return "fade"
    return t


# ── Concatenation with transitions ────────────────────────────────────────

def concat_with_transitions(
    video_paths: Sequence[str],
    transitions: Sequence[str],
    shot_durations: Sequence[float],
    output_path: str,
    project_id: str,
    output_dir: str,
) -> None:
    """Concatenate videos with optional xfade transitions into output_path.

    Port of concatWithTransitions: single clip → copy; all "cut" → fast concat
    demuxer with stream copy; otherwise an xfade filter chain re-encode.
    """
    # Single video: just copy
    if len(video_paths) == 1:
        shutil.copyfile(str(Path(video_paths[0]).resolve()), output_path)
        return

    ffmpeg_bin = _require_binary("ffmpeg")

    # All cuts: use the fast concat demuxer
    if all(t == "cut" for t in transitions):
        concat_list_path = Path(output_dir).resolve() / f"{project_id}-concat.txt"
        concat_list_path.write_text(
            "\n".join(f"file '{Path(p).resolve()}'" for p in video_paths),
            encoding="utf-8",
        )
        try:
            _run(
                [
                    ffmpeg_bin, "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(concat_list_path),
                    "-c", "copy",
                    output_path,
                ],
                "FFmpeg concat failed",
            )
        finally:
            # TS removed the list file on success; also clean up on failure.
            concat_list_path.unlink(missing_ok=True)
        return

    # Mixed transitions: use an xfade filter chain
    cmd = [ffmpeg_bin, "-y"]
    for vp in video_paths:
        cmd += ["-i", str(Path(vp).resolve())]

    filter_parts: list[str] = []
    prev_label = "0:v"
    cumulative_offset = 0.0

    for i, t in enumerate(transitions):
        duration = shot_durations[i]
        out_label = f"v{i}" if i < len(transitions) - 1 else "vout"

        if t == "cut":
            # For cut: use xfade with duration=0 to simulate a hard cut
            offset = cumulative_offset + duration
            filter_parts.append(
                f"[{prev_label}][{i + 1}:v]xfade=transition=fade:duration=0:offset={offset:.3f}[{out_label}]"
            )
            cumulative_offset = offset
        else:
            xfade_dur = DEFAULT_XFADE_DURATION
            offset = cumulative_offset + duration - xfade_dur
            xfade_name = map_transition_name(t)
            filter_parts.append(
                f"[{prev_label}][{i + 1}:v]xfade=transition={xfade_name}:duration={xfade_dur}:offset={offset:.3f}[{out_label}]"
            )
            cumulative_offset = offset

        prev_label = out_label

    complex_filter = ";".join(filter_parts)

    cmd += [
        "-filter_complex", complex_filter,
        "-map", "[vout]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",
        output_path,
    ]
    _run(cmd, "FFmpeg xfade concat failed")


# ── Full assembly pipeline ────────────────────────────────────────────────

def assemble_video(
    video_paths: Sequence[str],
    subtitles: Sequence[dict[str, Any]],
    project_id: str,
    shot_durations: Sequence[float],
    *,
    transitions: Sequence[str] | None = None,
    title_card: dict[str, Any] | None = None,
    credits_card: dict[str, Any] | None = None,
    bgm_path: str | None = None,
    bgm_volume: float | None = None,
) -> dict[str, str | None]:
    """Assemble the final video. Port of assembleVideo.

    Steps: optional title/credits cards → concat with transitions → burn
    subtitles (with graceful fallback if the burn fails) → mix BGM (graceful
    fallback). Returns {"video_path": ..., "srt_path": ...} with paths
    relative to the CWD (uploadUrl compatibility, matching the TS source);
    files live under settings.UPLOAD_DIR/videos.
    """
    all_paths = list(video_paths)
    all_durations = [float(d) for d in shot_durations]

    output_dir = Path(settings.UPLOAD_DIR).resolve() / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepend title card if specified
    if title_card:
        title_path = generate_title_card(title_card["text"], title_card["duration"], str(output_dir))
        all_paths.insert(0, title_path)
        all_durations.insert(0, float(title_card["duration"]))

    # Append credits card if specified
    if credits_card:
        credits_path = generate_title_card(credits_card["text"], credits_card["duration"], str(output_dir))
        all_paths.append(credits_path)
        all_durations.append(float(credits_card["duration"]))

    if transitions is None:
        transitions = ["cut"] * max(len(all_paths) - 1, 0)

    concat_output_path = str(output_dir / f"{project_id}-concat-{new_id()}.mp4")
    output_path = str(output_dir / f"{project_id}-final-{new_id()}.mp4")

    # Step 1: Concatenate video clips (with transitions)
    concat_with_transitions(all_paths, transitions, all_durations, concat_output_path, project_id, str(output_dir))

    # Step 2: Burn in subtitles if any
    srt_path: str | None = None
    if subtitles:
        srt_path = generate_srt_file(subtitles, all_durations, output_path)
        escaped_srt_path = escape_subtitle_path(str(Path(srt_path).resolve()))
        try:
            ffmpeg_bin = _require_binary("ffmpeg")
            _run(
                [
                    ffmpeg_bin, "-y",
                    "-i", concat_output_path,
                    "-vf", f"subtitles='{escaped_srt_path}'",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-c:a", "aac",
                    output_path,
                ],
                "FFmpeg subtitle burn failed",
            )
            os.unlink(concat_output_path)
            # Keep the SRT file for external subtitle export
        except Exception as err:
            # Fallback: skip subtitle burn, use the concat output directly
            print(f"[FFmpeg] Subtitle burn failed, using concat output: {err}")
            os.replace(concat_output_path, output_path)
    else:
        # No subtitles, just rename
        os.replace(concat_output_path, output_path)

    # Step 3: Mix background music if provided
    if bgm_path and Path(bgm_path).resolve().exists():
        bgm_output_path = output_path[:-4] + "-bgm.mp4" if output_path.endswith(".mp4") else output_path + "-bgm.mp4"
        vol = 0.3 if bgm_volume is None else bgm_volume
        try:
            ffmpeg_bin = _require_binary("ffmpeg")
            _run(
                [
                    ffmpeg_bin, "-y",
                    "-i", output_path,
                    "-i", str(Path(bgm_path).resolve()),
                    "-map", "0:v",
                    "-map", "1:a",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-af", f"volume={vol}",
                    "-shortest",
                    bgm_output_path,
                ],
                "FFmpeg BGM mix failed",
            )
            os.unlink(output_path)
            os.replace(bgm_output_path, output_path)
        except Exception as err:
            print(f"[FFmpeg] BGM mix failed, skipping: {err}")

    # Return relative paths for uploadUrl compatibility (mirrors the TS source)
    cwd = os.getcwd()
    return {
        "video_path": os.path.relpath(output_path, cwd),
        "srt_path": os.path.relpath(srt_path, cwd) if srt_path else None,
    }


# ── Probing & frame extraction (video subsystem utilities) ────────────────

def probe_duration(video_path: str) -> float:
    """Return the media duration in seconds using ffprobe."""
    ffprobe_bin = _require_binary("ffprobe")
    proc = subprocess.run(
        [
            ffprobe_bin,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(Path(video_path).resolve()),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        tail = "\n".join(proc.stderr.strip().splitlines()[-4:])
        raise RuntimeError(f"ffprobe duration failed for {video_path}: {tail}")
    return float(proc.stdout.strip())


def extract_frames(
    video_path: str,
    timestamps: Sequence[float] | None = None,
    *,
    fps: float | None = None,
    output_dir: str | None = None,
) -> list[str]:
    """Extract frames as PNGs and return their paths (under UPLOAD_DIR/frames by default).

    - timestamps: exact seek positions (one PNG each), or
    - fps: sample at a fixed rate (defaults to 1 fps when neither is given).
    """
    ffmpeg_bin = _require_binary("ffmpeg")
    src = str(Path(video_path).resolve())
    out_dir = Path(output_dir).resolve() if output_dir else Path(settings.UPLOAD_DIR).resolve() / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(video_path).stem}-{new_id()}"

    if timestamps is not None:
        results: list[str] = []
        for i, ts in enumerate(timestamps):
            out_path = str(out_dir / f"{stem}-{i:04d}.png")
            _run(
                [ffmpeg_bin, "-y", "-ss", str(ts), "-i", src, "-frames:v", "1", out_path],
                f"Frame extraction failed at {ts}s",
            )
            results.append(out_path)
        return results

    rate = fps if fps is not None else 1.0
    pattern = str(out_dir / f"{stem}-%04d.png")
    _run(
        [ffmpeg_bin, "-y", "-i", src, "-vf", f"fps={rate}", pattern],
        "Frame extraction failed",
    )
    return sorted(str(p) for p in out_dir.glob(f"{stem}-*.png"))
