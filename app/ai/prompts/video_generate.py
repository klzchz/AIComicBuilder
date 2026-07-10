"""Video-prompt builders (keyframe interpolation + reference mode).

Python port of src/lib/ai/prompts/video-generate.ts.

PORT NOTE: This module performs RUNTIME language selection (zh vs en) — it is
NOT prompt text to translate. The "zh" label tokens (角色形象, 对白口型, 画外音,
起始帧, 结束帧, 帧锚点, etc.) are kept EXACTLY as the Chinese originals, and the
"en" tokens as-is. The line-hint parse keys passed to the extractor helpers
("画内对白", "画外旁白", "首帧", "尾帧") are literal keys used to locate a line
inside a user-overridden slot template — they must stay Chinese literals. The
Chinese DEFAULT fallback strings (e.g. "从起始帧到结束帧进行平滑插值。") ARE the zh
runtime default and are kept in Chinese; the en branch uses English, exactly as
the TS source.
"""
import re
from typing import Optional

from app.ai.prompts.registry import get_prompt_definition

_CJK_RE = re.compile(r"[一-鿿]")


def _get(obj, key, alt=None, default=None):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        if alt is not None and alt in obj:
            return obj[alt]
        return default
    val = getattr(obj, key, None)
    if val is None and alt is not None:
        val = getattr(obj, alt, None)
    return val if val is not None else default


def _detect_language(text: str) -> str:
    chinese_chars = _CJK_RE.findall(text or "")
    return "zh" if chinese_chars and len(chinese_chars) > len(text) * 0.1 else "en"


def _get_labels(lang: str) -> dict:
    if lang == "zh":
        return {
            "characterAppearance": "角色形象",
            "dialogueLipSync": "对白口型",
            "offscreenVoice": "画外音",
            "camera": "镜头运动",
            "duration": "时长",
            "interpolation": "关键帧插值",
            "openingFrame": "起始帧",
            "closingFrame": "结束帧",
            "videoScript": "视频脚本",
            "frameAnchors": "帧锚点",
            "separator": "，",
            "period": "。",
            "colon": "：",
            "paren_open": "（",
            "paren_close": "）",
        }
    return {
        "characterAppearance": "Character Appearance",
        "dialogueLipSync": "Dialogue Lip Sync",
        "offscreenVoice": "Off-screen Voice",
        "camera": "Camera Movement",
        "duration": "Duration",
        "interpolation": "Keyframe Interpolation",
        "openingFrame": "Opening Frame",
        "closingFrame": "Closing Frame",
        "videoScript": "Video Script",
        "frameAnchors": "Frame Anchors",
        "separator": ", ",
        "period": ".",
        "colon": ": ",
        "paren_open": "(",
        "paren_close": ")",
    }


def _build_character_line(characters: Optional[list], lang: str = "zh") -> Optional[str]:
    with_hints = [c for c in (characters or []) if _get(c, "visual_hint", "visualHint")]
    if not with_hints:
        return None
    label = _get_labels(lang)
    return label["separator"].join(
        f"{_get(c, 'name')}{label['paren_open']}{_get(c, 'visual_hint', 'visualHint')}{label['paren_close']}"
        for c in with_hints
    )


def _resolve_slot(
    slot_contents: Optional[dict],
    prompt_key: str,
    slot_key: str,
    hardcoded_fallback: str,
) -> str:
    """Resolve a single slot value: slotContents override, then registry
    default, then hardcoded fallback."""
    if slot_contents and slot_key in slot_contents:
        return slot_contents[slot_key]
    definition = get_prompt_definition(prompt_key)
    if definition:
        for slot in definition.slots:
            if slot.key == slot_key:
                return slot.default_content
    return hardcoded_fallback


# -- Helpers for extracting labels from slot content ----------------------

_BRACKET_RE = re.compile(r"(【[^】]+】)")
_ANCHOR_HEADER_RE = re.compile(r"^\[([^\]]+)\]", re.MULTILINE)
_BEFORE_PLACEHOLDER_RE = re.compile(r"^([^{]+)")


def _extract_label(slot_text: str, line_hint: str, fallback: str) -> str:
    """Extract a dialogue label (e.g. 【对白口型】 or 【画外音】) from the slot
    format text."""
    if not slot_text:
        return fallback
    for line in slot_text.split("\n"):
        if line_hint in line:
            match = _BRACKET_RE.search(line)
            if match:
                return match.group(1)
    return fallback


def _extract_anchor_header(slot_text: str, fallback: str) -> str:
    """Extract the anchor section header (e.g. [FRAME ANCHORS] or [帧锚点]) from
    slot text."""
    if not slot_text:
        return fallback
    match = _ANCHOR_HEADER_RE.search(slot_text)
    if match:
        return f"[{match.group(1)}]"
    return fallback


def _extract_frame_label(slot_text: str, line_hint: str, fallback: str) -> str:
    """Extract a frame label (e.g. "Opening frame:" or "首帧：") from slot text."""
    if not slot_text:
        return fallback
    for line in slot_text.split("\n"):
        if line_hint in line:
            match = _BEFORE_PLACEHOLDER_RE.search(line)
            if match:
                return match.group(1).strip()
    return fallback


def build_reference_video_prompt(
    video_script: str,
    camera_direction: str,
    duration: Optional[int] = None,
    characters: Optional[list] = None,
    dialogues: Optional[list] = None,
    slot_contents: Optional[dict] = None,
) -> str:
    """Prompt for reference-image-based video generation (Toonflow/Kling
    reference mode). Seedance-style format: shot description (prose) → camera →
    【对白口型】. No frame-interpolation header, no [FRAME ANCHORS] — the reference
    image provides visual context."""
    lang = _detect_language(video_script)
    label = _get_labels(lang)
    lines: list[str] = []

    if duration:
        lines.append(f"{label['duration']}{label['colon']}{duration}s{label['period']}")
        lines.append("")

    char_line = _build_character_line(characters, lang)
    if char_line:
        lines.append(f"{label['characterAppearance']}{label['colon']}{char_line}{label['period']}")
        lines.append("")

    lines.append(video_script)

    lines.append("")
    lines.append(f"{label['camera']}{label['colon']}{camera_direction}{label['period']}")

    if dialogues:
        dialogue_format_text = _resolve_slot(
            slot_contents, "ref_video_generate", "dialogue_format", ""
        )

        default_on_screen = "【对白口型】" if lang == "zh" else "[Dialogue Lip Sync]"
        default_off_screen = "【画外音】" if lang == "zh" else "[Off-screen Voice]"
        on_screen_label = _extract_label(dialogue_format_text, "画内对白", default_on_screen)
        off_screen_label = _extract_label(dialogue_format_text, "画外旁白", default_off_screen)

        lines.append("")
        for d in dialogues:
            name = _get(d, "character_name", "characterName")
            text = _get(d, "text")
            if _get(d, "offscreen"):
                lines.append(f'{off_screen_label}{name}: "{text}"')
            else:
                vh = _get(d, "visual_hint", "visualHint")
                lbl = f"{name}{label['paren_open']}{vh}{label['paren_close']}" if vh else name
                lines.append(f'{on_screen_label}{lbl}: "{text}"')

    return "\n".join(lines)


def build_video_prompt(
    video_script: str,
    camera_direction: str,
    start_frame_desc: Optional[str] = None,
    end_frame_desc: Optional[str] = None,
    scene_description: Optional[str] = None,  # kept for call-site compatibility, not used in output
    duration: Optional[int] = None,
    characters: Optional[list] = None,
    dialogues: Optional[list] = None,
    slot_contents: Optional[dict] = None,
) -> str:
    lang = _detect_language(video_script)
    label = _get_labels(lang)
    lines: list[str] = []

    if duration:
        lines.append(f"{label['duration']}{label['colon']}{duration}s{label['period']}")
        lines.append("")

    char_line = _build_character_line(characters, lang)
    if char_line:
        lines.append(f"{label['characterAppearance']}{label['colon']}{char_line}{label['period']}")
        lines.append("")

    # Interpolation header from slot or registry default
    default_interpolation = (
        "从起始帧到结束帧进行平滑插值。"
        if lang == "zh"
        else "Smoothly interpolate from the opening frame to the closing frame."
    )
    interpolation_header = _resolve_slot(
        slot_contents, "video_generate", "interpolation_header", default_interpolation
    )
    lines.append(interpolation_header)
    lines.append("")

    lines.append(video_script)

    lines.append("")
    lines.append(f"{label['camera']}{label['colon']}{camera_direction}{label['period']}")

    has_start = bool(start_frame_desc)
    has_end = bool(end_frame_desc)
    if has_start or has_end:
        frame_anchors_text = _resolve_slot(
            slot_contents, "video_generate", "frame_anchors", ""
        )

        default_anchor_header = "[帧锚点]" if lang == "zh" else "[FRAME ANCHORS]"
        default_opening_label = "起始帧：" if lang == "zh" else "Opening frame:"
        default_closing_label = "结束帧：" if lang == "zh" else "Closing frame:"
        anchor_header = _extract_anchor_header(frame_anchors_text, default_anchor_header)
        opening_label = _extract_frame_label(frame_anchors_text, "首帧", default_opening_label)
        closing_label = _extract_frame_label(frame_anchors_text, "尾帧", default_closing_label)

        lines.append("")
        lines.append(anchor_header)
        if has_start:
            lines.append(f"{opening_label} {start_frame_desc}")
        if has_end:
            lines.append(f"{closing_label} {end_frame_desc}")

    if dialogues:
        dialogue_format_text = _resolve_slot(
            slot_contents, "video_generate", "dialogue_format", ""
        )

        default_on_screen = "【对白口型】" if lang == "zh" else "[Dialogue Lip Sync]"
        default_off_screen = "【画外音】" if lang == "zh" else "[Off-screen Voice]"
        on_screen_label = _extract_label(dialogue_format_text, "画内对白", default_on_screen)
        off_screen_label = _extract_label(dialogue_format_text, "画外旁白", default_off_screen)

        lines.append("")
        for d in dialogues:
            name = _get(d, "character_name", "characterName")
            text = _get(d, "text")
            if _get(d, "offscreen"):
                lines.append(f'{off_screen_label}{name}: "{text}"')
            else:
                vh = _get(d, "visual_hint", "visualHint")
                lbl = f"{name}{label['paren_open']}{vh}{label['paren_close']}" if vh else name
                lines.append(f'{on_screen_label}{lbl}: "{text}"')

    return "\n".join(lines)
