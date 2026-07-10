"""User-message builder for the `ref_video_prompt` AI call.

Python port of src/lib/ai/prompts/ref-video-prompt-generate.ts. The original
Chinese prose has been translated to English, but the literal Seedance / 即梦
inline reference tokens `@图片N` are KEPT INTACT — they are the exact syntax the
downstream Seedance model expects, not translatable prose.

PORT NOTE: `图片` inside `@图片N` is deliberately NOT translated. The system
prompt is NOT defined here — it lives in the registry under `ref_video_prompt`.

Output style follows the official 即梦 / Seedance inline syntax:
  - References are written as `@图片N`
  - Flowing natural-language prose, no structured mapping header, no
    "beat 1/2/3" labels, no [Dialogue Lip Sync] tags
  - Dialogue inline as "character line: ..." appended after the action prose
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class SceneFrameInfo:
    label: str      # e.g. "宫殿外", "竹林"
    index: int      # 1-based position in the ordered reference list


@dataclass
class CharacterRefInfo:
    name: str
    index: int      # 1-based position in the ordered reference list
    visual_hint: Optional[str] = None


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


def build_ref_video_prompt_request(
    motion_script: str,
    camera_direction: str,
    duration: int,
    characters: list,
    scene_frames: list,
    dialogues: Optional[list] = None,
) -> str:
    lines: list[str] = []

    lines.append(
        "You will receive the following reference images (the order strictly corresponds to @图片1, @图片2, @图片3 ...; "
        "you MUST use the `@图片N` form, you must **NOT** write it as `@图片N` in any other way):"
    )
    for c in characters:
        name = _get(c, "name")
        index = _get(c, "index")
        vh = _get(c, "visual_hint", "visualHint")
        hint = f"（{vh}）" if vh else ""
        lines.append(f"  @图片{index} = Character: {name}{hint}")
    for s in scene_frames:
        lines.append(f"  @图片{_get(s, 'index')} = Scene: {_get(s, 'label')}")
    lines.append("")

    if len(scene_frames) > 1:
        lines.append(
            f"This shot has {len(scene_frames)} scene reference images, corresponding in order to the spatial transitions within the shot. "
            "The prose must pass through these scenes in order and clearly describe the transitions."
        )
        lines.append("")

    if len(characters) == 0:
        lines.append(
            "Note: no character appears in this shot; only describe the scene/environment changes and camera movement, "
            "do not fabricate any people."
        )
        lines.append("")

    lines.append(f"Script action: {motion_script}")
    lines.append(f"Camera instruction: {camera_direction}")
    lines.append(f"Duration: {duration}s")

    if dialogues:
        joined = "; ".join(
            f'{_get(d, "character_name", "characterName")}: "{_get(d, "text")}"'
            for d in dialogues
        )
        lines.append(
            "Dialogue (keep the original language, embed directly at the end of the prose, "
            'in the format "character name line: ..."): '
            + joined
        )

    lines.append("")
    lines.append("Strict requirements:")
    lines.append("1. Use the `@图片N` form to reference all characters and scenes (e.g., @图片1, @图片2); never write it any other way")
    lines.append('2. The writing style is coherent natural prose; embed @图片N directly into the description; no structured "beat 1/2/3" labels')
    lines.append('3. Do not write a standalone mapping declaration line at the start of the prompt like "Image mapping: @图片1 is X, @图片2 is Y" — the information must be woven into the prose')
    lines.append("4. After every @图片N there must be a parenthetical annotation of the character/scene name, in the format @图片N（name）")
    lines.append("5. Dialogue (if any) is written directly at the end of the prose: character name line: the original line (no tags such as 【Dialogue Lip Sync】)")
    lines.append("6. Output only the prompt body, with no preamble and no markdown")

    return "\n".join(lines)
