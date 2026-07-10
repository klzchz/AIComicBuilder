"""User-side prompt builder for keyframe (first/last frame) image-prompt generation.

Python port of src/lib/ai/prompts/keyframe-prompts.ts. The original Chinese
user builder has been translated to English. The system prompt lives in the
registry under the `shot_split_keyframe_assets` key.

PORT NOTE: TS read camelCase keys (motionScript, cameraDirection, visualHint).
Here we prefer snake_case keys (motion_script, camera_direction, visual_hint)
via the _get helper, falling back to camelCase.
"""
from typing import Optional


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


def build_keyframe_prompts_request(
    shots: list,
    characters: list,
    visual_style: Optional[str] = None,
) -> str:
    char_descriptions = "\n".join(
        f"{_get(c, 'name')}（{_get(c, 'visual_hint', 'visualHint') or 'no visual identifier'}）: {_get(c, 'description', default='') or ''}"
        for c in characters
    )

    shot_blocks = []
    for s in shots:
        seq = _get(s, "sequence")
        prompt = _get(s, "prompt", default="")
        motion = _get(s, "motion_script", "motionScript")
        camera = _get(s, "camera_direction", "cameraDirection")
        block = f"Shot {seq}: {prompt}"
        if motion:
            block += f"\nAction: {motion}"
        if camera:
            block += f"\nCamera movement: {camera}"
        shot_blocks.append(block)
    shot_descriptions = "\n\n".join(shot_blocks)

    style_prefix = f"Visual style: {visual_style}\n\n" if visual_style else ""
    return f"{style_prefix}Characters:\n{char_descriptions}\n\nStoryboard:\n{shot_descriptions}"
