"""User-message builder for the `ref_image_prompts` AI call.

Python port of src/lib/ai/prompts/ref-image-prompts.ts. The original Chinese
user builder has been translated to English. The system prompt is NOT defined
here — it lives in the registry under `ref_image_prompts` (single source of
truth, also exposed in the prompt-management UI so users can override it).
This file only constructs the per-request user payload: visual style,
character context (for reasoning, not drawing), and shot list.

PORT NOTE: TS read camelCase keys (motionScript, cameraDirection). Here we
prefer snake_case (motion_script, camera_direction) via _get, falling back
to camelCase.
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


def build_ref_image_prompts_request(
    shots: list,
    characters: list,
    visual_style: Optional[str] = None,
) -> str:
    # Characters are passed as CONTEXT for the AI to reason about which
    # characters will act in which shot -> populates the `characters` field
    # in the JSON output. The scene prompts themselves must NOT depict any
    # characters.
    char_context = "\n".join(
        f"- {_get(c, 'name')}" + (f"：{_get(c, 'description')}" if _get(c, 'description') else "")
        for c in characters
    )

    shot_blocks = []
    for s in shots:
        seq = _get(s, "sequence")
        prompt = _get(s, "prompt", default="")
        duration = _get(s, "duration", default=None)
        if duration is None:
            duration = 10
        motion = _get(s, "motion_script", "motionScript")
        camera = _get(s, "camera_direction", "cameraDirection")
        lines = [f"Shot {seq} (duration {duration}s): {prompt}"]
        if motion:
            lines.append(
                f"  Plot action (used to judge the physical location the characters are in, do not draw people): {motion}"
            )
        if camera:
            lines.append(f"  Camera movement: {camera}")
        shot_blocks.append("\n".join(lines))
    shot_descriptions = "\n\n".join(shot_blocks)

    parts = [
        f"Project visual style keynote: {visual_style}" if visual_style else "",
        "",
        "Character list (for reasoning only: (1) the physical location they are in determines the scene (2) judging which characters appear in each shot. Do NOT mention them in the image prompts):",
        char_context or "(none)",
        "",
        '## What is a "scene image"',
        "A scene image = **the physical location / environmental space the characters are in** (e.g., the plaza before the Hall of Supreme Harmony, deep in a bamboo forest, the edge of a cliff, in front of a ruined palace gate, the interior of a meditation room).",
        "A scene image is **NOT**: abstract effects (energy light, a glowing brand), a standalone prop close-up (only a sword, only a talisman), a character portrait, a character's accessory.",
        'Judgment criterion: if by looking at this image alone you can say "this is a XX place", then it is a scene image; if you can only say "this is a blob of light / an object", then it is not.',
        "",
        "## Number of scene images (default 1, maximum 4)",
        "**By default generate only 1 scene image per shot** — the location the characters are in is the scene for this shot.",
        "Generate multiple (up to 4) only in the following cases:",
        "- **The characters cross different physical locations within the shot**: e.g., a fight moving from the ground to the air (bamboo-forest ground → high above the bamboo tops), a chase rushing from indoors to outdoors (study → corridor → courtyard), jumping from a bridge into the water (bridge deck → underwater)",
        "- **Large shifts in scene light/time**: dusk → deep night, dim indoors → walking out into bright light",
        "General beats within a single location — dialogue, standing, a close-up, gathering strength, a punch bursting out, opening a door, turning around — only need 1 scene image; the subsequent video generation will complete all beats within this same location.",
        "",
        "## Shot list",
        shot_descriptions,
        "",
        "To emphasize again:",
        "- Default 1 scene image per shot; only when characters cross physical locations should there be >1, **maximum 4**",
        '- A scene image must be a "location/environment", not an "effect/prop/light effect/symbol"',
        "- No people appear in the image (no person, no back view, no silhouette, no hands or feet)",
        "- The characters field must list the names of the characters who will appear in this shot, and the names must exactly match the character list above",
        "- No real person names (directors/actors/artists/brands/IP) — violating this causes the image API to return a 400 error",
        "- Output format strictly follows the scenes array required by the system prompt ({ name, prompt }), with no markdown wrapping",
    ]

    # PORT NOTE: TS uses .filter(Boolean) which drops every empty-string entry,
    # so the interspersed "" spacers collapse — output has no blank lines.
    return "\n".join(p for p in parts if p)
