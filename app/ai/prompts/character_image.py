"""Character four-view (turnaround) reference-sheet prompt.

Faithful port of src/lib/ai/prompts/character-image.ts
(`buildCharacterTurnaroundPrompt`). The pipeline's character_image stage
calls `build_character_turnaround_prompt(description, name)`. Illustrative
Chinese style examples in the source were translated to English equivalents.
"""
from __future__ import annotations

from typing import Optional


def build_character_turnaround_prompt(description: str, character_name: Optional[str] = None) -> str:
    name_line = f"Name: {character_name}\n" if character_name else ""
    name_label = (
        f'Display the character\'s name "{character_name}" as a clean typographic label '
        "below the four-view layout. Use a modern sans-serif font, dark text on white "
        "background, centered alignment."
        if character_name
        else "No character name label required."
    )

    return f"""Character four-view reference sheet — professional character design document.

=== CRITICAL: ART STYLE FIDELITY ===
The CHARACTER DESCRIPTION below is authoritative. It may specify an art style explicitly, implicitly, or through a combination of modifiers (e.g. "3D Chinese-animation CG render", "ink-wash freehand", "cyberpunk pixel art", "cel-shaded anime", "oil painting portrait", "PBR realtime render").

Rules for interpreting style:
1. Treat the FULL style phrase as one atomic instruction. Do NOT cherry-pick individual words and map them to a default bucket. "3D realistic Chinese-animation render" is NOT the same as "photorealistic" — it is a stylized 3D CG render in the Chinese animation idiom.
2. Style modifiers like "realistic / high-definition / refined" describe RENDERING FIDELITY, not medium. They raise detail level within the chosen medium; they never convert the medium to live-action photography.
3. The medium (2D illustration / 3D CG / photograph / painting / pixel / etc.) is determined ONLY by explicit medium words such as "photograph / live-action / real-person footage". In the ABSENCE of such explicit photographic words, DO NOT output a photograph or live-action render, even if "realistic" appears.
4. When multiple style words are present, the most specific / most restrictive one wins. "Chinese-animation" + "3D" + "realistic" → stylized 3D CG in Chinese animation style with high rendering fidelity.
5. Color palette, lighting mood, and era references in the description (e.g. "low-saturation muted tones", "cinematic historical-drama texture") are MANDATORY and must be honored exactly — they are not decorative.
6. If no style is mentioned at all, infer the most appropriate stylized illustration from the character's setting and genre. Default to stylized illustration, NOT photography.

=== CHARACTER DESCRIPTION (authoritative) ===
{name_line}{description}

=== FACE — HIGH DETAIL ===
Render the face with precision appropriate to the chosen medium and style:
- Consistent facial bone structure, eye shape, nose, mouth — matching the description exactly
- Eyes expressive and detailed, rendered in the chosen medium's idiom
- Hair with defined volume, color and flow, rendered in the chosen medium's idiom
- Skin and surface shading rendered in the chosen medium's idiom (cel-shading, subsurface, PBR, painterly, etc.)
- The face must be striking, memorable, and instantly recognizable across all four views

=== WEAPONS, COSTUME & EQUIPMENT ===
- All props, armor, clothing and equipment must be rendered in the SAME medium and style as the character
- Material detail must match the style (painterly strokes for paintings, PBR materials for 3D CG, clean flats for anime, etc.)
- Scale and anatomy must be correct relative to the body

=== FOUR-VIEW LAYOUT ===
Four views arranged LEFT to RIGHT on a clean pure white canvas, consistent medium shot (waist to crown) across all four:
1. FRONT — facing viewer directly, showing full outfit and any held items
2. THREE-QUARTER — rotated ~45° right, showing face depth and dimensional form
3. SIDE PROFILE — perfect 90° facing right, clear silhouette
4. BACK — fully facing away, hairstyle and clothing back detail

=== LIGHTING & RENDERING ===
- Clean professional key/fill/rim lighting, consistent direction across all four views
- Pure white background for clean character separation
- Honor any mood/tone/palette constraints from the description (if it says "low-saturation muted", the output MUST be low-saturation and muted, NOT bright)
- Highest quality achievable WITHIN the chosen medium and style — never break medium to chase fidelity

=== CONSISTENCY ACROSS ALL FOUR VIEWS ===
- Identical character identity, proportions and colors in every view
- Identical outfit, accessories, weapon placement, hair
- Heads aligned at the same top edge, waist at the same bottom edge
- Consistent expression across all views

=== CHARACTER NAME LABEL ===
{name_label}

=== FINAL OUTPUT STANDARD ===
Professional character design reference sheet. This is the single canonical reference — all future generated frames MUST reproduce this exact character in this exact medium and style. Zero medium drift, zero style drift, zero AI artifacts."""
