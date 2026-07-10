"""First/last frame image-prompt builders.

Python port of src/lib/ai/prompts/frame-generate.ts. Delegates to the registry
definitions `frame_generate_first` / `frame_generate_last`; keeps a hardcoded
fallback (translated from the original Chinese) for when the definition is
missing.

PORT NOTE: params passed to build_full_prompt use camelCase keys
(sceneDescription, startFrameDesc, ...) matching the registry build functions,
which mirror the original TypeScript param names.
"""
from typing import Optional

from app.ai.prompts.registry import get_prompt_definition


def build_first_frame_prompt(
    scene_description: str,
    start_frame_desc: str,
    character_descriptions: str,
    previous_last_frame: Optional[str] = None,
    slot_contents: Optional[dict] = None,
) -> str:
    definition = get_prompt_definition("frame_generate_first")
    if definition:
        return definition.build_full_prompt(
            slot_contents or {},
            {
                "sceneDescription": scene_description,
                "startFrameDesc": start_frame_desc,
                "characterDescriptions": character_descriptions,
                "previousLastFrame": previous_last_frame,
            },
        )

    # Fallback: hardcoded prompt (should not be reached if registry is intact)
    lines: list[str] = []

    lines.append("Generate the opening frame of this shot as a high-quality image.")
    lines.append("")
    lines.append("=== CRITICAL: ART STYLE (highest priority) ===")
    lines.append("Read the character descriptions and scene description below; they specify or imply an art style.")
    lines.append("You MUST match that art style exactly. Do NOT default to a realistic style.")
    lines.append("- If the descriptions mention anime/manga/动漫/漫画/cartoon/卡通 → generate an anime/manga-style illustration")
    lines.append("- If the descriptions mention realistic/live-action/写实/真人/photorealistic → generate a photorealistic image")
    lines.append("- If reference images are attached, their visual style is the standard — you must match it exactly")
    lines.append("- The output art style must stay consistent with the character reference images")
    lines.append("")
    lines.append("=== SCENE ENVIRONMENT ===")
    lines.append(scene_description)
    lines.append("")
    lines.append("=== FRAME DESCRIPTION ===")
    lines.append(start_frame_desc)
    lines.append("")
    lines.append("=== CHARACTER DESCRIPTIONS ===")
    lines.append(character_descriptions)
    lines.append("")
    lines.append("=== REFERENCE IMAGES (character model sheets) ===")
    lines.append("Each attached reference image is a character model sheet showing 4 views (front, three-quarter side, side, back).")
    lines.append("The character name is printed at the bottom of each model sheet — use it to identify the corresponding character.")
    lines.append("Mandatory consistency rules:")
    lines.append("- Match the character names on the model sheets with the character names in the scene description")
    lines.append("- The costume must exactly match the reference — same garment type, color, material, accessories. No substitutions (e.g., do NOT replace a cyan everyday robe with a dragon robe)")
    lines.append("- Face, hairstyle, hair color, body type, and skin tone must match exactly")
    lines.append("- All accessories shown in the reference images (hats, sabers, hairpins, jewelry) must appear")
    lines.append("- The art style must exactly match the reference images")
    lines.append("")

    if previous_last_frame:
        lines.append("=== CONTINUITY REQUIREMENTS ===")
        lines.append("This shot follows directly from the previous shot. The attached reference contains the last frame of the previous shot. Maintain visual continuity:")
        lines.append("- The same characters must wear consistent costumes and keep consistent proportions")
        lines.append("- Same art style — do NOT switch between anime and realistic")
        lines.append("- Ambient lighting and color temperature should transition smoothly")
        lines.append("- Character positions should naturally continue from where the previous shot ended")
        lines.append("")

    lines.append("=== RENDERING ===")
    lines.append("Quality: rich detail appropriate to the art style")
    lines.append("Lighting: cinematic lighting with plausible light sources. Use rim light to separate the character.")
    lines.append("Background: a fully rendered, detail-rich environment. Do NOT use a blank or abstract background.")
    lines.append("Character: appearance and art style exactly matching the reference images. Vivid expression, natural dynamic pose.")
    lines.append("Composition: cinematic framing with a clear focal point and depth of field.")

    return "\n".join(lines)


def build_last_frame_prompt(
    scene_description: str,
    end_frame_desc: str,
    character_descriptions: str,
    first_frame_path: Optional[str] = None,
    slot_contents: Optional[dict] = None,
) -> str:
    definition = get_prompt_definition("frame_generate_last")
    if definition:
        return definition.build_full_prompt(
            slot_contents or {},
            {
                "sceneDescription": scene_description,
                "endFrameDesc": end_frame_desc,
                "characterDescriptions": character_descriptions,
            },
        )

    # Fallback: hardcoded prompt (should not be reached if registry is intact)
    lines: list[str] = []

    lines.append("Generate the closing frame of this shot as a high-quality image.")
    lines.append("")
    lines.append("=== CRITICAL: ART STYLE (highest priority) ===")
    lines.append("You MUST match the art style of the first-frame image (already attached) exactly.")
    lines.append("If the first frame is anime/manga style → this frame must also be anime/manga style.")
    lines.append("If the first frame is photorealistic → this frame must also be photorealistic.")
    lines.append("Do NOT change or mix the art style. This is non-negotiable.")
    lines.append("")
    lines.append("=== SCENE ENVIRONMENT ===")
    lines.append(scene_description)
    lines.append("")
    lines.append("=== FRAME DESCRIPTION ===")
    lines.append(end_frame_desc)
    lines.append("")
    lines.append("=== CHARACTER DESCRIPTIONS ===")
    lines.append(character_descriptions)
    lines.append("")
    lines.append("=== REFERENCE IMAGES ===")
    lines.append("The first attached image is this shot's opening frame — use it as your visual anchor.")
    lines.append("The remaining attached images are character model sheets (4 views each, name printed at the bottom).")
    lines.append("Match the name on each character model sheet with the characters in the scene.")
    lines.append("")
    lines.append("=== RELATIONSHIP TO THE FIRST FRAME ===")
    lines.append("This closing frame shows the terminal state after the shot's action completes. Compared with the first frame:")
    lines.append("- Same environment, lighting setup, and color scheme")
    lines.append("- Same art style — absolutely do NOT change the style")
    lines.append("- Costumes fully consistent — the characters wear exactly the same clothing as in the reference model sheets and the first frame. Do NOT change costumes.")
    lines.append("- Same face, hairstyle, accessories — only pose/expression/position change")
    lines.append("- The character's position, pose, and expression have changed per the frame description above")
    lines.append("")
    lines.append("=== AS THE STARTING POINT FOR THE NEXT SHOT ===")
    lines.append("This frame will be reused as the opening frame of the next shot. Ensure:")
    lines.append("- The pose is stable — not an in-motion intermediate state or blurred")
    lines.append("- The composition is complete and stands on its own as an independent frame")
    lines.append("- The framing allows a natural transition to a different camera angle")
    lines.append("")
    lines.append("=== RENDERING ===")
    lines.append("Quality: rich detail matching the first frame's style")
    lines.append("Lighting: the same lighting setup as the first frame. Change only when the action requires it.")
    lines.append("Background: must be consistent with the first frame's environment.")
    lines.append("Character: exactly matching the reference images. Show the emotional state at the end of the shot's action.")
    lines.append("Composition: a natural resolution of the shot, ready to cut to the next shot.")

    return "\n".join(lines)
