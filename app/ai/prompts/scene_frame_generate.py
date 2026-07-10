"""Scene-frame image-prompt builder (reference mode).

Python port of src/lib/ai/prompts/scene-frame-generate.ts. Delegates entirely
to the registry definition `scene_frame_generate`.

PORT NOTE: params passed to build_full_prompt use camelCase keys matching the
registry build functions (which mirror the original TypeScript param names).
"""
from typing import Optional

from app.ai.prompts.registry import get_prompt_definition


def build_scene_frame_prompt(
    scene_description: str,
    char_ref_mapping: str,
    character_descriptions: str,
    camera_direction: Optional[str] = None,
    start_frame_desc: Optional[str] = None,
    motion_script: Optional[str] = None,
    slot_contents: Optional[dict] = None,
) -> str:
    definition = get_prompt_definition("scene_frame_generate")
    if not definition:
        raise RuntimeError("scene_frame_generate prompt definition not found in registry")

    return definition.build_full_prompt(
        slot_contents or {},
        {
            "sceneDescription": scene_description,
            "charRefMapping": char_ref_mapping,
            "characterDescriptions": character_descriptions,
            "cameraDirection": camera_direction or "",
            "startFrameDesc": start_frame_desc or "",
            "motionScript": motion_script or "",
        },
    )
