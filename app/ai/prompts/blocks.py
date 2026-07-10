"""Reusable prompt building blocks.

Ported from src/lib/ai/prompts/blocks.ts. All Chinese prompt text has been
translated to natural, faithful English while preserving structure, headers,
bullets, and examples.
"""

from __future__ import annotations

from typing import Optional


def art_style_block() -> str:
    return """## Art-Style Consistency
- Maintain the visual style defined in the project's "Visual Style" section across every generated image
- Style elements include: rendering technique, color scheme, lighting mood, texture quality
- Do not mix styles within a single project (e.g. do not place a photorealistic character on a cartoon background)
- If a specific art style is declared (anime, photorealistic, watercolor, etc.), every frame must match it"""


def reference_image_block() -> str:
    return """## Reference-Image Usage Rules
- The reference image defines the character's canonical appearance
- Must match: face shape, hairstyle/hair color, eye color, skin tone, clothing details, accessories
- May adjust: pose, expression, angle — these change from shot to shot
- Never contradict the core identity traits shown in the reference image"""


def language_rule_block(default_lang: Optional[str] = None) -> str:
    tail = f"\nDefault language when the language is ambiguous: {default_lang}" if default_lang else ""
    return (
        "## Critical Language Rule\n"
        "The output must match the language of the input. If the user writes in Chinese, reply entirely in Chinese. "
        "If they write in English, reply entirely in English. Do not mix languages within the output."
        + tail
    )


def theme_style_mapping_block() -> str:
    """Shared theme -> art-style mapping used by character_image, ref_image_prompts,
    frame_generate_first, and scene_frame_generate. Single source of truth to
    prevent style drift across the pipeline (character sheet / reference frame /
    first frame).
    """
    return """**Theme -> Art-Style Auto-Mapping Table** (shared across the whole pipeline to keep the character sheet / reference frame / first frame stylistically consistent):
- Xianxia / cultivation / xuanhuan -> 3D Chinese-animation render style, Chinese xianxia concept design, delicate materials and volumetric light
- Guofeng / historical -> Chinese gongbi fine-line painting / ink wash / classical painting, emphasizing linework and negative space
- Cyberpunk / futuristic / sci-fi -> futuristic sci-fi photorealistic CG, concept design, hard surfaces and glowing materials
- Realistic / urban / character-driven -> cinematic photorealistic style, film grain texture, natural skin
- Fantasy / Western magic -> Western-fantasy concept art, oil-painting texture
- Japanese anime -> anime cel shading / Makoto Shinkai soft light / Ghibli natural style (refine per the description)
- Chinese animation (guoman) -> guoman 3D render / new-wave Chinese animation style
- Chibi / cartoon -> super-deformed chibi (three-heads-tall), Disney/Pixar cartoon style
- Food / advertising -> commercial advertising photography, macro, softbox studio lighting

Art-style judgment principles:
1. Prioritize any art style explicitly specified in the script or description
2. If none is specified, match the theme keywords against the table above
3. Never default to photorealistic — you must actively determine the theme category"""


def physics_realism_block() -> str:
    """Shared physics/realism constraints used by any image prompt that depicts
    human figures in realistic settings. Extracted from ref_image_prompts so it
    can be shared with frame_generate_first/last and scene_frame_generate.
    """
    return """[!! Strict Physical-Common-Sense Constraints (Highest Priority)]
Image-generation models interpret every word literally. Obey the following iron rules:

1. **Never use any metaphor** (both action metaphors and appearance metaphors are forbidden): no "like...", "as if...", "resembling...", "似...", "as though..." or any other simile construction — the image model will literally draw the AI as the actual metaphorical object.
   - [BAD] Action metaphor: "Xiao Chen emerged from the cave mouth like an agile leopard" -> [GOOD] "Xiao Chen braced both hands on the ground, knelt on one knee, and crawled out of the cave mouth, body leaning forward"
   - [BAD] Appearance metaphor: "hair messy like weeds" -> [GOOD] "short black hair, uneven, tangled and sticking up in several places, split ends"
   - [BAD] Appearance metaphor: "jawline sharp as if carved by a knife" -> [GOOD] "jawline straight and sharp, angular and well-defined"
   - [BAD] Appearance metaphor: "eyes sharp like an eagle's" -> [GOOD] "eyes narrowed, outer corners slightly raised, gaze focused"
   - [BAD] Appearance metaphor: "figure like bamboo" -> [GOOD] "slender, straight figure, shoulders about 40 cm wide"
   - If you must convey an abstract texture ("soft as silk", "hard as iron"), rewrite it as a concrete material + sensory description ("smooth, glossy black hair", "hard-textured metal surface")

2. **In realistic scenes, no anti-physics behavior**:
   - Characters must stand / sit / walk / run / lie prone / kneel — feet must touch the ground
   - No "mid-air", "flying up", "floating", "hovering" — unless the subject is sci-fi/fantasy
   - Jumps must specify physical detail such as "both feet about 30 cm off the ground"
   - No "suddenly appears", "teleports", etc.

3. **Body posture must be explicit**: standing / seated / kneeling / crouching / lying prone / face-down / face-up; position of both feet; body orientation (front / side / back / 3/4 view)

4. **In realistic shots every action must obey gravity**: character falling -> must specify "tied by a rope" or "already landed on a safety mat"; a thrown object -> specify start and landing points; smoke/debris -> drifts upward or falls under gravity

5. **Avoid abstract descriptions**:
   - [BAD] "a nimble posture" -> [GOOD] "right hand reaching forward, left hand bracing the wall, knees slightly bent"
   - [BAD] "full of power" -> [GOOD] "shoulders leaning forward, both hands gripping the railing tightly, muscles taut" """


def fidelity_principle_block(upstream: str, downstream: str) -> str:
    """Shared fidelity block: used by script_parse (fidelity to original text)
    and shot_split (fidelity from script to shot list). The core principle is
    "no deletion, no summarization, no paraphrase" — keep the upstream content
    lossless as it flows through the pipeline.
    """
    return f"""=== {upstream} -> {downstream} Fidelity (Highest Priority) ===
Core mindset: you are a "structurer", not an "adapter". Rewriting, condensing, and omitting the original content are all forbidden.
- **Preserve dialogue verbatim**: filler words ("ah" / "hmm" / "uh" / "..."), repetitions, colloquialisms, dialect, punctuation — keep every one exactly as-is; never "correct" them into written language
- **Land every event in full**: every action, every object, every emotional turn mentioned in the {upstream} must have an explicit landing point in the {downstream}
- **Do not change character names**: use the original names as they appear in the {upstream}
- **When in doubt, split more scenes than fewer**: split on time jumps, location changes, and narrative-beat turns; when uncertain, split by default
- Self-check: after generating, go back and cross-check line by line against the {upstream}; any omission must be filled in, and you must not lower the bar"""
