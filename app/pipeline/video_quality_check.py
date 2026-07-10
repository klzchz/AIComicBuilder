"""Video quality check — Python port of src/lib/pipeline/video-quality-check.ts.

Scores a generated video frame (0-100) for common generation defects, optionally
comparing against a reference frame. Best-effort: any failure defaults to a pass
so it never blocks generation.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

# Translated from the original Chinese prompt.
QUALITY_CHECK_PROMPT = """Analyze this generated video frame for quality issues. Score 0-100.

Check the following:
1. Facial integrity (no distortion, correct proportions, natural facial features)
2. Limb integrity (correct finger count, natural poses, no extra limbs)
3. Visual coherence (no artifacts, no glitches, no object clipping/merging)
4. Overall image quality (sharpness, plausible lighting, no color banding)

If a reference frame is provided (the second image), also check:
5. Character consistency with the reference (similar face, costume, hairstyle)

Output ONLY valid JSON (no markdown, no code blocks):
{"score": <number 0-100>, "issues": ["<problem description>", ...], "pass": <boolean>}

Score >= 60 passes. Only fail on severe visual defects such as facial \
distortion, missing/extra limbs, or serious artifacts."""


async def check_video_quality(
    video_frame_url: str,
    reference_frame_url: Optional[str] = None,
) -> dict[str, Any]:
    """Return {"pass": bool, "score": int, "issues": [str]}. Port of checkVideoQuality."""
    from app.pipeline._helpers import ai_generate_text  # lazy

    try:
        images = [video_frame_url]
        if reference_frame_url:
            images.append(reference_frame_url)

        result = await ai_generate_text(QUALITY_CHECK_PROMPT, images=images)

        json_match = re.search(r"\{[\s\S]*\}", result)
        if not json_match:
            return {"pass": True, "score": 100, "issues": []}
        parsed = json.loads(json_match.group(0))
        score = parsed.get("score", 0)
        return {
            "pass": parsed.get("pass", score >= 60),
            "score": score,
            "issues": parsed.get("issues", []),
        }
    except Exception:  # noqa: BLE001 — default to pass, don't block generation
        return {"pass": True, "score": 100, "issues": []}
