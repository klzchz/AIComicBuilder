"""Continuity check — Python port of src/lib/pipeline/continuity-check.ts.

Compares the last frame of one shot against the first frame of the next and
asks the vision model whether continuity holds. Best-effort: any failure
defaults to a pass so it never blocks generation.
"""
from __future__ import annotations

import json
import re
from typing import Any

# Translated from the original Chinese prompt.
CONTINUITY_PROMPT = """Compare these two consecutive frames from an animated film.
The first frame is the last frame of the previous shot.
The second frame is the first frame of the next shot.

Check for continuity problems:
1. Character costume consistency (same clothing, accessories, hairstyle)
2. Logical character position continuity (natural motion transition)
3. Lighting direction consistency (same light-source angle)
4. Color-tone consistency (matching color grading)
5. Background continuity (if it is the same location)

Output ONLY valid JSON (no markdown):
{"pass": true/false, "issues": ["description of each problem found"]}

Pass if there is no significant continuity break. Slight perspective changes \
caused by different camera angles are normal and expected."""


async def check_continuity(last_frame_url: str, next_first_frame_url: str) -> dict[str, Any]:
    """Return {"pass": bool, "issues": [str]}. Port of checkContinuity."""
    from app.pipeline._helpers import ai_generate_text  # lazy

    try:
        result = await ai_generate_text(
            CONTINUITY_PROMPT,
            images=[last_frame_url, next_first_frame_url],
        )
        json_match = re.search(r"\{[\s\S]*\}", result)
        if not json_match:
            return {"pass": True, "issues": []}
        parsed = json.loads(json_match.group(0))
        return {
            "pass": parsed.get("pass", True),
            "issues": parsed.get("issues", []),
        }
    except Exception:  # noqa: BLE001 — default to pass, don't block generation
        return {"pass": True, "issues": []}
