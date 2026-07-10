"""Model duration limits — Python port of src/lib/ai/model-limits.ts."""
from __future__ import annotations

from typing import Optional

MODEL_MAX_DURATIONS: dict[str, int] = {
    "veo-2.0-generate-001": 8,
    "veo-3.0-generate-001": 8,
    "veo-3.0-fast-generate-001": 8,
    "veo-3.1-generate-001": 8,
    "veo-3.1-fast-generate-001": 8,
    "kling-v1": 10,
    "kling-v1-5": 10,
    "kling-v2.5-turbo": 10,
    "kling-v3": 15,
    "doubao-seedance-1-5-pro-250528": 12,
    "doubao-seedance-1-5-pro-251215": 12,
    "doubao-seedance-1-0-lite-250528": 5,
    "wan2.7-t2v": 15,
    "wan2.7-r2v": 15,
    "wan2.6-t2v": 15,
    "wan2.6-i2v-flash": 15,
    "wan2.6-i2v": 10,
    "wan2.6-r2v": 10,
    "wan2.6-r2v-flash": 10,
}

# Family-level fallback: if modelId contains this substring, use this duration.
# Order matters — more specific first.
FAMILY_MAX_DURATIONS: list[tuple[str, int]] = [
    ("veo", 8),
    ("kling-v3", 15),
    ("kling", 10),
    ("seedance-1-0", 5),
    ("seedance", 12),
    ("wan2.7", 15),
    ("wan2.6", 15),
    ("wan", 15),
]

DEFAULT_MAX_DURATION = 12


def get_model_max_duration(model_id: Optional[str]) -> int:
    """Return the maximum supported video duration (seconds) for a model ID.

    Unknown models return DEFAULT_MAX_DURATION (12).
    """
    if not model_id:
        return DEFAULT_MAX_DURATION

    lower_model_id = model_id.lower()

    # Exact match
    if lower_model_id in MODEL_MAX_DURATIONS:
        return MODEL_MAX_DURATIONS[lower_model_id]

    # Prefix match (longest keys first)
    for key in sorted(MODEL_MAX_DURATIONS.keys(), key=len, reverse=True):
        if lower_model_id.startswith(key) or key.startswith(lower_model_id):
            return MODEL_MAX_DURATIONS[key]

    # Family substring match (order matters — more specific first)
    for family, duration in FAMILY_MAX_DURATIONS:
        if family in lower_model_id:
            return duration

    return DEFAULT_MAX_DURATION
