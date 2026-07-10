"""AI SDK helpers — Python port of src/lib/ai/ai-sdk.ts.

The TS file also constructed Vercel-AI-SDK language models (createLanguageModel).
In the Python port the concrete SDK clients live inside each provider class
(app.ai.providers.*), so here we only keep the provider-config dataclass and the
JSON-extraction helper that the rest of the pipeline relies on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProviderConfig:
    protocol: str
    base_url: str
    api_key: str
    model_id: str
    secret_key: Optional[str] = None


# Control chars that break json.loads, except \n \r \t.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


def extract_json(text: str) -> str:
    """Strip markdown code fences from an AI response if present."""
    match = _CODE_FENCE_RE.search(text)
    raw = match.group(1).strip() if match else text.strip()
    # Remove control characters that break JSON parsing (except \n \r \t)
    return _CONTROL_CHARS_RE.sub("", raw)
