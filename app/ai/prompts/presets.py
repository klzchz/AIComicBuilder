"""Built-in prompt presets.

Python port of src/lib/ai/prompts/presets.ts. Empty for now — preset content
will be authored later.
"""
from dataclasses import dataclass, field


@dataclass
class BuiltInPreset:
    id: str
    name: str
    name_key: str
    description_key: str
    prompt_key: str
    slots: dict = field(default_factory=dict)


BUILT_IN_PRESETS: list = []
