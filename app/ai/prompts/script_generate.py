"""User-side prompt builder for script generation.

Python port of src/lib/ai/prompts/script-generate.ts.

The authoritative SYSTEM prompt lives in the prompt registry under the
`script_generate` key. This module only builds the per-request user message
so the two are not duplicated.
"""
import re

_CJK = re.compile(r"[一-鿿]")
_KANA = re.compile(r"[぀-ゟ゠-ヿ]")
_HANGUL = re.compile(r"[가-힯]")


def _detect_language(text: str) -> str:
    if _CJK.search(text):
        return "Chinese (中文)"
    if _KANA.search(text):
        return "Japanese (日本語)"
    if _HANGUL.search(text):
        return "Korean (한국어)"
    return "English"


def build_script_generate_prompt(idea: str) -> str:
    lang = _detect_language(idea)

    return f"""Write a complete, production-ready screenplay based on this creative concept:

"{idea}"

OUTPUT LANGUAGE: {lang}. You MUST write EVERY word of your output in {lang}, including all section headers, character descriptions, stage directions, and dialogue. Do NOT use English if the language is not English.

**STRICT FORMAT REMINDER** (details are in the system prompt — do not violate):
- Sections 1 (视觉风格) and 2 (角色描述) are machine-readable key:value blocks with fixed Chinese field labels. No markdown, no bullets, no code fences. One field per line.
- Section 3 (场景) is free-form screenplay prose.
- Field labels stay in Chinese verbatim regardless of the output language of the rest of the screenplay.

Content quality:
- Respect user-specified art style if given; otherwise infer the most fitting style from the concept.
- CHARACTERS section must cover every named character with all 5 required fields — downstream AI image generators rely on these to produce consistent visuals.
- Each scene description must be vivid enough for an AI image generator to produce a frame directly.
- Write RICHLY and in DETAIL — every scene needs specific visual descriptions, character actions, emotional beats, and dialogue. Avoid rushing through the story."""
