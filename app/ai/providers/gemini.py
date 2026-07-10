"""Gemini provider — Python port of src/lib/ai/providers/gemini.ts."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from app.ai.types import ImageOptions, TextOptions
from app.config import settings
from app.core.ids import new_id

_REFERENCE_RULES = """
[END OF REFERENCE IMAGES — {count} character sheets total]
CRITICAL CHARACTER CONSISTENCY RULES:
- Each reference image is a CHARACTER SHEET (turnaround view) showing FRONT, THREE-QUARTER, SIDE PROFILE, and BACK views
- The character's NAME is printed at the bottom of each reference sheet — use it to match characters in the scene
- You MUST reproduce EXACTLY: face shape, hairstyle, hair color, clothing design, clothing colors, accessories, body proportions
- CLOTHING MUST NOT CHANGE — if the reference shows a specific outfit, the character MUST wear that same outfit in the generated frame, NOT any other outfit
- If a character's reference shows specific accessories (hat, sword, hairpin), they MUST appear in the generated frame
- Art style must match the reference images exactly

"""


def _strip_base_url(url: str) -> str:
    return re.sub(r"/v\d[^/]*$", "", url.rstrip("/"))


def _read_image(path: Path) -> tuple[bytes, str]:
    ext = path.suffix.lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"
    return path.read_bytes(), mime_type


class GeminiProvider:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        upload_dir: Optional[str] = None,
    ):
        client_kwargs: dict = {"api_key": api_key or settings.GEMINI_API_KEY or ""}
        if base_url:
            client_kwargs["http_options"] = types.HttpOptions(base_url=_strip_base_url(base_url))
        self.client = genai.Client(**client_kwargs)
        self.default_model = model or "gemini-2.0-flash"
        self.upload_dir = upload_dir or settings.UPLOAD_DIR or "./uploads"

    async def generate_text(self, prompt: str, options: Optional[TextOptions] = None) -> str:
        model = (options.model if options and options.model else self.default_model)

        parts: list[types.Part] = []
        if options and options.images:
            for img_path in options.images:
                try:
                    resolved = Path(img_path).resolve()
                    if resolved.exists():
                        data, mime_type = _read_image(resolved)
                        parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
                except Exception:
                    pass  # skip
        parts.append(types.Part.from_text(text=prompt))

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=(options.temperature if options and options.temperature is not None else 0.7),
                max_output_tokens=(options.max_tokens if options else None),
                system_instruction=(options.system_prompt if options else None),
            ),
        )
        return response.text or ""

    async def generate_image(self, prompt: str, options: Optional[ImageOptions] = None) -> str:
        model = (options.model if options and options.model else self.default_model)

        parts: list[types.Part] = []

        ref_images = options.reference_images if options else None
        ref_labels = options.reference_labels if options else None

        if ref_images:
            img_index = 0
            for ri, img_path in enumerate(ref_images):
                try:
                    resolved = Path(img_path).resolve()
                    if resolved.exists():
                        data, mime_type = _read_image(resolved)
                        img_index += 1
                        label_name = ref_labels[ri] if ref_labels and ri < len(ref_labels) else None
                        label = (
                            f"[Character Reference: {label_name}]"
                            if label_name
                            else f"[Reference Image {img_index}]"
                        )
                        parts.append(types.Part.from_text(text=label))
                        parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
                except Exception:
                    pass  # Skip unreadable images
            if img_index > 0:
                parts.append(types.Part.from_text(
                    text=_REFERENCE_RULES.format(count=img_index) + prompt
                ))
            else:
                parts.append(types.Part.from_text(text=prompt))
        else:
            parts.append(types.Part.from_text(text=prompt))

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(response_modalities=["image", "text"]),
        )

        candidates = response.candidates
        response_parts = (
            candidates[0].content.parts if candidates and candidates[0].content else None
        )
        if not response_parts:
            raise Exception("No image returned from Gemini")

        for part in response_parts:
            inline = part.inline_data
            if inline and inline.data:
                buffer = inline.data  # raw bytes from SDK
                ext = "png" if (inline.mime_type and "png" in inline.mime_type) else "jpg"
                filename = f"{new_id()}.{ext}"
                directory = Path(self.upload_dir) / "frames"
                directory.mkdir(parents=True, exist_ok=True)
                filepath = directory / filename
                filepath.write_bytes(buffer)
                return str(filepath)
        raise Exception("No image data found in Gemini response")
