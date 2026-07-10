"""OpenAI provider — Python port of src/lib/ai/providers/openai.ts."""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

import httpx
from openai import AsyncOpenAI

from app.ai.types import ImageOptions, TextOptions
from app.config import settings
from app.core.ids import new_id


class OpenAIProvider:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        upload_dir: Optional[str] = None,
    ):
        self.client = AsyncOpenAI(
            api_key=api_key or settings.OPENAI_API_KEY,
            base_url=(base_url or settings.OPENAI_BASE_URL) or None,
        )
        self.default_model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o"
        self.upload_dir = upload_dir or settings.UPLOAD_DIR or "./uploads"

    async def generate_text(self, prompt: str, options: Optional[TextOptions] = None) -> str:
        messages: list[dict] = []
        if options and options.system_prompt:
            messages.append({"role": "system", "content": options.system_prompt})

        if options and options.images:
            content: list[dict] = []
            for img_path in options.images:
                try:
                    resolved = Path(img_path).resolve()
                    if resolved.exists():
                        data = base64.b64encode(resolved.read_bytes()).decode("ascii")
                        ext = resolved.suffix.lower()
                        mime_type = "image/png" if ext == ".png" else "image/jpeg"
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{data}"},
                        })
                except Exception:
                    pass  # skip unreadable
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=(options.model if options and options.model else self.default_model),
            messages=messages,
            temperature=(options.temperature if options and options.temperature is not None else 0.7),
            max_tokens=(options.max_tokens if options else None),
        )
        return (response.choices[0].message.content if response.choices else "") or ""

    async def generate_image(self, prompt: str, options: Optional[ImageOptions] = None) -> str:
        model = (options.model if options and options.model else self.default_model)
        is_dalle = model.startswith("dall-e")

        opt_size = options.size if options else None
        opt_ratio = options.aspect_ratio if options else None
        opt_quality = options.quality if options else None

        # Build extra params for non-DALL-E OpenAI-compatible providers (e.g. seedream, doubao).
        compat_params: dict = {}
        if not is_dalle:
            if opt_size:
                compat_params["size"] = opt_size
            if opt_ratio:
                compat_params["aspect_ratio"] = opt_ratio
            if not opt_size and not opt_ratio:
                compat_params["aspect_ratio"] = "16:9"

        params: dict = {"model": model, "prompt": prompt, "n": 1}
        if is_dalle:
            params["size"] = (
                opt_size if opt_size in ("1024x1024", "1792x1024", "1024x1792") else "1792x1024"
            )
            params["quality"] = opt_quality or "standard"
        params.update(compat_params)

        response = await self.client.images.generate(**params)

        image_url = response.data[0].url if response.data else None
        if not image_url:
            raise Exception("No image URL returned from OpenAI")

        async with httpx.AsyncClient(timeout=300.0) as http:
            image_response = await http.get(image_url)
            buffer = image_response.content

        filename = f"{new_id()}.png"
        directory = Path(self.upload_dir) / "frames"
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / filename
        filepath.write_bytes(buffer)

        return str(filepath)
