"""DashScope image provider — Python port of src/lib/ai/providers/dashscope-image.ts."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import httpx

from app.ai.types import ImageOptions, TextOptions
from app.config import settings
from app.core.ids import new_id

logger = logging.getLogger(__name__)

# ── Model family detection ──────────────────────────────────────────────────


def get_model_family(model: str) -> str:
    if model.startswith("wan"):
        return "wan"
    if model.startswith("z-image"):
        return "zimage"
    return "qwen"  # qwen-image-*


# ── Aspect-ratio → pixel size mappings ──────────────────────────────────────

WAN_ASPECT_RATIO_MAP = {
    "1:1": "1024*1024",
    "16:9": "1280*720",
    "9:16": "720*1280",
    "4:3": "1024*768",
    "3:4": "768*1024",
    "3:2": "1080*720",
    "2:3": "720*1080",
}

QWEN_ASPECT_RATIO_MAP = {
    "1:1": "2048*2048",
    "16:9": "2048*1152",
    "9:16": "1152*2048",
    "4:3": "2048*1536",
    "3:4": "1536*2048",
    "3:2": "2048*1365",
    "2:3": "1365*2048",
}

ZIMAGE_ASPECT_RATIO_MAP = {
    "1:1": "1024*1024",
    "16:9": "1536*1024",
    "9:16": "1024*1536",
    "4:3": "1024*768",
    "3:4": "768*1024",
    "3:2": "1536*1024",
    "2:3": "1024*1536",
}


def resolve_size(family: str, size: Optional[str], aspect_ratio: Optional[str]) -> Optional[str]:
    # If explicit size is given, pass through (caller knows best)
    if size:
        return size

    if aspect_ratio:
        if family == "wan":
            return WAN_ASPECT_RATIO_MAP.get(aspect_ratio)
        if family == "qwen":
            return QWEN_ASPECT_RATIO_MAP.get(aspect_ratio)
        if family == "zimage":
            return ZIMAGE_ASPECT_RATIO_MAP.get(aspect_ratio)

    # Return family-specific defaults
    if family == "wan":
        return "1024*1024"
    if family == "qwen":
        return "2048*2048"
    if family == "zimage":
        return "1024*1536"
    return None


class DashScopeImageProvider:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        upload_dir: Optional[str] = None,
    ):
        self.api_key = api_key or settings.DASHSCOPE_API_KEY or ""
        self.base_url = (
            base_url
            or os.environ.get("DASHSCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/api/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("DASHSCOPE_IMAGE_MODEL") or "qwen-image-2.0-pro"
        self.upload_dir = upload_dir or settings.UPLOAD_DIR or "./uploads"

    async def generate_text(self, prompt: str, options: Optional[TextOptions] = None) -> str:
        raise Exception("DashScope image models do not support text generation")

    async def generate_image(self, prompt: str, options: Optional[ImageOptions] = None) -> str:
        model = (options.model if options and options.model else self.model)
        family = get_model_family(model)
        size = resolve_size(
            family,
            options.size if options else None,
            options.aspect_ratio if options else None,
        )

        parameters: dict = {}
        if size:
            parameters["size"] = size

        if family == "wan":
            parameters["n"] = 1
        elif family == "qwen":
            parameters["n"] = 1
        elif family == "zimage":
            pass  # z-image-turbo does not support n parameter

        body = {
            "model": model,
            "input": {
                "messages": [
                    {"role": "user", "content": [{"text": prompt}]},
                ],
            },
            "parameters": parameters,
        }

        logger.info("[DashScopeImage] Generating: model=%s, family=%s, size=%s", model, family, size)

        async with httpx.AsyncClient(timeout=300.0) as http:
            res = await http.post(
                f"{self.base_url}/services/aigc/multimodal-generation/generation",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json=body,
            )
            if res.status_code >= 400:
                raise Exception(f"DashScope image request failed: {res.status_code} {res.text}")

            data = res.json()

            # Check for API-level error
            if data.get("code"):
                raise Exception(
                    f"DashScope image error [{data['code']}]: {data.get('message') or 'unknown'}"
                )

            image_url = None
            choices = (data.get("output") or {}).get("choices")
            if choices:
                content = (choices[0].get("message") or {}).get("content")
                if content:
                    image_url = content[0].get("image")
            if not image_url:
                raise Exception(f"DashScope image: no image URL in response: {data}")

            image_res = await http.get(image_url)
            if image_res.status_code >= 400:
                raise Exception(f"DashScope image: failed to download image ({image_res.status_code})")
            buffer = image_res.content

        ext = image_url.split("?")[0].split(".")[-1] or "png"
        filename = f"{new_id()}.{ext}"
        directory = Path(self.upload_dir) / "images"
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / filename
        filepath.write_bytes(buffer)

        logger.info("[DashScopeImage] Saved to %s", filepath)
        return str(filepath)
