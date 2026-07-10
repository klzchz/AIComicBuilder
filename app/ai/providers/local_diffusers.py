"""Local Stable Diffusion provider (free, GPU).

Thin client to a local diffusers microservice (see anime-forge/sd_service.py)
that runs SDXL/anime checkpoints on the machine's GPU. This provider only does
image generation; text still goes through the cloud text providers.

Enable by setting AICB_IMAGE_PROVIDER=local (and running the SD service). The
service URL defaults to http://localhost:8500 (override with AICB_SD_URL).
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

import httpx

from app.ai.types import ImageOptions, TextOptions
from app.config import settings
from app.core.ids import new_id

# A sensible anime-oriented negative prompt applied when the caller gives none.
DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, jpeg artifacts, signature, "
    "watermark, username, blurry, deformed, disfigured"
)


def _dims_from_options(options: Optional[ImageOptions]) -> tuple[int, int]:
    """Map aspect ratio / size to SDXL-friendly dimensions (multiples of 8)."""
    ratio = (options.aspect_ratio if options else None) or ""
    presets = {
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "1:1": (1024, 1024),
        "2.35:1": (1344, 576),
        "4:3": (1152, 896),
        "3:4": (896, 1152),
    }
    if ratio in presets:
        return presets[ratio]
    if options and options.size and "x" in options.size:
        try:
            w, h = options.size.lower().split("x")
            return int(w), int(h)
        except ValueError:
            pass
    return 1024, 1024


class LocalDiffusersProvider:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        secret_key: Optional[str] = None,
        model: Optional[str] = None,
        upload_dir: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("AICB_SD_URL") or "http://localhost:8500").rstrip("/")
        self.model = model  # optional model override passed to the service
        self.upload_dir = upload_dir or settings.UPLOAD_DIR

    async def generate_text(self, prompt: str, options: Optional[TextOptions] = None) -> str:  # noqa: D401
        raise NotImplementedError(
            "LocalDiffusersProvider is image-only; use a cloud provider for text."
        )

    async def generate_image(self, prompt: str, options: Optional[ImageOptions] = None) -> str:
        width, height = _dims_from_options(options)
        payload = {
            "prompt": prompt,
            "negative_prompt": DEFAULT_NEGATIVE,
            "width": width,
            "height": height,
        }
        if self.model:
            payload["model"] = self.model

        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(f"{self.base_url}/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()

        image_b64 = data["image_b64"]
        raw = base64.b64decode(image_b64)

        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{new_id()}.png"
        out_path = Path(self.upload_dir) / filename
        out_path.write_bytes(raw)
        return str(out_path)
