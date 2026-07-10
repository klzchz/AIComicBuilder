"""Kling image provider — Python port of src/lib/ai/providers/kling-image.ts."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from pathlib import Path
from typing import Optional

import httpx

from app.ai.types import ImageOptions, TextOptions
from app.config import settings
from app.core.ids import new_id

logger = logging.getLogger(__name__)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_kling_token(access_key: str, secret_key: str) -> str:
    now = int(time.time())
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64url(
        json.dumps({"iss": access_key, "exp": now + 1800, "nbf": now - 5}, separators=(",", ":")).encode()
    )
    signing_input = f"{header}.{payload}".encode()
    signature = _b64url(hmac.new(secret_key.encode(), signing_input, hashlib.sha256).digest())
    return f"{header}.{payload}.{signature}"


class KlingImageProvider:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        upload_dir: Optional[str] = None,
    ):
        self.api_key = api_key or settings.KLING_ACCESS_KEY or ""
        self.secret_key = secret_key or settings.KLING_SECRET_KEY or ""
        self.base_url = (base_url or "https://api.klingai.com").rstrip("/")
        self.model = model or "kling-v1"
        self.upload_dir = upload_dir or settings.UPLOAD_DIR or "./uploads"

    def _get_auth_header(self) -> str:
        if self.secret_key:
            return f"Bearer {generate_kling_token(self.api_key, self.secret_key)}"
        return f"Bearer {self.api_key}"

    async def generate_text(self, prompt: str, options: Optional[TextOptions] = None) -> str:
        raise Exception("Kling does not support text generation")

    async def generate_image(self, prompt: str, options: Optional[ImageOptions] = None) -> str:
        aspect_ratio = (options.aspect_ratio if options and options.aspect_ratio else "16:9")

        async with httpx.AsyncClient(timeout=300.0) as http:
            submit_res = await http.post(
                f"{self.base_url}/v1/images/generations",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": self._get_auth_header(),
                },
                json={"model": self.model, "prompt": prompt, "n": 1, "aspect_ratio": aspect_ratio},
            )
            if submit_res.status_code >= 400:
                raise Exception(f"Kling image submit failed: {submit_res.status_code}")

            submit_json = submit_res.json()
            if submit_json.get("code") != 0:
                raise Exception(f"Kling image error: {submit_json.get('message')}")

            task_id = submit_json["data"]["task_id"]
            logger.info("[Kling Image] Task submitted: %s", task_id)

            image_url = await self._poll_for_result(http, task_id)

            image_res = await http.get(image_url)
            buffer = image_res.content

        ext = image_url.split("?")[0].split(".")[-1] or "png"
        filename = f"{new_id()}.{ext}"
        directory = Path(self.upload_dir) / "images"
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / filename
        filepath.write_bytes(buffer)

        logger.info("[Kling Image] Saved to %s", filepath)
        return str(filepath)

    async def _poll_for_result(self, http: httpx.AsyncClient, task_id: str) -> str:
        max_attempts = 60

        for i in range(max_attempts):
            await asyncio.sleep(5.0)

            res = await http.get(
                f"{self.base_url}/v1/images/generations/{task_id}",
                headers={"Authorization": self._get_auth_header()},
            )
            if res.status_code >= 400:
                raise Exception(f"Kling image poll failed: {res.status_code}")

            data = res.json()
            if data.get("code") != 0:
                raise Exception(f"Kling image poll error: {data.get('message')}")

            task = data["data"]
            task_status = task["task_status"]
            logger.info("[Kling Image] Poll %s: status=%s", i + 1, task_status)

            if task_status == "succeed":
                images = (task.get("task_result") or {}).get("images") or []
                url = images[0]["url"] if images else None
                if not url:
                    raise Exception("Kling image: no URL in result")
                return url
            if task_status == "failed":
                raise Exception(f"Kling image generation failed: {task.get('task_status_msg')}")

        raise Exception("Kling image generation timed out after 5 minutes")
