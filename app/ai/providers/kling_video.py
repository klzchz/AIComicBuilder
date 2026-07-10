"""Kling video provider — Python port of src/lib/ai/providers/kling-video.ts."""
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

from app.ai.types import VideoGenerateParams, VideoGenerateResult
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


def to_base64(file_path: str) -> str:
    try:
        data = Path(file_path).read_bytes()
    except Exception:
        raise Exception(f"Kling: frame file not found: {file_path}")
    return base64.b64encode(data).decode("ascii")


async def to_base64_from_path_or_url(http: httpx.AsyncClient, path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        res = await http.get(path_or_url)
        if res.status_code >= 400:
            raise Exception(f"Failed to fetch image: {path_or_url} ({res.status_code})")
        return base64.b64encode(res.content).decode("ascii")
    return to_base64(path_or_url)


class KlingVideoProvider:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        upload_dir: Optional[str] = None,
    ):
        self.api_key = (api_key or settings.KLING_ACCESS_KEY or "").strip()
        self.secret_key = (secret_key or settings.KLING_SECRET_KEY or "").strip()
        self.base_url = (base_url or "https://api.klingai.com").rstrip("/")
        self.model = model or "kling-v1"
        self.upload_dir = upload_dir or settings.UPLOAD_DIR or "./uploads"

    def _get_auth_header(self) -> str:
        if self.secret_key:
            return f"Bearer {generate_kling_token(self.api_key, self.secret_key)}"
        return f"Bearer {self.api_key}"

    def _map_duration(self, duration: int) -> int:
        if self.model == "kling-v3":
            return max(3, min(15, duration))
        return 5 if duration <= 5 else 10

    async def generate_video(self, params: VideoGenerateParams) -> VideoGenerateResult:
        duration = self._map_duration(params.duration)
        aspect_ratio = params.ratio

        is_keyframe = params.first_frame is not None

        async with httpx.AsyncClient(timeout=300.0) as http:
            if is_keyframe:
                # ── Keyframe mode: image2video ──
                image_data = to_base64(params.first_frame)
                tail_image_data = to_base64(params.last_frame)

                logger.info(
                    "[Kling Video] image2video: model=%s, duration=%ss, ratio=%s",
                    self.model, duration, aspect_ratio,
                )

                submit_res = await http.post(
                    f"{self.base_url}/v1/videos/image2video",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": self._get_auth_header(),
                    },
                    json={
                        "model": self.model,
                        "prompt": params.prompt,
                        "image": image_data,
                        "tail_image": tail_image_data,
                        "duration": duration,
                        "aspect_ratio": aspect_ratio,
                        "sound": "on",
                    },
                )
                if submit_res.status_code >= 400:
                    raise Exception(
                        f"Kling image2video submit failed: {submit_res.status_code} {submit_res.text}"
                    )
                submit_json = submit_res.json()
                if submit_json.get("code") != 0:
                    raise Exception(f"Kling image2video error: {submit_json.get('message')}")
                task_id = submit_json["data"]["task_id"]
                logger.info("[Kling Video] image2video task submitted: %s", task_id)
            else:
                # ── Reference image mode: text2video with initial image ──
                ref_image = await to_base64_from_path_or_url(http, params.initial_image)

                logger.info(
                    "[Kling Video] text2video: model=%s, duration=%ss, ratio=%s",
                    self.model, duration, aspect_ratio,
                )

                submit_res = await http.post(
                    f"{self.base_url}/v1/videos/text2video",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": self._get_auth_header(),
                    },
                    json={
                        "model": self.model,
                        "prompt": params.prompt,
                        "reference_image": [ref_image],
                        "duration": duration,
                        "aspect_ratio": aspect_ratio,
                    },
                )

                # Fallback: if reference_image is unsupported (400/422), retry without it
                if submit_res.status_code in (400, 422):
                    fallback_body = submit_res.text
                    logger.warning(
                        "[Kling Video] text2video reference_image rejected (%s: %s), retrying without ref images",
                        submit_res.status_code, fallback_body,
                    )
                    submit_res = await http.post(
                        f"{self.base_url}/v1/videos/text2video",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": self._get_auth_header(),
                        },
                        json={
                            "model": self.model,
                            "prompt": params.prompt,
                            "duration": duration,
                            "aspect_ratio": aspect_ratio,
                        },
                    )

                if submit_res.status_code >= 400:
                    raise Exception(
                        f"Kling text2video submit failed: {submit_res.status_code} {submit_res.text}"
                    )
                submit_json = submit_res.json()
                if submit_json.get("code") != 0:
                    raise Exception(f"Kling text2video error: {submit_json.get('message')}")
                task_id = submit_json["data"]["task_id"]
                logger.info("[Kling Video] text2video task submitted: %s", task_id)

            task_type = "image2video" if is_keyframe else "text2video"
            video_url = await self._poll_for_result(http, task_id, task_type)

            video_res = await http.get(video_url)
            buffer = video_res.content

        filename = f"{new_id()}.mp4"
        directory = Path(self.upload_dir) / "videos"
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / filename
        filepath.write_bytes(buffer)

        logger.info("[Kling Video] Saved to %s", filepath)
        return VideoGenerateResult(file_path=str(filepath))

    async def _poll_for_result(
        self, http: httpx.AsyncClient, task_id: str, task_type: str
    ) -> str:
        max_attempts = 120

        for i in range(max_attempts):
            await asyncio.sleep(5.0)

            res = await http.get(
                f"{self.base_url}/v1/videos/{task_type}/{task_id}",
                headers={"Authorization": self._get_auth_header()},
            )
            if res.status_code >= 400:
                raise Exception(f"Kling video poll failed: {res.status_code}")

            data = res.json()
            if data.get("code") != 0:
                raise Exception(f"Kling video poll error: {data.get('message')}")

            task = data["data"]
            task_status = task["task_status"]
            logger.info("[Kling Video] Poll %s: status=%s", i + 1, task_status)

            if task_status == "succeed":
                videos = (task.get("task_result") or {}).get("videos") or []
                url = videos[0]["url"] if videos else None
                if not url:
                    raise Exception("Kling video: no URL in result")
                return url
            if task_status == "failed":
                raise Exception(f"Kling video generation failed: {task.get('task_status_msg')}")

        raise Exception("Kling video generation timed out after 10 minutes")
