"""Seedance video provider — Python port of src/lib/ai/providers/seedance.ts."""
from __future__ import annotations

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Optional

import httpx

from app.ai.types import VideoGenerateParams, VideoGenerateResult
from app.config import settings
from app.core.ids import new_id

logger = logging.getLogger(__name__)


def to_data_url(file_path: str) -> str:
    ext = Path(file_path).suffix.lower().replace(".", "")
    if ext in ("jpg", "jpeg"):
        mime = "image/jpeg"
    elif ext == "png":
        mime = "image/png"
    elif ext == "webp":
        mime = "image/webp"
    else:
        mime = "image/png"
    b64 = base64.b64encode(Path(file_path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def to_image_url(image_path_or_url: str) -> str:
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        return image_path_or_url
    return to_data_url(image_path_or_url)


class SeedanceProvider:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        upload_dir: Optional[str] = None,
    ):
        self.api_key = api_key or settings.SEEDANCE_API_KEY or ""
        self.base_url = (
            base_url
            or os.environ.get("SEEDANCE_BASE_URL")
            or "https://ark.cn-beijing.volces.com/api/v3"
        ).rstrip("/")
        self.model = (
            model or os.environ.get("SEEDANCE_MODEL") or "doubao-seedance-1-5-pro-250528"
        )
        self.upload_dir = upload_dir or settings.UPLOAD_DIR or "./uploads"

    async def generate_video(self, params: VideoGenerateParams) -> VideoGenerateResult:
        if params.first_frame is not None:
            body = self._build_keyframe_body(params)
        else:
            body = self._build_reference_body(params)

        logger.info(
            "[Seedance] Submitting task: model=%s, duration=%s, ratio=%s",
            body["model"], body["duration"], body["ratio"],
        )

        async with httpx.AsyncClient(timeout=300.0) as http:
            submit_response = await http.post(
                f"{self.base_url}/contents/generations/tasks",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json=body,
            )
            if submit_response.status_code >= 400:
                raise Exception(
                    f"Seedance submit failed: {submit_response.status_code} {submit_response.text}"
                )
            submit_result = submit_response.json()
            task_id = submit_result["id"]
            logger.info("[Seedance] Task submitted: %s", task_id)

            video_url, last_frame_url = await self._poll_for_result(http, task_id)

            video_response = await http.get(video_url)
            buffer = video_response.content

        filename = f"{new_id()}.mp4"
        directory = Path(self.upload_dir) / "videos"
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / filename
        filepath.write_bytes(buffer)

        return VideoGenerateResult(file_path=str(filepath), last_frame_url=last_frame_url)

    def _build_keyframe_body(self, params: VideoGenerateParams) -> dict:
        is_seedance2 = "seedance-2" in self.model
        body = {
            "model": self.model,
            "content": [
                {"type": "text", "text": params.prompt},
                {"type": "image_url", "image_url": {"url": to_data_url(params.first_frame)}, "role": "first_frame"},
                {"type": "image_url", "image_url": {"url": to_data_url(params.last_frame)}, "role": "last_frame"},
            ],
            "duration": params.duration or 5,
            "ratio": params.ratio or "16:9",
            "watermark": False,
        }
        if is_seedance2:
            body["generate_audio"] = True
        return body

    def _build_reference_body(self, params: VideoGenerateParams) -> dict:
        is_seedance2 = "seedance-2" in self.model

        content: list[dict] = [{"type": "text", "text": params.prompt}]

        if params.reference_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": to_image_url(params.initial_image)},
                "role": "reference_image",
            })
            for ref_img in params.reference_images[:8]:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": to_image_url(ref_img)},
                    "role": "reference_image",
                })
        else:
            content.append({
                "type": "image_url",
                "image_url": {"url": to_image_url(params.initial_image)},
            })

        body = {
            "model": self.model,
            "content": content,
            "duration": params.duration or 5,
            "ratio": params.ratio or "16:9",
            "return_last_frame": True,
            "watermark": False,
        }
        if is_seedance2:
            body["generate_audio"] = True
        return body

    async def _poll_for_result(
        self, http: httpx.AsyncClient, task_id: str
    ) -> tuple[str, Optional[str]]:
        max_attempts = 120
        interval = 5.0

        for i in range(max_attempts):
            await asyncio.sleep(interval)

            response = await http.get(
                f"{self.base_url}/contents/generations/tasks/{task_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            if response.status_code >= 400:
                continue

            result = response.json()
            logger.info("[Seedance] Poll %s: status=%s", i + 1, result.get("status"))

            if result.get("status") == "succeeded" and (result.get("content") or {}).get("video_url"):
                content = result["content"]
                return content["video_url"], content.get("last_frame_url")
            if result.get("status") == "failed":
                message = (result.get("error") or {}).get("message") or "unknown"
                raise Exception(f"Seedance generation failed: {message}")

        raise Exception("Seedance generation timed out after 10 minutes")
