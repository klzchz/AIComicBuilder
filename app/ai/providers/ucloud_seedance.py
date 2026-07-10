"""UCloud ModelVerse Seedance provider — Python port of ucloud-seedance.ts.

API docs: https://docs.ucloud.cn/modelverse/api_doc/video_api/doubao-seedance-1-5-pro-251215

Submit:  POST {baseUrl}/v1/tasks/submit
Poll:    GET  {baseUrl}/v1/tasks/status?task_id=<id>

Supports both Seedance 1.5 and Seedance 2.0 models.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import quote

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


class UCloudSeedanceProvider:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        upload_dir: Optional[str] = None,
    ):
        self.api_key = api_key or ""
        self.base_url = (base_url or "https://api.modelverse.cn").rstrip("/")
        self.model = model or "doubao-seedance-1-5-pro-251215"
        self.upload_dir = upload_dir or settings.UPLOAD_DIR or "./uploads"

    async def generate_video(self, params: VideoGenerateParams) -> VideoGenerateResult:
        if params.first_frame is not None:
            body = self._build_keyframe_body(params)
        else:
            body = self._build_reference_body(params)

        logger.info(
            "[UCloudSeedance] Submitting task: model=%s, duration=%s",
            self.model, (body.get("parameters") or {}).get("duration"),
        )

        async with httpx.AsyncClient(timeout=300.0) as http:
            submit_response = await http.post(
                f"{self.base_url}/v1/tasks/submit",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": self.api_key,
                },
                json=body,
            )
            if submit_response.status_code >= 400:
                raise Exception(
                    f"UCloudSeedance submit failed: {submit_response.status_code} {submit_response.text}"
                )

            submit_result = submit_response.json()
            task_id = (submit_result.get("output") or {}).get("task_id")
            if not task_id:
                raise Exception(f"UCloudSeedance: no task_id in response: {submit_result}")
            logger.info("[UCloudSeedance] Task submitted: %s", task_id)

            video_url = await self._poll_for_result(http, task_id)

            video_response = await http.get(video_url)
            buffer = video_response.content

        filename = f"{new_id()}.mp4"
        directory = Path(self.upload_dir) / "videos"
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / filename
        filepath.write_bytes(buffer)

        return VideoGenerateResult(file_path=str(filepath))

    def _build_keyframe_body(self, params: VideoGenerateParams) -> dict:
        is_seedance2 = "seedance-2" in self.model
        parameters = {
            "duration": params.duration or 5,
            "ratio": params.ratio or "16:9",
            "resolution": "720p",
            "watermark": False,
        }
        if is_seedance2:
            parameters["generate_audio"] = True
        return {
            "model": self.model,
            "input": {
                "content": [
                    {"type": "text", "text": params.prompt},
                    {"type": "image_url", "image_url": {"url": to_data_url(params.first_frame)}, "role": "first_frame"},
                    {"type": "image_url", "image_url": {"url": to_data_url(params.last_frame)}, "role": "last_frame"},
                ],
            },
            "parameters": parameters,
        }

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
                "role": "first_frame",
            })

        parameters = {
            "duration": params.duration or 5,
            "ratio": params.ratio or "16:9",
            "resolution": "720p",
            "watermark": False,
        }
        if is_seedance2:
            parameters["generate_audio"] = True

        return {
            "model": self.model,
            "input": {"content": content},
            "parameters": parameters,
        }

    async def _poll_for_result(self, http: httpx.AsyncClient, task_id: str) -> str:
        max_attempts = 360  # 30 min
        interval = 5.0

        for i in range(max_attempts):
            await asyncio.sleep(interval)

            res = await http.get(
                f"{self.base_url}/v1/tasks/status?task_id={quote(task_id, safe='')}",
                headers={"Authorization": self.api_key},
            )
            if res.status_code >= 400:
                logger.warning("[UCloudSeedance] Poll %s: HTTP %s, retrying…", i + 1, res.status_code)
                continue

            result = res.json()
            output = result.get("output") or {}
            status = output.get("task_status") or "UNKNOWN"
            logger.info("[UCloudSeedance] Poll %s: status=%s", i + 1, status)

            if status == "Success":
                urls = output.get("urls")
                if not urls:
                    raise Exception(f"UCloudSeedance: Success but no urls in response: {result}")
                return urls[0]

            if status in ("Failure", "Expired"):
                raise Exception(
                    f"UCloudSeedance generation {status.lower()}: {output.get('error_message') or 'unknown error'}"
                )

            # Pending / Running → keep polling

        raise Exception("UCloudSeedance generation timed out after 30 minutes")
