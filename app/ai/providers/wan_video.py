"""Wan video provider — Python port of src/lib/ai/providers/wan-video.ts."""
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


def to_image_url(image_path_or_url: str) -> str:
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        return image_path_or_url
    ext = Path(image_path_or_url).suffix.lower().replace(".", "")
    if ext in ("jpg", "jpeg"):
        mime = "image/jpeg"
    elif ext == "png":
        mime = "image/png"
    elif ext == "webp":
        mime = "image/webp"
    else:
        mime = "image/png"
    b64 = base64.b64encode(Path(image_path_or_url).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def ratio_to_size(ratio: str) -> str:
    mapping = {
        "16:9": "1280*720",
        "9:16": "720*1280",
        "1:1": "960*960",
        "4:3": "1088*832",
        "3:4": "832*1088",
    }
    return mapping.get(ratio, "1280*720")


def normalise_ratio(ratio: str) -> str:
    supported = ["16:9", "9:16", "1:1", "4:3", "3:4"]
    return ratio if ratio in supported else "16:9"


class WanVideoProvider:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        upload_dir: Optional[str] = None,
    ):
        self.api_key = (
            api_key
            or os.environ.get("WAN_API_KEY")
            or settings.DASHSCOPE_API_KEY
            or ""
        )
        self.base_url = (
            base_url
            or os.environ.get("WAN_BASE_URL")
            or "https://dashscope.aliyuncs.com/api/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("WAN_MODEL") or "wan2.1-i2v-plus"
        self.upload_dir = upload_dir or settings.UPLOAD_DIR or "./uploads"

    @property
    def is_wan27(self) -> bool:
        return self.model.startswith("wan2.7")

    async def generate_video(self, params: VideoGenerateParams) -> VideoGenerateResult:
        if params.first_frame is not None:
            body = self._build_keyframe_body(params)
        elif params.initial_image:
            body = self._build_reference_body(params)
        else:
            body = self._build_text_body(params)

        logger.info(
            "[WanVideo] Submitting task: model=%s, ratio=%s", self.model, params.ratio
        )

        async with httpx.AsyncClient(timeout=300.0) as http:
            submit_res = await http.post(
                f"{self.base_url}/services/aigc/video-generation/video-synthesis",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "X-DashScope-Async": "enable",
                },
                json=body,
            )
            if submit_res.status_code >= 400:
                raise Exception(f"WanVideo submit failed: {submit_res.status_code} {submit_res.text}")

            submit_result = submit_res.json()
            task_id = (submit_result.get("output") or {}).get("task_id")
            if not task_id:
                raise Exception(f"WanVideo: no task_id in response: {submit_result}")

            logger.info("[WanVideo] Task submitted: %s", task_id)

            video_url = await self._poll_for_result(http, task_id)

            video_res = await http.get(video_url)
            if video_res.status_code >= 400:
                raise Exception(f"WanVideo: failed to download video ({video_res.status_code})")
            buffer = video_res.content

        filename = f"{new_id()}.mp4"
        directory = Path(self.upload_dir) / "videos"
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / filename
        filepath.write_bytes(buffer)

        logger.info("[WanVideo] Saved to %s", filepath)
        return VideoGenerateResult(file_path=str(filepath))

    # ── Body builders ──────────────────────────────────────────────────────────

    def _build_keyframe_body(self, params: VideoGenerateParams) -> dict:
        if self.is_wan27:
            return {
                "model": "wan2.7-r2v",
                "input": {
                    "prompt": params.prompt,
                    "media": [
                        {"type": "first_frame", "url": to_image_url(params.first_frame)},
                        {"type": "last_frame", "url": to_image_url(params.last_frame)},
                    ],
                },
                "parameters": {
                    "resolution": "720P",
                    "ratio": normalise_ratio(params.ratio),
                    "duration": params.duration or 5,
                },
            }

        return {
            "model": self.model,
            "input": {
                "prompt": params.prompt,
                "img_url": to_image_url(params.first_frame),
            },
            "parameters": {
                "size": ratio_to_size(params.ratio),
                "duration": params.duration or 5,
            },
        }

    def _build_reference_body(self, params: VideoGenerateParams) -> dict:
        if self.is_wan27:
            media = [{"type": "reference_image", "url": to_image_url(params.initial_image)}]
            if params.reference_images:
                for ref_img in params.reference_images[:8]:
                    media.append({"type": "reference_image", "url": to_image_url(ref_img)})

            return {
                "model": "wan2.7-r2v",
                "input": {"prompt": params.prompt, "media": media},
                "parameters": {
                    "resolution": "720P",
                    "ratio": normalise_ratio(params.ratio),
                    "duration": params.duration or 5,
                },
            }

        return {
            "model": self.model,
            "input": {
                "prompt": params.prompt,
                "img_url": to_image_url(params.initial_image),
            },
            "parameters": {
                "size": ratio_to_size(params.ratio),
                "duration": params.duration or 5,
            },
        }

    def _build_text_body(self, params: VideoGenerateParams) -> dict:
        model = "wan2.7-t2v" if self.is_wan27 else self.model

        if self.is_wan27:
            return {
                "model": model,
                "input": {"prompt": params.prompt},
                "parameters": {
                    "resolution": "720P",
                    "ratio": normalise_ratio(params.ratio),
                    "duration": params.duration or 5,
                },
            }

        return {
            "model": model,
            "input": {"prompt": params.prompt},
            "parameters": {
                "size": ratio_to_size(params.ratio),
                "duration": params.duration or 5,
            },
        }

    # ── Polling ────────────────────────────────────────────────────────────────

    async def _poll_for_result(self, http: httpx.AsyncClient, task_id: str) -> str:
        max_attempts = 360  # 30 min — Wan models are slower
        interval = 5.0

        for i in range(max_attempts):
            await asyncio.sleep(interval)

            res = await http.get(
                f"{self.base_url}/tasks/{task_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            if res.status_code >= 400:
                logger.warning("[WanVideo] Poll %s: HTTP %s, retrying…", i + 1, res.status_code)
                continue

            result = res.json()
            output = result.get("output") or {}
            status = output.get("task_status") or "UNKNOWN"
            logger.info("[WanVideo] Poll %s: status=%s", i + 1, status)

            if status == "SUCCEEDED":
                video_url = output.get("video_url")
                if not video_url:
                    raise Exception(f"WanVideo: SUCCEEDED but no video_url in response: {result}")
                return video_url

            if status == "FAILED":
                raise Exception(
                    f"WanVideo generation failed: {output.get('message') or 'unknown error'}"
                )

            # PENDING / RUNNING → keep polling

        raise Exception("WanVideo generation timed out after 30 minutes")
