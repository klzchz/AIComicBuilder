"""Veo video provider — Python port of src/lib/ai/providers/veo.ts."""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from app.ai.types import VideoGenerateParams, VideoGenerateResult
from app.config import settings
from app.core.ids import new_id

logger = logging.getLogger(__name__)

VALID_DURATIONS = [4, 6, 8]


def clamp_duration(duration: int) -> int:
    best = VALID_DURATIONS[0]
    for curr in VALID_DURATIONS:
        if abs(curr - duration) < abs(best - duration):
            best = curr
    return best


def to_aspect_ratio(ratio: Optional[str]) -> str:
    if ratio == "9:16":
        return "9:16"
    return "16:9"


def read_image_data(file_path: str) -> types.Image:
    ext = Path(file_path).suffix.lower()
    if ext == ".png":
        mime_type = "image/png"
    elif ext == ".webp":
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"
    return types.Image(image_bytes=Path(file_path).read_bytes(), mime_type=mime_type)


def _strip_base_url(url: str) -> str:
    return re.sub(r"/v\d[^/]*$", "", url.rstrip("/"))


class VeoProvider:
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
        self.model = model or "veo-2.0-generate-001"
        self.upload_dir = upload_dir or settings.UPLOAD_DIR or "./uploads"

    def _is_veo31(self) -> bool:
        return "3.1" in self.model or "3-1" in self.model

    async def generate_video(self, params: VideoGenerateParams) -> VideoGenerateResult:
        duration_seconds = clamp_duration(params.duration)
        aspect_ratio = to_aspect_ratio(params.ratio)

        is_keyframe = params.first_frame is not None and bool(params.first_frame)
        is_reference = params.initial_image is not None and bool(params.initial_image)
        has_char_ref_images = bool(params.reference_images)
        can_use_reference_images = self._is_veo31() and has_char_ref_images

        # Reference mode + Veo 3.1: use referenceImages API (no image/firstFrame)
        # Reference mode + non-3.1: fall back to image-to-video (initialImage as firstFrame)
        # Keyframe mode: always use image + optional lastFrame
        if is_reference and can_use_reference_images:
            return await self._generate_with_reference_images(params, duration_seconds, aspect_ratio)

        # image-to-video mode
        if not is_keyframe and not is_reference:
            raise Exception("Veo requires an image input (firstFrame or initialImage)")

        image_source = params.first_frame if is_keyframe else params.initial_image
        image_data = read_image_data(image_source)

        config_kwargs: dict = {
            "duration_seconds": duration_seconds,
            "aspect_ratio": aspect_ratio,
        }

        # lastFrame only supported by Veo 2.x and 3.1+, NOT Veo 3.0
        is_veo30 = "3.0" in self.model or "3-0" in self.model
        if is_keyframe and params.last_frame and not is_veo30:
            config_kwargs["last_frame"] = read_image_data(params.last_frame)

        mode_label = "keyframe" if is_keyframe else "image2video"
        logger.info(
            "[Veo] mode=%s, model=%s, duration=%ss, ratio=%s",
            mode_label, self.model, duration_seconds, aspect_ratio,
        )

        operation = await self.client.aio.models.generate_videos(
            model=self.model,
            prompt=params.prompt,
            image=image_data,
            config=types.GenerateVideosConfig(**config_kwargs),
        )

        return await self._finish_generation(operation)

    async def _generate_with_reference_images(
        self, params: VideoGenerateParams, duration_seconds: int, aspect_ratio: str
    ) -> VideoGenerateResult:
        initial_image = params.initial_image

        # Build reference images: scene frame + character refs (max 3 total)
        all_ref_paths = [initial_image, *(params.reference_images or [])][:3]
        reference_images = [
            types.VideoGenerationReferenceImage(
                image=read_image_data(img_path),
                reference_type="asset",
            )
            for img_path in all_ref_paths
        ]

        # referenceImages requires duration=8
        logger.info(
            "[Veo] mode=referenceImages, model=%s, refCount=%s, ratio=%s",
            self.model, len(reference_images), aspect_ratio,
        )

        operation = await self.client.aio.models.generate_videos(
            model=self.model,
            prompt=params.prompt,
            config=types.GenerateVideosConfig(
                duration_seconds=8,
                aspect_ratio=aspect_ratio,
                reference_images=reference_images,
            ),
        )

        return await self._finish_generation(operation)

    async def _finish_generation(self, operation) -> VideoGenerateResult:
        operation = await self._poll_for_result(operation)

        response = operation.response

        if (getattr(response, "rai_media_filtered_count", 0) or 0) > 0:
            reasons = getattr(response, "rai_media_filtered_reasons", None)
            raise Exception(f"Veo generation blocked by safety filter: {reasons}")

        generated = getattr(response, "generated_videos", None) if response else None
        if not generated or not generated[0]:
            raise Exception("No video returned from Veo")
        video_file = generated[0].video
        if not video_file:
            raise Exception("No video URI returned from Veo")

        directory = Path(self.upload_dir) / "videos"
        directory.mkdir(parents=True, exist_ok=True)
        download_path = directory / f"{new_id()}.mp4"

        # PORT NOTE: the google-genai Python SDK's files.download returns the file
        # bytes rather than writing to a path; write them out ourselves.
        data = await self.client.aio.files.download(file=video_file)
        download_path.write_bytes(data)

        logger.info("[Veo] Video saved to %s", download_path)
        return VideoGenerateResult(file_path=str(download_path))

    async def _poll_for_result(self, initial):
        max_attempts = 60
        operation = initial

        for i in range(max_attempts):
            logger.info("[Veo] Poll %s: done=%s", i + 1, operation.done)

            if operation.done:
                if operation.error:
                    raise Exception(f"Veo generation failed: {operation.error}")
                return operation

            await asyncio.sleep(10.0)
            operation = await self.client.aio.operations.get(operation)

        raise Exception("Veo generation timed out after 10 minutes")
