"""AI type contracts — Python port of src/lib/ai/types.ts.

These are the stable interfaces every provider and the pipeline rely on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Optional


@dataclass
class TextOptions:
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    images: list[str] = field(default_factory=list)  # local file paths for vision input


@dataclass
class ImageOptions:
    model: Optional[str] = None
    size: Optional[str] = None
    aspect_ratio: Optional[str] = None
    quality: Optional[str] = None
    reference_images: list[str] = field(default_factory=list)
    # Labels for reference images (e.g. character names); must match order.
    reference_labels: list[str] = field(default_factory=list)


@dataclass
class VideoGenerateParams:
    prompt: str
    duration: int
    ratio: str
    # Keyframe mode: both first_frame and last_frame set.
    first_frame: Optional[str] = None
    last_frame: Optional[str] = None
    # Reference-image mode: a single initial image (local path or http URL).
    initial_image: Optional[str] = None
    # Character/style reference images for consistency (e.g. Veo 3.1).
    reference_images: list[str] = field(default_factory=list)


@dataclass
class VideoGenerateResult:
    file_path: str
    last_frame_url: Optional[str] = None


class AIProvider(Protocol):
    async def generate_text(self, prompt: str, options: Optional[TextOptions] = None) -> str: ...

    async def generate_image(self, prompt: str, options: Optional[ImageOptions] = None) -> str: ...


class VideoProvider(Protocol):
    async def generate_video(self, params: VideoGenerateParams) -> VideoGenerateResult: ...
