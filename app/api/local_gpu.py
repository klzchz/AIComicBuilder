"""Local GPU generation API — capability status + on/off toggle.

- GET  /local-gpu/status  -> detected GPU, SD service health, whether enabled.
- POST /local-gpu/toggle  -> { enabled: bool } to turn local generation on/off.

Lets the Settings UI show "You have a capable GPU — generate for free on your
graphics card" and flip it on.
"""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/local-gpu/status")
def local_gpu_status() -> dict:
    from app.ai.local_gpu import status

    return status()


@router.post("/local-gpu/toggle")
async def local_gpu_toggle(request: Request) -> dict:
    from app.ai.local_gpu import set_local_images_enabled, status

    body = await request.json()
    set_local_images_enabled(bool(body.get("enabled", False)))
    return status()
