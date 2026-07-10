"""Local GPU image generation — capability detection + on/off toggle.

Feature: if the user's machine has a capable GPU, they can generate images for
free on their own graphics card (via the optional diffusers SD service in
anime-forge/sd_service.py) instead of paying a cloud provider.

This module:
- detects the GPU (nvidia-smi) and whether the local SD service is reachable/ready,
- decides whether local generation is "recommended" (NVIDIA GPU >= MIN_VRAM_MB),
- persists a simple on/off toggle so the choice survives restarts.

The app's `generate_image` consults `local_images_enabled()`; when on and the
service is ready, images route to the GPU, otherwise it falls back to the
configured cloud provider.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import httpx

from app.config import settings

MIN_VRAM_MB = 6000  # SDXL is comfortable from ~6 GB with offload
SD_URL = (os.environ.get("AICB_SD_URL") or "http://127.0.0.1:8500").rstrip("/")
_STATE_FILE = Path(settings.UPLOAD_DIR).parent / "local_gpu.json"


def detect_gpu() -> Optional[dict]:
    """Return {name, vram_mb, driver} for the first NVIDIA GPU, or None."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        name, vram, driver = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        return {"name": name, "vram_mb": int(float(vram)), "driver": driver}
    except Exception:
        return None


def service_health() -> dict:
    """Ping the local SD service. Returns {reachable, ready, model, ...}."""
    try:
        r = httpx.get(f"{SD_URL}/health", timeout=3.0)
        if r.status_code == 200:
            data = r.json()
            data["reachable"] = True
            return data
    except Exception:
        pass
    return {"reachable": False, "ready": False, "model": None}


def _read_toggle() -> bool:
    # Env override always wins (useful for headless/CI).
    if os.environ.get("AICB_IMAGE_PROVIDER", "").lower() == "local":
        return True
    try:
        return bool(json.loads(_STATE_FILE.read_text()).get("enabled", False))
    except Exception:
        return False


def set_local_images_enabled(enabled: bool) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps({"enabled": bool(enabled)}))


def local_images_enabled() -> bool:
    """True only when the toggle is on AND the SD service is ready to serve."""
    if not _read_toggle():
        return False
    return service_health().get("ready", False)


def status() -> dict:
    """Full capability report for the Settings UI."""
    gpu = detect_gpu()
    health = service_health()
    recommended = bool(gpu and gpu["vram_mb"] >= MIN_VRAM_MB)
    return {
        "gpu": gpu,
        "recommended": recommended,
        "min_vram_mb": MIN_VRAM_MB,
        "service": {"url": SD_URL, **health},
        "enabled": _read_toggle(),
        "active": _read_toggle() and health.get("ready", False),
    }
