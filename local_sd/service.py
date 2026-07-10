"""Local Stable Diffusion microservice for AI Comic Builder.

Loads an anime SDXL checkpoint once on the GPU (tuned for 8 GB VRAM) and serves
text-to-image over HTTP. The AIComicBuilder `local_diffusers` provider calls
POST /generate and saves the returned PNG under the app's uploads dir.

Run:  ~/anime-forge/.venv/bin/python ~/anime-forge/sd_service.py
Env:  AICB_SD_MODEL (HF id), AICB_SD_PORT (default 8500)
"""
from __future__ import annotations

import base64
import io
import os
import time
import threading

import torch
from fastapi import FastAPI
from pydantic import BaseModel

# Model candidates, tried in order. First that loads wins. sd-turbo is already
# cached from the proof run, so the service always comes up even offline.
MODEL_CANDIDATES = [
    os.environ.get("AICB_SD_MODEL") or "cagliostrolab/animagine-xl-3.1",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/sd-turbo",
]

_state = {"pipe": None, "model": None, "ready": False, "error": None, "loading": True}
_lock = threading.Lock()


def _load_model() -> None:
    from diffusers import AutoPipelineForText2Image

    for model_id in MODEL_CANDIDATES:
        try:
            print(f"[sd_service] loading {model_id} ...", flush=True)
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id, torch_dtype=torch.float16, use_safetensors=True
            )
            # 8 GB VRAM tuning: offload modules to CPU as needed + slice VAE.
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
            _state.update(pipe=pipe, model=model_id, ready=True, loading=False)
            print(f"[sd_service] READY on {model_id}", flush=True)
            return
        except Exception as e:  # try the next candidate
            print(f"[sd_service] {model_id} failed: {e}", flush=True)
            _state["error"] = f"{model_id}: {e}"
    _state["loading"] = False
    print("[sd_service] ERROR: no model could be loaded", flush=True)


app = FastAPI(title="AICB Local SD")


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_load_model, daemon=True).start()


class GenReq(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance: float = 6.0
    seed: int | None = None
    model: str | None = None  # accepted for parity; active model is the loaded one


@app.get("/health")
def health() -> dict:
    return {
        "ready": _state["ready"],
        "loading": _state["loading"],
        "model": _state["model"],
        "error": _state["error"],
        "cuda": torch.cuda.is_available(),
    }


@app.post("/generate")
def generate(req: GenReq) -> dict:
    if not _state["ready"]:
        return {"error": "model not ready", "loading": _state["loading"]}

    pipe = _state["pipe"]
    steps = req.steps
    # sd-turbo is a 1-step distilled model; clamp so the fallback still works.
    if _state["model"] and "turbo" in _state["model"]:
        steps, req.guidance = 1, 0.0

    generator = None
    if req.seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(req.seed))

    with _lock:  # single GPU: serialize requests
        t0 = time.time()
        image = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=steps,
            guidance_scale=req.guidance,
            generator=generator,
        ).images[0]
        torch.cuda.synchronize()
        elapsed = time.time() - t0

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"image_b64": b64, "model": _state["model"], "seconds": round(elapsed, 2)}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("AICB_SD_PORT", "8500"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
