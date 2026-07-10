# Local GPU image generation (free)

If your machine has a capable NVIDIA GPU, you can generate images **for free on
your own graphics card** instead of paying a cloud provider. This runs an anime
SDXL checkpoint locally via [diffusers](https://github.com/huggingface/diffusers).

**Recommended:** a recent NVIDIA GPU with **6 GB+ VRAM** (SDXL runs comfortably
from ~6 GB with model offloading; tested on an 8 GB RTX 5050 Laptop GPU).

## Setup

```bash
# 1) Create a venv and install torch for your CUDA (example: CUDA 12.8)
python3 -m venv .venv-gpu && source .venv-gpu/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) Install the service deps
pip install -r local_sd/requirements-gpu.txt

# 3) Run the service (downloads the model on first start, ~6.5 GB)
python local_sd/service.py
```

The service listens on `http://127.0.0.1:8500`. Pick a different model or port:

```bash
AICB_SD_MODEL=cagliostrolab/animagine-xl-3.1 AICB_SD_PORT=8500 python local_sd/service.py
```

## Enable it in the app

1. Start the SD service (above) and the app (`uvicorn app.main:app`).
2. Open **Settings** — the "Local GPU Generation" card auto-detects your GPU and
   the service. Flip the toggle on.
3. Image generation now runs on your GPU at no cost. Text still uses your
   configured cloud provider.

You can also force it headless with `AICB_IMAGE_PROVIDER=local` (and optionally
`AICB_SD_URL` / `AICB_SD_MODEL`) in the app's environment.

## How it works

- `local_sd/service.py` — FastAPI service; loads the model once, serves
  `POST /generate` and `GET /health`. Tuned for 8 GB VRAM (fp16 + model CPU
  offload + VAE slicing/tiling).
- `app/ai/providers/local_diffusers.py` — the app-side provider; posts prompts
  to the service and saves the PNG under `uploads/`.
- `app/ai/local_gpu.py` — GPU detection (`nvidia-smi`) + on/off toggle.
- The model falls back gracefully: anime SDXL → SDXL base → sd-turbo, so the
  service always starts even offline (once a model is cached).
