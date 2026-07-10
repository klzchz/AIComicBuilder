# AI Comic Builder (Python)

AI-driven anime/comic generator — a fully automated pipeline from script to animated video.

> Python rewrite (v0.3.0-py). The original Next.js/TypeScript app is preserved under [`legacy-nextjs/`](./legacy-nextjs).

## Features

- **Script import** — upload TXT/DOCX/PDF; AI parses the text, extracts characters, and splits it into episodes.
- **Episode management** — project-level episode list, characters associated per episode, manual or auto split.
- **Character management** — main/guest partitions, cross-episode reuse, per-episode parsing.
- **Script authoring** — write manually or generate with AI assistance.
- **Character extraction** — AI extracts characters from the script and writes detailed visual descriptions.
- **Character four-view sheet** — a front / three-quarter / profile / back turnaround reference per character to keep later frames consistent.
- **Smart storyboard** — AI breaks the script into a professional shot list (composition, lighting, camera direction).
- **Keyframe generation** — first/last frame per shot (keyframe mode) or scene reference frame (reference mode).
- **Video prompts** — AI writes a video prompt per shot from the storyboard and reference frames; editable.
- **Video generation** — animated clips generated per shot (keyframe interpolation or reference).
- **Video assembly** — concatenate all clips into a full animation with burned-in subtitles and BGM.
- **Storyboard workflow** — list view and kanban view (auto-columned by generation progress), per-shot editing, version control.
- **Multi-model** — OpenAI, Gemini, Kling, Seedance, Veo, Wan, DashScope, UCloud Seedance — configurable per project.
- **Local GPU generation (free)** — if you have a capable NVIDIA GPU, generate images for free on your own graphics card instead of a cloud provider. Auto-detected in Settings. See [`local_sd/`](./local_sd).

## Tech stack

| Layer | Technology |
|------|------|
| Framework | FastAPI (ASGI) |
| UI | Server-rendered Jinja2 + HTMX + Tailwind (CDN), English |
| Database | SQLite + SQLAlchemy 2.0 |
| AI text | OpenAI / Gemini (via SDK / httpx) |
| AI image | OpenAI / Gemini Imagen / Kling / DashScope |
| AI video | Seedance / Kling / Veo / Wan |
| Video processing | system FFmpeg |
| Background jobs | DB-backed task queue (daemon worker thread) |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# system ffmpeg is required for video assembly
#   Debian/Ubuntu: sudo apt install ffmpeg   |   macOS: brew install ffmpeg

cp .env.example .env          # optional: set AI provider keys
uvicorn app.main:app --host 0.0.0.0 --port 3000
```

Open http://localhost:3000 and configure your AI providers in **Settings**. The database schema is created automatically on first boot.

## Docker

```bash
docker build -t ai-comic-builder .
docker run -d --name ai-comic-builder -p 3000:3000 \
  -v ./data:/app/data -v ./uploads:/app/uploads \
  ai-comic-builder
```

- `./data` — SQLite database file
- `./uploads` — uploaded files and generated assets (images, videos)

## Generation pipeline

```
Script input -> Script parse -> Character extract -> Character four-view
                                                          |
                                                     Smart storyboard
                                                          |
                                        Reference / first-last frames (per shot)
                                                          |
                                             Video prompt (per shot)
                                                          |
                                             Video generation (per shot)
                                                          |
                                             Video assembly + subtitles
```

Each stage can be triggered individually or in batch. Long-running image/video jobs run on the background task queue with retries; text stages run inline.

## Project structure

```
app/
├── main.py            # FastAPI entrypoint + bootstrap (migrations, providers, worker)
├── config.py          # env/config
├── core/ids.py        # short unique ids
├── db/                # SQLAlchemy models + session (SQLite, WAL)
├── ai/                # providers, prompt builders, provider factory, agent caller
│   ├── providers/     # openai, gemini, seedance, kling, veo, wan, dashscope, ucloud
│   └── prompts/       # prompt registry + builders
├── pipeline/          # the 8 generation stages + inline dispatcher + checks
├── video/ffmpeg.py    # FFmpeg wrapper (concat, subtitles, BGM, probe)
├── task_queue/        # DB-backed queue + daemon worker
├── api/               # FastAPI routers (mirrors the original REST API)
└── web/               # Jinja2 templates + HTMX pages
```

## Data model

- **Project** / **Episode** — script, status, generation mode
- **Character** — description, four-view reference, costumes, relations
- **Scene** / **Shot** — shot list with composition, camera, transitions
- **ShotAsset** — versioned per-shot artifacts (frames + video), active/history
- **Dialogue** — per-shot lines
- **Task** — background job queue

## License

[Apache License 2.0](./LICENSE)

---

Ported from the original Next.js/TypeScript project (forked from LingyiChen-AI/AIComicBuilder). The TypeScript app remains available under `legacy-nextjs/` for reference.
