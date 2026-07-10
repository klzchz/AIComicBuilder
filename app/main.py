"""FastAPI entrypoint — Python port of the Next.js app.

Replaces Next.js App Router: FastAPI serves the JSON API (app/api/*) and the
server-rendered UI (app/web). Bootstrap mirrors src/lib/bootstrap.ts:
run migrations, init AI providers, register pipeline handlers, start worker.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.session import run_migrations


def _bootstrap() -> None:
    print("[Bootstrap] Running database migrations...")
    run_migrations()

    print("[Bootstrap] Initializing AI providers...")
    try:
        from app.ai.setup import initialize_providers

        initialize_providers()
    except Exception as e:  # pragma: no cover
        print(f"[Bootstrap] AI providers init skipped: {e}")

    print("[Bootstrap] Registering pipeline handlers...")
    try:
        from app.pipeline import register_pipeline_handlers

        register_pipeline_handlers()
    except Exception as e:  # pragma: no cover
        print(f"[Bootstrap] Pipeline handlers skipped: {e}")

    print("[Bootstrap] Starting task worker...")
    try:
        from app.task_queue import start_worker

        start_worker()
    except Exception as e:  # pragma: no cover
        print(f"[Bootstrap] Worker start skipped: {e}")

    print("[Bootstrap] Ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _bootstrap()
    yield


app = FastAPI(title="AI Comic Builder", version="0.3.0-py", lifespan=lifespan)


# Serve uploaded assets (mirrors /api/uploads/[...path] + /uploads).
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR, check_dir=False), name="uploads")


def _include_routers() -> None:
    """Wire API routers + web UI. Imported lazily so a partial subsystem
    doesn't break boot; each include is best-effort and logged."""
    # JSON API
    try:
        from app.api import api_router

        app.include_router(api_router, prefix="/api")
    except Exception as e:  # pragma: no cover
        print(f"[Routers] API router skipped: {e}")

    # Server-rendered UI
    try:
        from app.web import web_router, mount_static

        mount_static(app)
        app.include_router(web_router)
    except Exception as e:  # pragma: no cover
        print(f"[Routers] Web router skipped: {e}")


_include_routers()


@app.get("/healthz")
def healthz():
    return {"ok": True, "app": "ai-comic-builder", "runtime": "python"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=False)
