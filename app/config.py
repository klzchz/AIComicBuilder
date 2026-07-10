"""Application configuration — port of env handling (.env.example)."""
import os
from pathlib import Path

# Load .env if python-dotenv is available (optional).
try:  # pragma: no cover
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _db_path() -> str:
    # DATABASE_URL=file:./data/aicomic.db  (mirrors src/lib/db/index.ts)
    raw = os.environ.get("DATABASE_URL", "file:./data/aicomic.db")
    return str(Path(raw.replace("file:", "")).resolve())


class Settings:
    DATABASE_PATH: str = _db_path()
    DATABASE_URL: str = f"sqlite:///{_db_path()}"
    UPLOAD_DIR: str = str(Path(os.environ.get("UPLOAD_DIR", "./uploads")).resolve())
    HOST: str = os.environ.get("HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("PORT", "3000"))
    ENV: str = os.environ.get("NODE_ENV", os.environ.get("APP_ENV", "development"))

    # AI provider keys (configured at runtime via settings UI too; env is fallback)
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "")
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
    SEEDANCE_API_KEY: str = os.environ.get("SEEDANCE_API_KEY", "")
    KLING_ACCESS_KEY: str = os.environ.get("KLING_ACCESS_KEY", "")
    KLING_SECRET_KEY: str = os.environ.get("KLING_SECRET_KEY", "")
    DASHSCOPE_API_KEY: str = os.environ.get("DASHSCOPE_API_KEY", "")


settings = Settings()

# Ensure runtime dirs exist.
Path(settings.DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
