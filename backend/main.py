"""
FastAPI bridge for the CS329 Financial Report Analyzer.

This module DOES NOT modify any existing Python file. It only imports the
public functions exported by the original modules and exposes them over a
typed REST API so the React frontend can consume them.

Run (from the project root):
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import sys
from io import UnsupportedOperation
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load project-root .env so ANTHROPIC_API_KEY etc. are available to the
# hybrid LLM engine without requiring the developer to export it manually.
try:
    from dotenv import load_dotenv  # noqa: WPS433

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def _ensure_utf8_stdio() -> None:
    """Use UTF-8 for stdout/stderr when the host allows it.

    On Windows the legacy pipeline prints Unicode (e.g. ✓, →) and SEC company
    titles; the default console encoding (cp1252) raises UnicodeEncodeError
    during ``print()`` and makes ``Run pipeline`` fail from the React UI."""
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError, AttributeError, UnsupportedOperation):
                pass


_ensure_utf8_stdio()

from backend.routers import evaluation, pipeline, sec, signals  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backend.main")

app = FastAPI(
    title="CS329 Financial Report Analyzer API",
    description=(
        "REST bridge over the existing Python backend (signal engine, "
        "scraping pipeline, SEC lookups, PhraseBank evaluation). "
        "Consumed by the React frontend in /frontend."
    ),
    version="0.1.0",
)

# CORS: permissive for local dev; restrict in production.
_DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEV_ORIGINS,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["meta"])
def health() -> dict:
    """Cheap liveness probe used by the frontend when it boots."""
    return {
        "status": "ok",
        "service": "cs329-financial-analyzer-api",
        "project_root": str(PROJECT_ROOT),
    }


app.include_router(signals.router, prefix="/api/signals", tags=["signals"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(sec.router, prefix="/api/sec", tags=["sec"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["evaluation"])
