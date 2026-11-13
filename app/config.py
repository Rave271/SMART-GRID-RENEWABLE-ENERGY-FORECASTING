"""Configuration management for Smart Grid Renewable Energy Forecasting."""

import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path
from dotenv import load_dotenv
from fastapi import Header, HTTPException, Request

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FRONTEND_DIR = BASE_DIR / "frontend"

for folder in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'smart_grid.db'}")

# External APIs
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
ALLOWED_CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(",")
    if origin.strip()
]

# API authentication is opt-in for local development and required when enabled.
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() in ("true", "1", "yes")
API_KEY = os.getenv("API_KEY", "").strip()
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
RATE_LIMIT_UPLOAD = int(os.getenv("RATE_LIMIT_UPLOAD", "5"))
RATE_LIMIT_TRAIN = int(os.getenv("RATE_LIMIT_TRAIN", "3"))
RATE_LIMIT_REPORT = int(os.getenv("RATE_LIMIT_REPORT", "10"))
RATE_LIMIT_PREDICT = int(os.getenv("RATE_LIMIT_PREDICT", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

_rate_limit_events: dict[str, deque[float]] = defaultdict(deque)

# App settings
APP_HOST = os.getenv("HOST", "127.0.0.1")
APP_PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

# Default Sample Data Path
DEFAULT_SAMPLE_PATH = SAMPLE_DATA_DIR / "smart_grid_sample.csv"


def validate_auth_config() -> None:
    """Fail startup when authentication is enabled without a configured key."""
    if AUTH_ENABLED and not API_KEY:
        raise RuntimeError("AUTH_ENABLED is true but API_KEY is not configured.")


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Validate the configured API key for protected routes."""
    if not AUTH_ENABLED:
        return
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key is required.")
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key.")


def rate_limit(limit: int, scope: str):
    """Create an in-memory IP limiter for a route group."""
    async def dependency(request: Request) -> None:
        client_ip = request.client.host if request.client else "unknown"
        key = f"{scope}:{client_ip}"
        now = time.monotonic()
        events = _rate_limit_events[key]
        cutoff = now - RATE_LIMIT_WINDOW_SECONDS
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
        events.append(now)

    return dependency


rate_limit_upload = rate_limit(RATE_LIMIT_UPLOAD, "upload")
rate_limit_train = rate_limit(RATE_LIMIT_TRAIN, "train")
rate_limit_report = rate_limit(RATE_LIMIT_REPORT, "report")
rate_limit_predict = rate_limit(RATE_LIMIT_PREDICT, "predict")
