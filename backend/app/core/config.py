import tempfile
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List

# Platform-safe base dir: C:\Users\{user}\AppData\Local\Temp\autovision on Windows
_BASE = Path(tempfile.gettempdir()) / "autovision"


class Settings(BaseSettings):
    APP_NAME: str = "AutoVision MLOps"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    UPLOAD_DIR: Path = _BASE / "uploads"
    MODEL_DIR: Path = _BASE / "models"
    LOG_DIR: Path = _BASE / "logs"

    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173", "http://localhost:80"]

    # Ollama settings (free local LLM)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # Redis (optional for WebSocket pub/sub)
    REDIS_URL: str = "redis://localhost:6379"

    # Training defaults
    DEFAULT_EPOCHS: int = 10
    PILOT_EPOCHS: int = 3
    EARLY_STOPPING_PATIENCE: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Ensure dirs exist
for _d in (settings.UPLOAD_DIR, settings.MODEL_DIR, settings.LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
