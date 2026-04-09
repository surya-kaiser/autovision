import shutil
import zipfile
import json
from pathlib import Path
from typing import Optional
import uuid

from app.utils.logger import get_logger

logger = get_logger(__name__)

from app.core.config import settings as _settings
UPLOAD_DIR = _settings.UPLOAD_DIR
MODEL_DIR = _settings.MODEL_DIR


def save_upload(file_bytes: bytes, filename: str, session_id: Optional[str] = None) -> Path:
    sid = session_id or str(uuid.uuid4())
    dest = UPLOAD_DIR / sid
    dest.mkdir(parents=True, exist_ok=True)
    fp = dest / filename
    fp.write_bytes(file_bytes)
    logger.info(f"Saved upload: {fp}")
    return fp


def extract_zip(zip_path: Path, dest_dir: Optional[Path] = None) -> Path:
    out = dest_dir or zip_path.parent / zip_path.stem
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out)
    logger.info(f"Extracted {zip_path} → {out}")
    return out


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def clean_session(session_id: str) -> None:
    p = UPLOAD_DIR / session_id
    if p.exists():
        shutil.rmtree(p)
        logger.info(f"Cleaned session {session_id}")
