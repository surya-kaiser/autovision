"""
Persistent metadata store.
Each session gets a metadata.json with dataset info and a list of all training runs.
Training logs are saved to models/{session_id}/{model_type}/train.log
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Session metadata ──────────────────────────────────────────────────────────

def _meta_path(session_id: str) -> Path:
    p = settings.UPLOAD_DIR / session_id / "metadata.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_session_meta(session_id: str) -> Dict[str, Any]:
    p = _meta_path(session_id)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"session_id": session_id, "created_at": datetime.now().isoformat(), "training_runs": []}


def save_session_meta(session_id: str, meta: Dict[str, Any]) -> None:
    _meta_path(session_id).write_text(json.dumps(meta, indent=2, default=str))


def save_upload_name(session_id: str, filename: str) -> None:
    """Store uploaded filename as initial dataset name (called right after upload)."""
    meta = load_session_meta(session_id)
    # Only set if not already preprocessed (don't overwrite richer metadata)
    if not meta.get("dataset", {}).get("task_type"):
        existing = meta.get("dataset", {})
        existing["name"] = filename
        meta["dataset"] = existing
    save_session_meta(session_id, meta)


def update_dataset_meta(
    session_id: str,
    dataset_name: str,
    format: str,
    task_type: str,
    num_samples: int,
    classes: List[str],
    preprocess_report: Optional[Dict[str, Any]] = None,
) -> None:
    meta = load_session_meta(session_id)
    meta["dataset"] = {
        "name": dataset_name,
        "format": format,
        "task_type": task_type,
        "num_samples": num_samples,
        "classes": classes,
        "preprocessed_at": datetime.now().isoformat(),
        "preprocess_report": preprocess_report or {},
    }
    save_session_meta(session_id, meta)


def record_training_run(
    session_id: str,
    model_type: str,
    task_type: str,
    metrics: Dict[str, Any],
    checkpoint_path: str,
    training_time_s: float,
    is_pilot: bool = False,
) -> None:
    meta = load_session_meta(session_id)
    runs: List[Dict[str, Any]] = meta.get("training_runs", [])

    # Update if same model already trained, else append
    existing = next((r for r in runs if r["model_type"] == model_type and r.get("pilot") == is_pilot), None)
    entry = {
        "model_type": model_type,
        "task_type": task_type,
        "metrics": metrics,
        "checkpoint_path": checkpoint_path,
        "training_time_s": round(training_time_s, 2),
        "pilot": is_pilot,
        "trained_at": datetime.now().isoformat(),
    }
    if existing:
        runs[runs.index(existing)] = entry
    else:
        runs.append(entry)

    meta["training_runs"] = runs
    save_session_meta(session_id, meta)


def get_all_sessions() -> List[Dict[str, Any]]:
    """Return summary of all sessions."""
    sessions = []
    if not settings.UPLOAD_DIR.exists():
        return sessions
    for sid_dir in sorted(settings.UPLOAD_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not sid_dir.is_dir():
            continue
        meta_file = sid_dir / "metadata.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                display_name = meta.get("dataset", {}).get("name", "unknown")
                # Fall back to .meta.json rename if still unknown
                if display_name in ("unknown", "", None):
                    dot_meta = sid_dir / ".meta.json"
                    if dot_meta.exists():
                        try:
                            dm = json.loads(dot_meta.read_text())
                            display_name = dm.get("dataset_name", "unknown")
                        except Exception:
                            pass
                sessions.append({
                    "session_id": sid_dir.name,
                    "dataset": display_name,
                    "task_type": meta.get("dataset", {}).get("task_type", "unknown"),
                    "num_samples": meta.get("dataset", {}).get("num_samples", 0),
                    "num_training_runs": len(meta.get("training_runs", [])),
                    "created_at": meta.get("created_at", ""),
                })
            except Exception:
                pass
    return sessions


# ── Per-model training log files ──────────────────────────────────────────────

def get_training_log_path(session_id: str, model_type: str) -> Path:
    log_dir = settings.MODEL_DIR / session_id / model_type
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "train.log"


def append_training_log(session_id: str, model_type: str, message: str) -> None:
    log_path = get_training_log_path(session_id, model_type)
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")
    except Exception as e:
        logger.warning(f"Could not write training log: {e}")


def read_training_log(session_id: str, model_type: str) -> str:
    log_path = get_training_log_path(session_id, model_type)
    if log_path.exists():
        return log_path.read_text(encoding="utf-8")
    return ""


def get_session_summary(session_id: str) -> Dict[str, Any]:
    """Full session summary — dataset info + all training runs + log snippets."""
    meta = load_session_meta(session_id)
    runs = meta.get("training_runs", [])

    # Attach log tail to each run
    for run in runs:
        mt = run.get("model_type", "")
        log = read_training_log(session_id, mt)
        run["log_tail"] = log[-2000:] if len(log) > 2000 else log  # last 2000 chars

    return {
        "session_id": session_id,
        "dataset": meta.get("dataset", {}),
        "training_runs": runs,
        "created_at": meta.get("created_at", ""),
    }
