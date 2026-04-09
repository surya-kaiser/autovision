"""
Dataset upload, preview, and preprocessing endpoints.
"""
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.core.llm_engine import get_recommendation
from app.models.schemas import (
    APIResponse, DatasetInfo, PreprocessConfig, TaskType
)
from app.services.preprocessor import (
    detect_format, detect_task_type, preprocess_dataset, get_dataset_summary as _get_summary
)
from app.services.metadata_store import update_dataset_meta, save_upload_name, get_all_sessions, get_session_summary
from app.utils.file_handler import save_upload, extract_zip
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/dataset", tags=["dataset"])


def _locate_dataset(session_dir: Path):
    """Return (dataset_path, is_csv) for a session directory."""
    csvs = list(session_dir.rglob("*.csv"))
    if csvs:
        return csvs[0], True
    return session_dir, False


@router.post("/upload", response_model=APIResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    """Upload a CSV, ZIP, or image file."""
    sid = session_id or str(uuid.uuid4())
    filename = file.filename or "upload"

    try:
        content = await file.read()
        saved_path = save_upload(content, filename, session_id=sid)

        # Auto-extract ZIP
        if saved_path.suffix.lower() == ".zip":
            extracted = extract_zip(saved_path)
            dataset_path = extracted
        else:
            dataset_path = saved_path

        fmt = detect_format(dataset_path)
        task = detect_task_type(fmt, dataset_path)

        # Get sample/class counts immediately so the UI can display them
        from app.services.preprocessor import get_dataset_summary as _get_summary
        try:
            summary = _get_summary(dataset_path, sid)
            num_samples = summary.get("num_samples", 0)
            num_classes = summary.get("num_classes", 0)
            class_names = list(summary.get("class_distribution", {}).keys())
        except Exception:
            num_samples = 0
            num_classes = 0
            class_names = []

        # Persist upload name immediately so sessions are never "unknown"
        clean_name = Path(filename).stem.replace("_", " ").replace("-", " ").title()
        save_upload_name(sid, clean_name)

        return APIResponse(
            status="success",
            data={
                "session_id": sid,
                "filename": filename,
                "format": fmt.value,
                "task_type": task.value,
                "num_samples": num_samples,
                "num_classes": num_classes,
                "class_names": class_names,
                "path": str(dataset_path),
            },
            message=f"Uploaded {filename} as {fmt.value}",
        )
    except Exception as exc:
        logger.error(f"Upload error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/preview/{session_id}", response_model=APIResponse)
async def preview_dataset(session_id: str):
    """Return a preview of the uploaded dataset."""
    session_dir = settings.UPLOAD_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        dataset_path, is_csv = _locate_dataset(session_dir)
        if is_csv:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            preview = {
                "type": "csv",
                "rows": len(df),
                "columns": list(df.columns),
                "sample": df.head(10).to_dict(orient="records"),
                "dtypes": {k: str(v) for k, v in df.dtypes.items()},
                "missing": df.isna().sum().to_dict(),
            }
        else:
            # Check for class subdirs (for structured image datasets)
            from app.services.preprocessor import _find_images, ImagePreprocessor
            processor = ImagePreprocessor(PreprocessConfig(session_id=session_id))
            root = processor._find_dataset_root(session_dir)
            class_dirs = sorted([d for d in root.iterdir() if d.is_dir()]) if root.is_dir() else []

            if class_dirs:
                class_counts = {}
                sample_names = []
                for cls_dir in class_dirs:
                    imgs = _find_images(cls_dir)
                    class_counts[cls_dir.name] = len(imgs)
                    sample_names.extend([f"{cls_dir.name}/{p.name}" for p in imgs[:3]])
                preview = {
                    "type": "images",
                    "count": sum(class_counts.values()),
                    "classes": list(class_counts.keys()),
                    "class_counts": class_counts,
                    "sample_names": sample_names[:20],
                }
            else:
                images = _find_images(session_dir)
                preview = {
                    "type": "images",
                    "count": len(images),
                    "sample_names": [p.name for p in images[:20]],
                }

        return APIResponse(status="success", data=preview, message="Preview ready")
    except Exception as exc:
        logger.error(f"Preview error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/preprocess", response_model=APIResponse)
async def preprocess(config: PreprocessConfig):
    """Run the full preprocessing pipeline on an uploaded dataset."""
    session_dir = settings.UPLOAD_DIR / config.session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        dataset_path, _ = _locate_dataset(session_dir)
        # Pass user-selected task type as hint if provided
        task_hint = None
        if config.task_type_hint:
            try:
                task_hint = TaskType(config.task_type_hint)
            except ValueError:
                pass
        info, report = preprocess_dataset(dataset_path, config, task_type_hint=task_hint)

        # Persist dataset metadata for this session
        update_dataset_meta(
            session_id=config.session_id,
            dataset_name=dataset_path.name,
            format=info.format.value,
            task_type=info.task_type.value,
            num_samples=info.num_samples,
            classes=info.class_names,
            preprocess_report=report.model_dump(),
        )

        return APIResponse(
            status="success",
            data={"info": info.model_dump(), "report": report.model_dump()},
            message="Preprocessing complete",
        )
    except Exception as exc:
        logger.error(f"Preprocessing error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/recommend/{session_id}", response_model=APIResponse)
async def recommend(session_id: str, task_type: Optional[str] = None):
    """Get LLM model recommendation based on dataset summary."""
    session_dir = settings.UPLOAD_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        dataset_path, _ = _locate_dataset(session_dir)
        summary = _get_summary(dataset_path, session_id)

        if task_type:
            tt = TaskType(task_type)
        else:
            fmt = detect_format(dataset_path)
            tt = detect_task_type(fmt, dataset_path)

        recommendation = get_recommendation(tt, summary)
        return APIResponse(
            status="success",
            data=recommendation.model_dump(),
            message="Recommendation ready",
        )
    except Exception as exc:
        logger.error(f"Recommendation error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


class DetectRequest(BaseModel):
    session_id: str


@router.post("/detect", response_model=APIResponse)
async def detect_dataset(body: DetectRequest):
    """Detect format and task type for an already-uploaded session."""
    session_dir = settings.UPLOAD_DIR / body.session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        dataset_path, _ = _locate_dataset(session_dir)
        fmt = detect_format(dataset_path)
        task = detect_task_type(fmt, dataset_path)
        return APIResponse(
            status="success",
            data={"session_id": body.session_id, "format": fmt.value, "task_type": task.value},
            message="",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/upload-folder", response_model=APIResponse)
async def upload_folder(
    files: List[UploadFile] = File(...),
    session_id: str = Form(...),
):
    """
    Accept a batch of files from a folder upload (webkitdirectory).
    Each file's filename carries the relative path (e.g. 'cats/img001.jpg').
    Files are saved preserving that relative path under the session directory.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    saved = 0
    errors = []
    for upload in files:
        rel_path = upload.filename or "unknown"
        # Sanitize: strip leading slashes / drive letters
        rel_path = rel_path.lstrip("/\\").replace("\\", "/")
        dest = settings.UPLOAD_DIR / session_id / rel_path
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            content = await upload.read()
            dest.write_bytes(content)
            saved += 1
        except Exception as e:
            errors.append(f"{rel_path}: {e}")

    if errors:
        logger.warning(f"Folder upload errors for {session_id}: {errors[:5]}")

    return APIResponse(
        status="success",
        data={"session_id": session_id, "files_saved": saved, "errors": len(errors)},
        message=f"Saved {saved} files",
    )


@router.get("/sessions", response_model=APIResponse)
async def list_sessions():
    """List all dataset sessions with their training history."""
    sessions = get_all_sessions()
    return APIResponse(status="success", data=sessions, message=f"{len(sessions)} session(s)")


@router.get("/session/{session_id}/summary", response_model=APIResponse)
async def session_summary(session_id: str):
    """Full session detail: dataset info + all training runs + log tails."""
    summary = get_session_summary(session_id)
    return APIResponse(status="success", data=summary, message="")


@router.delete("/{session_id}", response_model=APIResponse)
async def delete_session(session_id: str):
    """Delete an uploaded dataset session and all its files."""
    import shutil
    session_dir = settings.UPLOAD_DIR / session_id
    model_dir = settings.MODEL_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        shutil.rmtree(session_dir, ignore_errors=True)
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
        logger.info(f"Deleted session {session_id}")
        return APIResponse(status="success", data={"session_id": session_id}, message="Session deleted")
    except Exception as exc:
        logger.error(f"Delete error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


class RenameRequest(BaseModel):
    dataset_name: str


@router.post("/{session_id}/rename", response_model=APIResponse)
async def rename_dataset(session_id: str, body: RenameRequest):
    """Set a human-readable name for a session."""
    session_dir = settings.UPLOAD_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        meta_path = session_dir / ".meta.json"
        meta = {}
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text())
        meta["dataset_name"] = body.dataset_name
        import json
        meta_path.write_text(json.dumps(meta, indent=2))
        return APIResponse(status="success", data={"dataset_name": body.dataset_name}, message="Renamed")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
