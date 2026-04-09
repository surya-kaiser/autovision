"""
Inference endpoints — run predictions on uploaded data.
"""
import base64
import io
import json
import pickle
import uuid
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.models.schemas import APIResponse, PredictionResult, TaskType
from app.services.yolo_service import predict_yolo
from app.utils.file_handler import save_upload
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/inference", tags=["inference"])

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}

# PyTorch DL model types (saved as .pt, not .pkl)
_DL_MODEL_TYPES = {"cnn", "resnet", "unet", "deeplabv3"}
_DL_SEG_MODEL_TYPES = {"unet", "deeplabv3"}


def _is_dl_model(model_type: str) -> bool:
    return model_type in _DL_MODEL_TYPES


def _is_dl_seg_model(model_type: str) -> bool:
    return model_type in _DL_SEG_MODEL_TYPES


def _load_model(session_id: str, model_type: str):
    """Load a trained sklearn model. Checks both new and legacy checkpoint paths."""
    # New layout: models/{session_id}/{model_type}/best.pkl
    ckpt = settings.MODEL_DIR / session_id / model_type / "best.pkl"
    if not ckpt.exists():
        # Legacy flat layout
        ckpt = settings.MODEL_DIR / session_id / f"{model_type}_best.pkl"
    if not ckpt.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for '{model_type}'. Train a model first.",
        )
    with open(ckpt, "rb") as f:
        return pickle.load(f)


def _load_class_info(session_id: str):
    """Load class names from preprocessing manifest."""
    for fname in ("class_info.json", "split_manifest.json"):
        p = settings.UPLOAD_DIR / session_id / "preprocessed" / fname
        if p.exists():
            data = json.loads(p.read_text())
            return data.get("classes", [])
    return []


@router.post("/predict", response_model=APIResponse)
async def predict(
    session_id: str = Form(...),
    model_type: str = Form(...),
    task_type: str = Form(...),
    file: Optional[UploadFile] = File(None),
    data_json: Optional[str] = Form(None),
):
    """Run inference on an uploaded image or tabular data."""
    try:
        tt = TaskType(task_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown task type: {task_type}")

    result = PredictionResult()

    # ── Object detection (YOLO) ───────────────────────────────────────────────
    if tt == TaskType.OBJECT_DETECTION:
        if file is None:
            raise HTTPException(status_code=400, detail="Image file required for object detection")
        content = await file.read()
        tmp_path = save_upload(content, file.filename or "image.jpg", session_id=str(uuid.uuid4()))
        boxes = predict_yolo(session_id, tmp_path, model_type=model_type)
        result.bounding_boxes = boxes
        return APIResponse(status="success", data=result.model_dump(), message=f"Detected {len(boxes)} object(s)")

    # ── Segmentation ────────────────────────────────────────────────────────────
    if tt == TaskType.SEGMENTATION:
        if file is None:
            raise HTTPException(status_code=400, detail="Image file required for segmentation")
        content = await file.read()

        # ── PyTorch segmentation (U-Net / DeepLabV3) ─────────────────────────
        if _is_dl_seg_model(model_type):
            try:
                from app.services.dl_trainer import load_dl_checkpoint, predict_segmentation
                ckpt = load_dl_checkpoint(session_id, model_type)
                if ckpt is None:
                    raise HTTPException(status_code=404, detail=f"No trained {model_type} model found")
                pred = predict_segmentation(ckpt, content)
                result.segmented_image = pred.get("segmented_image")
                result.label = pred.get("label")
                return APIResponse(status="success", data=result.model_dump(), message="Segmentation complete")
            except ImportError:
                raise HTTPException(status_code=500, detail="PyTorch not installed. Run: pip install torch torchvision")
            except HTTPException:
                raise
            except Exception as exc:
                logger.error(f"DL segmentation error: {exc}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(exc))

        # ── YOLO-seg or pre-trained yolov8n-seg ─────────────────────────────
        tmp_path = save_upload(content, file.filename or "image.jpg", session_id=str(uuid.uuid4()))
        try:
            from ultralytics import YOLO
            from PIL import Image as _PIL

            trained_pt = settings.MODEL_DIR / session_id / model_type / "train" / "weights" / "best.pt"
            seg_model = None

            if trained_pt.exists():
                try:
                    seg_model = YOLO(str(trained_pt))
                except Exception as e:
                    logger.warning(f"Failed to load trained model: {e}")

            if seg_model is None:
                seg_weights = model_type if model_type.endswith("-seg") else "yolov8n-seg"
                seg_model = YOLO(f"{seg_weights}.pt")

            seg_results = seg_model.predict(str(tmp_path), conf=0.25, verbose=False)
            if seg_results and seg_results[0] is not None:
                annotated_bgr = seg_results[0].plot()
                annotated_rgb = annotated_bgr[..., ::-1]
                img = _PIL.fromarray(annotated_rgb)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                result.segmented_image = f"data:image/png;base64,{b64}"
                masks_count = len(seg_results[0].masks) if seg_results[0].masks is not None else 0
                result.label = f"{masks_count} segment(s) detected"
            else:
                result.label = "No segments detected"

            return APIResponse(status="success", data=result.model_dump(), message="Segmentation complete")
        except ImportError:
            raise HTTPException(status_code=500, detail="ultralytics not installed. Run: pip install ultralytics")
        except Exception as exc:
            logger.error(f"Segmentation error: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(exc)}")

    # ── Deep learning image classification (CNN / ResNet — PyTorch) ─────────────
    if file and _is_dl_model(model_type) and Path(file.filename or "").suffix.lower() in IMAGE_EXTS:
        content = await file.read()
        try:
            from app.services.dl_trainer import load_dl_checkpoint, predict_classification
            ckpt = load_dl_checkpoint(session_id, model_type)
            if ckpt is None:
                raise HTTPException(status_code=404, detail=f"No trained {model_type} model found")
            pred = predict_classification(ckpt, content)
            result.label = pred.get("label")
            result.confidence = pred.get("confidence")
            result.class_probabilities = pred.get("class_probabilities")
            return APIResponse(status="success", data=result.model_dump(), message="Prediction complete")
        except ImportError:
            raise HTTPException(status_code=500, detail="PyTorch not installed. Run: pip install torch torchvision")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"DL classification error: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Image classification: CNN/ResNet only (no pixel flattening) ───────────
    if file and Path(file.filename or "").suffix.lower() in IMAGE_EXTS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Image inference requires CNN/ResNet for classification or U-Net/DeepLabV3/YOLO-seg "
                "for segmentation. Tabular sklearn models are not valid for image pixels."
            ),
        )

    # ── Tabular classification / regression ───────────────────────────────────
    model = _load_model(session_id, model_type)

    if data_json:
        row = json.loads(data_json)
        X = np.array(list(row.values()), dtype=float).reshape(1, -1)
    elif file:
        import pandas as pd
        content = await file.read()
        tmp = save_upload(content, file.filename or "input.csv")
        df = pd.read_csv(tmp)
        X = df.values
    else:
        raise HTTPException(status_code=400, detail="Provide an image, JSON data, or a CSV file")

    try:
        preds = model.predict(X)
        if tt == TaskType.CLASSIFICATION:
            result.label = str(preds[0])
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                classes_list = [str(c) for c in model.classes_]
                result.class_probabilities = {c: round(float(p), 4) for c, p in zip(classes_list, proba)}
                result.confidence = float(max(proba))
        else:
            result.value = float(preds[0])

        return APIResponse(status="success", data=result.model_dump(), message="Prediction complete")
    except Exception as exc:
        logger.error(f"Inference error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/models/{session_id}", response_model=APIResponse)
async def list_models(session_id: str):
    """List all trained models available for a session."""
    model_dir = settings.MODEL_DIR / session_id
    models = []
    if model_dir.exists():
        for sub in model_dir.iterdir():
            # Accept both .pkl (sklearn) and .pt (PyTorch) checkpoints
            has_model = (
                sub.is_dir()
                and ((sub / "best.pkl").exists() or (sub / "best.pt").exists())
            )
            if has_model:
                result_file = sub / "results.json"
                metrics = {}
                if result_file.exists():
                    try:
                        metrics = json.loads(result_file.read_text()).get("metrics", {})
                    except Exception:
                        pass
                models.append({"model_type": sub.name, "metrics": metrics})
    return APIResponse(status="success", data=models, message=f"{len(models)} model(s) available")


@router.post("/export/{session_id}", response_model=APIResponse)
async def export_model(session_id: str, model_type: str, format: str = "pkl"):
    """Export a trained model."""
    if model_type.startswith("yolov8"):
        from app.services.yolo_service import export_yolo_model
        path = export_yolo_model(session_id, format=format)
        if path:
            return APIResponse(status="success", data={"path": path}, message=f"Exported to {path}")
        raise HTTPException(status_code=500, detail="YOLO export failed")

    # Check .pt (PyTorch) then .pkl (sklearn)
    for ext in ("best.pt", "best.pkl"):
        ckpt = settings.MODEL_DIR / session_id / model_type / ext
        if ckpt.exists():
            fmt = "pt" if ext.endswith(".pt") else "pkl"
            return APIResponse(status="success", data={"path": str(ckpt), "format": fmt}, message="Model ready")
    # Legacy flat layout
    ckpt = settings.MODEL_DIR / session_id / f"{model_type}_best.pkl"
    if ckpt.exists():
        return APIResponse(status="success", data={"path": str(ckpt), "format": "pkl"}, message="Model ready")
    raise HTTPException(status_code=404, detail="Model not found")


@router.get("/chat", response_model=APIResponse)
async def llm_chat(session_id: str, message: str, history_json: Optional[str] = None):
    """Chat with the LLM about the dataset."""
    from app.core.llm_engine import chat, check_ollama_available
    from app.models.schemas import ChatMessage

    history = []
    if history_json:
        try:
            raw = json.loads(history_json)
            history = [ChatMessage(**m) for m in raw]
        except Exception:
            pass

    session_dir = settings.UPLOAD_DIR / session_id
    context = ""
    if session_dir.exists():
        csvs = list(session_dir.rglob("*.csv"))
        if csvs:
            import pandas as pd
            df = pd.read_csv(csvs[0])
            context = f"CSV dataset with {len(df)} rows, columns: {', '.join(df.columns[:10])}"
        else:
            class_info = session_dir / "preprocessed" / "class_info.json"
            if class_info.exists():
                ci = json.loads(class_info.read_text())
                context = f"Image dataset with classes: {', '.join(ci.get('classes', []))}"

    response = chat(message, history, dataset_context=context)
    return APIResponse(
        status="success",
        data={"response": response, "ollama_online": check_ollama_available()},
        message="",
    )
