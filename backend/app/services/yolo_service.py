"""
YOLOv8 service: validation, inference, bounding-box prediction.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def validate_yolo_dataset(dataset_dir: Path) -> Dict[str, Any]:
    """Validate a YOLO dataset directory structure."""
    result: Dict[str, Any] = {"valid": True, "errors": [], "warnings": [], "stats": {}}

    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.exists():
        result["errors"].append("Missing 'images' directory")
        result["valid"] = False
    if not labels_dir.exists():
        result["errors"].append("Missing 'labels' directory")
        result["valid"] = False

    if not result["valid"]:
        return result

    from app.services.preprocessor import _find_images
    images = _find_images(images_dir)
    labels = list(labels_dir.rglob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels if p.name != "classes.txt"}
    unlabelled = image_stems - label_stems
    extra_labels = label_stems - image_stems

    if unlabelled:
        result["warnings"].append(f"{len(unlabelled)} images have no labels")
    if extra_labels:
        result["warnings"].append(f"{len(extra_labels)} label files have no matching image")

    result["stats"] = {
        "num_images": len(images),
        "num_labels": len(labels),
        "unlabelled_images": len(unlabelled),
        "extra_labels": len(extra_labels),
    }
    return result


def predict_yolo(
    session_id: str,
    image_path: Path,
    model_type: str = "yolov8n",
    conf_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """Run YOLOv8 inference and return bounding boxes."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed")
        return []

    ckpt = settings.MODEL_DIR / session_id / "train" / "weights" / "best.pt"
    if not ckpt.exists():
        # Try default model
        ckpt = f"{model_type}.pt"

    try:
        model = YOLO(str(ckpt))
        results = model(str(image_path), conf=conf_threshold, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append({
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "confidence": round(float(box.conf[0]), 4),
                    "class_id": int(box.cls[0]),
                    "class_name": r.names[int(box.cls[0])],
                })
        return boxes
    except Exception as e:
        logger.error(f"YOLO inference error: {e}")
        return []


def export_yolo_model(
    session_id: str,
    format: str = "onnx",
) -> Optional[str]:
    """Export YOLO model to ONNX or other format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return None

    ckpt = settings.MODEL_DIR / session_id / "train" / "weights" / "best.pt"
    if not ckpt.exists():
        return None

    try:
        model = YOLO(str(ckpt))
        exported = model.export(format=format)
        return str(exported)
    except Exception as e:
        logger.error(f"YOLO export error: {e}")
        return None
