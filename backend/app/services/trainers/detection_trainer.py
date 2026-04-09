"""
Object Detection Trainer — hard-gate wrapper for YOLO-based detection tasks.

ALLOWED models: YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8n-seg, YOLOv8s-seg
FORBIDDEN: Tabular models, CNN (without detection head), U-Net

Delegates actual training to trainer._train_yolo().
Returns metrics: {"map50", "map50_95"}
"""
import asyncio
from typing import Any, Dict

from app.core.config import settings
from app.models.schemas import ModelType, TaskType, TrainingConfig, ModelResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_MODELS = {
    ModelType.YOLOV8N,
    ModelType.YOLOV8S,
    ModelType.YOLOV8M,
    ModelType.YOLOV8N_SEG,
    ModelType.YOLOV8S_SEG,
}

ALLOWED_MODEL_VALUES = {m.value for m in ALLOWED_MODELS}


async def train_detection(
    config: TrainingConfig,
    log_queue: "asyncio.Queue[str]",
) -> ModelResult:
    """
    Train a YOLO object detection / instance segmentation model.

    Raises:
        ValueError: If model_type is not allowed for detection,
                    or if data.yaml is missing.
    """
    def log(msg: str):
        log_queue.put_nowait(msg)

    # ── Hard gate ────────────────────────────────────────────────────────────
    if config.model_type not in ALLOWED_MODELS:
        raise ValueError(
            f"Object detection requires one of: {sorted(ALLOWED_MODEL_VALUES)}. "
            f"Got: '{config.model_type.value}'."
        )

    # ── Validate YOLO data.yaml ───────────────────────────────────────────────
    yaml_path = settings.UPLOAD_DIR / config.session_id / "preprocessed" / "data.yaml"
    if not yaml_path.exists():
        raise ValueError(
            "Object detection requires a YOLO-format dataset with data.yaml. "
            f"No data.yaml found at {yaml_path}. "
            "Please preprocess a YOLO or COCO dataset first."
        )

    log(f"[detection_trainer] Model: {config.model_type.value}")

    # ── Delegate to existing _train_yolo ─────────────────────────────────────
    from app.services.trainer import _train_yolo

    result = ModelResult(
        session_id=config.session_id,
        model_type=config.model_type.value,
        task_type=TaskType.OBJECT_DETECTION.value,
    )

    loop = asyncio.get_event_loop()
    metrics = await loop.run_in_executor(
        None,
        lambda: _train_yolo(
            config.model_type,
            config.session_id,
            config.epochs,
            config.hyperparams,
            log_queue,
        ),
    )

    result.metrics = metrics
    result.map50 = metrics.get("map50")

    log(
        f"[detection_trainer] Done — "
        f"mAP50={metrics.get('map50', 0):.4f}  "
        f"mAP50-95={metrics.get('map50_95', 0):.4f}"
    )
    return result
