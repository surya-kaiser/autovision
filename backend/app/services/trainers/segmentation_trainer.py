"""
Segmentation Trainer — hard-gate wrapper for semantic segmentation tasks.

ALLOWED models: U-Net, DeepLabV3, YOLOv8n-seg, YOLOv8s-seg
FORBIDDEN: ANY tabular model (XGBoost, LightGBM, RandomForest, etc.)

Delegates actual training to dl_trainer.train_dl_segmentation().
Returns metrics: {"iou", "dice", "pixel_accuracy"}
"""
import asyncio
from typing import Any, Dict, Optional

from app.core.config import settings
from app.models.schemas import ModelType, TaskType, TrainingConfig, ModelResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_MODELS = {
    ModelType.UNET,
    ModelType.DEEPLABV3,
    ModelType.YOLOV8N_SEG,
    ModelType.YOLOV8S_SEG,
}

ALLOWED_MODEL_VALUES = {m.value for m in ALLOWED_MODELS}


async def train_segmentation(
    config: TrainingConfig,
    log_queue: "asyncio.Queue[str]",
) -> ModelResult:
    """
    Train a segmentation model.

    Raises:
        ValueError: If model_type is not allowed for segmentation,
                    or if seg_manifest.json is missing (no masks uploaded).
    """
    def log(msg: str):
        log_queue.put_nowait(msg)

    # ── Hard gate: reject wrong model types ──────────────────────────────────
    if config.model_type not in ALLOWED_MODELS:
        raise ValueError(
            f"Segmentation task requires one of: {sorted(ALLOWED_MODEL_VALUES)}. "
            f"Got: '{config.model_type.value}'. "
            "Tabular models (XGBoost, LightGBM, RandomForest) cannot process image data — "
            "they flatten spatial information and produce meaningless outputs."
        )

    # ── Validate dataset has been preprocessed for segmentation ──────────────
    manifest_path = (
        settings.UPLOAD_DIR / config.session_id / "preprocessed" / "seg_manifest.json"
    )
    if not manifest_path.exists():
        raise ValueError(
            "Segmentation training requires a dataset with images/ + masks/ directories. "
            f"No seg_manifest.json found at {manifest_path}. "
            "Please upload a dataset containing both images/ and masks/ subdirectories, "
            "then run preprocessing before training."
        )

    log(f"[segmentation_trainer] Model: {config.model_type.value}")
    log(f"[segmentation_trainer] Manifest: {manifest_path}")

    # ── Delegate to existing dl_trainer ──────────────────────────────────────
    from app.services.dl_trainer import train_dl_segmentation

    result = ModelResult(
        session_id=config.session_id,
        model_type=config.model_type.value,
        task_type=TaskType.SEGMENTATION.value,
    )

    loop = asyncio.get_event_loop()
    ckpt_path, metrics = await loop.run_in_executor(
        None,
        lambda: train_dl_segmentation(
            config.model_type,
            config.session_id,
            config.epochs,
            config.batch_size,
            config.learning_rate,
            config.hyperparams,
            log_queue,
        ),
    )

    result.checkpoint_path = ckpt_path or ""
    result.metrics = metrics
    # Segmentation metrics: iou, dice, pixel_accuracy — NOT accuracy or RMSE
    result.accuracy = metrics.get("iou")  # use IoU as primary score for compatibility

    log(
        f"[segmentation_trainer] Done — "
        f"IoU={metrics.get('iou', 0):.4f}  "
        f"Dice={metrics.get('dice', 0):.4f}  "
        f"PixelAcc={metrics.get('pixel_accuracy', 0):.4f}"
    )
    return result
