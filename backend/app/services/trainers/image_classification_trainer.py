"""
Image Classification Trainer — hard-gate wrapper for image classification tasks.

ALLOWED models: CNN, ResNet
FORBIDDEN: Tabular models (these flatten pixel data and destroy spatial information)

Delegates actual training to dl_trainer.train_dl_classification().
Returns metrics: {"accuracy", "f1"}
"""
import asyncio
from typing import Any, Dict

from app.core.config import settings
from app.models.schemas import ModelType, TaskType, TrainingConfig, ModelResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_MODELS = {
    ModelType.CNN,
    ModelType.RESNET,
}

ALLOWED_MODEL_VALUES = {m.value for m in ALLOWED_MODELS}


async def train_classification(
    config: TrainingConfig,
    log_queue: "asyncio.Queue[str]",
) -> ModelResult:
    """
    Train an image classification model.

    Raises:
        ValueError: If model_type is not allowed for image classification,
                    or if split_manifest.json is missing.
    """
    def log(msg: str):
        log_queue.put_nowait(msg)

    # ── Hard gate ────────────────────────────────────────────────────────────
    if config.model_type not in ALLOWED_MODELS:
        raise ValueError(
            f"Image classification requires one of: {sorted(ALLOWED_MODEL_VALUES)}. "
            f"Got: '{config.model_type.value}'. "
            "Use tabular_trainer for CSV/tabular datasets."
        )

    # ── Validate manifest ─────────────────────────────────────────────────────
    manifest_path = (
        settings.UPLOAD_DIR / config.session_id / "preprocessed" / "split_manifest.json"
    )
    if not manifest_path.exists():
        raise ValueError(
            "Image classification requires a preprocessed image folder dataset. "
            f"No split_manifest.json found at {manifest_path}. "
            "Please preprocess the dataset first."
        )

    log(f"[classification_trainer] Model: {config.model_type.value}")

    # ── Delegate to existing dl_trainer ──────────────────────────────────────
    from app.services.dl_trainer import train_dl_classification

    result = ModelResult(
        session_id=config.session_id,
        model_type=config.model_type.value,
        task_type=TaskType.CLASSIFICATION.value,
    )

    loop = asyncio.get_event_loop()
    ckpt_path, metrics = await loop.run_in_executor(
        None,
        lambda: train_dl_classification(
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
    result.accuracy = metrics.get("accuracy")
    result.f1_score = metrics.get("f1")

    log(
        f"[classification_trainer] Done — "
        f"Accuracy={metrics.get('accuracy', 0):.4f}  "
        f"F1={metrics.get('f1', 0):.4f}"
    )
    return result
