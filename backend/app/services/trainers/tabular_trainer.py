"""
Tabular Trainer — hard-gate wrapper for CSV/tabular ML tasks.

ALLOWED models: XGBoost, LightGBM, RandomForest, LinearRegression, Ridge
FORBIDDEN: CNN, ResNet, U-Net, DeepLabV3, YOLO (image models cannot train on CSV)

Delegates actual training to trainer._train_sklearn_model().
Returns metrics:
  Regression    → {"rmse", "r2"}
  Classification → {"accuracy", "f1"}
"""
import asyncio
from typing import Any, Dict

import numpy as np
import pandas as pd

from app.core.config import settings
from app.models.schemas import ModelType, TaskType, TrainingConfig, ModelResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_MODELS = {
    ModelType.XGBOOST,
    ModelType.LIGHTGBM,
    ModelType.RANDOM_FOREST,
    ModelType.LINEAR,
    ModelType.RIDGE,
}

ALLOWED_MODEL_VALUES = {m.value for m in ALLOWED_MODELS}

_IMAGE_MODELS = {
    ModelType.CNN, ModelType.RESNET,
    ModelType.UNET, ModelType.DEEPLABV3,
    ModelType.YOLOV8N, ModelType.YOLOV8S, ModelType.YOLOV8M,
    ModelType.YOLOV8N_SEG, ModelType.YOLOV8S_SEG,
}


async def train_tabular(
    config: TrainingConfig,
    log_queue: "asyncio.Queue[str]",
) -> ModelResult:
    """
    Train a tabular ML model (sklearn / XGBoost / LightGBM).

    Raises:
        ValueError: If model_type is an image model (cannot process CSV),
                    or if preprocessed CSV files are missing.
    """
    def log(msg: str):
        log_queue.put_nowait(msg)

    # ── Hard gate: image models cannot run on tabular data ───────────────────
    if config.model_type in _IMAGE_MODELS:
        raise ValueError(
            f"Tabular training cannot use image model '{config.model_type.value}'. "
            f"Allowed tabular models: {sorted(ALLOWED_MODEL_VALUES)}. "
            "Use image_classification_trainer or segmentation_trainer for image datasets."
        )

    if config.model_type not in ALLOWED_MODELS:
        raise ValueError(
            f"Unknown model '{config.model_type.value}' for tabular training. "
            f"Allowed: {sorted(ALLOWED_MODEL_VALUES)}."
        )

    # ── Validate preprocessed CSV files ──────────────────────────────────────
    base = settings.UPLOAD_DIR / config.session_id / "preprocessed"
    train_csv = base / "train.csv"
    val_csv = base / "val.csv"

    if not train_csv.exists():
        raise ValueError(
            f"No preprocessed CSV data found at {train_csv}. "
            "Please upload a CSV dataset and run preprocessing first."
        )

    log(f"[tabular_trainer] Model: {config.model_type.value}")
    log(f"[tabular_trainer] Loading: {train_csv}")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv) if val_csv.exists() else train_df.head(max(1, len(train_df) // 5))

    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    X_val = val_df.iloc[:, :-1].values
    y_val = val_df.iloc[:, -1].values

    log(f"[tabular_trainer] Train: {len(X_train)} samples | Val: {len(X_val)} samples")

    # ── Delegate to existing _train_sklearn_model ────────────────────────────
    from app.services.trainer import _train_sklearn_model

    result = ModelResult(
        session_id=config.session_id,
        model_type=config.model_type.value,
        task_type=config.task_type.value,
    )

    loop = asyncio.get_event_loop()
    model, metrics = await loop.run_in_executor(
        None,
        lambda: _train_sklearn_model(
            config.model_type,
            X_train,
            y_train,
            X_val,
            y_val,
            config.task_type,
            config.hyperparams,
            log_queue,
        ),
    )

    # Save checkpoint
    import pickle
    ckpt_dir = settings.MODEL_DIR / config.session_id / config.model_type.value
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(model, f)

    result.checkpoint_path = str(ckpt_path)
    result.metrics = metrics
    result.accuracy = metrics.get("accuracy")
    result.f1_score = metrics.get("f1")

    if config.task_type == TaskType.REGRESSION:
        log(
            f"[tabular_trainer] Done — "
            f"RMSE={metrics.get('rmse', 0):.4f}  "
            f"R²={metrics.get('r2', 0):.4f}"
        )
    else:
        log(
            f"[tabular_trainer] Done — "
            f"Accuracy={metrics.get('accuracy', 0):.4f}  "
            f"F1={metrics.get('f1', 0):.4f}"
        )
    return result
