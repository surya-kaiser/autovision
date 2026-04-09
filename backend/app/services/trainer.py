"""
Multi-model trainer supporting classification, regression, and object detection.
Streams real-time logs via asyncio queue. Never crashes — wraps all errors.
"""
import asyncio
import time
import json
import pickle
import importlib
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.config import settings
from app.models.schemas import ModelResult, ModelType, TaskType, TrainingConfig, TrainingStatus
from app.services.metadata_store import append_training_log, record_training_run
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Shared registry: session_id -> TrainingStatus
TRAINING_REGISTRY: Dict[str, TrainingStatus] = {}


def _detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_preprocessed_data(session_id: str) -> Tuple[
    Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]
]:
    base = settings.UPLOAD_DIR / session_id / "preprocessed"
    try:
        train = pd.read_csv(base / "train.csv")
        val = pd.read_csv(base / "val.csv")
        test = pd.read_csv(base / "test.csv")
        return train, val, test
    except FileNotFoundError:
        return None, None, None




def _split_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


# ── Sklearn / XGBoost / LightGBM trainer ────────────────────────────────────

def _train_sklearn_model(
    model_type: ModelType,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task_type: TaskType,
    hyperparams: Dict[str, Any],
    log_queue: "asyncio.Queue[str]",
) -> Tuple[Any, Dict[str, float]]:

    async def _log(msg: str):
        await log_queue.put(msg)

    def log_sync(msg: str):
        log_queue.put_nowait(msg)

    log_sync(f"Initialising {model_type.value} model...")

    model: Any = None

    if model_type == ModelType.RANDOM_FOREST:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestClassifier if task_type == TaskType.CLASSIFICATION else RandomForestRegressor
        model = cls(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", None),
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == ModelType.XGBOOST:
        try:
            import xgboost as xgb
            from xgboost import XGBClassifier, XGBRegressor

            n_est = hyperparams.get("n_estimators", 200)
            period = max(1, n_est // 20)  # log ~20 times

            class _XGBLog(xgb.callback.TrainingCallback):
                def __init__(self, lf, tot, p):
                    super().__init__()
                    self._lf, self._tot, self._p = lf, tot, p
                def after_iteration(self, model, epoch, evals_log):
                    if (epoch + 1) % self._p == 0 or epoch == self._tot - 1:
                        parts = []
                        for _, metrics in evals_log.items():
                            for m, vals in metrics.items():
                                parts.append(f"{m}={vals[-1]:.4f}")
                        self._lf(f"  [Round {epoch+1}/{self._tot}] {' | '.join(parts)}")
                    return False

            cls = XGBClassifier if task_type == TaskType.CLASSIFICATION else XGBRegressor
            model = cls(
                n_estimators=n_est,
                max_depth=hyperparams.get("max_depth", 6),
                learning_rate=hyperparams.get("learning_rate", 0.1),
                random_state=42,
                verbosity=0,
                eval_metric="logloss" if task_type == TaskType.CLASSIFICATION else "rmse",
                callbacks=[_XGBLog(log_sync, n_est, period)],
            )
            log_sync(f"  XGBoost — {n_est} boosting rounds")
        except ImportError:
            log_sync("XGBoost not installed — falling back to RandomForest")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state=42)
    elif model_type == ModelType.LIGHTGBM:
        try:
            import lightgbm as lgb
            from lightgbm import LGBMClassifier, LGBMRegressor

            n_est = hyperparams.get("n_estimators", 200)
            period = max(1, n_est // 20)

            def _lgbm_cb(log_fn, tot, p):
                def callback(env):
                    it = env.iteration
                    if (it + 1) % p == 0 or it == tot - 1:
                        parts = []
                        for item in env.evaluation_result_list:
                            if len(item) >= 3:
                                parts.append(f"{item[1]}={item[2]:.4f}")
                        log_fn(f"  [Round {it+1}/{tot}] {' | '.join(parts)}")
                callback.order = 10
                return callback

            cls = LGBMClassifier if task_type == TaskType.CLASSIFICATION else LGBMRegressor
            model = cls(
                n_estimators=n_est,
                num_leaves=hyperparams.get("num_leaves", 31),
                learning_rate=hyperparams.get("learning_rate", 0.1),
                random_state=42,
                verbose=-1,
            )
            model._lgbm_extra_fit_kwargs = {
                "eval_set": None,  # filled in at fit time
                "callbacks": [lgb.log_evaluation(-1), _lgbm_cb(log_sync, n_est, period)],
            }
            log_sync(f"  LightGBM — {n_est} boosting rounds")
        except ImportError:
            log_sync("LightGBM not installed — falling back to RandomForest")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state=42)
    elif model_type == ModelType.LINEAR:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif model_type == ModelType.RIDGE:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=hyperparams.get("alpha", 1.0))
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)

    log_sync(f"  Fitting on {len(X_train)} train / {len(X_val)} val samples…")
    t0 = time.time()
    # XGBoost: callbacks already in model, just pass eval_set
    fit_kwargs: Dict[str, Any] = {}
    if hasattr(model, "get_params") and "eval_metric" in (model.get_params() or {}):
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False
    # LightGBM: extra kwargs stored on model
    elif hasattr(model, "_lgbm_extra_fit_kwargs"):
        extra = model._lgbm_extra_fit_kwargs
        extra["eval_set"] = [(X_val, y_val)]
        fit_kwargs.update(extra)
        del model._lgbm_extra_fit_kwargs
    model.fit(X_train, y_train, **fit_kwargs)
    elapsed = time.time() - t0
    log_sync(f"  Training done in {elapsed:.1f}s — evaluating…")

    metrics: Dict[str, float] = {}
    try:
        if task_type == TaskType.CLASSIFICATION:
            from sklearn.metrics import accuracy_score, f1_score
            preds = model.predict(X_val)
            metrics["accuracy"] = float(accuracy_score(y_val, preds))
            metrics["f1"] = float(f1_score(y_val, preds, average="weighted", zero_division=0))
            log_sync(f"  Val accuracy={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}")
        else:
            from sklearn.metrics import mean_squared_error, r2_score
            preds = model.predict(X_val)
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_val, preds)))
            metrics["r2"] = float(r2_score(y_val, preds))
            log_sync(f"  Val rmse={metrics['rmse']:.4f}  r2={metrics['r2']:.4f}")
    except Exception as e:
        log_sync(f"Metric calculation warning: {e}")

    return model, metrics


# ── YOLO trainer ──────────────────────────────────────────────────────────────

def _train_yolo(
    model_type: ModelType,
    session_id: str,
    epochs: int,
    hyperparams: Dict[str, Any],
    log_queue: "asyncio.Queue[str]",
) -> Dict[str, float]:

    def log_sync(msg: str):
        log_queue.put_nowait(msg)

    try:
        from ultralytics import YOLO
    except ImportError:
        log_sync("ultralytics not installed. Run: pip install ultralytics")
        return {}

    model_type_str = model_type.value
    is_segmentation = "seg" in model_type_str.lower()

    # Check if segmentation model but no mask annotations available
    if is_segmentation:
        labels_dir = settings.UPLOAD_DIR / session_id / "preprocessed" / "labels"
        if not labels_dir.exists():
            log_sync(
                "⚠️  Segmentation task detected but no annotated masks found.")
            log_sync(
                "   Segmentation requires manually annotated polygon/mask data.")
            log_sync(
                "   Using pre-trained YOLO segmentation model instead of training.")
            log_sync(f"   Model: {model_type_str}")
            # Return empty metrics — will use pre-trained model during inference
            return {"model_type": model_type_str, "pre_trained": True}

    device = _detect_device()
    log_sync(f"Using device: {device}")

    yaml_path = settings.UPLOAD_DIR / session_id / "preprocessed" / "data.yaml"
    if not yaml_path.exists():
        log_sync(f"data.yaml not found at {yaml_path}. Please preprocess dataset first.")
        return {}

    model_name = model_type.value  # e.g. "yolov8n"
    model = YOLO(f"{model_name}.pt")
    log_sync(f"Loaded {model_name}")

    out_dir = settings.MODEL_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=hyperparams.get("batch_size", 8),
            imgsz=hyperparams.get("imgsz", 640),
            device=device,
            project=str(out_dir),
            name="train",
            verbose=True,
        )
        metrics = {
            "map50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
            "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        }
        log_sync(f"YOLO training done. mAP50={metrics.get('map50', 0):.4f}")
        return metrics
    except Exception as e:
        log_sync(f"YOLO training error: {e}")
        return {}


# ── Public Training API ───────────────────────────────────────────────────────

async def train_model(
    config: TrainingConfig,
    log_queue: "asyncio.Queue[str]",
) -> ModelResult:
    """Full async training entry-point."""
    sid = config.session_id
    epochs = settings.PILOT_EPOCHS if config.pilot else config.epochs

    status = TrainingStatus(
        session_id=sid,
        model_type=config.model_type.value,
        status="running",
        total_epochs=epochs,
    )
    TRAINING_REGISTRY[sid] = status

    result = ModelResult(
        session_id=sid,
        model_type=config.model_type.value,
        task_type=config.task_type.value,
    )

    t_start = time.time()
    model_type_str = config.model_type.value

    def _log(msg: str) -> None:
        """Send to WebSocket queue AND write to persistent log file."""
        log_queue.put_nowait(msg)
        append_training_log(sid, model_type_str, msg)

    _log(f"=== Training started: {model_type_str} | session={sid[:8]} ===")

    # Model-type sets for routing
    _YOLO_TYPES = {
        ModelType.YOLOV8N, ModelType.YOLOV8S, ModelType.YOLOV8M,
        ModelType.YOLOV8N_SEG, ModelType.YOLOV8S_SEG,
    }
    _DL_CLASSIFICATION = {ModelType.CNN, ModelType.RESNET}
    _DL_SEGMENTATION = {ModelType.UNET, ModelType.DEEPLABV3}
    _TABULAR_ONLY = {
        ModelType.RANDOM_FOREST,
        ModelType.XGBOOST,
        ModelType.LIGHTGBM,
        ModelType.LINEAR,
        ModelType.RIDGE,
    }

    try:
        if config.task_type == TaskType.SEGMENTATION and config.model_type not in (_DL_SEGMENTATION | _YOLO_TYPES):
            _log(
                "ERROR: Segmentation requires CNN-based models (U-Net/DeepLabV3/YOLO-seg). "
                f"Received: {config.model_type.value}"
            )
            status.status = "failed"
            return result

        if config.model_type in _YOLO_TYPES:
            _log(f"Starting YOLO training ({model_type_str})")
            loop = asyncio.get_event_loop()
            metrics = await loop.run_in_executor(
                None,
                lambda: _train_yolo(config.model_type, sid, epochs, config.hyperparams, log_queue),
            )
            result.metrics = metrics
            result.map50 = metrics.get("map50")

        elif config.model_type in _DL_CLASSIFICATION:
            # ── PyTorch deep-learning classification ─────────────────────────
            from app.services.dl_trainer import train_dl_classification
            _log(f"Starting PyTorch training ({model_type_str})")
            loop = asyncio.get_event_loop()
            ckpt_path, metrics = await loop.run_in_executor(
                None,
                lambda: train_dl_classification(
                    config.model_type, sid, epochs,
                    config.batch_size, config.learning_rate,
                    config.hyperparams, log_queue,
                ),
            )
            result.checkpoint_path = ckpt_path or ""
            result.metrics = metrics
            result.accuracy = metrics.get("accuracy")
            result.f1_score = metrics.get("f1")

        elif config.model_type in _DL_SEGMENTATION:
            # ── PyTorch deep-learning segmentation ───────────────────────────
            from app.services.dl_trainer import train_dl_segmentation
            _log(f"Starting PyTorch segmentation training ({model_type_str})")
            loop = asyncio.get_event_loop()
            ckpt_path, metrics = await loop.run_in_executor(
                None,
                lambda: train_dl_segmentation(
                    config.model_type, sid, epochs,
                    config.batch_size, config.learning_rate,
                    config.hyperparams, log_queue,
                ),
            )
            result.checkpoint_path = ckpt_path or ""
            result.metrics = metrics

        else:
            # ── Tabular models (CSV only) ───────────────────────────────────
            train_df, val_df, _ = _get_preprocessed_data(sid)
            if train_df is not None:
                X_train, y_train = _split_xy(train_df)
                X_val, y_val = _split_xy(val_df)

                # Handle extremely small datasets: use training data for validation if needed
                if len(y_val) == 0 and len(y_train) > 0:
                    _log(f"  Validation set empty — using subset of training data for validation")
                    split_idx = max(1, len(X_train) // 2)
                    X_val, y_val = X_train[:split_idx], y_train[:split_idx]
            else:
                if config.model_type in _TABULAR_ONLY:
                    _log(
                        "ERROR: Tabular models require CSV preprocessed data. "
                        "Image tasks must use CNN/segmentation models."
                    )
                else:
                    _log("ERROR: No tabular preprocessed data found. Run preprocessing first.")
                status.status = "failed"
                return result

            _log(f"Training {model_type_str} | {len(X_train)} train samples | {len(X_val)} val samples")
            loop = asyncio.get_event_loop()
            model, metrics = await loop.run_in_executor(
                None,
                lambda: _train_sklearn_model(
                    config.model_type, X_train, y_train, X_val, y_val,
                    config.task_type, config.hyperparams, log_queue,
                ),
            )

            # Save checkpoint under models/{sid}/{model_type}/best.pkl
            ckpt_dir = settings.MODEL_DIR / sid / model_type_str
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "best.pkl"
            with open(ckpt_path, "wb") as f:
                pickle.dump(model, f)
            result.checkpoint_path = str(ckpt_path)
            result.metrics = metrics
            result.accuracy = metrics.get("accuracy")
            result.f1_score = metrics.get("f1")

        result.training_time_s = time.time() - t_start
        status.status = "completed"
        status.metrics = result.metrics
        _log(f"=== Training finished in {result.training_time_s:.1f}s ===")
        for k, v in result.metrics.items():
            _log(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # Save result JSON per model
        model_dir = settings.MODEL_DIR / sid / model_type_str
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "results.json").write_text(
            json.dumps(result.model_dump(), indent=2, default=str)
        )

        # Also keep a top-level results.json (latest result)
        (settings.MODEL_DIR / sid / "results.json").write_text(
            json.dumps(result.model_dump(), indent=2, default=str)
        )

        # Record in session metadata
        record_training_run(
            session_id=sid,
            model_type=model_type_str,
            task_type=config.task_type.value,
            metrics=result.metrics,
            checkpoint_path=result.checkpoint_path,
            training_time_s=result.training_time_s,
            is_pilot=config.pilot,
        )

    except Exception as exc:
        logger.exception(f"Training error for session {sid}: {exc}")
        _log(f"ERROR: {exc}")
        status.status = "failed"

    return result


async def auto_train_with_retry(
    config: TrainingConfig,
    log_queue: "asyncio.Queue[str]",
) -> ModelResult:
    """Train with automatic fallback on failure and low-accuracy retry.

    Strategy:
      1. Train the recommended model.
      2. If it FAILS → retry with a safer fallback model.
      3. If it SUCCEEDS but accuracy < 65% (classification) → auto-tune:
         - Image tasks: retry with ResNet (transfer learning)
         - Tabular tasks: retry with RandomForest with stronger hyperparameters
    """
    import copy

    FALLBACKS = {
        ModelType.XGBOOST: ModelType.RANDOM_FOREST,
        ModelType.LIGHTGBM: ModelType.RANDOM_FOREST,
        ModelType.CNN: ModelType.RESNET,
        ModelType.YOLOV8S: ModelType.YOLOV8N,
        ModelType.YOLOV8M: ModelType.YOLOV8N,
    }
    _DL_IMAGE_MODELS = {ModelType.CNN, ModelType.RESNET, ModelType.UNET, ModelType.DEEPLABV3}
    ACCURACY_THRESHOLD = 0.65

    log_queue.put_nowait(f"=== Auto-train: {config.model_type.value} (with auto-fix) ===")
    result = await train_model(config, log_queue)
    status = TRAINING_REGISTRY.get(config.session_id)

    # ── Step 2: failed → fallback model ──────────────────────────────────────
    if status and status.status == "failed":
        fallback = FALLBACKS.get(config.model_type)
        if fallback:
            log_queue.put_nowait(
                f"=== Auto-fix: {config.model_type.value} failed — retrying with {fallback.value} ==="
            )
            config.model_type = fallback
            result = await train_model(config, log_queue)
            status = TRAINING_REGISTRY.get(config.session_id)

    # ── Step 3: low accuracy → tune hyperparameters ───────────────────────────
    if (
        status
        and status.status == "completed"
        and config.task_type == TaskType.CLASSIFICATION
    ):
        accuracy = status.metrics.get("accuracy", 1.0)
        f1 = status.metrics.get("f1", 1.0)
        score = min(accuracy, f1) if f1 < 1.0 else accuracy

        if score < ACCURACY_THRESHOLD:
            # For image models, retry with ResNet (transfer learning)
            # For tabular models, retry with RandomForest + stronger hyperparams
            if config.model_type in _DL_IMAGE_MODELS:
                log_queue.put_nowait(
                    f"=== Score {score:.1%} below threshold {ACCURACY_THRESHOLD:.0%} "
                    f"— retrying with ResNet transfer learning ==="
                )
                config2 = copy.deepcopy(config)
                config2.model_type = ModelType.RESNET
                config2.hyperparams = {
                    "weight_decay": 1e-4,
                }
                config2.epochs = max(config.epochs, 15)
                config2.learning_rate = 0.0005
            else:
                log_queue.put_nowait(
                    f"=== Score {score:.1%} below threshold {ACCURACY_THRESHOLD:.0%} "
                    f"— auto-tuning with stronger hyperparameters ==="
                )
                config2 = copy.deepcopy(config)
                config2.model_type = ModelType.RANDOM_FOREST
                config2.hyperparams = {
                    "n_estimators": 400,
                    "max_depth": 25,
                }
            result2 = await train_model(config2, log_queue)
            status2 = TRAINING_REGISTRY.get(config.session_id)

            if status2 and status2.status == "completed":
                new_score = min(
                    status2.metrics.get("accuracy", 0),
                    status2.metrics.get("f1", 1.0) if status2.metrics.get("f1", 1.0) < 1.0
                    else status2.metrics.get("accuracy", 0),
                )
                if new_score > score:
                    log_queue.put_nowait(
                        f"=== Improved: {new_score:.1%} accuracy (was {score:.1%}) ==="
                    )
                    result = result2
                else:
                    log_queue.put_nowait(
                        f"=== No improvement ({new_score:.1%} vs {score:.1%}) — keeping original model ==="
                    )

    return result


def get_training_status(session_id: str) -> Optional[TrainingStatus]:
    return TRAINING_REGISTRY.get(session_id)


def load_model(session_id: str, model_type: str) -> Any:
    # New layout: models/{session_id}/{model_type}/best.pkl
    ckpt = settings.MODEL_DIR / session_id / model_type / "best.pkl"
    if not ckpt.exists():
        # Fallback: old flat layout
        ckpt = settings.MODEL_DIR / session_id / f"{model_type}_best.pkl"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint found for {model_type} in session {session_id}")
    with open(ckpt, "rb") as f:
        return pickle.load(f)
