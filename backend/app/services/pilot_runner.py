"""
Autonomous ML Pipeline: Uses AI agents to drive intelligent model selection,
experiment planning, training, and iterative improvement.

Pipeline:
1. Research Agent: Analyzes dataset → recommends models
2. Experiment Planner: Converts recommendations → training configs
3. Training Loop: Train models with early stopping
4. Evaluation: Compute metrics
5. Best model selection
"""
import asyncio
import time
import json
import traceback
from typing import Dict, Any, Optional, List
from pathlib import Path

from app.core.config import settings
from app.models.schemas import ModelResult, TrainingConfig, TaskType, ModelType
from app.services.trainer import train_model
from app.services.preprocessor import detect_format, detect_task_type, preprocess_dataset
from app.models.schemas import PreprocessConfig
from app.agents.research_agent import get_research_agent
from app.agents.experiment_planner import get_experiment_planner, ExperimentConfig
from app.agents.improvement_agent import get_improvement_agent
from app.experiments.experiment_tracker import get_experiment_tracker
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _canonical_to_task_type(canonical: str) -> Optional[TaskType]:
    """Convert task_router canonical string to TaskType enum."""
    mapping = {
        "image_segmentation": TaskType.SEGMENTATION,
        "image_classification": TaskType.CLASSIFICATION,
        "object_detection": TaskType.OBJECT_DETECTION,
        "tabular": TaskType.CLASSIFICATION,  # refined by column analysis later
    }
    return mapping.get(canonical)


# Map model name keywords → ModelType supported by trainer
_MODEL_NAME_MAP = {
    # Tabular ML
    "xgboost": ModelType.XGBOOST,
    "xgb": ModelType.XGBOOST,
    "lightgbm": ModelType.LIGHTGBM,
    "lgbm": ModelType.LIGHTGBM,
    "randomforest": ModelType.RANDOM_FOREST,
    # Deep learning — classification
    "cnn": ModelType.CNN,
    "resnet": ModelType.RESNET,
    "efficientnet": ModelType.RESNET,   # map to ResNet (closest available)
    "mobilenet": ModelType.CNN,         # map to CNN (lightweight)
    "vgg": ModelType.RESNET,
    "inception": ModelType.RESNET,
    "densenet": ModelType.RESNET,
    # Deep learning — segmentation
    "unet": ModelType.UNET,
    "deeplabv3": ModelType.DEEPLABV3,
    "deeplab": ModelType.DEEPLABV3,
    # YOLO — detection & segmentation
    "yolov8n": ModelType.YOLOV8N,
    "yolov8s": ModelType.YOLOV8S,
    "yolov8m": ModelType.YOLOV8M,
    "yolov8nseg": ModelType.YOLOV8N_SEG,
    "yolov8sseg": ModelType.YOLOV8S_SEG,
    "yolo": ModelType.YOLOV8N,
}


_TABULAR_ONLY_TYPES = {
    ModelType.XGBOOST,
    ModelType.LIGHTGBM,
    ModelType.RANDOM_FOREST,
    ModelType.LINEAR,
    ModelType.RIDGE,
}
_IMAGE_ONLY_TYPES = {
    ModelType.CNN,
    ModelType.RESNET,
    ModelType.UNET,
    ModelType.DEEPLABV3,
    ModelType.YOLOV8N,
    ModelType.YOLOV8S,
    ModelType.YOLOV8M,
    ModelType.YOLOV8N_SEG,
    ModelType.YOLOV8S_SEG,
}


def _map_model_name(model_name: str, task_type: Optional[TaskType] = None) -> ModelType:
    """Map a model name string to a supported ModelType.

    Enforces task-type constraints BEFORE returning:
    - Segmentation task: tabular models (XGBoost/LGBM/RF) are hard-rejected → U-Net
    - Tabular task: image-only models are hard-rejected → RandomForest
    """
    name = (model_name or "").lower().replace(" ", "").replace("-", "").replace("_", "")
    for key, mtype in _MODEL_NAME_MAP.items():
        if key in name:
            # Hard rejection: tabular model for segmentation/detection task
            if task_type == TaskType.SEGMENTATION and mtype in _TABULAR_ONLY_TYPES:
                logger.warning(
                    f"_map_model_name: rejecting tabular model '{model_name}' for "
                    "segmentation task — substituting U-Net"
                )
                return ModelType.UNET
            if task_type == TaskType.OBJECT_DETECTION and mtype in _TABULAR_ONLY_TYPES:
                logger.warning(
                    f"_map_model_name: rejecting tabular model '{model_name}' for "
                    "detection task — substituting YOLOv8n"
                )
                return ModelType.YOLOV8N
            # Hard rejection: image-only model for tabular task
            if task_type in (TaskType.REGRESSION, None) and mtype in _IMAGE_ONLY_TYPES:
                # Only reject for explicit regression; classification may use CNN (images)
                pass  # allow — classification could be image-based
            return mtype

    # Fallback by task type when no keyword matched
    if task_type == TaskType.SEGMENTATION:
        return ModelType.UNET
    if task_type == TaskType.OBJECT_DETECTION:
        return ModelType.YOLOV8N
    if task_type == TaskType.REGRESSION:
        return ModelType.XGBOOST
    return ModelType.RANDOM_FOREST


class AutonomousMLPipeline:
    """Fully autonomous ML pipeline using AI agents."""

    def __init__(self, max_experiments: int = 3):
        self.research_agent = get_research_agent()
        self.planner = get_experiment_planner()
        self.improvement_agent = get_improvement_agent()
        self.tracker = get_experiment_tracker()
        self.max_experiments = max_experiments

    async def run_autonomous_training(
        self,
        session_id: str,
        dataset_path: str,
        log_queue: "asyncio.Queue[str]",
        task_type_override: Optional[str] = None,
        dataset_name: str = "Dataset",
    ) -> Optional[Dict[str, Any]]:
        """Run full autonomous ML pipeline."""

        def log(msg: str):
            logger.info(msg)
            log_queue.put_nowait(msg)

        dataset_path_obj = Path(dataset_path)

        try:
            log("🤖 AUTONOMOUS ML ENGINEER STARTING")
            log("=" * 60)
            log(f"  Dataset: {dataset_name}")
            log(f"  Session: {session_id[:12]}...")

            # ── Step 1: Analyze dataset ──────────────────────────────────────
            log("\n📊 STEP 1: Dataset Analysis")
            log("-" * 40)

            dataset_info = await self._analyze_dataset(
                session_id, dataset_path_obj, log, task_type_override
            )
            if not dataset_info:
                log("❌ Dataset analysis failed — check that files were uploaded correctly")
                return None

            log(f"  Samples : {dataset_info['num_samples']}")
            log(f"  Classes : {dataset_info['num_classes']}")
            log(f"  Task    : {dataset_info['task_type'].value}")
            log(f"  Format  : {dataset_info['format']}")

            # ── Step 1.5: Preprocess ─────────────────────────────────────────
            log("\n🧹 STEP 1.5: Preprocessing")
            log("-" * 40)
            try:
                loop = asyncio.get_event_loop()
                task_hint = dataset_info["task_type"]  # pass user override to preprocessor
                preprocess_result = await loop.run_in_executor(
                    None,
                    lambda: preprocess_dataset(
                        dataset_path_obj,
                        PreprocessConfig(session_id=session_id),
                        task_type_hint=task_hint,
                    ),
                )
                # preprocess_dataset returns (DatasetInfo, PreprocessReport).
                # If segmentation was requested but no masks were found, it downgrades
                # task_type to classification. Sync that back into dataset_info.
                if preprocess_result:
                    actual_info, preprocess_report = preprocess_result
                    if actual_info.task_type != dataset_info["task_type"]:
                        log(
                            f"  ⚠️  Task downgraded: {dataset_info['task_type'].value} → "
                            f"{actual_info.task_type.value} (dataset has no masks)"
                        )
                        dataset_info["task_type"] = actual_info.task_type
                    for warning in preprocess_report.warnings:
                        log(f"  ⚠️  {warning}")
                log("  ✓ Preprocessing complete")
            except Exception as exc:
                log(f"  ❌ Preprocessing failed: {exc}")
                log(f"  Detail: {traceback.format_exc().splitlines()[-1]}")
                return None

            # ── Step 2: LLM model research ───────────────────────────────────
            log("\n🧠 STEP 2: Model Research (LLM)")
            log("-" * 40)

            recommendations = self._research_models(dataset_info, log)
            if not recommendations:
                log("  ⚠ No recommendations — using default models")
                from app.agents.research_agent import ModelRecommendation
                if dataset_info["task_type"] == TaskType.SEGMENTATION:
                    recommendations = [
                        ModelRecommendation("UNet", "Default segmentation architecture", 1, 0.80),
                        ModelRecommendation("DeepLabV3", "Advanced segmentation backbone", 2, 0.84),
                    ]
                else:
                    recommendations = [
                        ModelRecommendation("RandomForest", "Reliable baseline for tabular features", 1, 0.75),
                        ModelRecommendation("XGBoost", "Gradient boosting alternative", 2, 0.78),
                    ]

            for i, rec in enumerate(recommendations, 1):
                log(f"  {i}. {rec.model}: {rec.reason[:80]}")

            # ── Step 3: Plan experiments ─────────────────────────────────────
            log("\n📋 STEP 3: Experiment Planning")
            log("-" * 40)

            experiment_configs = self.planner.plan_experiments(
                models=[r.model for r in recommendations],
                task_type=dataset_info["task_type"],
                dataset_size=dataset_info["num_samples"],
                num_classes=dataset_info["num_classes"],
                dataset_format=dataset_info["format"],
                hardware="cpu",
                max_experiments=self.max_experiments,
            )

            if not experiment_configs:
                log("  ❌ No experiment configs produced")
                return None

            log(f"  Planned {len(experiment_configs)} experiment(s)")

            # ── Step 4: Train & evaluate ─────────────────────────────────────
            log("\n⚙️  STEP 4: Training Loop")
            log("-" * 40)

            best_result = await self._run_experiment_loop(
                session_id=session_id,
                dataset_path=dataset_path,
                experiment_configs=experiment_configs,
                dataset_info=dataset_info,
                log=log,
                log_queue=log_queue,
            )

            # ── Step 5: Results ──────────────────────────────────────────────
            log("\n🏆 STEP 5: Summary")
            log("-" * 40)

            summary = self.tracker.get_experiment_summary(session_id)
            best_model = summary.get("best_model", "unknown")
            completed = summary.get("completed_experiments", 0)
            total_time = summary.get("total_training_time_s", 0.0)
            task = dataset_info["task_type"]
            score_val = 0.0

            # Pull score from best_result (tracker may lack r2/IoU awareness)
            if best_result:
                best_model = best_result.get("model", best_model) or best_model
                metrics = best_result.get("metrics", {})
                if task == TaskType.SEGMENTATION:
                    score_val = metrics.get("iou", 0.0)
                    dice = metrics.get("dice", 0.0)
                    score_label = f"IoU={score_val:.4f}  Dice={dice:.4f}"
                elif task == TaskType.REGRESSION:
                    score_val = metrics.get("r2", 0.0)
                    score_label = f"R² = {score_val:.4f}"
                else:
                    score_val = metrics.get("accuracy", 0.0)
                    score_label = f"{score_val:.4f} ({score_val*100:.1f}%)"
            else:
                score_val = 0.0
                score_label = "N/A"

            log(f"  Best Model  : {best_model}")
            log(f"  Best Score  : {score_label}")
            log(f"  Experiments : {completed}")
            log(f"  Total Time  : {total_time:.1f}s")
            log("\n" + "=" * 60)
            log("✅ AUTONOMOUS TRAINING COMPLETE")

            return {
                "status": "completed",
                "best_model": best_model,
                "best_accuracy": score_val,
                "experiments_count": completed,
                "total_training_time": total_time,
                "summary": summary,
            }

        except Exception as exc:
            log(f"❌ Pipeline error: {exc}")
            log(f"   {traceback.format_exc().splitlines()[-1]}")
            logger.error("Autonomous pipeline error", exc_info=True)
            return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _analyze_dataset(
        self,
        session_id: str,
        dataset_path: Path,
        log,
        task_type_override: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            from app.services.preprocessor import get_dataset_summary
            from app.core.task_router import detect_task, TASK_IMAGE_SEGMENTATION, TASK_IMAGE_CLASSIFICATION

            # Strict task detection — fail-fast on invalid dataset structure
            try:
                canonical_task = detect_task(dataset_path)
                log(f"  Task router: detected '{canonical_task}'")
            except ValueError as ve:
                log(f"  ❌ Dataset structure error: {ve}")
                log("  Cannot proceed — fix the dataset structure and re-upload.")
                return None

            summary = get_dataset_summary(dataset_path, session_id)

            # Determine task type: user override takes precedence, but validate against detected
            if task_type_override:
                try:
                    task_type = TaskType(task_type_override)
                    # Warn if override conflicts with detected task
                    detected_schema_value = _canonical_to_task_type(canonical_task)
                    if detected_schema_value and task_type != detected_schema_value:
                        log(
                            f"  ⚠️  Task override '{task_type.value}' differs from detected "
                            f"'{canonical_task}' — proceeding with override"
                        )
                except ValueError:
                    task_type = _canonical_to_task_type(canonical_task) or TaskType.CLASSIFICATION
            else:
                # Use router result as authoritative source
                task_type = _canonical_to_task_type(canonical_task) or TaskType(
                    summary.get("task_type", "classification")
                )

            num_samples = summary.get("num_samples", 0)
            num_classes = summary.get("num_classes", 0)

            if num_samples == 0:
                log("  ⚠ No samples detected — check dataset structure")

            return {
                "num_samples": num_samples,
                "num_classes": num_classes,
                "task_type": task_type,
                "format": summary.get("format", "image_folder"),
                "image_resolution": summary.get("image_resolution", (224, 224)),
                "imbalance_ratio": summary.get("imbalance_ratio", 1.0),
            }
        except Exception as exc:
            log(f"  ❌ Analysis error: {exc}")
            logger.error("Dataset analysis error", exc_info=True)
            return None

    def _research_models(self, dataset_info: Dict[str, Any], log) -> Optional[List]:
        try:
            return self.research_agent.analyze_dataset(
                task_type=dataset_info["task_type"],
                dataset_size=dataset_info["num_samples"],
                num_classes=dataset_info["num_classes"],
                image_resolution=dataset_info.get("image_resolution"),
                imbalance_ratio=dataset_info["imbalance_ratio"],
            )
        except Exception as exc:
            log(f"  ⚠ Research agent error: {exc}")
            return None

    async def _run_experiment_loop(
        self,
        session_id: str,
        dataset_path: str,
        experiment_configs: List[ExperimentConfig],
        dataset_info: Dict[str, Any],
        log,
        log_queue: "asyncio.Queue[str]",
    ) -> Optional[Dict[str, Any]]:

        best_result = None
        seg_allowed = {ModelType.UNET, ModelType.DEEPLABV3, ModelType.YOLOV8N_SEG, ModelType.YOLOV8S_SEG}

        for i, config in enumerate(experiment_configs, 1):
            model_type = _map_model_name(config.model, dataset_info["task_type"])
            if dataset_info["task_type"] == TaskType.SEGMENTATION and model_type not in seg_allowed:
                log(f"\n  [{i}/{len(experiment_configs)}] Skipping {config.model} (not a segmentation model)")
                continue
            log(f"\n  [{i}/{len(experiment_configs)}] {config.model} → {model_type.value}")
            log(f"    lr={config.learning_rate}  batch={config.batch_size}  epochs={config.epochs}")

            exp = self.tracker.create_experiment(session_id, config.model, config.to_dict())

            try:
                result = await self._train_single_experiment(
                    exp_id=exp.id,
                    session_id=session_id,
                    config=config,
                    model_type=model_type,
                    task_type=dataset_info["task_type"],
                    log_queue=log_queue,
                )
            except Exception as exc:
                log(f"    ❌ Experiment exception: {exc}")
                logger.error(f"Experiment {i} error", exc_info=True)
                result = None

            if result:
                metrics = result.get("metrics", {})
                task = dataset_info["task_type"]
                if task == TaskType.SEGMENTATION:
                    iou = metrics.get("iou", 0.0)
                    dice = metrics.get("dice", 0.0)
                    log(f"    ✓ Done  IoU={iou:.4f}  Dice={dice:.4f}")
                    primary_score = iou
                elif task == TaskType.REGRESSION:
                    r2 = metrics.get("r2", 0.0)
                    rmse = metrics.get("rmse", 0.0)
                    log(f"    ✓ Done  r2={r2:.4f}  rmse={rmse:.4f}")
                    primary_score = r2
                else:
                    accuracy = metrics.get("accuracy", 0.0)
                    f1 = metrics.get("f1", 0.0)
                    log(f"    ✓ Done  accuracy={accuracy:.4f}  f1={f1:.4f}")
                    primary_score = accuracy
                result["_primary_score"] = primary_score
                self.tracker.update_metrics(exp.id, metrics, "completed",
                                            result.get("training_time", 0))
                if not best_result or primary_score > best_result.get("_primary_score", 0):
                    best_result = result
            else:
                log(f"    ✗ Experiment failed")
                self.tracker.update_metrics(exp.id, {}, "failed")

        return best_result

    async def _train_single_experiment(
        self,
        exp_id: str,
        session_id: str,
        config: ExperimentConfig,
        model_type: ModelType,
        task_type: TaskType,
        log_queue: "asyncio.Queue[str]",
    ) -> Optional[Dict[str, Any]]:
        """Train one experiment, streaming all epoch logs to the frontend queue."""
        training_config = TrainingConfig(
            session_id=session_id,
            model_type=model_type,
            task_type=task_type,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            hyperparams=config.to_dict(),
        )

        t_start = time.time()
        # Pass the main log_queue so every epoch line flows directly to the UI
        result = await train_model(training_config, log_queue)
        training_time = time.time() - t_start

        metrics = result.metrics if result else {}
        return {
            "model": config.model,
            "model_type": model_type.value,
            "metrics": metrics,
            "training_time": training_time,
            "checkpoint": result.checkpoint_path if result else "",
        }


# ── Public API ────────────────────────────────────────────────────────────────

async def run_autonomous_pipeline(
    session_id: str,
    dataset_path: str,
    log_queue: "asyncio.Queue[str]",
    max_experiments: int = 3,
    task_type_override: Optional[str] = None,
    dataset_name: str = "Dataset",
) -> Optional[Dict[str, Any]]:
    """Run full autonomous ML pipeline."""
    pipeline = AutonomousMLPipeline(max_experiments=max_experiments)
    return await pipeline.run_autonomous_training(
        session_id=session_id,
        dataset_path=dataset_path,
        log_queue=log_queue,
        task_type_override=task_type_override,
        dataset_name=dataset_name,
    )


async def run_pilot(
    config: TrainingConfig,
    log_queue: "asyncio.Queue[str]",
) -> Dict[str, Any]:
    """Legacy pilot run function (for compatibility)."""
    await log_queue.put(f"Starting pilot run ({settings.PILOT_EPOCHS} epochs)...")
    pilot_config = config.model_copy(update={"epochs": settings.PILOT_EPOCHS, "pilot": True})
    t0 = time.time()
    result = await train_model(pilot_config, log_queue)
    pilot_time = time.time() - t0
    estimate_seconds = (pilot_time / settings.PILOT_EPOCHS) * config.epochs
    summary: Dict[str, Any] = {
        "pilot_epochs": settings.PILOT_EPOCHS,
        "pilot_time_s": round(pilot_time, 1),
        "estimated_full_training_s": round(estimate_seconds, 1),
        "estimated_full_training_min": round(estimate_seconds / 60, 1),
        "early_metrics": result.metrics if result else {},
        "checkpoint_path": result.checkpoint_path if result else "",
    }
    await log_queue.put(
        f"Pilot complete. Estimated full training: {summary['estimated_full_training_min']:.1f} min."
    )
    logger.info(f"Pilot run summary: {summary}")
    return summary
