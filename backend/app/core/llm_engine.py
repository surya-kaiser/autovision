"""
LLM engine using Ollama (free local inference).
Falls back to rule-based recommendations if Ollama is not available.
"""
import json
import requests
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.models.schemas import LLMRecommendation, TaskType, ModelType, ChatMessage
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an expert ML engineering assistant embedded in AutoVision, an MLOps platform.
Your job is to analyze datasets and recommend the best ML approach.
Always respond with valid JSON when asked for structured recommendations.
Be concise, practical, and explain your reasoning clearly."""

# Hard-coded allowed models per task — LLM output is validated against this.
# Any LLM suggestion that violates these rules is rejected and replaced with
# the rule-based fallback. This prevents hallucinated tabular models from
# being used on image/segmentation tasks.
ALLOWED_MODELS_BY_TASK: Dict[str, set] = {
    "segmentation":      {"unet", "deeplabv3", "yolov8n-seg", "yolov8s-seg"},
    "object_detection":  {"yolov8n", "yolov8s", "yolov8m", "yolov8n-seg", "yolov8s-seg"},
    "classification":    {"cnn", "resnet", "random_forest", "xgboost", "lightgbm"},
    "regression":        {"xgboost", "lightgbm", "random_forest", "linear_regression", "ridge"},
}


def _validate_model_for_task(model_type_str: str, task_type: "TaskType") -> bool:
    """Return True if model_type_str is allowed for task_type, False otherwise."""
    allowed = ALLOWED_MODELS_BY_TASK.get(task_type.value if hasattr(task_type, "value") else str(task_type))
    if allowed is None:
        return True  # UNKNOWN task — don't block
    return model_type_str.lower().replace(" ", "_").replace("-", "_") in {
        m.replace("-", "_") for m in allowed
    }


def _call_ollama(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Call local Ollama API. Returns text response."""
    try:
        payload = {
            "model": settings.OLLAMA_MODEL,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 1024},
        }
        resp = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        logger.warning(f"Ollama unavailable: {e}. Using rule-based fallback.")
        return ""


def _rule_based_recommendation(
    task_type: TaskType,
    num_samples: int,
    num_features: int,
    num_classes: int,
    dataset_format: str = "csv",
) -> LLMRecommendation:
    """Fallback: rule-based model recommendation."""
    if task_type == TaskType.OBJECT_DETECTION:
        return LLMRecommendation(
            model_type=ModelType.YOLOV8N.value,
            hyperparams={"epochs": 50, "batch_size": 16, "imgsz": 640},
            preprocessing_strategy="YOLO format with mosaic augmentation",
            explanation="YOLOv8n is lightweight and fast for object detection. Upgrade to YOLOv8s for higher mAP.",
            estimated_training_minutes=10,
        )
    elif task_type == TaskType.REGRESSION:
        model = ModelType.XGBOOST if num_samples > 1000 else ModelType.RIDGE
        return LLMRecommendation(
            model_type=model.value,
            hyperparams={"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
            preprocessing_strategy="StandardScaler + outlier removal (IQR)",
            explanation=f"{'XGBoost' if num_samples > 1000 else 'Ridge regression'} suits this regression dataset size best.",
            estimated_training_minutes=2,
        )
    elif task_type == TaskType.SEGMENTATION:
        return LLMRecommendation(
            model_type=ModelType.UNET.value,
            hyperparams={"epochs": 30, "batch_size": 8, "learning_rate": 0.001},
            preprocessing_strategy="Resize to 224x224, normalize with ImageNet stats, horizontal flip + rotation augmentation",
            explanation=(
                "U-Net is the standard architecture for semantic segmentation. "
                "It preserves spatial information through skip connections between encoder and decoder. "
                "Requires images/ and masks/ directories. Use DeepLabV3 for multi-scale context."
            ),
            estimated_training_minutes=10,
        )
    elif task_type == TaskType.CLASSIFICATION:
        # Image datasets — use CNN/ResNet (NOT tabular models on flattened pixels)
        if dataset_format in ("image_folder", "zip"):
            if num_samples > 5000:
                model = ModelType.RESNET
                hparams = {"epochs": 15, "batch_size": 32, "learning_rate": 0.001, "weight_decay": 1e-4}
                minutes = 8
                strategy = "Resize to 224x224, ImageNet normalization, random flip + rotation + color jitter"
                explanation = (
                    f"Image classification with {num_classes} classes and {num_samples} samples. "
                    "ResNet18 with ImageNet pre-trained weights provides strong transfer learning. "
                    "Spatial features are preserved through convolutional layers."
                )
            else:
                model = ModelType.CNN
                hparams = {"epochs": 20, "batch_size": 32, "learning_rate": 0.001, "weight_decay": 1e-4}
                minutes = 5
                strategy = "Resize to 224x224, ImageNet normalization, random flip + rotation augmentation"
                explanation = (
                    f"Image classification with {num_classes} classes and {num_samples} samples. "
                    "Custom 3-block CNN is lightweight and less prone to overfitting on small datasets. "
                    "Convolutional layers preserve spatial relationships in images."
                )
            return LLMRecommendation(
                model_type=model.value,
                hyperparams=hparams,
                preprocessing_strategy=strategy,
                explanation=explanation,
                estimated_training_minutes=minutes,
            )
        # Tabular datasets
        if num_samples < 500:
            model, hparams, mins = ModelType.RANDOM_FOREST, {"n_estimators": 150, "max_depth": 12}, 1
        elif num_samples < 5000:
            model, hparams, mins = ModelType.XGBOOST, {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05}, 2
        else:
            model, hparams, mins = ModelType.LIGHTGBM, {"n_estimators": 300, "num_leaves": 63, "learning_rate": 0.05}, 3
        return LLMRecommendation(
            model_type=model.value,
            hyperparams=hparams,
            preprocessing_strategy="StandardScaler + one-hot encoding",
            explanation=f"{num_samples} samples suggests {model.value} for the best accuracy/speed trade-off.",
            estimated_training_minutes=mins,
        )
    else:
        return LLMRecommendation(
            model_type=ModelType.RANDOM_FOREST.value,
            hyperparams={"n_estimators": 100},
            preprocessing_strategy="StandardScaler",
            explanation="Random Forest is a robust default across most task types.",
            estimated_training_minutes=2,
        )


def get_recommendation(
    task_type: TaskType,
    dataset_summary: Dict[str, Any],
) -> LLMRecommendation:
    """Get LLM or rule-based model recommendation."""
    num_samples = dataset_summary.get("num_samples", 0)
    num_features = dataset_summary.get("num_features", 0)
    num_classes = dataset_summary.get("num_classes", 0)
    class_dist = dataset_summary.get("class_distribution", {})

    prompt = f"""Analyze this dataset and recommend the best ML approach.

Dataset Summary:
- Task type: {task_type.value}
- Number of samples: {num_samples}
- Number of features: {num_features}
- Number of classes: {num_classes}
- Class distribution: {json.dumps(class_dist)}
- Dataset format: {dataset_summary.get('format', 'csv')}
- Sample stats: {json.dumps(dataset_summary.get('column_stats', {}))}

Respond ONLY with a JSON object in this exact format:
{{
  "model_type": "<one of: random_forest, xgboost, lightgbm, cnn, resnet, unet, deeplabv3, yolov8n, yolov8s, yolov8n-seg, yolov8s-seg, linear_regression, ridge>",
  "hyperparams": {{"key": "value"}},
  "preprocessing_strategy": "<description>",
  "explanation": "<2-3 sentence explanation>",
  "estimated_training_minutes": <integer>
}}"""

    raw = _call_ollama(prompt)

    if raw:
        try:
            # Extract JSON from response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
                rec = LLMRecommendation(**data)
                # Validate LLM output — reject hallucinated models for the task
                if not _validate_model_for_task(rec.model_type, task_type):
                    logger.warning(
                        f"LLM returned '{rec.model_type}' for task '{task_type.value}' — "
                        f"not in allowed set {ALLOWED_MODELS_BY_TASK.get(task_type.value)}. "
                        "Falling back to rule-based recommendation."
                    )
                    # Fall through to rule-based below
                else:
                    return rec
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}. Using rule-based fallback.")

    dataset_format = dataset_summary.get("format", "csv")
    return _rule_based_recommendation(task_type, num_samples, num_features, num_classes, dataset_format)


def chat(message: str, history: List[ChatMessage], dataset_context: str = "") -> str:
    """Chat with the LLM about the dataset."""
    context = f"\nDataset context: {dataset_context}\n" if dataset_context else ""

    history_text = "\n".join(
        f"{m.role.upper()}: {m.content}" for m in history[-6:]  # last 6 turns
    )

    prompt = f"""{context}
Conversation history:
{history_text}

USER: {message}

ASSISTANT:"""

    raw = _call_ollama(prompt)

    if not raw:
        return _fallback_chat_response(message)

    return raw.strip()


def _fallback_chat_response(message: str) -> str:
    msg_lower = message.lower()
    if "why" in msg_lower and "model" in msg_lower:
        return (
            "The model was selected based on your dataset size and task type. "
            "Larger datasets benefit from gradient boosting methods like XGBoost/LightGBM, "
            "while smaller datasets do well with Random Forest."
        )
    elif "faster" in msg_lower:
        return (
            "For faster training, try reducing epochs, using a smaller model variant, "
            "or reducing the dataset size with stratified sampling."
        )
    elif "accuracy" in msg_lower or "better" in msg_lower:
        return (
            "To improve accuracy: try feature engineering, increase training epochs, "
            "tune hyperparameters, or use an ensemble of models."
        )
    else:
        return (
            "I'm AutoVision's ML assistant. I can help you understand model choices, "
            "hyperparameter tuning, and preprocessing strategies. "
            "Note: Ollama LLM is currently offline — connect it for richer responses."
        )


def check_ollama_available() -> bool:
    try:
        resp = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False
