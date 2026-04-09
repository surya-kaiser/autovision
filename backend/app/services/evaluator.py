"""
Model evaluation and metrics aggregation.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.models.schemas import ModelResult, TaskType
from app.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    session_id: str,
    model_type: str,
    task_type: TaskType,
) -> Dict[str, Any]:
    """Load test data and evaluate a saved model."""
    results: Dict[str, Any] = {"session_id": session_id, "model_type": model_type}

    test_csv = settings.UPLOAD_DIR / session_id / "preprocessed" / "test.csv"
    ckpt = settings.MODEL_DIR / session_id / f"{model_type}_best.pkl"

    if not test_csv.exists() or not ckpt.exists():
        results["error"] = "Test data or model checkpoint not found"
        return results

    import pickle
    with open(ckpt, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(test_csv)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    try:
        preds = model.predict(X)

        if task_type == TaskType.CLASSIFICATION:
            from sklearn.metrics import (
                accuracy_score, f1_score,
                confusion_matrix, classification_report,
            )
            results["accuracy"] = float(accuracy_score(y, preds))
            results["f1"] = float(f1_score(y, preds, average="weighted", zero_division=0))
            cm = confusion_matrix(y, preds)
            results["confusion_matrix"] = cm.tolist()
            report = classification_report(y, preds, output_dict=True, zero_division=0)
            results["per_class"] = report

            # Try to get probabilities
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                results["has_proba"] = True

        elif task_type == TaskType.REGRESSION:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            results["rmse"] = float(np.sqrt(mean_squared_error(y, preds)))
            results["mae"] = float(mean_absolute_error(y, preds))
            results["r2"] = float(r2_score(y, preds))

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        results["error"] = str(e)

    return results


def get_all_results(session_id: str) -> List[Dict[str, Any]]:
    """Return all model results for a session."""
    results_path = settings.MODEL_DIR / session_id / "results.json"
    if results_path.exists():
        try:
            data = json.loads(results_path.read_text())
            return [data] if isinstance(data, dict) else data
        except Exception:
            pass
    return []


def compare_models(session_id: str) -> List[Dict[str, Any]]:
    """Compare all trained models in a session by accuracy/mAP."""
    model_dir = settings.MODEL_DIR / session_id
    all_results = []

    if not model_dir.exists():
        return all_results

    for ckpt in model_dir.glob("*_best.pkl"):
        model_type = ckpt.stem.replace("_best", "")
        results_file = model_dir / "results.json"
        if results_file.exists():
            try:
                data = json.loads(results_file.read_text())
                if isinstance(data, dict):
                    all_results.append(data)
            except Exception:
                pass

    return all_results
