"""
Tests for LLM engine — mocks Ollama to avoid network calls.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from app.core.llm_engine import (
    get_recommendation,
    chat,
    check_ollama_available,
    _rule_based_recommendation,
    _fallback_chat_response,
)
from app.models.schemas import TaskType, ModelType


class TestRuleBasedFallback:
    def test_classification_small(self):
        rec = _rule_based_recommendation(TaskType.CLASSIFICATION, 100, 5, 2)
        assert rec.model_type == ModelType.RANDOM_FOREST

    def test_classification_medium(self):
        rec = _rule_based_recommendation(TaskType.CLASSIFICATION, 2000, 10, 5)
        assert rec.model_type == ModelType.XGBOOST

    def test_classification_large(self):
        rec = _rule_based_recommendation(TaskType.CLASSIFICATION, 10000, 20, 3)
        assert rec.model_type == ModelType.LIGHTGBM

    def test_regression(self):
        rec = _rule_based_recommendation(TaskType.REGRESSION, 200, 5, 1)
        assert rec.model_type in (ModelType.XGBOOST, ModelType.RIDGE)

    def test_object_detection(self):
        rec = _rule_based_recommendation(TaskType.OBJECT_DETECTION, 500, 0, 5)
        assert rec.model_type == ModelType.YOLOV8N

    def test_explanation_not_empty(self):
        rec = _rule_based_recommendation(TaskType.CLASSIFICATION, 1000, 5, 3)
        assert len(rec.explanation) > 10


class TestLLMRecommendation:
    def test_uses_ollama_when_available(self):
        mock_response = {
            "model_type": "xgboost",
            "hyperparams": {"n_estimators": 200},
            "preprocessing_strategy": "StandardScaler",
            "explanation": "XGBoost is great for tabular data.",
            "estimated_training_minutes": 3,
        }
        with patch("app.core.llm_engine._call_ollama", return_value=json.dumps(mock_response)):
            rec = get_recommendation(
                TaskType.CLASSIFICATION,
                {"num_samples": 5000, "num_features": 20, "num_classes": 3},
            )
            assert rec.model_type == "xgboost"

    def test_falls_back_on_bad_json(self):
        with patch("app.core.llm_engine._call_ollama", return_value="not json"):
            rec = get_recommendation(
                TaskType.CLASSIFICATION,
                {"num_samples": 500, "num_features": 5, "num_classes": 2},
            )
            assert rec.model_type is not None

    def test_falls_back_on_ollama_offline(self):
        with patch("app.core.llm_engine._call_ollama", return_value=""):
            rec = get_recommendation(
                TaskType.REGRESSION,
                {"num_samples": 100, "num_features": 3, "num_classes": 0},
            )
            assert rec.model_type is not None


class TestChat:
    def test_chat_with_ollama(self):
        with patch("app.core.llm_engine._call_ollama", return_value="Use XGBoost for best results."):
            response = chat("What model?", [], dataset_context="CSV with 1000 rows")
            assert "XGBoost" in response

    def test_fallback_chat(self):
        response = _fallback_chat_response("why did you choose this model?")
        assert len(response) > 10

    def test_ollama_status_offline(self):
        with patch("requests.get", side_effect=Exception("connection refused")):
            assert check_ollama_available() is False


class TestOllamaCheck:
    def test_ollama_online(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("requests.get", return_value=mock_resp):
            assert check_ollama_available() is True
