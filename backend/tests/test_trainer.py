"""
Tests for the training pipeline.
"""
import asyncio
import pytest
from pathlib import Path

from app.models.schemas import ModelType, TaskType, TrainingConfig
from app.services.trainer import train_model, get_training_status, load_model


def _make_config(session_id, model_type=ModelType.RANDOM_FOREST, pilot=True):
    return TrainingConfig(
        session_id=session_id,
        model_type=model_type,
        task_type=TaskType.CLASSIFICATION,
        epochs=3,
        batch_size=16,
        pilot=pilot,
        hyperparams={"n_estimators": 10},
    )


class TestTrainer:
    def test_random_forest_pilot(self, trained_session):
        """Random Forest pilot run should complete without error."""
        config = _make_config(trained_session, ModelType.RANDOM_FOREST, pilot=True)
        q: asyncio.Queue = asyncio.Queue()

        result = asyncio.get_event_loop().run_until_complete(train_model(config, q))
        assert result.session_id == trained_session
        # RF doesn't produce 'epoch' metrics but should have a checkpoint
        assert result.checkpoint_path

    def test_checkpoint_saved(self, trained_session):
        config = _make_config(trained_session, ModelType.RANDOM_FOREST, pilot=True)
        q: asyncio.Queue = asyncio.Queue()
        asyncio.get_event_loop().run_until_complete(train_model(config, q))

        model = load_model(trained_session, ModelType.RANDOM_FOREST.value)
        assert model is not None

    def test_training_status_registered(self, trained_session):
        config = _make_config(trained_session, ModelType.RANDOM_FOREST, pilot=True)
        q: asyncio.Queue = asyncio.Queue()
        asyncio.get_event_loop().run_until_complete(train_model(config, q))

        status = get_training_status(trained_session)
        assert status is not None
        assert status.status == "completed"

    def test_metrics_captured(self, trained_session):
        config = _make_config(trained_session, ModelType.RANDOM_FOREST, pilot=True)
        q: asyncio.Queue = asyncio.Queue()
        result = asyncio.get_event_loop().run_until_complete(train_model(config, q))

        # Should have accuracy for classification
        assert result.accuracy is not None or result.metrics

    def test_log_queue_populated(self, trained_session):
        config = _make_config(trained_session, ModelType.RANDOM_FOREST, pilot=True)
        q: asyncio.Queue = asyncio.Queue()
        asyncio.get_event_loop().run_until_complete(train_model(config, q))
        assert not q.empty()

    def test_missing_data_graceful(self, sample_session_id):
        """Training without preprocessed data should fail gracefully."""
        config = _make_config(sample_session_id, ModelType.RANDOM_FOREST, pilot=True)
        q: asyncio.Queue = asyncio.Queue()
        result = asyncio.get_event_loop().run_until_complete(train_model(config, q))
        # Should not raise, result should be a ModelResult
        assert result.session_id == sample_session_id
