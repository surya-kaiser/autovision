"""
Shared pytest fixtures for AutoVision test suite.
"""
import io
import csv
import json
import uuid
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient


# ── Paths ─────────────────────────────────────────────────────────────────────

TEST_DIR = Path("/tmp/autovision/test_data")
TEST_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_csv_path() -> Path:
    """Simple iris-like classification CSV."""
    path = TEST_DIR / "sample.csv"
    rows = [
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "species": "setosa"},
        {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3, "species": "virginica"},
        {"sepal_length": 5.8, "sepal_width": 2.7, "petal_length": 4.1, "petal_width": 1.0, "species": "versicolor"},
    ] * 30  # 90 rows
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_csv_with_missing(tmp_path) -> Path:
    """CSV with missing values and categoricals."""
    path = tmp_path / "missing.csv"
    data = {
        "age": [25, None, 35, 28, None],
        "income": [50000, 60000, None, 45000, 70000],
        "city": ["NYC", "LA", "NYC", None, "Chicago"],
        "target": [1, 0, 1, 0, 1],
    }
    pd.DataFrame(data).to_csv(path, index=False)
    return path


@pytest.fixture
def sample_session_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def preprocess_config(sample_session_id):
    from app.models.schemas import PreprocessConfig
    return PreprocessConfig(session_id=sample_session_id)


# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


# ── Training fixture ──────────────────────────────────────────────────────────

@pytest.fixture
def trained_session(sample_csv_path, sample_session_id, tmp_path):
    """Pre-process a CSV and return session_id with data ready."""
    from app.services.preprocessor import preprocess_dataset
    from app.models.schemas import PreprocessConfig
    from app.core.config import settings

    # Copy to upload dir
    dest = settings.UPLOAD_DIR / sample_session_id / "sample.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(sample_csv_path.read_bytes())

    config = PreprocessConfig(session_id=sample_session_id)
    preprocess_dataset(dest, config)
    return sample_session_id
