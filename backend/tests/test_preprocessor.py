"""
Tests for the preprocessing pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from app.models.schemas import PreprocessConfig, DatasetFormat, TaskType
from app.services.preprocessor import (
    detect_format,
    detect_task_type,
    preprocess_dataset,
    CSVPreprocessor,
    get_dataset_summary,
)


class TestDetection:
    def test_detect_csv(self, sample_csv_path):
        assert detect_format(sample_csv_path) == DatasetFormat.CSV

    def test_detect_task_type_classification(self, sample_csv_path):
        fmt = detect_format(sample_csv_path)
        task = detect_task_type(fmt, sample_csv_path)
        assert task == TaskType.CLASSIFICATION

    def test_detect_task_type_regression(self, tmp_path):
        path = tmp_path / "reg.csv"
        df = pd.DataFrame({"x": range(100), "y": np.random.randn(100)})
        df.to_csv(path, index=False)
        fmt = detect_format(path)
        task = detect_task_type(fmt, path)
        # y has many unique values → regression
        assert task == TaskType.REGRESSION


class TestCSVPreprocessor:
    def test_basic_pipeline(self, sample_csv_path, preprocess_config):
        proc = CSVPreprocessor(preprocess_config)
        train, val, test, report = proc.run(sample_csv_path)

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(report.steps_applied) > 0
        assert report.train_size + report.val_size + report.test_size == len(train) + len(val) + len(test)

    def test_split_ratios(self, sample_csv_path, preprocess_config):
        proc = CSVPreprocessor(preprocess_config)
        train, val, test, _ = proc.run(sample_csv_path)
        total = len(train) + len(val) + len(test)
        # Allow ±5 rows slack
        assert abs(len(train) / total - 0.70) < 0.10

    def test_missing_value_handling(self, sample_csv_with_missing, sample_session_id):
        config = PreprocessConfig(session_id=sample_session_id)
        proc = CSVPreprocessor(config)
        train, val, test, report = proc.run(sample_csv_with_missing)
        # After handling, no NaN in numeric columns
        combined = pd.concat([train, val, test])
        numeric = combined.select_dtypes(include=[np.number])
        assert not numeric.isnull().any().any()
        assert report.missing_handled  # some columns were addressed

    def test_outlier_removal(self, sample_session_id, tmp_path):
        # Create data with obvious outlier
        path = tmp_path / "outlier.csv"
        data = {"x": list(range(50)) + [10000], "y": [0, 1] * 25 + [1]}
        pd.DataFrame(data).to_csv(path, index=False)
        config = PreprocessConfig(session_id=sample_session_id)
        proc = CSVPreprocessor(config)
        _, _, _, report = proc.run(path)
        assert report.outliers_removed >= 1

    def test_categorical_encoding(self, sample_csv_path, preprocess_config):
        proc = CSVPreprocessor(preprocess_config)
        train, _, _, report = proc.run(sample_csv_path)
        # 'species' column should be encoded
        assert "species" in report.encodings or len(report.encodings) > 0


class TestPreprocessDataset:
    def test_full_pipeline_csv(self, sample_csv_path, preprocess_config):
        info, report = preprocess_dataset(sample_csv_path, preprocess_config)
        assert info.num_samples > 0
        assert info.format == DatasetFormat.CSV
        assert len(report.steps_applied) > 0

    def test_get_summary(self, sample_csv_path, sample_session_id):
        summary = get_dataset_summary(sample_csv_path, sample_session_id)
        assert "num_samples" in summary
        assert summary["num_samples"] > 0
        assert "num_classes" in summary
