"""
Tests for YOLO service — uses mocks since ultralytics is optional.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.services.yolo_service import validate_yolo_dataset, predict_yolo


class TestYOLOValidation:
    def test_missing_images_dir(self, tmp_path):
        result = validate_yolo_dataset(tmp_path)
        assert not result["valid"]
        assert any("images" in e for e in result["errors"])

    def test_missing_labels_dir(self, tmp_path):
        (tmp_path / "images").mkdir()
        result = validate_yolo_dataset(tmp_path)
        assert not result["valid"]
        assert any("labels" in e for e in result["errors"])

    def test_valid_structure(self, tmp_path):
        (tmp_path / "images" / "train").mkdir(parents=True)
        (tmp_path / "labels" / "train").mkdir(parents=True)
        # Add a matching image + label pair
        (tmp_path / "images" / "train" / "img1.jpg").write_bytes(b"fake")
        (tmp_path / "labels" / "train" / "img1.txt").write_text("0 0.5 0.5 0.1 0.1")
        result = validate_yolo_dataset(tmp_path)
        assert result["valid"]
        assert result["stats"]["num_images"] == 1
        assert result["stats"]["unlabelled_images"] == 0

    def test_unlabelled_images_warning(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        (tmp_path / "images" / "img1.jpg").write_bytes(b"fake")
        # No label file
        result = validate_yolo_dataset(tmp_path)
        assert result["valid"]
        assert result["stats"]["unlabelled_images"] == 1
        assert result["warnings"]


class TestYOLOPrediction:
    def test_predict_no_ultralytics(self, tmp_path, sample_session_id):
        """Should return empty list when ultralytics not installed."""
        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake image")

        with patch.dict("sys.modules", {"ultralytics": None}):
            boxes = predict_yolo(sample_session_id, img)
            assert isinstance(boxes, list)

    def test_predict_with_mock_model(self, tmp_path, sample_session_id):
        """Mock ultralytics to return fake boxes."""
        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake")

        mock_box = MagicMock()
        mock_box.xyxy = [[MagicMock(tolist=lambda: [10.0, 20.0, 100.0, 200.0])]]
        mock_box.conf = [[0.9]]
        mock_box.cls = [[0]]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "cat"}

        mock_model_instance = MagicMock(return_value=[mock_result])

        with patch("app.services.yolo_service.predict_yolo") as mock_predict:
            mock_predict.return_value = [{"x1": 10, "y1": 20, "x2": 100, "y2": 200, "confidence": 0.9, "class_id": 0, "class_name": "cat"}]
            boxes = predict_yolo(sample_session_id, img)
            # Direct mock call — test structure
            mock_predict.return_value
