"""
API endpoint tests using FastAPI TestClient.
"""
import io
import json
import pytest
import pandas as pd
from pathlib import Path


class TestHealth:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_system_info(self, client):
        r = client.get("/api/v1/system/info")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"

    def test_llm_status(self, client):
        r = client.get("/api/v1/llm/status")
        assert r.status_code == 200
        assert "ollama_online" in r.json()["data"]


class TestDatasetEndpoints:
    def test_upload_csv(self, client, sample_csv_path):
        with open(sample_csv_path, "rb") as f:
            r = client.post(
                "/api/v1/dataset/upload",
                files={"file": ("sample.csv", f, "text/csv")},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert "session_id" in data["data"]

    def test_upload_then_preview(self, client, sample_csv_path):
        with open(sample_csv_path, "rb") as f:
            r = client.post(
                "/api/v1/dataset/upload",
                files={"file": ("sample.csv", f, "text/csv")},
            )
        sid = r.json()["data"]["session_id"]

        r2 = client.get(f"/api/v1/dataset/preview/{sid}")
        assert r2.status_code == 200
        prev = r2.json()["data"]
        assert "columns" in prev or "count" in prev

    def test_preprocess_endpoint(self, client, sample_csv_path):
        # Upload first
        with open(sample_csv_path, "rb") as f:
            r = client.post("/api/v1/dataset/upload", files={"file": ("sample.csv", f, "text/csv")})
        sid = r.json()["data"]["session_id"]

        # Preprocess
        r2 = client.post("/api/v1/dataset/preprocess", json={
            "session_id": sid,
            "scale_method": "standard",
            "handle_missing": "auto",
            "augmentation": True,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        })
        assert r2.status_code == 200
        assert r2.json()["status"] == "success"

    def test_preview_not_found(self, client):
        r = client.get("/api/v1/dataset/preview/nonexistent-session")
        assert r.status_code == 404

    def test_recommend_endpoint(self, client, sample_csv_path):
        with open(sample_csv_path, "rb") as f:
            r = client.post("/api/v1/dataset/upload", files={"file": ("sample.csv", f, "text/csv")})
        sid = r.json()["data"]["session_id"]

        r2 = client.get(f"/api/v1/dataset/recommend/{sid}", params={"task_type": "classification"})
        assert r2.status_code == 200
        rec = r2.json()["data"]
        assert "model_type" in rec
        assert "explanation" in rec


class TestTrainingEndpoints:
    def test_start_training(self, client, trained_session):
        r = client.post("/api/v1/training/start", json={
            "session_id": trained_session,
            "model_type": "random_forest",
            "task_type": "classification",
            "epochs": 3,
            "batch_size": 16,
            "pilot": True,
            "hyperparams": {"n_estimators": 10},
        })
        assert r.status_code == 200
        assert r.json()["status"] == "success"

    def test_results_endpoint(self, client, trained_session):
        r = client.get(f"/api/v1/training/results/{trained_session}")
        assert r.status_code == 200


class TestInferenceEndpoints:
    def test_chat_endpoint(self, client):
        r = client.get("/api/v1/inference/chat", params={
            "session_id": "test",
            "message": "What model should I use?",
        })
        assert r.status_code == 200
        assert "response" in r.json()["data"]
