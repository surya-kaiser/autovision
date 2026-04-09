"""
Tests for task_router.py — strict task detection and model routing.

Verifies:
  1. Classification dataset  → correct pipeline detected
  2. Segmentation dataset    → U-Net used (not XGBoost)
  3. Tabular dataset         → XGBoost used
  4. Fail-fast: seg without masks → ValueError
  5. LLM validation: tabular rejected for segmentation task
"""
import io
import struct
import zlib
import pytest
from pathlib import Path

from app.core.task_router import (
    TASK_IMAGE_CLASSIFICATION,
    TASK_IMAGE_SEGMENTATION,
    TASK_OBJECT_DETECTION,
    TASK_TABULAR,
    detect_task,
)
from app.models.schemas import ModelType, TaskType


# ── Minimal PNG helper ────────────────────────────────────────────────────────

def _make_png(path: Path) -> None:
    """Write a valid 1×1 red PNG to *path*."""
    def _crc(data: bytes) -> bytes:
        return struct.pack(">I", zlib.crc32(data) & 0xFFFFFFFF)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + _crc(tag + data)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1×1, 8-bit RGB
    ihdr = _chunk(b"IHDR", ihdr_data)
    raw_row = b"\x00\xFF\x00\x00"  # filter=0, R=255 G=0 B=0
    idat = _chunk(b"IDAT", zlib.compress(raw_row))
    iend = _chunk(b"IEND", b"")
    path.write_bytes(sig + ihdr + idat + iend)


def _make_gray_png(path: Path) -> None:
    """Write a valid 1×1 grayscale PNG to *path* (suitable as a segmentation mask)."""
    def _crc(data: bytes) -> bytes:
        return struct.pack(">I", zlib.crc32(data) & 0xFFFFFFFF)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + _crc(tag + data)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)  # 1×1, 8-bit grayscale
    ihdr = _chunk(b"IHDR", ihdr_data)
    raw_row = b"\x00\x01"  # filter=0, pixel=1
    idat = _chunk(b"IDAT", zlib.compress(raw_row))
    iend = _chunk(b"IEND", b"")
    path.write_bytes(sig + ihdr + idat + iend)


# ── Test 1: Classification dataset ────────────────────────────────────────────

def test_classification_dataset_routes_correctly(tmp_path):
    """Image folder with class subdirs → image_classification."""
    (tmp_path / "cats").mkdir()
    (tmp_path / "dogs").mkdir()
    _make_png(tmp_path / "cats" / "cat1.png")
    _make_png(tmp_path / "dogs" / "dog1.png")

    task = detect_task(tmp_path)
    assert task == TASK_IMAGE_CLASSIFICATION, (
        f"Expected 'image_classification', got '{task}'"
    )


# ── Test 2: Segmentation dataset → U-Net, NOT XGBoost ────────────────────────

def test_segmentation_dataset_routes_to_unet(tmp_path):
    """Dataset with images/ + masks/ → image_segmentation."""
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    _make_png(tmp_path / "images" / "img1.png")
    _make_gray_png(tmp_path / "masks" / "img1.png")

    task = detect_task(tmp_path)
    assert task == TASK_IMAGE_SEGMENTATION, (
        f"Expected 'image_segmentation', got '{task}'"
    )


def test_map_model_name_rejects_xgboost_for_segmentation():
    """_map_model_name must return U-Net, not XGBoost, for segmentation task."""
    from app.services.pilot_runner import _map_model_name

    result = _map_model_name("xgboost", task_type=TaskType.SEGMENTATION)
    assert result == ModelType.UNET, (
        f"Expected ModelType.UNET for xgboost+segmentation, got {result}"
    )

    result2 = _map_model_name("lightgbm", task_type=TaskType.SEGMENTATION)
    assert result2 == ModelType.UNET, (
        f"Expected ModelType.UNET for lightgbm+segmentation, got {result2}"
    )

    result3 = _map_model_name("random_forest", task_type=TaskType.SEGMENTATION)
    assert result3 == ModelType.UNET, (
        f"Expected ModelType.UNET for random_forest+segmentation, got {result3}"
    )


def test_map_model_name_allows_unet_for_segmentation():
    """_map_model_name must pass unet through for segmentation task."""
    from app.services.pilot_runner import _map_model_name

    result = _map_model_name("unet", task_type=TaskType.SEGMENTATION)
    assert result == ModelType.UNET

    result2 = _map_model_name("deeplabv3", task_type=TaskType.SEGMENTATION)
    assert result2 == ModelType.DEEPLABV3


# ── Test 3: Tabular dataset → XGBoost ────────────────────────────────────────

def test_tabular_dataset_routes_correctly(tmp_path):
    """CSV file → tabular task string."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("feature1,feature2,label\n1.0,2.0,cat\n3.0,4.0,dog\n")

    task = detect_task(csv_path)
    assert task == TASK_TABULAR, (
        f"Expected 'tabular', got '{task}'"
    )


def test_tabular_trainer_raises_for_segmentation_model(tmp_path):
    """tabular_trainer must raise ValueError if given an image/segmentation model."""
    import asyncio
    from app.services.trainers.tabular_trainer import train_tabular
    from app.models.schemas import TrainingConfig, ModelType, TaskType

    config = TrainingConfig(
        session_id="test-session",
        model_type=ModelType.UNET,
        task_type=TaskType.CLASSIFICATION,
    )

    async def _run():
        q = asyncio.Queue()
        await train_tabular(config, q)

    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(ValueError, match="image model"):
            loop.run_until_complete(_run())
    finally:
        loop.close()


# ── Test 4: Fail-fast — segmentation without masks ───────────────────────────

def test_fail_fast_segmentation_without_masks(tmp_path):
    """Dataset with only images/ (no masks/) must raise ValueError."""
    (tmp_path / "images").mkdir()
    _make_png(tmp_path / "images" / "img1.png")
    # No masks/ directory

    with pytest.raises(ValueError):
        detect_task(tmp_path)


def test_fail_fast_empty_dataset(tmp_path):
    """Completely empty directory must raise ValueError."""
    with pytest.raises(ValueError):
        detect_task(tmp_path)


def test_fail_fast_subdirs_without_images(tmp_path):
    """Subdirectories with no image files must raise ValueError."""
    (tmp_path / "cats").mkdir()
    (tmp_path / "dogs").mkdir()
    # No image files inside

    with pytest.raises(ValueError):
        detect_task(tmp_path)


# ── Test 5: LLM validation rejects tabular for segmentation ──────────────────

def test_llm_validation_rejects_tabular_for_segmentation():
    """_validate_model_for_task must block tabular models for segmentation."""
    from app.core.llm_engine import _validate_model_for_task

    assert _validate_model_for_task("xgboost", TaskType.SEGMENTATION) is False
    assert _validate_model_for_task("lightgbm", TaskType.SEGMENTATION) is False
    assert _validate_model_for_task("random_forest", TaskType.SEGMENTATION) is False


def test_llm_validation_allows_segmentation_models():
    """_validate_model_for_task must allow correct segmentation models."""
    from app.core.llm_engine import _validate_model_for_task

    assert _validate_model_for_task("unet", TaskType.SEGMENTATION) is True
    assert _validate_model_for_task("deeplabv3", TaskType.SEGMENTATION) is True
    assert _validate_model_for_task("yolov8n-seg", TaskType.SEGMENTATION) is True


def test_llm_validation_allows_tabular_for_classification():
    """_validate_model_for_task must allow XGBoost for tabular classification."""
    from app.core.llm_engine import _validate_model_for_task

    assert _validate_model_for_task("xgboost", TaskType.CLASSIFICATION) is True
    assert _validate_model_for_task("lightgbm", TaskType.CLASSIFICATION) is True


def test_llm_validation_allows_unknown_task():
    """_validate_model_for_task must not block when task is UNKNOWN."""
    from app.core.llm_engine import _validate_model_for_task

    assert _validate_model_for_task("xgboost", TaskType.UNKNOWN) is True


# ── Test 6: Segmentation trainer rejects wrong model ─────────────────────────

def test_segmentation_trainer_raises_for_xgboost():
    """segmentation_trainer.train_segmentation must raise for tabular model."""
    import asyncio
    from app.services.trainers.segmentation_trainer import train_segmentation
    from app.models.schemas import TrainingConfig, ModelType, TaskType

    config = TrainingConfig(
        session_id="test-session",
        model_type=ModelType.XGBOOST,
        task_type=TaskType.SEGMENTATION,
    )

    async def _run():
        q = asyncio.Queue()
        await train_segmentation(config, q)

    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(ValueError, match="Segmentation task requires"):
            loop.run_until_complete(_run())
    finally:
        loop.close()
