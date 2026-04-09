"""
Task Router — strict gateway for dataset task detection.

Examines dataset structure and returns one of four canonical task strings.
Raises ValueError immediately on ambiguous or invalid structure.
Delegates to preprocessor.py's existing detection logic — no duplication.
"""
from pathlib import Path
from typing import Union

from app.services.preprocessor import detect_format, _has_segmentation_layout, _find_images
from app.models.schemas import DatasetFormat
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Canonical task strings (avoid circular imports with schemas/enums)
TASK_IMAGE_SEGMENTATION = "image_segmentation"
TASK_IMAGE_CLASSIFICATION = "image_classification"
TASK_OBJECT_DETECTION = "object_detection"
TASK_TABULAR = "tabular"

_SKIP_DIRS = {"preprocessed", "__pycache__", "__macosx", ".ds_store", "models", ".git"}


def detect_task(dataset_path: Union[str, Path]) -> str:
    """
    Examine dataset structure and return one canonical task string.

    Detection priority:
      1. CSV file                              → "tabular"
      2. YOLO (yaml+txt+images) or COCO JSON  → "object_detection"
      3. images/ + masks/ with paired files   → "image_segmentation"
      4. Image folder with class subdirs      → "image_classification"
      5. Anything else                        → raises ValueError

    Raises:
        ValueError: Dataset structure is ambiguous or invalid for task detection.
    """
    path = Path(dataset_path)

    if not path.exists():
        raise ValueError(f"Dataset path does not exist: {path}")

    fmt = detect_format(path)
    logger.info(f"detect_task: format={fmt.value}  path={path}")

    # ── 1. CSV / Tabular ─────────────────────────────────────────────────────
    if fmt == DatasetFormat.CSV:
        try:
            import pandas as pd
            df = pd.read_csv(path, nrows=5)
            if df.empty:
                raise ValueError(f"CSV file is empty: {path}")
        except Exception as exc:
            raise ValueError(f"CSV file is unreadable: {path} — {exc}") from exc
        return TASK_TABULAR

    # ── 2. Object Detection (YOLO / COCO) ────────────────────────────────────
    if fmt in (DatasetFormat.YOLO, DatasetFormat.COCO):
        return TASK_OBJECT_DETECTION

    # ── 3 & 4. Image-based tasks ─────────────────────────────────────────────
    if fmt == DatasetFormat.IMAGE_FOLDER:
        if not path.is_dir():
            raise ValueError(f"Expected a directory for image dataset, got: {path}")

        # Check for images/ dir without masks/ — likely incomplete segmentation dataset
        img_dir = path / "images"
        mask_dir = path / "masks"
        if img_dir.is_dir() and not mask_dir.is_dir():
            raise ValueError(
                f"Dataset at {path} has an images/ directory but no masks/ directory. "
                "If this is a segmentation dataset, add a masks/ directory with "
                "per-image mask files (same filename as images). "
                "If this is a classification dataset, rename the 'images/' folder to "
                "the class name (e.g. 'cats/', 'dogs/')."
            )

        # Check segmentation layout first (images/ + masks/)
        if _has_segmentation_layout(path):
            _validate_segmentation_layout(path)
            return TASK_IMAGE_SEGMENTATION

        # Check for class subdirectories → classification
        subdirs = [
            d for d in path.iterdir()
            if d.is_dir() and d.name.lower() not in _SKIP_DIRS
        ]
        if subdirs:
            # Verify at least one subdir contains images
            has_images = any(_find_images(d) for d in subdirs)
            if has_images:
                return TASK_IMAGE_CLASSIFICATION
            raise ValueError(
                f"Found subdirectories in {path} but none contain image files. "
                "For classification: organize images into class-named subdirectories. "
                "For segmentation: create images/ + masks/ subdirectories."
            )

        # No structure found
        raise ValueError(
            f"Cannot determine task type from dataset at {path}. "
            "Expected one of:\n"
            "  • Classification: class-named subdirs (e.g. cats/, dogs/)\n"
            "  • Segmentation:   images/ + masks/ subdirs\n"
            "  • Detection:      YOLO yaml+labels or COCO annotations.json\n"
            "  • Tabular:        a .csv file"
        )

    # ZIP or unknown format — not directly handled here
    raise ValueError(
        f"Invalid dataset structure for task detection (format={fmt.value}). "
        "Please extract ZIP files before uploading, or use CSV/image folder/YOLO format."
    )


def _validate_segmentation_layout(path: Path) -> None:
    """
    Fail-fast validation for segmentation datasets.
    Raises ValueError if images/ or masks/ directories are empty,
    or if no image-mask pairs share the same filename stem.
    """
    # Walk one level deep (mirrors _has_segmentation_layout logic)
    candidates = [path]
    subdirs = [d for d in path.iterdir() if d.is_dir() and d.name.lower() not in _SKIP_DIRS]
    if len(subdirs) == 1:
        candidates.append(subdirs[0])

    for d in candidates:
        img_dir = d / "images"
        mask_dir = d / "masks"
        if img_dir.is_dir() and mask_dir.is_dir():
            imgs = _find_images(img_dir)
            masks = _find_images(mask_dir)

            if not imgs:
                raise ValueError(
                    f"Segmentation dataset error: images/ directory is empty at {img_dir}. "
                    "Add image files before training."
                )
            if not masks:
                raise ValueError(
                    f"Segmentation dataset error: masks/ directory is empty at {mask_dir}. "
                    "Each image must have a corresponding mask file. "
                    "Cannot train segmentation without masks."
                )

            # Verify at least some pairs share the same stem
            img_stems = {p.stem for p in imgs}
            mask_stems = {p.stem for p in masks}
            pairs = img_stems & mask_stems
            if not pairs:
                raise ValueError(
                    f"Segmentation dataset error: no image-mask filename pairs found. "
                    f"Image stems: {sorted(img_stems)[:5]}... "
                    f"Mask stems: {sorted(mask_stems)[:5]}... "
                    "Image and mask files must share the same filename (e.g. cat.jpg + cat.png)."
                )
            return  # Valid

    raise ValueError(
        f"Segmentation layout detected but images/ or masks/ dirs could not be located under {path}."
    )
