"""
Robust preprocessing pipeline for CSV, image folders, COCO, YOLO, and Pascal VOC datasets.
Every step is logged and the config is saved so inference can reuse the same pipeline.
"""
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

from app.core.config import settings
from app.models.schemas import (
    DatasetFormat,
    DatasetInfo,
    PreprocessConfig,
    PreprocessReport,
    TaskType,
)
from app.utils.file_handler import save_json, load_json
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def _find_images(directory: Path) -> List[Path]:
    return [f for f in directory.rglob("*") if _is_image(f)]


# ── Dataset Detection ─────────────────────────────────────────────────────────

def detect_format(path: Path) -> DatasetFormat:
    if path.is_file():
        if path.suffix.lower() == ".csv":
            return DatasetFormat.CSV
        if path.suffix.lower() == ".zip":
            return DatasetFormat.ZIP
        if path.suffix.lower() == ".json":
            # Check for COCO
            try:
                data = json.loads(path.read_text())
                if "annotations" in data and "images" in data:
                    return DatasetFormat.COCO
            except Exception:
                pass
    if path.is_dir():
        yamls = list(path.rglob("*.yaml")) + list(path.rglob("*.yml"))
        txts = list(path.rglob("*.txt"))
        images = _find_images(path)
        if yamls and txts and images:
            return DatasetFormat.YOLO
        if images:
            return DatasetFormat.IMAGE_FOLDER
    return DatasetFormat.IMAGE_FOLDER


def _has_segmentation_layout(path: Path) -> bool:
    """Check if dataset has images/ + masks/ subdirectories (segmentation format)."""
    if not path.is_dir():
        return False
    # Walk one level deep in case there's a wrapper directory
    candidates = [path]
    subdirs = [d for d in path.iterdir() if d.is_dir() and d.name.lower() not in _SKIP_DIRS_DETECT]
    if len(subdirs) == 1:
        candidates.append(subdirs[0])
    for d in candidates:
        img_dir = d / "images"
        mask_dir = d / "masks"
        if img_dir.is_dir() and mask_dir.is_dir():
            imgs = _find_images(img_dir)
            masks = _find_images(mask_dir)
            if imgs and masks:
                return True
    return False


_SKIP_DIRS_DETECT = {"preprocessed", "__pycache__", "__macosx", ".ds_store", "models"}


def detect_task_type(fmt: DatasetFormat, path: Path) -> TaskType:
    if fmt == DatasetFormat.CSV:
        try:
            df = pd.read_csv(path, nrows=200)
            last_col = df.columns[-1]
            col = df[last_col].dropna()
            n_unique = col.nunique()
            n_total = len(col)
            # Float column with high cardinality ratio → regression
            is_float = pd.api.types.is_float_dtype(col)
            high_cardinality = n_unique > 20 or (n_total > 0 and n_unique / n_total > 0.5)
            if is_float and high_cardinality:
                return TaskType.REGRESSION
            if n_unique <= 20:
                return TaskType.CLASSIFICATION
            return TaskType.REGRESSION
        except Exception:
            return TaskType.UNKNOWN
    if fmt in (DatasetFormat.YOLO, DatasetFormat.COCO):
        return TaskType.OBJECT_DETECTION
    if fmt == DatasetFormat.IMAGE_FOLDER:
        # Check for segmentation layout (images/ + masks/) first
        if _has_segmentation_layout(Path(path)):
            return TaskType.SEGMENTATION
        # If subdirectories exist treat as classification
        p = Path(path)
        subdirs = [x for x in p.iterdir() if x.is_dir()] if p.is_dir() else []
        if subdirs:
            return TaskType.CLASSIFICATION
        return TaskType.UNKNOWN
    return TaskType.UNKNOWN


# ── CSV Preprocessing ────────────────────────────────────────────────────────

class CSVPreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.report = PreprocessReport(session_id=config.session_id)
        self._pipeline: Dict[str, Any] = {}

    def run(self, csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PreprocessReport]:
        logger.info(f"CSV preprocessing: {csv_path}")
        df = pd.read_csv(csv_path)
        self.report.steps_applied.append(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        df = self._handle_missing(df)
        df = self._remove_outliers(df)
        df = self._encode_categoricals(df)
        df = self._scale_numerics(df)
        train, val, test = self._split(df)

        self.report.train_size = len(train)
        self.report.val_size = len(val)
        self.report.test_size = len(test)

        out_dir = settings.UPLOAD_DIR / self.config.session_id / "preprocessed"
        out_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(out_dir / "train.csv", index=False)
        val.to_csv(out_dir / "val.csv", index=False)
        test.to_csv(out_dir / "test.csv", index=False)
        save_json(self._pipeline, out_dir / "pipeline.json")

        logger.info("CSV preprocessing complete")
        return train, val, test, self.report

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_info: Dict[str, Any] = {}
        for col in df.columns:
            pct = df[col].isna().mean()
            if pct == 0:
                continue
            if pct > 0.5:
                df = df.drop(columns=[col])
                missing_info[col] = "dropped (>50% missing)"
                self.report.steps_applied.append(f"Dropped column '{col}' (>50% missing)")
            elif df[col].dtype in [np.float64, np.int64]:
                fill = df[col].median() if self.config.handle_missing == "median" else df[col].mean()
                df[col] = df[col].fillna(fill)
                missing_info[col] = f"filled with {self.config.handle_missing or 'mean'} ({fill:.4f})"
                self._pipeline[f"fill_{col}"] = float(fill)
            else:
                fill = df[col].mode()[0] if not df[col].mode().empty else "MISSING"
                df[col] = df[col].fillna(fill)
                missing_info[col] = f"filled with mode ('{fill}')"
                self._pipeline[f"fill_{col}"] = fill
        self.report.missing_handled = missing_info
        self.report.steps_applied.append("Handled missing values")
        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return df
        original_len = len(df)
        for col in num_cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        removed = original_len - len(df)
        self.report.outliers_removed = removed
        self.report.steps_applied.append(f"Removed {removed} outlier rows (IQR method)")
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        enc: Dict[str, str] = {}
        for col in cat_cols:
            n_unique = df[col].nunique()
            if n_unique == 2:
                mapping = {v: i for i, v in enumerate(df[col].unique())}
                df[col] = df[col].map(mapping)
                enc[col] = "binary"
                self._pipeline[f"enc_{col}"] = mapping
            elif n_unique <= 10:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                enc[col] = "one-hot"
                self._pipeline[f"enc_{col}"] = list(dummies.columns)
            else:
                mapping = {v: i for i, v in enumerate(df[col].unique())}
                df[col] = df[col].map(mapping).fillna(-1)
                enc[col] = "label"
                self._pipeline[f"enc_{col}"] = mapping
        self.report.encodings = enc
        self.report.steps_applied.append(f"Encoded {len(cat_cols)} categorical columns")
        return df

    def _scale_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Don't scale the target (last column)
        if len(num_cols) > 1:
            num_cols = num_cols[:-1]
        for col in num_cols:
            mean, std = df[col].mean(), df[col].std()
            if std == 0:
                continue
            if self.config.scale_method == "minmax":
                mn, mx = df[col].min(), df[col].max()
                df[col] = (df[col] - mn) / (mx - mn + 1e-8)
                self._pipeline[f"scale_{col}"] = {"method": "minmax", "min": mn, "max": mx}
            else:
                df[col] = (df[col] - mean) / (std + 1e-8)
                self._pipeline[f"scale_{col}"] = {"method": "standard", "mean": mean, "std": std}
        self.report.steps_applied.append(f"Scaled {len(num_cols)} numeric features ({self.config.scale_method})")
        return df

    def _split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df)
        
        # Handle small datasets: ensure train and val both have samples
        if n == 1:
            # Single sample: put in train, will be used for validation too (fallback)
            n_train = 1
            n_val = 0
        elif n == 2:
            # Two samples: 1 train, 1 val
            n_train = 1
            n_val = 1
        else:
            # Normal splitting with minimum samples
            n_train = max(1, int(n * self.config.train_ratio))
            n_val = max(1, int(n * self.config.val_ratio)) if n > 3 else max(1, n - n_train - 1)
        
        train = df.iloc[:n_train]
        val = df.iloc[n_train:n_train + n_val]
        test = df.iloc[n_train + n_val:]
        self.report.steps_applied.append(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test


# ── Image Preprocessing ───────────────────────────────────────────────────────

class ImagePreprocessor:
    """
    Preprocesses image classification datasets.
    Writes a split manifest JSON instead of copying all images —
    avoids Windows path-length issues with large datasets.
    Resizes only a small sample (up to 100 images) for validation.
    """

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.report = PreprocessReport(session_id=config.session_id)

    def run(self, dataset_dir: Path) -> PreprocessReport:
        logger.info(f"Image preprocessing: {dataset_dir}")

        # ── Find all images and determine class structure ─────────────────────
        # Detect top-level class dirs (buildings/forest/glacier pattern)
        root = self._find_dataset_root(dataset_dir)
        class_dirs = sorted([d for d in root.iterdir() if d.is_dir()]) if root.is_dir() else []

        if class_dirs:
            # Classification layout: each subdir = one class
            class_images: Dict[str, List[Path]] = {}
            for cls_dir in class_dirs:
                imgs = _find_images(cls_dir)
                if imgs:
                    class_images[cls_dir.name] = imgs
        else:
            # Flat image folder — treat as single class "images"
            all_imgs = _find_images(root)
            class_images = {"images": all_imgs} if all_imgs else {}

        total = sum(len(v) for v in class_images.values())
        self.report.steps_applied.append(f"Found {total} images in {len(class_images)} class(es)")

        if total == 0:
            self.report.warnings.append("No images found in dataset")
            return self.report

        # ── Stratified split — produce a manifest, not copies ─────────────────
        import random
        random.seed(42)

        manifest: Dict[str, Any] = {"classes": list(class_images.keys()), "splits": {"train": [], "val": [], "test": []}}

        for cls_name, imgs in class_images.items():
            shuffled = imgs.copy()
            random.shuffle(shuffled)
            n = len(shuffled)
            
            # Handle small datasets: ensure train and val both have samples
            if n == 1:
                # Single sample: put in train, will be used for validation too (fallback)
                n_train = 1
                n_val = 0
                n_test = 0
            elif n == 2:
                # Two samples: 1 train, 1 val
                n_train = 1
                n_val = 1
                n_test = 0
            else:
                # Normal splitting with minimum 1 for each split if dataset is very small
                n_train = max(1, int(n * self.config.train_ratio))
                n_val = max(1, int(n * self.config.val_ratio)) if n > 3 else max(1, n - n_train - 1)
                n_test = max(0, n - n_train - n_val)

            self.report.train_size += n_train
            self.report.val_size += n_val
            self.report.test_size += n_test

            for img in shuffled[:n_train]:
                manifest["splits"]["train"].append({"path": str(img), "class": cls_name})
            for img in shuffled[n_train:n_train + n_val]:
                manifest["splits"]["val"].append({"path": str(img), "class": cls_name})
            for img in shuffled[n_train + n_val:]:
                manifest["splits"]["test"].append({"path": str(img), "class": cls_name})

        # ── Save manifest (much cheaper than copying/resizing all images) ──────
        out_dir = settings.UPLOAD_DIR / self.config.session_id / "preprocessed"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(manifest, out_dir / "split_manifest.json")
        save_json(
            {"classes": list(class_images.keys()), "class_counts": {k: len(v) for k, v in class_images.items()}},
            out_dir / "class_info.json",
        )

        self.report.steps_applied.append(
            f"Stratified split — train={self.report.train_size}, "
            f"val={self.report.val_size}, test={self.report.test_size}"
        )
        self.report.augmentations = ["horizontal_flip", "rotation_15deg", "brightness_jitter"]

        # ── Resize a small sample to verify images are readable ───────────────
        try:
            from PIL import Image
            sample_imgs = list(class_images.values())[0][:5]
            ok = 0
            for img_path in sample_imgs:
                try:
                    img = Image.open(img_path).convert("RGB")
                    w, h = img.size
                    ok += 1
                except Exception:
                    pass
            self.report.steps_applied.append(f"Verified {ok}/{len(sample_imgs)} sample images readable")
        except ImportError:
            self.report.warnings.append("Pillow not installed — skipping image verification")

        return self.report

    def _find_dataset_root(self, dataset_dir: Path) -> Path:
        """
        Descend into a single subdirectory if no images exist at this level.
        Skips internal dirs (preprocessed/, __pycache__, etc.) so they are
        never mistaken for class directories.
        """
        _SKIP = {"preprocessed", "__pycache__", "__macosx", ".ds_store", "models"}
        if not dataset_dir.is_dir():
            return dataset_dir
        children = list(dataset_dir.iterdir())
        subdirs = [c for c in children if c.is_dir() and c.name.lower() not in _SKIP]
        # Only count images directly in this directory, NOT recursively
        images_here = [f for f in children if f.is_file() and _is_image(f)]
        if len(subdirs) == 1 and not images_here:
            return self._find_dataset_root(subdirs[0])
        return dataset_dir


# ── Segmentation Preprocessing ────────────────────────────────────────────────

class SegmentationPreprocessor:
    """
    Preprocesses segmentation datasets with images/ + masks/ layout.
    Pairs images with masks by filename stem, detects class count from masks,
    and writes a seg_manifest.json for the DL trainer.
    """

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.report = PreprocessReport(session_id=config.session_id)

    def run(self, dataset_dir: Path) -> PreprocessReport:
        logger.info(f"Segmentation preprocessing: {dataset_dir}")

        # Find the images/ and masks/ directories (may be one level deep)
        img_dir, mask_dir = self._find_dirs(dataset_dir)
        if img_dir is None or mask_dir is None:
            self.report.warnings.append(
                "Segmentation requires images/ and masks/ subdirectories. "
                "Falling back to classification."
            )
            return self.report

        # Pair images with masks by filename stem
        images = {p.stem: p for p in _find_images(img_dir)}
        masks = {p.stem: p for p in _find_images(mask_dir)}
        paired = [(images[s], masks[s]) for s in images if s in masks]

        self.report.steps_applied.append(
            f"Found {len(images)} images, {len(masks)} masks, {len(paired)} paired"
        )
        if not paired:
            self.report.warnings.append("No image-mask pairs found (filenames must match)")
            return self.report

        # Detect number of classes from a sample of masks
        num_classes = self._detect_num_classes([m for _, m in paired[:20]])
        self.report.steps_applied.append(f"Detected {num_classes} segmentation classes in masks")

        # Stratified split
        import random
        random.seed(42)
        shuffled = paired.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * self.config.train_ratio))
        n_val = max(1, int(n * self.config.val_ratio)) if n > 2 else max(0, n - n_train)

        train_pairs = shuffled[:n_train]
        val_pairs = shuffled[n_train:n_train + n_val]
        test_pairs = shuffled[n_train + n_val:]

        self.report.train_size = len(train_pairs)
        self.report.val_size = len(val_pairs)
        self.report.test_size = len(test_pairs)

        def to_items(pairs):
            return [{"image": str(img), "mask": str(msk)} for img, msk in pairs]

        manifest = {
            "num_classes": num_classes,
            "class_names": [f"class_{i}" for i in range(num_classes)],
            "splits": {
                "train": to_items(train_pairs),
                "val": to_items(val_pairs),
                "test": to_items(test_pairs),
            },
        }

        out_dir = settings.UPLOAD_DIR / self.config.session_id / "preprocessed"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(manifest, out_dir / "seg_manifest.json")

        self.report.steps_applied.append(
            f"Split — train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}"
        )
        self.report.augmentations = ["horizontal_flip", "rotation_15deg"]
        return self.report

    def _find_dirs(self, base: Path):
        """Locate images/ and masks/ directories, handling a wrapper dir."""
        candidates = [base]
        subdirs = [d for d in base.iterdir() if d.is_dir() and d.name.lower() not in _SKIP_DIRS_DETECT]
        if len(subdirs) == 1:
            candidates.append(subdirs[0])
        for d in candidates:
            img_dir = d / "images"
            mask_dir = d / "masks"
            if img_dir.is_dir() and mask_dir.is_dir():
                return img_dir, mask_dir
        return None, None

    @staticmethod
    def _detect_num_classes(mask_paths: List[Path]) -> int:
        """Read a sample of masks and find unique pixel values."""
        try:
            from PIL import Image as PILImg
            all_vals = set()
            for mp in mask_paths:
                m = PILImg.open(mp).convert("L")
                arr = np.array(m)
                all_vals.update(np.unique(arr).tolist())
            return max(len(all_vals), 2)
        except Exception:
            return 2


# ── YOLO Preprocessing ────────────────────────────────────────────────────────

class YOLOPreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.report = PreprocessReport(session_id=config.session_id)

    def run(self, dataset_dir: Path) -> PreprocessReport:
        logger.info(f"YOLO preprocessing: {dataset_dir}")
        images = _find_images(dataset_dir)
        labels = list(dataset_dir.rglob("*.txt"))
        self.report.steps_applied.append(f"Found {len(images)} images, {len(labels)} label files")

        # Validate pairing
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels if p.name != "classes.txt"}
        missing_labels = image_stems - label_stems
        if missing_labels:
            self.report.warnings.append(f"{len(missing_labels)} images missing label files")

        # Count classes and check imbalance
        class_counts: Dict[str, int] = {}
        for lf in labels:
            if lf.name == "classes.txt":
                continue
            try:
                for line in lf.read_text().strip().splitlines():
                    parts = line.strip().split()
                    if parts:
                        cls_id = parts[0]
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            except Exception:
                pass

        if class_counts:
            counts = list(class_counts.values())
            imbalance_ratio = max(counts) / (min(counts) + 1e-8)
            if imbalance_ratio > 10:
                self.report.warnings.append(
                    f"Severe class imbalance detected (ratio {imbalance_ratio:.1f}x). "
                    "Consider oversampling minority classes."
                )

        # Generate data.yaml
        classes_file = dataset_dir / "classes.txt"
        class_names: List[str] = []
        if classes_file.exists():
            class_names = [l.strip() for l in classes_file.read_text().splitlines() if l.strip()]

        yaml_content = {
            "path": str(dataset_dir),
            "train": "images/train",
            "val": "images/val",
            "nc": len(class_names) or len(class_counts),
            "names": class_names or list(class_counts.keys()),
        }

        out_dir = settings.UPLOAD_DIR / self.config.session_id / "preprocessed"
        out_dir.mkdir(parents=True, exist_ok=True)

        import yaml  # type: ignore
        yaml_path = out_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)
        save_json({"class_counts": class_counts, "yaml": yaml_content}, out_dir / "yolo_report.json")

        self.report.steps_applied.append(f"Generated data.yaml with {len(class_names) or len(class_counts)} classes")
        self.report.augmentations = ["mosaic", "hsv_augment", "scale", "fliplr"]
        return self.report


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess_dataset(
    dataset_path: Path,
    config: PreprocessConfig,
    task_type_hint: Optional[TaskType] = None,
) -> Tuple[DatasetInfo, PreprocessReport]:
    """
    Preprocess dataset. Pass ``task_type_hint`` when the user explicitly selected
    a task (e.g. segmentation) so the preprocessor tries the correct pipeline even
    when auto-detection would choose a different task type.
    """
    fmt = detect_format(dataset_path)
    task = detect_task_type(fmt, dataset_path)

    # User-selected task overrides auto-detection for image folders.
    # For CSV, auto-detection is authoritative (regression vs classification).
    if task_type_hint is not None and fmt != DatasetFormat.CSV:
        task = task_type_hint

    info = DatasetInfo(
        session_id=config.session_id,
        filename=dataset_path.name,
        format=fmt,
        task_type=task,
    )

    if fmt == DatasetFormat.CSV:
        preprocessor = CSVPreprocessor(config)
        df = pd.read_csv(dataset_path)
        info.num_samples = len(df)
        info.columns = list(df.columns)
        info.preview = df.head(10).to_dict(orient="records")
        info.num_classes = int(df.iloc[:, -1].nunique())
        _, _, _, report = preprocessor.run(dataset_path)

    elif fmt == DatasetFormat.YOLO:
        preprocessor = YOLOPreprocessor(config)
        images = _find_images(dataset_path)
        info.num_samples = len(images)
        report = preprocessor.run(dataset_path)

    elif fmt == DatasetFormat.IMAGE_FOLDER and task == TaskType.SEGMENTATION:
        preprocessor = SegmentationPreprocessor(config)
        images = _find_images(dataset_path)
        info.num_samples = len(images)
        info.task_type = TaskType.SEGMENTATION
        report = preprocessor.run(dataset_path)
        # Read back manifest for class count
        seg_manifest = settings.UPLOAD_DIR / config.session_id / "preprocessed" / "seg_manifest.json"
        if seg_manifest.exists():
            sm = json.loads(seg_manifest.read_text())
            info.num_classes = sm.get("num_classes", 2)
        else:
            # Failsafe: no masks/paired masks -> downgrade to classification
            report.warnings.append(
                "⚠️  Segmentation requires images/ + masks/ directories with matching filenames. "
                "Dataset has no masks — downgrading to image classification instead."
            )
            info.task_type = TaskType.CLASSIFICATION
            cls_pre = ImagePreprocessor(config)
            cls_report = cls_pre.run(dataset_path)
            report.steps_applied.extend(cls_report.steps_applied)
            report.warnings.extend(cls_report.warnings)
            report.train_size = cls_report.train_size
            report.val_size = cls_report.val_size
            report.test_size = cls_report.test_size
            root = cls_pre._find_dataset_root(dataset_path)
            class_dirs = [d for d in root.iterdir() if d.is_dir()] if root.is_dir() else []
            info.num_classes = len(class_dirs)
            info.class_names = [d.name for d in class_dirs]

    elif fmt == DatasetFormat.IMAGE_FOLDER:
        preprocessor = ImagePreprocessor(config)
        images = _find_images(dataset_path)
        info.num_samples = len(images)
        root = preprocessor._find_dataset_root(dataset_path)
        class_dirs = [d for d in root.iterdir() if d.is_dir()] if root.is_dir() else []
        info.num_classes = len(class_dirs)
        info.class_names = [d.name for d in class_dirs]
        report = preprocessor.run(dataset_path)

    else:
        report = PreprocessReport(session_id=config.session_id)
        report.warnings.append(f"Unsupported format: {fmt}")

    return info, report


_SKIP_DIRS = {"preprocessed", "__pycache__", "__macosx", ".ds_store", "models"}


def get_dataset_summary(dataset_path: Path, session_id: str) -> Dict[str, Any]:
    """Get a statistical summary for LLM recommendations."""
    fmt = detect_format(dataset_path)
    task = detect_task_type(fmt, dataset_path)
    summary: Dict[str, Any] = {
        "format": fmt.value,
        "task_type": task.value,
        "session_id": session_id,
    }

    if fmt == DatasetFormat.CSV:
        df = pd.read_csv(dataset_path)
        summary["num_samples"] = len(df)
        summary["num_features"] = len(df.columns) - 1
        summary["num_classes"] = int(df.iloc[:, -1].nunique())
        summary["class_distribution"] = df.iloc[:, -1].value_counts().to_dict()
        col_stats: Dict[str, Any] = {}
        for col in df.select_dtypes(include=[np.number]).columns[:10]:
            col_stats[col] = {
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "missing_pct": round(float(df[col].isna().mean()), 4),
            }
        summary["column_stats"] = col_stats
    elif fmt == DatasetFormat.IMAGE_FOLDER:
        images = _find_images(dataset_path)
        summary["num_samples"] = len(images)
        root = ImagePreprocessor(PreprocessConfig(session_id=session_id))._find_dataset_root(dataset_path)
        # Exclude system/internal directories from class count
        class_dirs = [
            d for d in root.iterdir()
            if d.is_dir() and d.name.lower() not in _SKIP_DIRS
        ] if root.is_dir() else []
        summary["num_classes"] = len(class_dirs)
        summary["class_distribution"] = {d.name: len(_find_images(d)) for d in class_dirs}
        # Compute imbalance ratio
        counts = [len(_find_images(d)) for d in class_dirs if _find_images(d)]
        if len(counts) >= 2:
            summary["imbalance_ratio"] = round(max(counts) / max(min(counts), 1), 2)
        else:
            summary["imbalance_ratio"] = 1.0
    elif fmt in (DatasetFormat.YOLO, DatasetFormat.COCO):
        images = _find_images(dataset_path)
        summary["num_samples"] = len(images)

    return summary
