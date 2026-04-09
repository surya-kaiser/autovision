from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
from enum import Enum


# ── Enums ────────────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


class ModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CNN = "cnn"
    RESNET = "resnet"
    UNET = "unet"
    DEEPLABV3 = "deeplabv3"
    YOLOV8N = "yolov8n"
    YOLOV8S = "yolov8s"
    YOLOV8M = "yolov8m"
    YOLOV8N_SEG = "yolov8n-seg"
    YOLOV8S_SEG = "yolov8s-seg"
    LINEAR = "linear_regression"
    RIDGE = "ridge"


class DatasetFormat(str, Enum):
    CSV = "csv"
    IMAGE_FOLDER = "image_folder"
    COCO = "coco"
    YOLO = "yolo"
    PASCAL_VOC = "pascal_voc"
    ZIP = "zip"


# ── Generic API Response ──────────────────────────────────────────────────────

class APIResponse(BaseModel):
    status: str = "success"
    data: Optional[Any] = None
    message: str = ""


# ── Dataset schemas ───────────────────────────────────────────────────────────

class DatasetInfo(BaseModel):
    session_id: str
    filename: str
    format: DatasetFormat
    task_type: TaskType
    num_samples: int = 0
    num_classes: int = 0
    class_names: List[str] = []
    columns: List[str] = []
    preview: Optional[Any] = None


class PreprocessConfig(BaseModel):
    session_id: str
    scale_method: str = "standard"  # standard | minmax
    handle_missing: str = "auto"    # auto | drop | mean
    encoding: str = "auto"
    augmentation: bool = True
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    task_type_hint: Optional[str] = None  # user-selected task override (e.g. "segmentation")


class PreprocessReport(BaseModel):
    session_id: str
    steps_applied: List[str] = []
    missing_handled: Dict[str, Any] = {}
    encodings: Dict[str, str] = {}
    outliers_removed: int = 0
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0
    augmentations: List[str] = []
    warnings: List[str] = []


# ── Training schemas ──────────────────────────────────────────────────────────

class TrainingConfig(BaseModel):
    session_id: str
    model_type: ModelType
    task_type: TaskType
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping: bool = True
    patience: int = 5
    pilot: bool = False
    hyperparams: Dict[str, Any] = {}


class TrainingStatus(BaseModel):
    session_id: str
    model_type: str
    status: str  # queued | running | completed | failed
    epoch: int = 0
    total_epochs: int = 0
    loss: float = 0.0
    accuracy: Optional[float] = None
    metrics: Dict[str, Any] = {}
    logs: List[str] = []
    eta_seconds: Optional[int] = None


class ModelResult(BaseModel):
    session_id: str
    model_type: str
    task_type: str
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    map50: Optional[float] = None
    loss: float = 0.0
    training_time_s: float = 0.0
    checkpoint_path: str = ""
    metrics: Dict[str, Any] = {}


# ── LLM schemas ───────────────────────────────────────────────────────────────

class LLMRecommendation(BaseModel):
    model_type: str
    hyperparams: Dict[str, Any]
    preprocessing_strategy: str
    explanation: str
    estimated_training_minutes: int = 5


class ChatMessage(BaseModel):
    role: str  # user | assistant
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: List[ChatMessage] = []


# ── Inference schemas ─────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    label: Optional[str] = None
    confidence: Optional[float] = None
    value: Optional[float] = None
    bounding_boxes: Optional[List[Dict[str, Any]]] = None
    class_probabilities: Optional[Dict[str, float]] = None
    segmented_image: Optional[str] = None   # base64 data-URI of annotated image
