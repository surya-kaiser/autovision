"""
Experiment Planner: Converts model recommendations into training configurations.
"""
import json
from typing import Dict, Any, Optional, List
from app.llm.ollama_client import get_ollama_client
from app.llm.prompt_templates import get_system_prompt, get_user_prompt
from app.models.schemas import TaskType
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentConfig:
    """Structured experiment configuration."""
    
    def __init__(
        self,
        model: str,
        learning_rate: float = 0.0003,
        batch_size: int = 32,
        epochs: int = 15,
        optimizer: str = "adam",
        augmentation: List[str] = None,
        regularization: Dict[str, Any] = None,
        fine_tune_layers: int = 20,
        early_stopping_patience: int = 5,
        warmup_epochs: int = 2,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.augmentation = augmentation or []
        self.regularization = regularization or {}
        self.fine_tune_layers = fine_tune_layers
        self.early_stopping_patience = early_stopping_patience
        self.warmup_epochs = warmup_epochs
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "optimizer": self.optimizer,
            "augmentation": self.augmentation,
            "regularization": self.regularization,
            "fine_tune_layers": self.fine_tune_layers,
            "early_stopping_patience": self.early_stopping_patience,
            "warmup_epochs": self.warmup_epochs,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(
            model=data.get("model", "unknown"),
            learning_rate=data.get("learning_rate", 0.0003),
            batch_size=data.get("batch_size", 32),
            epochs=data.get("epochs", 15),
            optimizer=data.get("optimizer", "adam"),
            augmentation=data.get("augmentation", []),
            regularization=data.get("regularization", {}),
            fine_tune_layers=data.get("fine_tune_layers", 20),
            early_stopping_patience=data.get("early_stopping_patience", 5),
            warmup_epochs=data.get("warmup_epochs", 2),
        )


class ExperimentPlanner:
    """Planner converts model recommendations into experiment configs."""
    
    def __init__(self):
        self.client = get_ollama_client()
        self.system_prompt = get_system_prompt("strategy")
        self.user_prompt_template = get_user_prompt("strategy")
    
    def plan_experiments(
        self,
        models: List[str],
        task_type: TaskType,
        dataset_size: int,
        num_classes: int,
        dataset_format: str = "image",
        hardware: str = "cpu",
        max_experiments: int = 3,
    ) -> List[ExperimentConfig]:
        """
        Plan experiments for given models.
        
        Args:
            models: List of model names to configure
            task_type: Classification, detection, regression
            dataset_size: Number of samples
            num_classes: Number of classes
            dataset_format: image, csv, etc.
            hardware: cpu or gpu
            max_experiments: Max number of experiments
            
        Returns:
            List of ExperimentConfig
        """
        logger.info(f"Planning experiments for {len(models)} models")
        
        configs = []

        if task_type == TaskType.SEGMENTATION:
            allowed = ("unet", "deeplabv3", "maskrcnn", "yolov8n-seg", "yolov8s-seg", "yolo")
            filtered = [m for m in models if any(k in m.lower() for k in allowed)]
            models = filtered or ["UNet", "DeepLabV3"]
        
        # Limit experiments
        models_to_plan = models[:max_experiments]
        
        for model_name in models_to_plan:
            config = self._plan_single_experiment(
                model=model_name,
                task_type=task_type,
                dataset_size=dataset_size,
                num_classes=num_classes,
                dataset_format=dataset_format,
                hardware=hardware,
            )
            
            if config:
                configs.append(config)
        
        logger.info(f"Planned {len(configs)} experiment configurations")
        return configs
    
    def _plan_single_experiment(
        self,
        model: str,
        task_type: TaskType,
        dataset_size: int,
        num_classes: int,
        dataset_format: str = "image",
        hardware: str = "cpu",
    ) -> Optional[ExperimentConfig]:
        """Plan single experiment using LLM."""
        
        prompt = self.user_prompt_template.format(
            model=model,
            dataset_size=dataset_size,
            num_classes=num_classes,
            task_type=task_type.value,
            hardware=hardware,
        )
        
        response_text = self.client.generate(
            prompt=prompt,
            system=self.system_prompt,
            temperature=0.3,
        )
        
        if not response_text:
            logger.warning(f"No response for model {model}")
            return self._fallback_config(model)
        
        # Parse JSON
        try:
            config_data = json.loads(response_text)
            config = ExperimentConfig.from_dict(config_data)
            logger.info(f"Planned config for {model}: lr={config.learning_rate}, bs={config.batch_size}")
            return config
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse config for {model}: {e}")
            return self._fallback_config(model)
    
    def _fallback_config(self, model: str) -> ExperimentConfig:
        """Fallback configuration if LLM unavailable."""
        logger.info(f"Using fallback config for {model}")
        name = model.lower()

        if "unet" in name:
            return ExperimentConfig(
                model=model,
                learning_rate=0.001,
                batch_size=8,
                epochs=30,
                augmentation=["flip", "rotation"],
                regularization={"weight_decay": 1e-4},
            )
        elif "deeplab" in name:
            return ExperimentConfig(
                model=model,
                learning_rate=0.0005,
                batch_size=4,
                epochs=25,
                augmentation=["flip", "rotation", "brightness"],
                regularization={"weight_decay": 1e-4},
            )
        elif "cnn" in name:
            return ExperimentConfig(
                model=model,
                learning_rate=0.001,
                batch_size=32,
                epochs=20,
                augmentation=["flip", "rotation", "brightness"],
                regularization={"dropout": 0.5, "weight_decay": 1e-4},
            )
        elif "resnet" in name or "vgg" in name or "efficient" in name or "mobile" in name:
            return ExperimentConfig(
                model=model,
                learning_rate=0.0003,
                batch_size=16,
                epochs=20,
                augmentation=["flip", "rotation", "brightness", "zoom"],
                regularization={"dropout": 0.5, "weight_decay": 1e-4},
                fine_tune_layers=50,
            )
        elif "yolo" in name:
            return ExperimentConfig(
                model=model,
                learning_rate=0.001,
                batch_size=16,
                epochs=100,
                augmentation=["flip", "rotation", "mosaic"],
                regularization={"dropout": 0.0, "l2": 0.0005},
            )
        else:
            # Default safe config (tabular models)
            return ExperimentConfig(
                model=model,
                learning_rate=0.0003,
                batch_size=32,
                epochs=10,
                augmentation=["flip", "rotation"],
                regularization={"dropout": 0.3, "l2": 0.0001},
            )


def get_experiment_planner() -> ExperimentPlanner:
    """Get experiment planner instance."""
    return ExperimentPlanner()
