"""
Improvement Agent: Analyzes training results and suggests optimizations.
Drives the autonomous improvement loop.
"""
import json
from typing import Dict, Any, Optional
from app.llm.ollama_client import get_ollama_client
from app.llm.prompt_templates import get_system_prompt, get_user_prompt
from app.agents.experiment_planner import ExperimentConfig
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ImprovementSuggestion:
    """Structured improvement suggestion."""
    
    def __init__(
        self,
        action: str,
        model: str,
        learning_rate: float,
        augmentation: list,
        regularization: dict,
        reasoning: str,
    ):
        self.action = action
        self.model = model
        self.learning_rate = learning_rate
        self.augmentation = augmentation
        self.regularization = regularization
        self.reasoning = reasoning
    
    def to_experiment_config(self, current_config: ExperimentConfig) -> ExperimentConfig:
        """Convert suggestion to experiment config."""
        config = ExperimentConfig.from_dict(current_config.to_dict())
        config.model = self.model
        config.learning_rate = self.learning_rate
        config.augmentation = self.augmentation
        config.regularization = self.regularization
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "model": self.model,
            "learning_rate": self.learning_rate,
            "augmentation": self.augmentation,
            "regularization": self.regularization,
            "reasoning": self.reasoning,
        }


class ImprovementAgent:
    """Agent that suggests improvements based on training results."""
    
    def __init__(self):
        self.client = get_ollama_client()
        self.system_prompt = get_system_prompt("improvement")
        self.user_prompt_template = get_user_prompt("improvement")
    
    def suggest_improvements(
        self,
        model: str,
        accuracy: float,
        f1_score: float,
        loss: float,
        training_time: float,
        epochs_trained: int,
        total_epochs: int,
        early_stop: bool,
        dataset_size: int,
        num_classes: int,
        imbalance_ratio: float,
        previous_config: ExperimentConfig,
    ) -> Optional[ImprovementSuggestion]:
        """
        Analyze training results and suggest improvements.
        
        Args:
            model: Model name that was trained
            accuracy: Final accuracy achieved
            f1_score: Final F1 score
            loss: Final loss value
            training_time: Training time in seconds
            epochs_trained: Number of epochs trained
            total_epochs: Total epochs to train
            early_stop: Whether early stopping triggered
            dataset_size: Number of training samples
            num_classes: Number of classes
            imbalance_ratio: Class imbalance ratio
            previous_config: Previous training config
            
        Returns:
            ImprovementSuggestion or None if failed
        """
        logger.info(f"Improvement agent analyzing: {model}, acc={accuracy:.4f}")
        
        # Build prompt
        prompt = self.user_prompt_template.format(
            model=model,
            accuracy=accuracy,
            f1_score=f1_score,
            loss=loss,
            training_time=training_time,
            epochs_trained=epochs_trained,
            total_epochs=total_epochs,
            early_stop=early_stop,
            dataset_size=dataset_size,
            num_classes=num_classes,
            imbalance_ratio=imbalance_ratio,
            previous_config=json.dumps(previous_config.to_dict(), indent=2),
        )
        
        response_text = self.client.generate(
            prompt=prompt,
            system=self.system_prompt,
            temperature=0.5,  # Higher creativity for suggestions
        )
        
        if not response_text:
            logger.warning("Improvement agent: empty response")
            return self._fallback_suggestion(accuracy, model, previous_config)
        
        # Parse JSON
        try:
            suggestion_data = json.loads(response_text)
            
            suggestion = ImprovementSuggestion(
                action=suggestion_data.get("action", "adjust_learning_rate"),
                model=suggestion_data.get("model", model),
                learning_rate=suggestion_data.get("learning_rate", previous_config.learning_rate),
                augmentation=suggestion_data.get("augmentation", previous_config.augmentation),
                regularization=suggestion_data.get("regularization", previous_config.regularization),
                reasoning=suggestion_data.get("reasoning", ""),
            )
            
            logger.info(f"Improvement suggestion: {suggestion.action} for {suggestion.model}")
            return suggestion
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse improvement suggestion: {e}")
            return self._fallback_suggestion(accuracy, model, previous_config)
    
    def _fallback_suggestion(
        self,
        accuracy: float,
        model: str,
        config: ExperimentConfig,
    ) -> ImprovementSuggestion:
        """Fallback suggestion based on accuracy."""
        logger.info("Using fallback improvement suggestion")
        
        # Determine improvement strategy based on accuracy
        if accuracy < 0.5:
            # Very low - try new model
            return ImprovementSuggestion(
                action="switch_model",
                model="ResNet50" if "mobile" in model.lower() else "MobileNetV3",
                learning_rate=0.0003,
                augmentation=["flip", "rotation", "brightness"],
                regularization={"dropout": 0.4, "l2": 0.0002},
                reasoning="Very low accuracy, switching to different architecture",
            )
        elif accuracy < 0.7:
            # Low - increase augmentation
            return ImprovementSuggestion(
                action="increase_augmentation",
                model=model,
                learning_rate=config.learning_rate,
                augmentation=["flip", "rotation", "brightness", "zoom", "contrast"],
                regularization=config.regularization,
                reasoning="Low accuracy, increasing data augmentation",
            )
        elif accuracy < 0.85:
            # Medium - reduce learning rate
            return ImprovementSuggestion(
                action="adjust_learning_rate",
                model=model,
                learning_rate=config.learning_rate * 0.5,
                augmentation=config.augmentation,
                regularization=config.regularization,
                reasoning="Medium accuracy, reducing learning rate for fine-tuning",
            )
        else:
            # Good - increase regularization
            return ImprovementSuggestion(
                action="increase_regularization",
                model=model,
                learning_rate=config.learning_rate,
                augmentation=config.augmentation,
                regularization={"dropout": min(0.5, config.regularization.get("dropout", 0.3) + 0.1)},
                reasoning="Good accuracy, increasing regularization to prevent overfitting",
            )


def get_improvement_agent() -> ImprovementAgent:
    """Get improvement agent instance."""
    return ImprovementAgent()
