"""
Research Agent: Analyzes datasets and recommends model architectures.
Uses LLM reasoning to understand data characteristics and suggest optimal models.
"""
import json
from typing import List, Dict, Any, Optional, Tuple
from app.llm.ollama_client import get_ollama_client
from app.llm.prompt_templates import get_system_prompt, get_user_prompt
from app.models.schemas import TaskType, DatasetFormat
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRecommendation:
    """Structured model recommendation."""
    
    def __init__(self, model: str, reason: str, priority: int = 1, 
                 estimated_accuracy: float = 0.8):
        self.model = model
        self.reason = reason
        self.priority = priority
        self.estimated_accuracy = estimated_accuracy
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "reason": self.reason,
            "priority": self.priority,
            "estimated_accuracy": self.estimated_accuracy,
        }


class ResearchAgent:
    """Autonomous research agent for model selection."""
    
    def __init__(self):
        self.client = get_ollama_client()
        self.system_prompt = get_system_prompt("research")
        self.user_prompt_template = get_user_prompt("research")
    
    def analyze_dataset(
        self,
        task_type: TaskType,
        dataset_size: int,
        num_classes: int,
        image_resolution: Optional[Tuple[int, int]] = None,
        imbalance_ratio: float = 1.0,
        dataset_format: str = "image",
        training_budget: int = 10,
    ) -> Optional[List[ModelRecommendation]]:
        """
        Analyze dataset characteristics and recommend models.
        
        Args:
            task_type: Classification, regression, detection, etc.
            dataset_size: Number of samples
            num_classes: Number of output classes
            image_resolution: For image tasks (height, width)
            imbalance_ratio: Class balance ratio (1.0 = balanced)
            dataset_format: image, csv, etc.
            training_budget: Max training time in minutes
            
        Returns:
            List of ModelRecommendation or None if failed
        """
        logger.info(f"Research agent analyzing dataset: {task_type}, {dataset_size} samples")
        
        # Format image resolution string
        resolution_str = str(list(image_resolution)) if image_resolution else "Unknown"
        
        # Build prompt
        try:
            prompt = self.user_prompt_template.format(
                task_type=task_type.value,
                dataset_size=dataset_size,
                num_classes=num_classes,
                image_resolution=resolution_str,
                imbalance_ratio=imbalance_ratio,
                training_budget=training_budget,
            )
        except Exception as e:
            logger.error(f"Research agent prompt formatting failed: {e}")
            return self._fallback_recommendations(task_type, dataset_size)
        
        # Call Ollama
        response_text = self.client.generate(
            prompt=prompt,
            system=self.system_prompt,
            temperature=0.3,
        )
        
        if not response_text:
            logger.warning("Research agent: empty response from Ollama")
            return self._fallback_recommendations(task_type, dataset_size)
        
        # Parse JSON response
        try:
            recommendations_data = json.loads(response_text)
            
            # Handle both list and wrapped responses
            if isinstance(recommendations_data, dict) and "recommendations" in recommendations_data:
                recommendations_data = recommendations_data["recommendations"]
            
            if not isinstance(recommendations_data, list):
                logger.warning(f"Invalid recommendation format: {type(recommendations_data)}")
                return self._fallback_recommendations(task_type, dataset_size)
            
            recommendations = [
                ModelRecommendation(
                    model=item.get("model", "Unknown"),
                    reason=item.get("reason", ""),
                    priority=item.get("priority", 1),
                    estimated_accuracy=item.get("estimated_accuracy", 0.8),
                )
                for item in recommendations_data[:3]  # Max 3 recommendations
            ]
            
            logger.info(f"Research agent recommended {len(recommendations)} models")
            return recommendations
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse research agent response: {e}")
            return self._fallback_recommendations(task_type, dataset_size)
    
    def _fallback_recommendations(
        self,
        task_type: TaskType,
        dataset_size: int,
    ) -> List[ModelRecommendation]:
        """Fallback rule-based recommendations if LLM unavailable."""
        logger.info("Using fallback recommendations")

        if task_type == TaskType.CLASSIFICATION:
            if dataset_size > 5000:
                return [
                    ModelRecommendation(
                        "ResNet18",
                        "Transfer-learning with ImageNet-pretrained ResNet18 — strong on medium/large datasets",
                        priority=1,
                        estimated_accuracy=0.90,
                    ),
                    ModelRecommendation(
                        "CNN",
                        "Custom 3-block CNN — lightweight and trains fast on any GPU/CPU",
                        priority=2,
                        estimated_accuracy=0.85,
                    ),
                ]
            else:
                return [
                    ModelRecommendation(
                        "CNN",
                        "Custom CNN for small dataset — less prone to overfitting than large models",
                        priority=1,
                        estimated_accuracy=0.82,
                    ),
                    ModelRecommendation(
                        "ResNet18",
                        "ResNet18 with pretrained backbone — transfer learning even on small data",
                        priority=2,
                        estimated_accuracy=0.85,
                    ),
                ]

        elif task_type == TaskType.SEGMENTATION:
            return [
                ModelRecommendation(
                    "UNet",
                    "Standard U-Net — reliable encoder-decoder architecture for pixel-wise segmentation",
                    priority=1,
                    estimated_accuracy=0.80,
                ),
                ModelRecommendation(
                    "DeepLabV3",
                    "DeepLabV3 with ResNet50 backbone — atrous convolutions capture multi-scale context",
                    priority=2,
                    estimated_accuracy=0.85,
                ),
            ]

        elif task_type == TaskType.OBJECT_DETECTION:
            return [
                ModelRecommendation(
                    "YOLOv8n",
                    "Nano model for real-time detection",
                    priority=1,
                    estimated_accuracy=0.75,
                ),
                ModelRecommendation(
                    "YOLOv8s",
                    "Small model for better accuracy",
                    priority=2,
                    estimated_accuracy=0.82,
                ),
            ]

        else:  # Regression or unknown
            return [
                ModelRecommendation(
                    "XGBoost",
                    "Gradient boosting for regression",
                    priority=1,
                    estimated_accuracy=0.85,
                ),
                ModelRecommendation(
                    "LightGBM",
                    "Fast gradient boosting",
                    priority=2,
                    estimated_accuracy=0.84,
                ),
                ModelRecommendation(
                    "RandomForest",
                    "Ensemble method",
                    priority=3,
                    estimated_accuracy=0.80,
                ),
            ]


def get_research_agent() -> ResearchAgent:
    """Get research agent instance."""
    return ResearchAgent()
