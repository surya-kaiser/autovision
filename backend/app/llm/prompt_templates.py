"""
Centralized prompt templates for LLM agents.
Ensures consistent prompting across all agents.
"""

RESEARCH_AGENT_SYSTEM = """You are an expert machine learning researcher specializing in computer vision and tabular data.
Your role is to analyze dataset characteristics and recommend optimal model architectures.
Provide recommendations in valid JSON format only.
Consider dataset size, complexity, computational constraints, and class imbalance."""

RESEARCH_AGENT_PROMPT = """Analyze this dataset and recommend the top 3 model architectures:

Dataset Info:
- Task Type: {task_type}
- Dataset Size: {dataset_size:,} samples
- Number of Classes: {num_classes}
- Image Resolution: {image_resolution}
- Class Imbalance Ratio: {imbalance_ratio:.2f}x
- Training Budget: {training_budget} minutes max

For image tasks, recommend architectures (e.g., MobileNetV3, EfficientNetB0, ResNet50).
For tabular tasks, recommend models (e.g., XGBoost, LightGBM, Random Forest).

Return ONLY valid JSON array with no markdown, no explanation:
[
  {{
    "model": "ModelName",
    "reason": "Why this model is suitable",
    "priority": 1,
    "estimated_accuracy": 0.92
  }},
  ...
]"""

TRAINING_STRATEGY_SYSTEM = """You are an expert in hyperparameter tuning and training strategies.
Your role is to suggest optimal training configurations for given models and datasets.
Provide recommendations in valid JSON format only."""

TRAINING_STRATEGY_PROMPT = """Suggest optimal training strategy for {model} on this dataset:

Dataset:
- Size: {dataset_size:,} samples
- Classes: {num_classes}
- Task: {task_type}
- Hardware: {hardware}

Return ONLY valid JSON configuration with no markdown:
{{
  "learning_rate": 0.0003,
  "batch_size": 32,
  "epochs": 15,
  "optimizer": "adam",
  "augmentation": ["flip", "rotation", "brightness"],
  "regularization": {{"dropout": 0.3, "l2": 0.0001}},
  "fine_tune_layers": 20,
  "early_stopping_patience": 5,
  "warmup_epochs": 2
}}"""

IMPROVEMENT_AGENT_SYSTEM = """You are an ML optimization expert.
Your role is to analyze failed or underperforming experiments and suggest improvements.
Provide recommendations in valid JSON format only."""

IMPROVEMENT_AGENT_PROMPT = """Analyze this training result and suggest improvements:

Training Result:
- Model: {model}
- Accuracy: {accuracy:.4f}
- F1 Score: {f1_score:.4f}
- Loss: {loss:.4f}
- Training Time: {training_time} sec
- Epochs Trained: {epochs_trained}/{total_epochs}
- Early Stop Triggered: {early_stop}

Dataset:
- Size: {dataset_size:,}
- Classes: {num_classes}
- Imbalance: {imbalance_ratio:.2f}x

Previous Config:
{previous_config}

Suggest next experiment with ONLY valid JSON (no markdown):
{{
  "action": "adjust_learning_rate|increase_augmentation|switch_model|increase_regularization",
  "model": "NewModelName or same",
  "learning_rate": 0.0001,
  "augmentation": ["flip", "rotation", "zoom"],
  "regularization": {{"dropout": 0.4, "l2": 0.0002}},
  "reasoning": "Why this change should improve results"
}}"""

EXPERIMENT_SUMMARIZER_SYSTEM = """You are an ML researcher summarizing experiment results."""

EXPERIMENT_SUMMARIZER_PROMPT = """Summarize the best experiment from this list:

Experiments:
{experiments_json}

Return ONLY valid JSON with:
{{
  "best_experiment_id": "uuid",
  "reason": "Why it's the best",
  "key_insight": "Main learning",
  "recommendations": ["recommendation1", "recommendation2"]
}}"""

# System prompts
SYSTEM_PROMPTS = {
    "research": RESEARCH_AGENT_SYSTEM,
    "strategy": TRAINING_STRATEGY_SYSTEM,
    "improvement": IMPROVEMENT_AGENT_SYSTEM,
    "summarizer": EXPERIMENT_SUMMARIZER_SYSTEM,
}

# User prompts (templates, use with .format())
USER_PROMPTS = {
    "research": RESEARCH_AGENT_PROMPT,
    "strategy": TRAINING_STRATEGY_PROMPT,
    "improvement": IMPROVEMENT_AGENT_PROMPT,
    "summarizer": EXPERIMENT_SUMMARIZER_PROMPT,
}


def get_system_prompt(agent_type: str) -> str:
    """Get system prompt for agent type."""
    return SYSTEM_PROMPTS.get(agent_type, "")


def get_user_prompt(agent_type: str) -> str:
    """Get user prompt template for agent type."""
    return USER_PROMPTS.get(agent_type, "")
