"""
Experiment Tracking: Stores and manages experiment metadata.
"""
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentMetadata:
    """Experiment metadata."""
    id: str
    session_id: str
    model: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    status: str  # queued, running, completed, failed
    created_at: str
    updated_at: str
    training_time_s: float = 0.0
    logs: List[str] = None
    improvement_suggestion: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["logs"] = data.get("logs") or []
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentMetadata":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            session_id=data.get("session_id", ""),
            model=data.get("model", ""),
            config=data.get("config", {}),
            metrics=data.get("metrics", {}),
            status=data.get("status", "queued"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            training_time_s=data.get("training_time_s", 0.0),
            logs=data.get("logs", []),
            improvement_suggestion=data.get("improvement_suggestion"),
        )


class ExperimentTracker:
    """Tracks and stores experiments."""
    
    def __init__(self):
        self.experiments_dir = settings.MODEL_DIR.parent / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Experiment tracker initialized at {self.experiments_dir}")
    
    def create_experiment(
        self,
        session_id: str,
        model: str,
        config: Dict[str, Any],
    ) -> ExperimentMetadata:
        """Create new experiment and save to disk immediately."""
        exp = ExperimentMetadata(
            id=str(uuid.uuid4()),
            session_id=session_id,
            model=model,
            config=config,
            metrics={},
            status="queued",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            logs=[],
        )
        self.save_experiment(exp)
        logger.info(f"Created experiment {exp.id}")
        return exp
    
    def save_experiment(self, experiment: ExperimentMetadata) -> bool:
        """Save experiment to disk."""
        try:
            exp_path = self.experiments_dir / f"{experiment.id}.json"
            
            # Update timestamp
            experiment.updated_at = datetime.now().isoformat()
            
            with open(exp_path, "w") as f:
                json.dump(experiment.to_dict(), f, indent=2)
            
            logger.debug(f"Saved experiment {experiment.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save experiment: {e}")
            return False
    
    def load_experiment(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Load experiment from disk."""
        try:
            exp_path = self.experiments_dir / f"{experiment_id}.json"
            
            if not exp_path.exists():
                logger.warning(f"Experiment not found: {experiment_id}")
                return None
            
            with open(exp_path, "r") as f:
                data = json.load(f)
            
            return ExperimentMetadata.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to load experiment: {e}")
            return None
    
    def get_session_experiments(self, session_id: str) -> List[ExperimentMetadata]:
        """Get all experiments for a session."""
        experiments = []
        
        try:
            for exp_file in self.experiments_dir.glob("*.json"):
                try:
                    with open(exp_file, "r") as f:
                        data = json.load(f)
                    
                    if data.get("session_id") == session_id:
                        experiments.append(ExperimentMetadata.from_dict(data))
                        
                except Exception as e:
                    logger.warning(f"Failed to load {exp_file}: {e}")
            
            # Sort by creation time (newest first)
            experiments.sort(
                key=lambda x: x.created_at,
                reverse=True
            )
            
            return experiments
            
        except Exception as e:
            logger.error(f"Failed to get session experiments: {e}")
            return []
    
    def get_best_experiment(self, session_id: str) -> Optional[ExperimentMetadata]:
        """Get best experiment by accuracy."""
        experiments = self.get_session_experiments(session_id)
        
        if not experiments:
            return None
        
        # Filter completed experiments
        completed = [e for e in experiments if e.status == "completed"]
        
        if not completed:
            return None
        
        # Sort by accuracy
        completed.sort(
            key=lambda x: x.metrics.get("accuracy", 0),
            reverse=True
        )
        
        return completed[0]
    
    def update_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, Any],
        status: str = "completed",
        training_time_s: float = 0.0,
    ) -> bool:
        """Update experiment metrics."""
        experiment = self.load_experiment(experiment_id)
        
        if not experiment:
            logger.warning(f"Experiment not found: {experiment_id}")
            return False
        
        experiment.metrics = metrics
        experiment.status = status
        experiment.training_time_s = training_time_s
        
        return self.save_experiment(experiment)
    
    def add_log(self, experiment_id: str, log_entry: str) -> bool:
        """Add log entry to experiment."""
        experiment = self.load_experiment(experiment_id)
        
        if not experiment:
            return False
        
        if experiment.logs is None:
            experiment.logs = []
        
        experiment.logs.append(f"[{datetime.now().isoformat()}] {log_entry}")
        
        return self.save_experiment(experiment)
    
    def get_experiment_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of all experiments in session."""
        experiments = self.get_session_experiments(session_id)
        
        if not experiments:
            return {
                "total_experiments": 0,
                "best_accuracy": 0.0,
                "best_model": None,
                "experiments": [],
            }
        
        completed = [e for e in experiments if e.status == "completed"]
        
        best = None
        best_acc = 0.0
        
        for exp in completed:
            acc = exp.metrics.get("accuracy", 0)
            if acc > best_acc:
                best_acc = acc
                best = exp
        
        return {
            "total_experiments": len(experiments),
            "completed_experiments": len(completed),
            "best_accuracy": best_acc,
            "best_model": best.model if best else None,
            "best_experiment_id": best.id if best else None,
            "total_training_time_s": sum(e.training_time_s for e in completed),
            "experiments": [
                {
                    "id": e.id,
                    "model": e.model,
                    "accuracy": e.metrics.get("accuracy", 0),
                    "status": e.status,
                    "created_at": e.created_at,
                }
                for e in experiments[:10]  # Latest 10
            ],
        }


# Global instance
_tracker: Optional[ExperimentTracker] = None


def get_experiment_tracker() -> ExperimentTracker:
    """Get experiment tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker()
    return _tracker
