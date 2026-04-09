"""
Task-specific trainer wrappers.

Each module provides a single async train_* function that:
  1. Validates the model type is correct for the task (hard gate)
  2. Validates the preprocessed data exists
  3. Delegates to the existing dl_trainer / trainer implementations

Import paths:
  from app.services.trainers.segmentation_trainer import train_segmentation
  from app.services.trainers.image_classification_trainer import train_classification
  from app.services.trainers.detection_trainer import train_detection
  from app.services.trainers.tabular_trainer import train_tabular
"""
from app.services.trainers.segmentation_trainer import train_segmentation
from app.services.trainers.image_classification_trainer import train_classification
from app.services.trainers.detection_trainer import train_detection
from app.services.trainers.tabular_trainer import train_tabular

__all__ = ["train_segmentation", "train_classification", "train_detection", "train_tabular"]
