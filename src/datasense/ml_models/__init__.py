"""Machine Learning models interface and training pipelines."""

from datasense.utilities.logger import get_logger
from datasense.ml_models.schemas import (
    TaskType,
    ClassificationAlgorithm,
    RegressionAlgorithm,
    ClassificationMetrics,
    RegressionMetrics,
    ModelEvaluationResult,
    TrainingConfig,
    ModelComparisonReport,
    PredictionRequest,
    PredictionResponse,
)
from datasense.ml_models.registry import BaseModelRegistry, LocalModelRegistry, MLflowModelRegistry
from datasense.ml_models.trainer import ModelTrainer, determine_task_type

logger = get_logger("ml_models")


class BaseModelPipeline:
    """Base abstract interface for machine learning pipelines."""

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        logger.info(f"Initialized BaseModelPipeline with model_type: {model_type}")

    def train(self, X, y):
        """Pipeline training interface stub."""
        logger.info(f"Training stub for {self.model_type}")
        return {"status": "trained", "model_type": self.model_type}


__all__ = [
    "ModelTrainer",
    "determine_task_type",
    "BaseModelRegistry",
    "LocalModelRegistry",
    "MLflowModelRegistry",
    "TaskType",
    "ClassificationAlgorithm",
    "RegressionAlgorithm",
    "ClassificationMetrics",
    "RegressionMetrics",
    "ModelEvaluationResult",
    "TrainingConfig",
    "ModelComparisonReport",
    "PredictionRequest",
    "PredictionResponse",
    "BaseModelPipeline",
]

