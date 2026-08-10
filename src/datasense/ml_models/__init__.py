"""Machine Learning models interface and training pipelines."""

from datasense.utilities.logger import get_logger

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
