"""Pydantic schemas and dataclasses for the Machine Learning Engine."""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone

from pydantic import BaseModel, Field, ConfigDict
from datasense.data_processing.schemas import PreprocessingConfig


class TaskType(str, Enum):
    """Machine Learning task classification."""

    AUTO = "auto"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ClassificationAlgorithm(str, Enum):
    """Supported Classification algorithms."""

    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    GRADIENT_BOOSTING = "gradient_boosting"


class RegressionAlgorithm(str, Enum):
    """Supported Regression algorithms."""

    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest_regressor"
    XGBOOST = "xgboost_regressor"
    GRADIENT_BOOSTING = "gradient_boosting_regressor"


class ClassificationMetrics(BaseModel):
    """Evaluation metrics for classification models."""

    accuracy: float = Field(..., description="Overall accuracy score")
    precision: float = Field(..., description="Precision score (weighted/binary)")
    recall: float = Field(..., description="Recall score (weighted/binary)")
    f1: float = Field(..., description="F1 score (weighted/binary)")
    roc_auc: Optional[float] = Field(None, description="Receiver Operating Characteristic AUC score")
    confusion_matrix: List[List[int]] = Field(default_factory=list, description="2D matrix of confusion counts")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RegressionMetrics(BaseModel):
    """Evaluation metrics for regression models."""

    mae: float = Field(..., description="Mean Absolute Error")
    mse: float = Field(..., description="Mean Squared Error")
    rmse: float = Field(..., description="Root Mean Squared Error")
    r2: float = Field(..., description="R-squared (coefficient of determination)")
    mape: Optional[float] = Field(None, description="Mean Absolute Percentage Error (%)")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModelEvaluationResult(BaseModel):
    """Model evaluation summary for single trained model."""

    model_id: str = Field(..., description="Unique model identifier")
    run_id: str = Field(..., description="Associated experiment run ID")
    model_name: str = Field(..., description="Human-readable model name")
    model_type: str = Field(..., description="Algorithm string key")
    task_type: TaskType = Field(..., description="Classification or Regression")
    training_time_seconds: float = Field(..., description="Training time in seconds")
    validation_metrics: Dict[str, Any] = Field(..., description="Metrics computed on validation set")
    test_metrics: Dict[str, Any] = Field(..., description="Metrics computed on holdout test set")
    cv_scores_mean: Optional[float] = Field(None, description="Mean cross-validation score on train split")
    cv_scores_std: Optional[float] = Field(None, description="Standard deviation of CV scores")
    best_params: Optional[Dict[str, Any]] = Field(None, description="Tuned hyperparameter dictionary")
    is_best: bool = Field(False, description="Whether model was chosen as top performer")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TrainingConfig(BaseModel):
    """Configuration for ModelTrainer execution."""

    target_column: str = Field(..., description="Target column name for predictive modeling")
    task_type: TaskType = Field(default=TaskType.AUTO, description="Task mode: auto, classification, or regression")
    selected_models: Optional[List[str]] = Field(
        default=None,
        description="Subset of model keys to train. If empty/None, all task-compatible models are trained.",
    )
    test_size: float = Field(default=0.15, ge=0.05, le=0.4, description="Proportion of dataset for test set")
    val_size: float = Field(default=0.15, ge=0.05, le=0.4, description="Proportion of dataset for validation set")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    cross_validation_folds: int = Field(default=5, ge=2, le=10, description="K-fold cross-validation split count")
    enable_tuning: bool = Field(default=False, description="Whether to execute hyperparameter search")
    n_tuning_trials: int = Field(default=10, ge=2, le=50, description="Number of randomized search iterations")
    preprocessing_config: Optional[PreprocessingConfig] = Field(
        default=None, description="Data preprocessing configuration"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModelComparisonReport(BaseModel):
    """Comparison report summarizing experiment results across all trained models."""

    run_id: str = Field(..., description="Unique experiment run ID")
    dataset_id: Optional[int] = Field(None, description="Registered dataset ID if applicable")
    target_column: str = Field(..., description="Target column name")
    task_type: TaskType = Field(..., description="Resolved task type")
    total_samples: int = Field(..., description="Total rows in dataset")
    train_samples: int = Field(..., description="Number of training samples")
    val_samples: int = Field(..., description="Number of validation samples")
    test_samples: int = Field(..., description="Number of test samples")
    feature_names: List[str] = Field(default_factory=list, description="Transformed feature names fed to models")
    results: List[ModelEvaluationResult] = Field(default_factory=list, description="Results per model")
    best_model_id: str = Field(..., description="Model ID of top performer")
    best_model_name: str = Field(..., description="Model name of top performer")
    selection_metric: str = Field(..., description="Primary metric used for best model selection")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Timestamp of report generation"
    )


    model_config = ConfigDict(arbitrary_types_allowed=True)


class PredictionRequest(BaseModel):
    """Payload for making real-time model predictions."""

    model_id: Optional[str] = Field(None, description="Model ID to use for inference. Uses latest best if omitted.")
    data: List[Dict[str, Any]] = Field(..., min_length=1, description="List of feature dictionary records")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PredictionResponse(BaseModel):
    """Response returned from model prediction endpoint."""

    model_id: str = Field(..., description="Model ID used for predictions")
    model_name: str = Field(..., description="Model algorithm name")
    task_type: TaskType = Field(..., description="Classification or Regression")
    predictions: List[Any] = Field(..., description="Predicted label or value per record")
    probabilities: Optional[List[Dict[str, float]]] = Field(
        None, description="Class probabilities for classification tasks"
    )
    row_count: int = Field(..., description="Number of predicted records")

    model_config = ConfigDict(arbitrary_types_allowed=True)
