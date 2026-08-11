"""Machine Learning Engine REST API Router."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import pandas as pd

from datasense.database.connection import get_db_session
from datasense.database.models import DatasetMetadata, SystemAuditLog
from datasense.ml_models.schemas import (
    TaskType,
    TrainingConfig,
    ModelComparisonReport,
    PredictionRequest,
    PredictionResponse,
    ModelEvaluationResult,
)
from datasense.ml_models.trainer import ModelTrainer
from datasense.ml_models.registry import LocalModelRegistry
from datasense.utilities.logger import get_logger

logger = get_logger("api.ml")

router = APIRouter(prefix="/api/v1/ml", tags=["Machine Learning Engine"])


class TrainMLRequest(BaseModel):
    """Payload to trigger model training pipeline."""

    dataset_id: Optional[int] = Field(None, description="Registered dataset ID to train on")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Direct feature records if not using dataset_id")
    target_column: str = Field(..., description="Target column name for prediction")
    task_type: TaskType = Field(default=TaskType.AUTO, description="Task mode: auto, classification, or regression")
    selected_models: Optional[List[str]] = Field(default=None, description="Subset of models to train")
    test_size: float = Field(default=0.15, ge=0.05, le=0.4, description="Holdout test set fraction")
    val_size: float = Field(default=0.15, ge=0.05, le=0.4, description="Validation set fraction")
    random_state: int = Field(default=42, description="Random seed")
    cross_validation_folds: int = Field(default=5, ge=2, le=10, description="CV fold count")
    enable_tuning: bool = Field(default=False, description="Enable hyperparameter search")


@router.post(
    "/train",
    response_model=ModelComparisonReport,
    status_code=status.HTTP_200_OK,
    summary="Train Machine Learning Models",
    description="Runs dataset splitting, data leakage prevention, cross-validation, training, tuning, evaluation, and generates model comparison report.",
)
def train_models_endpoint(
    payload: TrainMLRequest,
    db: Session = Depends(get_db_session),
) -> ModelComparisonReport:
    """Triggers ML training workflow on a registered dataset or inline records."""
    df: Optional[pd.DataFrame] = None

    if payload.dataset_id is not None:
        dataset_record = db.query(DatasetMetadata).filter(DatasetMetadata.id == payload.dataset_id).first()
        if not dataset_record:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {payload.dataset_id} not found.")

        if not dataset_record.preview_data or "rows" not in dataset_record.preview_data:
            raise HTTPException(
                status_code=400, detail=f"Dataset ID {payload.dataset_id} contains no readable rows for training."
            )

        df = pd.DataFrame(dataset_record.preview_data["rows"])

    elif payload.data:
        df = pd.DataFrame(payload.data)

    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="Must provide valid dataset_id or non-empty data list for training.")

    if payload.target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{payload.target_column}' not found in dataset columns: {list(df.columns)}",
        )

    config = TrainingConfig(
        target_column=payload.target_column,
        task_type=payload.task_type,
        selected_models=payload.selected_models,
        test_size=payload.test_size,
        val_size=payload.val_size,
        random_state=payload.random_state,
        cross_validation_folds=payload.cross_validation_folds,
        enable_tuning=payload.enable_tuning,
    )

    try:
        trainer = ModelTrainer(registry=LocalModelRegistry())
        report: ModelComparisonReport = trainer.train(df, config, dataset_id=payload.dataset_id)

        # Audit log
        audit = SystemAuditLog(
            event_type="ML_TRAINING_EXECUTED",
            description=f"Trained ML models on target '{payload.target_column}' for run_id '{report.run_id}'",
            status="SUCCESS",
            meta_data={
                "run_id": report.run_id,
                "dataset_id": payload.dataset_id,
                "target_column": payload.target_column,
                "best_model_id": report.best_model_id,
                "best_model_name": report.best_model_name,
            },
        )
        db.add(audit)
        db.commit()

        return report

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        logger.error(f"Error during ML model training: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(exc)}")


@router.get(
    "/models",
    response_model=List[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="List Registered Trained Models",
    description="Returns metadata list of all saved machine learning models in registry.",
)
def list_models_endpoint(
    task_type: Optional[str] = Query(None, description="Optional task type filter (classification/regression)"),
) -> List[Dict[str, Any]]:
    """List registered models from model registry."""
    registry = LocalModelRegistry()
    return registry.list_models(task_type=task_type)


@router.get(
    "/models/{model_id}",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get Trained Model Details",
    description="Retrieves metadata, hyperparameters, and evaluation metrics for a specific model ID.",
)
def get_model_endpoint(model_id: str) -> Dict[str, Any]:
    """Get detailed model metadata."""
    registry = LocalModelRegistry()
    meta = registry.get_model_metadata(model_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Model with ID '{model_id}' not found in registry.")
    return meta


@router.get(
    "/runs/{run_id}",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get Experiment Run Report",
    description="Retrieves complete comparison report for an experiment run ID.",
)
def get_run_report_endpoint(run_id: str) -> Dict[str, Any]:
    """Get comparison report for experiment run."""
    registry = LocalModelRegistry()
    report = registry.get_experiment_report(run_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Experiment run report with ID '{run_id}' not found.")
    return report


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Make Real-Time Predictions",
    description="Makes predictions using specified model_id (or latest best model if omitted) on input feature records.",
)
def predict_endpoint(
    payload: PredictionRequest,
    db: Session = Depends(get_db_session),
) -> PredictionResponse:
    """Inference endpoint for classification and regression models."""
    registry = LocalModelRegistry()
    model_id = payload.model_id

    if not model_id:
        models = registry.list_models()
        if not models:
            raise HTTPException(status_code=400, detail="No trained models available in registry for inference.")
        # Find latest best model
        best_models = [m for m in models if m.get("is_best")]
        target_model = best_models[0] if best_models else models[0]
        model_id = target_model["model_id"]

    try:
        df_features = pd.DataFrame(payload.data)
        if df_features.empty:
            raise HTTPException(status_code=400, detail="Prediction payload 'data' records list is empty.")

        trainer = ModelTrainer(registry=registry)
        response = trainer.predict(model_id=model_id, df_features=df_features)
        return response

    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as exc:
        logger.error(f"Prediction error for model_id '{model_id}': {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")


@router.delete(
    "/models/{model_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Registered Model",
    description="Deletes model artifact file and database registry entry.",
)
def delete_model_endpoint(model_id: str) -> Dict[str, str]:
    """Delete model artifact."""
    registry = LocalModelRegistry()
    success = registry.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found or already deleted.")
    return {"status": "deleted", "model_id": model_id}
