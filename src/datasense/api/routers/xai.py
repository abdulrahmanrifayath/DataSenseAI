"""Explainable AI (SHAP) REST API Router."""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from datasense.database.connection import get_db_session
from datasense.database.models import DatasetMetadata, XAIExplanationRecord, SystemAuditLog
from datasense.xai.schemas import XAIReport, XAIRequest, XAIResponse
from datasense.xai.service import XAIExplanationService
from datasense.ml_models.trainer import determine_task_type
from datasense.utilities.logger import get_logger

logger = get_logger("api.xai")

router = APIRouter(prefix="/api/v1/xai", tags=["Explainable AI"])


@router.post(
    "/explain",
    response_model=XAIResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate SHAP Model Explanations",
    description="Computes global feature importances and local per-instance explanations using SHAP.",
)
def explain_model_endpoint(
    payload: XAIRequest,
    db: Session = Depends(get_db_session),
) -> XAIResponse:
    """Triggers SHAP model explanation pipeline."""
    df: Optional[pd.DataFrame] = None

    if payload.dataset_id is not None:
        dataset_record = db.query(DatasetMetadata).filter(DatasetMetadata.id == payload.dataset_id).first()
        if not dataset_record:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {payload.dataset_id} not found.")

        if not dataset_record.preview_data or "rows" not in dataset_record.preview_data:
            raise HTTPException(
                status_code=400, detail=f"Dataset ID {payload.dataset_id} contains no readable rows for SHAP explanation."
            )

        df = pd.DataFrame(dataset_record.preview_data["rows"])

    elif payload.data:
        df = pd.DataFrame(payload.data)

    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="Must provide valid dataset_id or non-empty data list for SHAP explanation.")

    # Filter numerical feature columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != payload.target_column]
    if not num_cols:
        raise HTTPException(status_code=400, detail="No valid numerical feature columns found for SHAP calculation.")

    df_num = df[num_cols].fillna(df[num_cols].median())
    X = df_num.to_numpy()

    # Determine target & fit surrogate explainer model if custom model is not loaded
    if payload.target_column and payload.target_column in df.columns:
        y = df[payload.target_column].to_numpy()
        task_type = determine_task_type(df, payload.target_column)

    else:
        y = X[:, 0]
        task_type = "regression"
        X = X[:, 1:]
        num_cols = num_cols[1:]

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=50, random_state=42)

    model.fit(X, y)

    try:
        service = XAIExplanationService()
        report: XAIReport = service.explain(
            model=model,
            X_df=pd.DataFrame(X, columns=num_cols),
            feature_names=num_cols,
            task_type=task_type,
            model_id=payload.model_id,
            instance_indices=payload.instance_indices,
        )

        # Database Persistence
        db_record = XAIExplanationRecord(
            run_id=report.run_id,
            model_id=payload.model_id,
            task_type=task_type,
            report=report.model_dump(),
        )
        db.add(db_record)

        # Audit Log
        audit = SystemAuditLog(
            event_type="XAI_EXPLANATION_GENERATED",
            description=f"Generated SHAP explanations for model (run_id: {report.run_id})",
            status="SUCCESS",
            meta_data={"run_id": report.run_id, "model_id": payload.model_id, "task_type": task_type},
        )
        db.add(audit)
        db.commit()

        return XAIResponse(run_id=report.run_id, report=report)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        logger.error(f"XAI explanation failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"XAI explanation error: {str(exc)}")


@router.get(
    "/runs/{run_id}",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get XAI Explanation Report",
    description="Retrieves report for a past XAI run ID.",
)
def get_xai_run_endpoint(
    run_id: str,
    db: Session = Depends(get_db_session),
) -> Dict[str, Any]:
    """Retrieve XAI run report."""
    record = db.query(XAIExplanationRecord).filter(XAIExplanationRecord.run_id == run_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"XAI run with ID '{run_id}' not found.")
    return record.report
