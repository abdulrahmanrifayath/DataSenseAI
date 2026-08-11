"""Time-Series Forecasting REST API Router."""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import pandas as pd

from datasense.database.connection import get_db_session
from datasense.database.models import DatasetMetadata, ForecastingRecord, SystemAuditLog
from datasense.forecasting.schemas import (
    ForecastingConfig,
    ForecastingReport,
    ForecastRequest,
    ForecastResponse,
)
from datasense.forecasting.engine import ForecastingEngine
from datasense.utilities.logger import get_logger

logger = get_logger("api.forecasting")

router = APIRouter(prefix="/api/v1/forecasting", tags=["Time-Series Forecasting"])


@router.post(
    "/predict",
    response_model=ForecastResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Time-Series Forecast",
    description="Runs automated time-series preprocessing, frequency grid resampling, temporal feature engineering, model training, evaluation, and future forecasting.",
)
def predict_forecast_endpoint(
    payload: ForecastRequest,
    db: Session = Depends(get_db_session),
) -> ForecastResponse:
    """Triggers time-series forecasting pipeline."""
    df: Optional[pd.DataFrame] = None

    if payload.dataset_id is not None:
        dataset_record = db.query(DatasetMetadata).filter(DatasetMetadata.id == payload.dataset_id).first()
        if not dataset_record:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {payload.dataset_id} not found.")

        if not dataset_record.preview_data or "rows" not in dataset_record.preview_data:
            raise HTTPException(
                status_code=400, detail=f"Dataset ID {payload.dataset_id} contains no readable rows for forecasting."
            )

        df = pd.DataFrame(dataset_record.preview_data["rows"])

    elif payload.data:
        df = pd.DataFrame(payload.data)

    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="Must provide valid dataset_id or non-empty data list for forecasting.")

    config = ForecastingConfig(
        date_column=payload.date_column,
        target_column=payload.target_column,
        forecast_horizon=payload.forecast_horizon,
        frequency=payload.frequency,
        selected_models=payload.selected_models,
    )

    try:
        engine = ForecastingEngine()
        report: ForecastingReport = engine.run_forecasting(df, config, dataset_id=payload.dataset_id)

        # Database Persistence
        db_record = ForecastingRecord(
            run_id=report.run_id,
            dataset_id=payload.dataset_id,
            date_column=payload.date_column,
            target_column=payload.target_column,
            forecast_horizon=payload.forecast_horizon,
            report=report.model_dump(),
        )
        db.add(db_record)

        # Audit Log
        audit = SystemAuditLog(
            event_type="FORECASTING_EXECUTED",
            description=f"Generated forecast for target '{payload.target_column}' with horizon {payload.forecast_horizon} (run_id: {report.run_id})",
            status="SUCCESS",
            meta_data={
                "run_id": report.run_id,
                "dataset_id": payload.dataset_id,
                "best_model_name": report.best_model_name,
            },
        )
        db.add(audit)
        db.commit()

        return ForecastResponse(run_id=report.run_id, report=report)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        logger.error(f"Time-series forecasting failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Forecasting execution error: {str(exc)}")


@router.get(
    "/runs/{run_id}",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get Forecasting Run Report",
    description="Retrieves report for a past forecasting run ID.",
)
def get_forecasting_run_endpoint(
    run_id: str,
    db: Session = Depends(get_db_session),
) -> Dict[str, Any]:
    """Retrieve forecasting run report."""
    record = db.query(ForecastingRecord).filter(ForecastingRecord.run_id == run_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Forecasting run with ID '{run_id}' not found.")
    return record.report
