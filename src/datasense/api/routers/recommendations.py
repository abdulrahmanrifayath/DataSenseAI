"""Business Recommendations REST API Router."""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import pandas as pd

from datasense.database.connection import get_db_session
from datasense.database.models import DatasetMetadata, BusinessRecommendationRecord, SystemAuditLog
from datasense.recommendations.schemas import (
    RecommendationReport,
    RecommendationRequest,
    RecommendationResponse,
)
from datasense.recommendations.engine import BusinessRecommendationEngine
from datasense.utilities.logger import get_logger

logger = get_logger("api.recommendations")

router = APIRouter(prefix="/api/v1/recommendations", tags=["Business Recommendations"])


@router.post(
    "/generate",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Data-Grounded Business Recommendations",
    description="Synthesizes findings across EDA, ML Models, Time-Series Forecasts, Anomaly Detection, BI Analytics, and SHAP to generate actionable recommendations.",
)
def generate_recommendations_endpoint(
    payload: RecommendationRequest,
    db: Session = Depends(get_db_session),
) -> RecommendationResponse:
    """Triggers Business Recommendation synthesis engine."""
    df: Optional[pd.DataFrame] = None

    if payload.dataset_id is not None:
        dataset_record = db.query(DatasetMetadata).filter(DatasetMetadata.id == payload.dataset_id).first()
        if not dataset_record:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {payload.dataset_id} not found.")

        if dataset_record.preview_data and "rows" in dataset_record.preview_data:
            df = pd.DataFrame(dataset_record.preview_data["rows"])

    elif payload.data:
        df = pd.DataFrame(payload.data)

    try:
        engine = BusinessRecommendationEngine()
        report: RecommendationReport = engine.generate_recommendations(
            df=df,
            eda_report=payload.eda_report,
            ml_report=payload.ml_report,
            forecast_report=payload.forecast_report,
            anomaly_report=payload.anomaly_report,
            bi_report=payload.bi_report,
            xai_report=payload.xai_report,
            dataset_id=payload.dataset_id,
        )

        # Database Persistence
        db_record = BusinessRecommendationRecord(
            run_id=report.run_id,
            dataset_id=payload.dataset_id,
            total_recommendations=report.total_recommendations,
            critical_count=report.critical_count,
            report=report.model_dump(),
        )
        db.add(db_record)

        # Audit Log
        audit = SystemAuditLog(
            event_type="RECOMMENDATIONS_GENERATED",
            description=f"Generated {report.total_recommendations} business recommendations ({report.critical_count} Critical, {report.high_count} High) (run_id: {report.run_id})",
            status="SUCCESS",
            meta_data={
                "run_id": report.run_id,
                "dataset_id": payload.dataset_id,
                "total_recommendations": report.total_recommendations,
                "critical_count": report.critical_count,
            },
        )
        db.add(audit)
        db.commit()

        return RecommendationResponse(run_id=report.run_id, report=report)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        logger.error(f"Recommendation generation failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recommendation generation error: {str(exc)}")


@router.get(
    "/runs/{run_id}",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get Business Recommendation Run Report",
    description="Retrieves report for a past recommendation run ID.",
)
def get_recommendation_run_endpoint(
    run_id: str,
    db: Session = Depends(get_db_session),
) -> Dict[str, Any]:
    """Retrieve recommendation run report."""
    record = db.query(BusinessRecommendationRecord).filter(BusinessRecommendationRecord.run_id == run_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Recommendation run with ID '{run_id}' not found.")
    return record.report
