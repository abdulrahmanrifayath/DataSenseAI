"""Business Intelligence & Customer Analytics REST API Router."""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import pandas as pd

from datasense.database.connection import get_db_session
from datasense.database.models import DatasetMetadata, BIRecord, SystemAuditLog
from datasense.bi.schemas import (
    ColumnMappingConfig,
    BIAnalysisReport,
    BIAnalysisRequest,
    BIAnalysisResponse,
)
from datasense.bi.engine import BIEngine
from datasense.utilities.logger import get_logger

logger = get_logger("api.bi")

router = APIRouter(prefix="/api/v1/bi", tags=["Business Intelligence"])


@router.post(
    "/analyze",
    response_model=BIAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Run Business Intelligence & Customer Analytics Analysis",
    description="Runs automated column mapping resolution, executive KPI calculations, RFM persona segmentation, K-Means/Hierarchical clustering, cluster quality evaluation, churn risk prediction, CLV estimation, and generates insights and Plotly charts.",
)
def analyze_bi_endpoint(
    payload: BIAnalysisRequest,
    db: Session = Depends(get_db_session),
) -> BIAnalysisResponse:
    """Triggers Business Intelligence & Customer Analytics pipeline."""
    df: Optional[pd.DataFrame] = None

    if payload.dataset_id is not None:
        dataset_record = db.query(DatasetMetadata).filter(DatasetMetadata.id == payload.dataset_id).first()
        if not dataset_record:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {payload.dataset_id} not found.")

        if not dataset_record.preview_data or "rows" not in dataset_record.preview_data:
            raise HTTPException(
                status_code=400, detail=f"Dataset ID {payload.dataset_id} contains no readable rows for BI analysis."
            )

        df = pd.DataFrame(dataset_record.preview_data["rows"])

    elif payload.data:
        df = pd.DataFrame(payload.data)

    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="Must provide valid dataset_id or non-empty data list for BI analysis.")

    try:
        engine = BIEngine()
        report: BIAnalysisReport = engine.analyze(
            df=df,
            manual_mapping=payload.column_mapping,
            clustering_algorithm=payload.clustering_algorithm,
            dataset_id=payload.dataset_id,
        )

        # Database Persistence
        db_record = BIRecord(
            run_id=report.run_id,
            dataset_id=payload.dataset_id,
            total_customers=report.business_kpis.total_customers,
            total_revenue=report.business_kpis.total_revenue,
            report=report.model_dump(),
        )
        db.add(db_record)

        # Audit Log
        audit = SystemAuditLog(
            event_type="BI_ANALYSIS_EXECUTED",
            description=f"Ran BI Customer Analytics on dataset {payload.dataset_id or 'direct'} ($ {report.business_kpis.total_revenue:,.2f} sales, {report.business_kpis.total_customers} customers) (run_id: {report.run_id})",
            status="SUCCESS",
            meta_data={
                "run_id": report.run_id,
                "dataset_id": payload.dataset_id,
                "total_revenue": report.business_kpis.total_revenue,
                "total_customers": report.business_kpis.total_customers,
            },
        )
        db.add(audit)
        db.commit()

        return BIAnalysisResponse(run_id=report.run_id, report=report)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        logger.error(f"BI Analysis execution failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"BI Analysis execution error: {str(exc)}")


@router.get(
    "/runs/{run_id}",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get Business Intelligence Run Report",
    description="Retrieves report for a past BI analysis run ID.",
)
def get_bi_run_endpoint(
    run_id: str,
    db: Session = Depends(get_db_session),
) -> Dict[str, Any]:
    """Retrieve BI run report."""
    record = db.query(BIRecord).filter(BIRecord.run_id == run_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"BI analysis run with ID '{run_id}' not found.")
    return record.report
