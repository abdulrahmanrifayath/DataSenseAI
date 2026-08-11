"""Anomaly Detection REST API Router."""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import pandas as pd

from datasense.database.connection import get_db_session
from datasense.database.models import DatasetMetadata, AnomalyDetectionRecord, SystemAuditLog
from datasense.anomaly_detection.schemas import (
    AnomalyConfig,
    AnomalyReport,
    AnomalyRequest,
    AnomalyResponse,
)
from datasense.anomaly_detection.detector import AnomalyDetector
from datasense.utilities.logger import get_logger

logger = get_logger("api.anomaly")

router = APIRouter(prefix="/api/v1/anomaly", tags=["Anomaly Detection"])


@router.post(
    "/detect",
    response_model=AnomalyResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect Anomalies in Dataset",
    description="Runs Isolation Forest, Z-Score, IQR, or Ensemble anomaly detection and returns scores, severity, affected rows, and feature attribution.",
)
def detect_anomalies_endpoint(
    payload: AnomalyRequest,
    db: Session = Depends(get_db_session),
) -> AnomalyResponse:
    """Triggers anomaly detection pipeline."""
    df: Optional[pd.DataFrame] = None

    if payload.dataset_id is not None:
        dataset_record = db.query(DatasetMetadata).filter(DatasetMetadata.id == payload.dataset_id).first()
        if not dataset_record:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {payload.dataset_id} not found.")

        if not dataset_record.preview_data or "rows" not in dataset_record.preview_data:
            raise HTTPException(
                status_code=400, detail=f"Dataset ID {payload.dataset_id} contains no readable rows for anomaly detection."
            )

        df = pd.DataFrame(dataset_record.preview_data["rows"])

    elif payload.data:
        df = pd.DataFrame(payload.data)

    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="Must provide valid dataset_id or non-empty data list for anomaly detection.")

    config = AnomalyConfig(
        features=payload.features,
        method=payload.method,
        contamination=payload.contamination,
        z_threshold=payload.z_threshold,
        iqr_multiplier=payload.iqr_multiplier,
    )

    try:
        detector = AnomalyDetector()
        report: AnomalyReport = detector.detect(df, config, dataset_id=payload.dataset_id)

        # Database Persistence
        db_record = AnomalyDetectionRecord(
            run_id=report.run_id,
            dataset_id=payload.dataset_id,
            method=report.method,
            affected_rows_count=report.affected_rows_count,
            report=report.model_dump(),
        )
        db.add(db_record)

        # Audit Log
        audit = SystemAuditLog(
            event_type="ANOMALY_DETECTION_EXECUTED",
            description=f"Ran anomaly detection ({report.method}) flagging {report.affected_rows_count} anomalous rows (run_id: {report.run_id})",
            status="SUCCESS",
            meta_data={
                "run_id": report.run_id,
                "dataset_id": payload.dataset_id,
                "affected_count": report.affected_rows_count,
                "max_severity": report.max_severity.value if hasattr(report.max_severity, "value") else str(report.max_severity),
            },
        )
        db.add(audit)
        db.commit()

        return AnomalyResponse(run_id=report.run_id, report=report)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        logger.error(f"Anomaly detection failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Anomaly detection execution error: {str(exc)}")


@router.get(
    "/runs/{run_id}",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get Anomaly Detection Run Report",
    description="Retrieves report for a past anomaly detection run ID.",
)
def get_anomaly_run_endpoint(
    run_id: str,
    db: Session = Depends(get_db_session),
) -> Dict[str, Any]:
    """Retrieve anomaly detection run report."""
    record = db.query(AnomalyDetectionRecord).filter(AnomalyDetectionRecord.run_id == run_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Anomaly detection run with ID '{run_id}' not found.")
    return record.report
