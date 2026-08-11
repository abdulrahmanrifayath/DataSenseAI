"""Exploratory Data Analysis (EDA) API Router."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import pandas as pd

from datasense.database.connection import get_db_session
from datasense.database.models import DatasetMetadata, EDARecord, SystemAuditLog
from datasense.eda.engine import EDAEngine
from datasense.eda.schemas import EDARequest, EDAResponse, EDAReport
from datasense.utilities.logger import get_logger

logger = get_logger("api.eda")

router = APIRouter(prefix="/api/v1/eda", tags=["Exploratory Data Analysis Engine"])


@router.post(
    "/datasets/{dataset_id}/analyze",
    response_model=EDAResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger Automated Exploratory Data Analysis",
    description="Runs statistical profiling, correlation analysis, outlier detection, distribution analysis, time trends, target relationships, and automatic insight generation.",
)
def analyze_dataset_eda(
    dataset_id: int,
    payload: Optional[EDARequest] = None,
    db: Session = Depends(get_db_session),
) -> EDAResponse:
    """Triggers complete automated EDA on a registered dataset."""
    dataset_record = db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
    if not dataset_record:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found.")

    if not dataset_record.preview_data or "rows" not in dataset_record.preview_data:
        raise HTTPException(status_code=400, detail=f"Dataset ID {dataset_id} contains no readable rows for analysis.")

    target_column = payload.target_column if payload else None

    try:
        df = pd.DataFrame(dataset_record.preview_data["rows"])
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset for analysis is empty.")

        eda_engine = EDAEngine(df, target_column=target_column)
        report: EDAReport = eda_engine.generate_report()

        # Persist record in database
        eda_record = EDARecord(
            dataset_id=dataset_id,
            target_column=target_column,
            report=report.model_dump(),
        )
        db.add(eda_record)
        db.commit()
        db.refresh(eda_record)

        # Audit log
        audit = SystemAuditLog(
            event_type="DATASET_EDA_GENERATED",
            description=f"Generated EDA report for dataset ID {dataset_id} (Target: {target_column or 'None'})",
            status="SUCCESS",
            meta_data={"dataset_id": dataset_id, "eda_id": eda_record.id},
        )
        db.add(audit)
        db.commit()

        return EDAResponse(
            id=eda_record.id,
            dataset_id=eda_record.dataset_id,
            created_at=eda_record.created_at.isoformat(),
            target_column=eda_record.target_column,
            report=report,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing EDA analysis for dataset ID {dataset_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"EDA execution failed: {str(e)}",
        )


@router.get(
    "/datasets/{dataset_id}/report",
    response_model=EDAResponse,
    summary="Get Latest EDA Report for Dataset",
    description="Retrieves the most recent EDA analysis report for a specific dataset ID.",
)
def get_latest_dataset_eda_report(dataset_id: int, db: Session = Depends(get_db_session)) -> EDAResponse:
    """Get latest EDA report for a dataset ID."""
    eda_record = (
        db.query(EDARecord)
        .filter(EDARecord.dataset_id == dataset_id)
        .order_by(EDARecord.created_at.desc())
        .first()
    )
    if not eda_record:
        raise HTTPException(status_code=404, detail=f"No EDA report found for dataset ID {dataset_id}.")

    return EDAResponse(
        id=eda_record.id,
        dataset_id=eda_record.dataset_id,
        created_at=eda_record.created_at.isoformat(),
        target_column=eda_record.target_column,
        report=EDAReport(**eda_record.report),
    )


@router.get(
    "/{report_id}",
    response_model=EDAResponse,
    summary="Get EDA Report by ID",
    description="Retrieves a specific EDA analysis report by record ID.",
)
def get_eda_report_by_id(report_id: int, db: Session = Depends(get_db_session)) -> EDAResponse:
    """Get EDA report by record ID."""
    eda_record = db.query(EDARecord).filter(EDARecord.id == report_id).first()
    if not eda_record:
        raise HTTPException(status_code=404, detail=f"EDA report ID {report_id} not found.")

    return EDAResponse(
        id=eda_record.id,
        dataset_id=eda_record.dataset_id,
        created_at=eda_record.created_at.isoformat(),
        target_column=eda_record.target_column,
        report=EDAReport(**eda_record.report),
    )
