"""Dataset Preprocessing and Cleaning API Router."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import pandas as pd

from datasense.database.connection import get_db_session
from datasense.database.models import DatasetMetadata, PreprocessingRecord, SystemAuditLog
from datasense.data_processing.preprocessor import DataPreprocessor
from datasense.data_processing.schemas import (
    PreprocessingConfig,
    PreprocessingRequest,
    PreprocessingResponse,
    PreprocessingReport,
)
from datasense.utilities.logger import get_logger

logger = get_logger("api.preprocessing")

router = APIRouter(prefix="/api/v1/preprocessing", tags=["Dataset Preprocessing Engine"])


@router.post(
    "/datasets/{dataset_id}/preprocess",
    response_model=PreprocessingResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger Automated Dataset Preprocessing",
    description="Executes customizable preprocessing pipeline (imputation, scaling, encoding, deduplication, outlier clipping, feature filtering) on dataset.",
)
def preprocess_dataset(
    dataset_id: int,
    payload: Optional[PreprocessingRequest] = None,
    db: Session = Depends(get_db_session),
) -> PreprocessingResponse:
    """Triggers preprocessing on a registered dataset and stores execution metadata."""
    dataset_record = db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
    if not dataset_record:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found.")

    if not dataset_record.preview_data or "rows" not in dataset_record.preview_data:
        raise HTTPException(status_code=400, detail=f"Dataset ID {dataset_id} contains no processable data.")

    config = payload.config if payload and payload.config else PreprocessingConfig()

    try:
        # Load dataset rows into DataFrame
        df = pd.DataFrame(dataset_record.preview_data["rows"])
        if df.empty:
            raise HTTPException(status_code=400, detail="Target dataset is empty.")

        preprocessor = DataPreprocessor(config=config)
        transformed_df = preprocessor.fit_transform(df)
        report: PreprocessingReport = preprocessor.get_report()

        # Preview transformed dataset
        preview_rows = transformed_df.head(10).to_dict(orient="records")
        preview_data = {
            "columns": list(transformed_df.columns),
            "rows": preview_rows,
            "total_preview_rows": len(preview_rows),
        }

        # Store record in database
        prep_record = PreprocessingRecord(
            dataset_id=dataset_id,
            config=config.model_dump(),
            report=report.model_dump(),
            processed_preview=preview_data,
        )
        db.add(prep_record)
        db.commit()
        db.refresh(prep_record)

        # Audit log
        audit = SystemAuditLog(
            event_type="DATASET_PREPROCESSED",
            description=f"Preprocessed dataset ID {dataset_id} ({report.initial_shape} -> {report.final_shape})",
            status="SUCCESS",
            meta_data={"dataset_id": dataset_id, "preprocessing_id": prep_record.id},
        )
        db.add(audit)
        db.commit()

        return PreprocessingResponse(
            id=prep_record.id,
            dataset_id=prep_record.dataset_id,
            created_at=prep_record.created_at.isoformat(),
            config=config,
            report=report,
            preview_data=preview_data,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during preprocessing dataset ID {dataset_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Preprocessing execution failed: {str(e)}",
        )


@router.get(
    "/datasets/{dataset_id}/preprocessing",
    response_model=PreprocessingResponse,
    summary="Get Latest Preprocessing Report for Dataset",
    description="Retrieves the most recent preprocessing run and metadata for a specific dataset ID.",
)
def get_latest_dataset_preprocessing(dataset_id: int, db: Session = Depends(get_db_session)) -> PreprocessingResponse:
    """Get latest preprocessing report for a dataset ID."""
    prep_record = (
        db.query(PreprocessingRecord)
        .filter(PreprocessingRecord.dataset_id == dataset_id)
        .order_by(PreprocessingRecord.created_at.desc())
        .first()
    )
    if not prep_record:
        raise HTTPException(status_code=404, detail=f"No preprocessing history found for dataset ID {dataset_id}.")

    return PreprocessingResponse(
        id=prep_record.id,
        dataset_id=prep_record.dataset_id,
        created_at=prep_record.created_at.isoformat(),
        config=PreprocessingConfig(**prep_record.config),
        report=PreprocessingReport(**prep_record.report),
        preview_data=prep_record.processed_preview,
    )


@router.get(
    "/{preprocessing_id}",
    response_model=PreprocessingResponse,
    summary="Get Preprocessing Report by ID",
    description="Retrieves specific preprocessing run metadata and report by preprocessing record ID.",
)
def get_preprocessing_by_id(preprocessing_id: int, db: Session = Depends(get_db_session)) -> PreprocessingResponse:
    """Get preprocessing metadata by record ID."""
    prep_record = db.query(PreprocessingRecord).filter(PreprocessingRecord.id == preprocessing_id).first()
    if not prep_record:
        raise HTTPException(status_code=404, detail=f"Preprocessing record ID {preprocessing_id} not found.")

    return PreprocessingResponse(
        id=prep_record.id,
        dataset_id=prep_record.dataset_id,
        created_at=prep_record.created_at.isoformat(),
        config=PreprocessingConfig(**prep_record.config),
        report=PreprocessingReport(**prep_record.report),
        preview_data=prep_record.processed_preview,
    )
