"""Dataset Management and Ingestion/Validation API Router."""

import os
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from datasense.database.connection import get_db_session
from datasense.database.models import DatasetMetadata, SystemAuditLog
from datasense.data_processing.ingestion import DataIngestionService
from datasense.data_processing.validator import DataValidator
from datasense.data_processing.schemas import ValidationReport, DatasetMetadataResponse
from datasense.utilities.logger import get_logger

logger = get_logger("api.datasets")

router = APIRouter(prefix="/api/v1/datasets", tags=["Dataset Ingestion & Validation"])


class DatabaseLoadRequest(BaseModel):
    name: str = Field(..., description="Target dataset name")
    connection_url: str = Field(..., description="SQLAlchemy database connection string")
    query_or_table: str = Field(..., description="SQL SELECT query or table name")


@router.post(
    "/upload",
    response_model=DatasetMetadataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Validate Dataset",
    description="Uploads a CSV or Excel dataset file, performs validation and data profiling, and persists dataset metadata.",
)
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: Optional[str] = Form(None),
    db: Session = Depends(get_db_session),
) -> DatasetMetadataResponse:
    """Handles CSV and Excel file uploads, runs automated data validation, and stores metadata."""
    filename = file.filename or "uploaded_dataset.csv"
    name = dataset_name or os.path.splitext(filename)[0]

    logger.info(f"Received file upload request: {filename} (name: {name})")

    try:
        file_bytes = await file.read()
        file_size = len(file_bytes)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Ingest file into pandas dataframe
        df = DataIngestionService.ingest_file(file_bytes=file_bytes, filename=filename)

        # Run DataValidator
        validator = DataValidator(df)
        report: ValidationReport = validator.validate()

        # Extract head 10 rows for preview
        preview_rows = df.head(10).to_dict(orient="records")
        preview_data = {
            "columns": list(df.columns),
            "rows": preview_rows,
            "total_preview_rows": len(preview_rows),
        }

        # Store in Database
        metadata_record = DatasetMetadata(
            name=name,
            filename=filename,
            file_type="excel" if filename.endswith((".xlsx", ".xls")) else "csv",
            file_size_bytes=file_size,
            row_count=report.row_count,
            column_count=report.column_count,
            quality_score=report.quality_score,
            schema_info={"columns": list(df.columns), "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}},
            validation_report=report.model_dump(),
            preview_data=preview_data,
        )

        db.add(metadata_record)
        db.commit()
        db.refresh(metadata_record)

        # Audit log
        audit = SystemAuditLog(
            event_type="DATASET_UPLOAD",
            description=f"Uploaded dataset '{name}' ({report.row_count} rows, quality score: {report.quality_score})",
            status="SUCCESS",
            meta_data={"dataset_id": metadata_record.id, "filename": filename},
        )
        db.add(audit)
        db.commit()

        return DatasetMetadataResponse(
            id=metadata_record.id,
            name=metadata_record.name,
            filename=metadata_record.filename,
            file_type=metadata_record.file_type,
            file_size_bytes=metadata_record.file_size_bytes,
            row_count=metadata_record.row_count,
            column_count=metadata_record.column_count,
            quality_score=metadata_record.quality_score,
            created_at=metadata_record.created_at.isoformat(),
            preview_data=metadata_record.preview_data,
            validation_report=report,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing dataset upload '{filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process and validate dataset: {str(e)}",
        )


@router.get(
    "/",
    response_model=List[DatasetMetadataResponse],
    summary="List Ingested Datasets",
    description="Retrieves a list of all registered dataset metadata records.",
)
def list_datasets(db: Session = Depends(get_db_session)) -> List[DatasetMetadataResponse]:
    """List all registered dataset metadata records."""
    records = db.query(DatasetMetadata).order_by(DatasetMetadata.created_at.desc()).all()
    results = []
    for r in records:
        val_report = ValidationReport(**r.validation_report) if r.validation_report else None
        results.append(
            DatasetMetadataResponse(
                id=r.id,
                name=r.name,
                filename=r.filename,
                file_type=r.file_type,
                file_size_bytes=r.file_size_bytes,
                row_count=r.row_count,
                column_count=r.column_count,
                quality_score=r.quality_score,
                created_at=r.created_at.isoformat(),
                preview_data=r.preview_data,
                validation_report=val_report,
            )
        )
    return results


@router.get(
    "/{dataset_id}",
    response_model=DatasetMetadataResponse,
    summary="Get Dataset Metadata & Preview",
    description="Retrieves dataset metadata record, schema, and sample preview rows by ID.",
)
def get_dataset(dataset_id: int, db: Session = Depends(get_db_session)) -> DatasetMetadataResponse:
    """Get metadata for a specific dataset ID."""
    record = db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found.")

    val_report = ValidationReport(**record.validation_report) if record.validation_report else None

    return DatasetMetadataResponse(
        id=record.id,
        name=record.name,
        filename=record.filename,
        file_type=record.file_type,
        file_size_bytes=record.file_size_bytes,
        row_count=record.row_count,
        column_count=record.column_count,
        quality_score=record.quality_score,
        created_at=record.created_at.isoformat(),
        preview_data=record.preview_data,
        validation_report=val_report,
    )


@router.get(
    "/{dataset_id}/validation",
    response_model=ValidationReport,
    summary="Get Validation Report",
    description="Retrieves the detailed validation and data profiling report for a specific dataset ID.",
)
def get_dataset_validation(dataset_id: int, db: Session = Depends(get_db_session)) -> ValidationReport:
    """Get validation report for a specific dataset ID."""
    record = db.query(DatasetMetadata).filter(DatasetMetadata.id == dataset_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found.")
    if not record.validation_report:
        raise HTTPException(status_code=404, detail=f"Validation report for dataset ID {dataset_id} is unavailable.")

    return ValidationReport(**record.validation_report)


@router.post(
    "/load-db",
    response_model=DatasetMetadataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Load Dataset from Database Connection",
    description="Extracts data directly from a PostgreSQL or SQL database query, validates it, and registers metadata.",
)
def load_dataset_from_db(
    request: DatabaseLoadRequest,
    db: Session = Depends(get_db_session),
) -> DatasetMetadataResponse:
    """Load dataset directly from external database connection query."""
    try:
        df = DataIngestionService.load_from_db(
            connection_url=request.connection_url,
            query_or_table=request.query_or_table,
        )

        validator = DataValidator(df)
        report: ValidationReport = validator.validate()

        preview_rows = df.head(10).to_dict(orient="records")
        preview_data = {
            "columns": list(df.columns),
            "rows": preview_rows,
            "total_preview_rows": len(preview_rows),
        }

        metadata_record = DatasetMetadata(
            name=request.name,
            filename=f"db_{request.query_or_table[:30]}",
            file_type="database",
            file_size_bytes=0,
            row_count=report.row_count,
            column_count=report.column_count,
            quality_score=report.quality_score,
            schema_info={"columns": list(df.columns), "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}},
            validation_report=report.model_dump(),
            preview_data=preview_data,
        )

        db.add(metadata_record)
        db.commit()
        db.refresh(metadata_record)

        return DatasetMetadataResponse(
            id=metadata_record.id,
            name=metadata_record.name,
            filename=metadata_record.filename,
            file_type=metadata_record.file_type,
            file_size_bytes=metadata_record.file_size_bytes,
            row_count=metadata_record.row_count,
            column_count=metadata_record.column_count,
            quality_score=metadata_record.quality_score,
            created_at=metadata_record.created_at.isoformat(),
            preview_data=metadata_record.preview_data,
            validation_report=report,
        )
    except Exception as e:
        logger.error(f"Error loading dataset from DB query: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Database dataset extraction failed: {str(e)}",
        )
