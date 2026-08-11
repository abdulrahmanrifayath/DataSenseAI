"""SQLAlchemy database ORM models for metadata, validation reports, and audit logging."""

import datetime
from typing import Optional, Dict, Any
from sqlalchemy import String, DateTime, Text, Integer, Float, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from datasense.database.connection import Base


class SystemAuditLog(Base):
    """Audit log entity for tracking system activity and API invocations."""

    __tablename__ = "system_audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="SUCCESS")
    meta_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False
    )


class DatasetMetadata(Base):
    """Metadata and validation report entity for registered datasets."""

    __tablename__ = "dataset_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False, default="csv")
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)
    schema_info: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    validation_report: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    preview_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False
    )


class PreprocessingRecord(Base):
    """Stored dataset preprocessing run metadata, report, and config."""

    __tablename__ = "preprocessing_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(Integer, nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    report: Mapped[dict] = mapped_column(JSON, nullable=False)
    processed_preview: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False
    )


class EDARecord(Base):
    """Stored dataset EDA run report and metadata."""

    __tablename__ = "eda_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(Integer, nullable=False)
    target_column: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    report: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False
    )


class MLExperimentRecord(Base):
    """Stored Machine Learning training run experiment report."""

    __tablename__ = "ml_experiment_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    dataset_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    best_model_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    best_model_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    report: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False
    )


class MLModelRecord(Base):
    """Trained machine learning model metadata entity for the model registry."""

    __tablename__ = "ml_model_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    dataset_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_type: Mapped[str] = mapped_column(String(100), nullable=False)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_columns: Mapped[dict] = mapped_column(JSON, nullable=False)
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    training_time_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    model_path: Mapped[str] = mapped_column(String(500), nullable=False)
    is_best: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False
    )


class ForecastingRecord(Base):
    """Stored Time-Series Forecasting run metadata and report."""

    __tablename__ = "forecasting_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    dataset_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    date_column: Mapped[str] = mapped_column(String(255), nullable=False)
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    forecast_horizon: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    report: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False
    )


class AnomalyDetectionRecord(Base):
    """Stored Anomaly Detection run metadata and report."""

    __tablename__ = "anomaly_detection_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    dataset_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    method: Mapped[str] = mapped_column(String(100), nullable=False)
    affected_rows_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    report: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False
    )


class BIRecord(Base):
    """Stored Business Intelligence and Customer Analytics run metadata and report."""

    __tablename__ = "bi_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    dataset_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_customers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_revenue: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    report: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False
    )





