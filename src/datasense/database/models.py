"""SQLAlchemy database ORM models for metadata, validation reports, and audit logging."""

import datetime
from typing import Optional, Dict, Any
from sqlalchemy import String, DateTime, Text, Integer, Float, JSON
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
