"""Unit tests for database health check and ORM models."""

from datasense.database.connection import check_db_health
from datasense.database.models import SystemAuditLog, DatasetMetadata


def test_db_health_check_return_structure():
    """Verify check_db_health returns expected tuple structure."""
    is_healthy, message, details = check_db_health()
    assert isinstance(is_healthy, bool)
    assert isinstance(message, str)
    assert isinstance(details, dict)
    assert "status" in details


def test_models_tablename():
    """Verify SQLAlchemy ORM model table names."""
    assert SystemAuditLog.__tablename__ == "system_audit_logs"
    assert DatasetMetadata.__tablename__ == "dataset_metadata"
