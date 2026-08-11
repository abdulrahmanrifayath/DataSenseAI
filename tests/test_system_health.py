"""Integration tests for System Health Diagnostics API endpoint."""

import pytest
from fastapi.testclient import TestClient
from datasense.api.main import app

client = TestClient(app)


def test_system_health_endpoint_extended():
    """Test GET /api/v1/health endpoint for CPU, RAM, Disk, and DB count details."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200, response.text

    data = response.json()
    assert "status" in data
    assert "app_name" in data
    assert "version" in data
    assert "environment" in data
    assert "timestamp" in data
    assert "database" in data
    assert "system_metrics" in data

    sys_m = data["system_metrics"]
    assert "cpu_usage_pct" in sys_m
    assert "memory_usage_pct" in sys_m
    assert "disk_usage_pct" in sys_m
    assert isinstance(sys_m["cpu_usage_pct"], (int, float))

    assert "registered_datasets_count" in data
    assert "active_models_count" in data
