"""Integration tests for FastAPI health and root endpoints."""

from fastapi.testclient import TestClient
from datasense.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test GET / root metadata endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "DataSense AI"
    assert data["status"] == "online"
    assert "version" in data


def test_health_endpoint():
    """Test GET /health system status endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["app_name"] == "DataSense AI"
    assert data["status"] in ["healthy", "degraded", "unhealthy"]
    assert "database" in data
    assert "timestamp" in data
