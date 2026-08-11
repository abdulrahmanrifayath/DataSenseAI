"""Integration tests for Anomaly Detection FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from datasense.api.main import app

client = TestClient(app)


@pytest.fixture
def anomaly_payload():
    """Sample dataset payload for anomaly detection API testing."""
    records = []
    for i in range(50):
        records.append({"feat_x": float(i), "feat_y": float(i * 2)})

    # Outlier row
    records.append({"feat_x": 5000.0, "feat_y": -9000.0})

    return {
        "data": records,
        "features": ["feat_x", "feat_y"],
        "method": "ensemble",
        "contamination": 0.05,
    }


def test_detect_anomalies_endpoint(anomaly_payload):
    """Test POST /api/v1/anomaly/detect endpoint."""
    response = client.post("/api/v1/anomaly/detect", json=anomaly_payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert "run_id" in data
    report = data["report"]
    assert report["total_rows"] == 51
    assert report["affected_rows_count"] >= 1
    assert len(report["anomalous_records"]) >= 1


def test_get_anomaly_run_endpoint(anomaly_payload):
    """Test GET /api/v1/anomaly/runs/{run_id} endpoint."""
    res_train = client.post("/api/v1/anomaly/detect", json=anomaly_payload)
    assert res_train.status_code == 200
    run_id = res_train.json()["run_id"]

    res_get = client.get(f"/api/v1/anomaly/runs/{run_id}")
    assert res_get.status_code == 200
    report_data = res_get.json()
    assert report_data["run_id"] == run_id
