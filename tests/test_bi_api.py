"""Integration tests for Business Intelligence FastAPI endpoints."""

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from datasense.api.main import app

client = TestClient(app)


@pytest.fixture
def bi_payload():
    """Sample dataset payload for BI API testing."""
    records = []
    dates = pd.date_range(start="2025-01-01", periods=40, freq="D").strftime("%Y-%m-%d").tolist()
    for i, d in enumerate(dates):
        records.append({
            "customer_id": f"C_{i % 10}",
            "order_date": d,
            "revenue": 100.0 + i * 5.0,
            "profit": 30.0 + i * 1.5,
        })

    return {
        "data": records,
        "clustering_algorithm": "kmeans",
    }


def test_analyze_bi_endpoint(bi_payload):
    """Test POST /api/v1/bi/analyze endpoint."""
    response = client.post("/api/v1/bi/analyze", json=bi_payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert "run_id" in data
    report = data["report"]
    assert report["business_kpis"]["total_revenue"] > 0.0
    assert report["business_kpis"]["total_customers"] == 10
    assert len(report["rfm_segments"]) > 0


def test_get_bi_run_endpoint(bi_payload):
    """Test GET /api/v1/bi/runs/{run_id} endpoint."""
    res_train = client.post("/api/v1/bi/analyze", json=bi_payload)
    assert res_train.status_code == 200
    run_id = res_train.json()["run_id"]

    res_get = client.get(f"/api/v1/bi/runs/{run_id}")
    assert res_get.status_code == 200
    report_data = res_get.json()
    assert report_data["run_id"] == run_id
