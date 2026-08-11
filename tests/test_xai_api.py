"""Integration tests for Explainable AI (SHAP) FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from datasense.api.main import app

client = TestClient(app)


@pytest.fixture
def xai_payload():
    """Sample dataset payload for XAI API testing."""
    records = []
    for i in range(40):
        records.append({
            "age": float(20 + i),
            "income": float(30000 + i * 1000),
            "score": float(500 + i * 5),
            "target": 1 if i > 20 else 0,
        })

    return {
        "data": records,
        "target_column": "target",
        "instance_indices": [0, 2],
    }


def test_explain_model_endpoint(xai_payload):
    """Test POST /api/v1/xai/explain endpoint."""
    response = client.post("/api/v1/xai/explain", json=xai_payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert "run_id" in data
    report = data["report"]
    assert len(report["global_importance"]) > 0
    assert len(report["sample_local_explanations"]) == 2


def test_get_xai_run_endpoint(xai_payload):
    """Test GET /api/v1/xai/runs/{run_id} endpoint."""
    res_explain = client.post("/api/v1/xai/explain", json=xai_payload)
    assert res_explain.status_code == 200
    run_id = res_explain.json()["run_id"]

    res_get = client.get(f"/api/v1/xai/runs/{run_id}")
    assert res_get.status_code == 200
    report_data = res_get.json()
    assert report_data["run_id"] == run_id
