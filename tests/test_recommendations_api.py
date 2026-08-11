"""Integration tests for Business Recommendations FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from datasense.api.main import app

client = TestClient(app)


@pytest.fixture
def recommendation_payload():
    """Sample payload for recommendation API testing."""
    return {
        "data": [{"a": i, "b": i * 2} for i in range(20)],
        "eda_report": {"outliers_detected": 5},
        "forecast_report": {
            "best_model_name": "Baseline Exponential Smoothing",
            "results": [{
                "is_best": True,
                "future_forecast": [
                    {"predicted_value": 500.0},
                    {"predicted_value": 350.0},
                ]
            }]
        },
        "anomaly_report": {"affected_rows_count": 3, "anomaly_percentage": 2.5, "max_severity": "High"},
    }


def test_generate_recommendations_endpoint(recommendation_payload):
    """Test POST /api/v1/recommendations/generate endpoint."""
    response = client.post("/api/v1/recommendations/generate", json=recommendation_payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert "run_id" in data
    report = data["report"]
    assert report["total_recommendations"] >= 3
    assert len(report["items"]) >= 3


def test_get_recommendation_run_endpoint(recommendation_payload):
    """Test GET /api/v1/recommendations/runs/{run_id} endpoint."""
    res_gen = client.post("/api/v1/recommendations/generate", json=recommendation_payload)
    assert res_gen.status_code == 200
    run_id = res_gen.json()["run_id"]

    res_get = client.get(f"/api/v1/recommendations/runs/{run_id}")
    assert res_get.status_code == 200
    report_data = res_get.json()
    assert report_data["run_id"] == run_id
