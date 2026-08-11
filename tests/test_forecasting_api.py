"""Integration tests for Time-Series Forecasting FastAPI endpoints."""

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from datasense.api.main import app

client = TestClient(app)


@pytest.fixture
def forecasting_payload():
    """Sample time series payload for API testing."""
    dates = pd.date_range(start="2025-01-01", periods=30, freq="D").strftime("%Y-%m-%d").tolist()
    data = []
    for idx, d in enumerate(dates):
        data.append({"date": d, "val": 100.0 + idx * 2.0 + (idx % 7)})

    return {
        "data": data,
        "date_column": "date",
        "target_column": "val",
        "forecast_horizon": 7,
        "frequency": "D",
        "selected_models": ["baseline"],
    }


def test_predict_forecast_endpoint(forecasting_payload):
    """Test POST /api/v1/forecasting/predict endpoint."""
    response = client.post("/api/v1/forecasting/predict", json=forecasting_payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert "run_id" in data
    report = data["report"]
    assert report["forecast_horizon"] == 7
    assert len(report["results"]) == 1
    best_res = report["results"][0]
    assert len(best_res["future_forecast"]) == 7


def test_get_forecasting_run_endpoint(forecasting_payload):
    """Test GET /api/v1/forecasting/runs/{run_id} endpoint."""
    res_train = client.post("/api/v1/forecasting/predict", json=forecasting_payload)
    assert res_train.status_code == 200
    run_id = res_train.json()["run_id"]

    res_get = client.get(f"/api/v1/forecasting/runs/{run_id}")
    assert res_get.status_code == 200
    report_data = res_get.json()
    assert report_data["run_id"] == run_id
