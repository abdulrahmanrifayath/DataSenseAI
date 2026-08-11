"""Integration tests for Machine Learning Engine FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from datasense.api.main import app

client = TestClient(app)


@pytest.fixture
def sample_dataset_payload():
    """Sample records for training payload."""
    return {
        "data": [
            {"age": 25, "income": 50000, "bought": 0},
            {"age": 45, "income": 90000, "bought": 1},
            {"age": 35, "income": 60000, "bought": 0},
            {"age": 50, "income": 120000, "bought": 1},
            {"age": 23, "income": 40000, "bought": 0},
            {"age": 60, "income": 150000, "bought": 1},
            {"age": 29, "income": 52000, "bought": 0},
            {"age": 48, "income": 95000, "bought": 1},
            {"age": 31, "income": 61000, "bought": 0},
            {"age": 52, "income": 115000, "bought": 1},
            {"age": 24, "income": 41000, "bought": 0},
            {"age": 58, "income": 140000, "bought": 1},
            {"age": 28, "income": 49000, "bought": 0},
            {"age": 46, "income": 92000, "bought": 1},
            {"age": 33, "income": 63000, "bought": 0},
            {"age": 55, "income": 130000, "bought": 1},
        ],
        "target_column": "bought",
        "task_type": "classification",
        "selected_models": ["logistic_regression"],
        "cross_validation_folds": 2,
    }


def test_ml_train_endpoint(sample_dataset_payload):
    """Test POST /api/v1/ml/train endpoint."""
    response = client.post("/api/v1/ml/train", json=sample_dataset_payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert "run_id" in data
    assert data["target_column"] == "bought"
    assert data["task_type"] == "classification"
    assert "best_model_id" in data
    assert len(data["results"]) == 1


def test_ml_list_models_and_predict(sample_dataset_payload):
    """Test training, listing models, and making predictions via REST API."""
    # 1. Train model
    train_res = client.post("/api/v1/ml/train", json=sample_dataset_payload)
    assert train_res.status_code == 200
    report = train_res.json()
    model_id = report["best_model_id"]

    # 2. List models
    list_res = client.get("/api/v1/ml/models")
    assert list_res.status_code == 200
    models_list = list_res.json()
    assert len(models_list) >= 1

    # 3. Get single model details
    get_res = client.get(f"/api/v1/ml/models/{model_id}")
    assert get_res.status_code == 200
    meta = get_res.json()
    assert meta["model_id"] == model_id

    # 4. Make Prediction
    pred_payload = {
        "model_id": model_id,
        "data": [
            {"age": 30, "income": 55000},
            {"age": 52, "income": 110000},
        ],
    }
    pred_res = client.post("/api/v1/ml/predict", json=pred_payload)
    assert pred_res.status_code == 200
    pred_data = pred_res.json()

    assert pred_data["model_id"] == model_id
    assert len(pred_data["predictions"]) == 2
    assert pred_data["row_count"] == 2
