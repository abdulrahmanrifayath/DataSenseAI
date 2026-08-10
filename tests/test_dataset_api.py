"""Integration tests for dataset management and validation API endpoints."""

import io
import pytest
from fastapi.testclient import TestClient
from datasense.api.main import app

client = TestClient(app)


def test_upload_dataset_endpoint():
    """Test POST /api/v1/datasets/upload with CSV file upload."""
    csv_content = b"user_id,amount,category\n1,150.50,Retail\n2,200.00,Tech\n3,50.25,Retail\n"
    files = {"file": ("test_upload.csv", io.BytesIO(csv_content), "text/csv")}

    response = client.post("/api/v1/datasets/upload", files=files, data={"dataset_name": "Test Upload Dataset"})
    assert response.status_code == 201

    data = response.json()
    assert data["name"] == "Test Upload Dataset"
    assert data["row_count"] == 3
    assert data["column_count"] == 3
    assert "quality_score" in data
    assert "validation_report" in data
    assert data["validation_report"]["row_count"] == 3


def test_list_and_get_dataset_endpoints():
    """Test GET /api/v1/datasets/ and GET /api/v1/datasets/{id} endpoints."""
    # First upload a dataset
    csv_content = b"col1,col2\n10,20\n30,40\n"
    files = {"file": ("test_list.csv", io.BytesIO(csv_content), "text/csv")}
    upload_res = client.post("/api/v1/datasets/upload", files=files)
    dataset_id = upload_res.json()["id"]

    # Test list endpoint
    list_res = client.get("/api/v1/datasets/")
    assert list_res.status_code == 200
    datasets = list_res.json()
    assert len(datasets) >= 1
    assert any(d["id"] == dataset_id for d in datasets)

    # Test get by ID endpoint
    get_res = client.get(f"/api/v1/datasets/{dataset_id}")
    assert get_res.status_code == 200
    dataset = get_res.json()
    assert dataset["id"] == dataset_id
    assert dataset["row_count"] == 2

    # Test validation report endpoint
    val_res = client.get(f"/api/v1/datasets/{dataset_id}/validation")
    assert val_res.status_code == 200
    val_report = val_res.json()
    assert val_report["row_count"] == 2
    assert val_report["column_count"] == 2


def test_get_nonexistent_dataset_404():
    """Test 404 response for invalid dataset ID."""
    response = client.get("/api/v1/datasets/999999")
    assert response.status_code == 404
