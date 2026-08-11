"""Unit tests for Dataset Preprocessing API router and endpoints."""

import os
import pytest
from fastapi.testclient import TestClient

from datasense.api.main import create_app
from datasense.database.connection import init_db_tables, SessionLocal
from datasense.database.models import DatasetMetadata, PreprocessingRecord

app = create_app()


@pytest.fixture(scope="module", autouse=True)
def setup_test_database():
    init_db_tables()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def seed_dataset():
    db = SessionLocal()
    preview_data = {
        "columns": ["id", "age", "salary", "category"],
        "rows": [
            {"id": 1, "age": 25.0, "salary": 50000.0, "category": "A"},
            {"id": 2, "age": 30.0, "salary": 60000.0, "category": "B"},
            {"id": 3, "age": None, "salary": 70000.0, "category": "A"},
            {"id": 4, "age": 40.0, "salary": None, "category": None},
            {"id": 1, "age": 25.0, "salary": 50000.0, "category": "A"},  # duplicate of row 1
        ],
    }


    record = DatasetMetadata(
        name="Test Preprocessing Dataset",
        filename="test.csv",
        file_type="csv",
        file_size_bytes=1024,
        row_count=5,
        column_count=4,
        quality_score=90.0,
        preview_data=preview_data,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    dataset_id = record.id
    db.close()
    return dataset_id


def test_preprocess_dataset_endpoint(client, seed_dataset):
    payload = {
        "config": {
            "target_column": None,
            "identifier_columns": ["id"],
            "numerical_impute_strategy": "median",
            "categorical_impute_strategy": "most_frequent",
            "remove_duplicates": True,
            "numerical_scaling": "standard",
            "categorical_encoding": "onehot",
        }
    }

    response = client.post(f"/api/v1/preprocessing/datasets/{seed_dataset}/preprocess", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["dataset_id"] == seed_dataset
    assert "report" in data
    assert data["report"]["duplicates_removed"] == 1
    assert data["report"]["missing_values_fixed"] >= 2
    assert "preview_data" in data


def test_get_latest_preprocessing_endpoint(client, seed_dataset):
    # Preprocess first
    client.post(f"/api/v1/preprocessing/datasets/{seed_dataset}/preprocess", json={})

    response = client.get(f"/api/v1/preprocessing/datasets/{seed_dataset}/preprocessing")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["dataset_id"] == seed_dataset


def test_get_preprocessing_by_id_endpoint(client, seed_dataset):
    # Preprocess first
    client.post(f"/api/v1/preprocessing/datasets/{seed_dataset}/preprocess", json={})

    resp1 = client.get(f"/api/v1/preprocessing/datasets/{seed_dataset}/preprocessing")
    assert resp1.status_code == 200, resp1.text
    prep_id = resp1.json()["id"]

    resp2 = client.get(f"/api/v1/preprocessing/{prep_id}")
    assert resp2.status_code == 200
    assert resp2.json()["id"] == prep_id


def test_preprocess_nonexistent_dataset_404(client):
    response = client.post("/api/v1/preprocessing/datasets/999999/preprocess", json={})
    assert response.status_code == 404
