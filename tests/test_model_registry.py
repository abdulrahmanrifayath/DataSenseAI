"""Unit tests for ModelRegistry abstraction and persistence functionality."""

import pytest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from datasense.ml_models.registry import LocalModelRegistry, MLflowModelRegistry
from datasense.data_processing.preprocessor import DataPreprocessor


def test_local_model_registry_crud(tmp_path):
    """Test saving, loading, listing, and deleting models in LocalModelRegistry."""
    registry = LocalModelRegistry(storage_dir=str(tmp_path / "registry"))

    model_id = "test_model_001"
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit([[1, 2], [3, 4]], [0, 1])

    preprocessor = DataPreprocessor()
    df_sample = pd.DataFrame({"f1": [1, 3], "f2": [2, 4]})
    preprocessor.fit_transform(df_sample)

    artifact = {
        "estimator": dummy_clf,
        "preprocessor": preprocessor,
        "feature_names": ["f1", "f2"],
        "task_type": "classification",
        "target_column": "target",
    }

    metadata = {
        "model_id": model_id,
        "run_id": "run_123",
        "model_name": "Dummy Classifier",
        "model_type": "dummy",
        "task_type": "classification",
        "target_column": "target",
        "feature_columns": ["f1", "f2"],
        "metrics": {"accuracy": 1.0},
        "training_time_seconds": 0.01,
        "is_best": True,
    }

    # 1. Save model
    saved_path = registry.save_model(model_id, artifact, metadata)
    assert saved_path is not None

    # 2. Get metadata
    retrieved_meta = registry.get_model_metadata(model_id)
    assert retrieved_meta is not None
    assert retrieved_meta["model_name"] == "Dummy Classifier"
    assert retrieved_meta["is_best"] is True

    # 3. List models
    models_list = registry.list_models()
    model_ids = [m["model_id"] for m in models_list]
    assert model_id in model_ids


    # 4. Load model
    loaded_artifact = registry.load_model(model_id)
    assert loaded_artifact is not None
    assert hasattr(loaded_artifact["estimator"], "predict")

    # 5. Delete model
    deleted = registry.delete_model(model_id)
    assert deleted is True
    assert registry.get_model_metadata(model_id) is None


def test_mlflow_registry_wrapper_fallback(tmp_path):
    """Test MLflowModelRegistry delegation to LocalModelRegistry fallback."""
    registry = MLflowModelRegistry()
    registry._local_fallback = LocalModelRegistry(storage_dir=str(tmp_path / "mlflow_reg"))

    model_id = "test_mlflow_001"
    dummy_clf = DummyClassifier()
    dummy_clf.fit([[1], [2]], [0, 1])

    artifact = {"estimator": dummy_clf, "task_type": "classification"}
    metadata = {"model_id": model_id, "model_name": "MLflow Wrapped Model"}

    registry.save_model(model_id, artifact, metadata)
    loaded = registry.load_model(model_id)
    assert loaded is not None
