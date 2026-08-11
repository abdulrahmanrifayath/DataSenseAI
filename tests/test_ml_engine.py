"""Unit tests for the Machine Learning Engine - ModelTrainer & Task Auto-Detection."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from datasense.ml_models.trainer import ModelTrainer, determine_task_type
from datasense.ml_models.schemas import TaskType, TrainingConfig, ModelComparisonReport
from datasense.ml_models.registry import LocalModelRegistry


@pytest.fixture
def sample_classification_df():
    """Generates synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_classes=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = pd.Series(y).map({0: "ClassA", 1: "ClassB"})
    return df


@pytest.fixture
def sample_regression_df():
    """Generates synthetic regression dataset."""
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(5)])
    df["house_price"] = y
    return df


def test_determine_task_type(sample_classification_df, sample_regression_df):
    """Test task auto-detection logic."""
    assert determine_task_type(sample_classification_df, "target") == TaskType.CLASSIFICATION
    assert determine_task_type(sample_regression_df, "house_price") == TaskType.REGRESSION

    # Low unique integers should trigger classification
    df_discrete = pd.DataFrame({"age": np.random.randint(0, 100, 100), "status": np.random.choice([0, 1], 100)})
    assert determine_task_type(df_discrete, "status") == TaskType.CLASSIFICATION


def test_classification_training_pipeline(sample_classification_df, tmp_path):
    """Test complete classification training pipeline across models."""
    registry = LocalModelRegistry(storage_dir=str(tmp_path / "models"))
    trainer = ModelTrainer(registry=registry)

    config = TrainingConfig(
        target_column="target",
        task_type=TaskType.CLASSIFICATION,
        test_size=0.2,
        val_size=0.2,
        cross_validation_folds=3,
        enable_tuning=False,
    )

    report: ModelComparisonReport = trainer.train(sample_classification_df, config)

    assert report.task_type == TaskType.CLASSIFICATION
    assert report.total_samples == 200
    assert len(report.results) >= 3  # Logistic Regression, Random Forest, (XGBoost if present), Gradient Boosting
    assert report.best_model_id is not None
    assert report.best_model_name is not None


    for res in report.results:
        val_m = res.validation_metrics
        assert "accuracy" in val_m
        assert "precision" in val_m
        assert "recall" in val_m
        assert "f1" in val_m
        assert "confusion_matrix" in val_m
        assert res.training_time_seconds >= 0.0


def test_regression_training_pipeline(sample_regression_df, tmp_path):
    """Test complete regression training pipeline across models."""
    registry = LocalModelRegistry(storage_dir=str(tmp_path / "models"))
    trainer = ModelTrainer(registry=registry)

    config = TrainingConfig(
        target_column="house_price",
        task_type=TaskType.REGRESSION,
        test_size=0.2,
        val_size=0.2,
        cross_validation_folds=3,
        enable_tuning=False,
    )

    report: ModelComparisonReport = trainer.train(sample_regression_df, config)

    assert report.task_type == TaskType.REGRESSION
    assert len(report.results) >= 3  # Linear Regression, Random Forest, (XGBoost if present), Gradient Boosting
    assert report.best_model_id is not None


    for res in report.results:
        val_m = res.validation_metrics
        assert "mae" in val_m
        assert "mse" in val_m
        assert "rmse" in val_m
        assert "r2" in val_m
        assert "mape" in val_m


def test_hyperparameter_tuning(sample_classification_df, tmp_path):
    """Test hyperparameter tuning execution."""
    registry = LocalModelRegistry(storage_dir=str(tmp_path / "models"))
    trainer = ModelTrainer(registry=registry)

    config = TrainingConfig(
        target_column="target",
        task_type=TaskType.CLASSIFICATION,
        selected_models=["random_forest"],
        enable_tuning=True,
        n_tuning_trials=3,
        cross_validation_folds=2,
    )

    report = trainer.train(sample_classification_df, config)
    assert len(report.results) == 1
    res = report.results[0]
    assert res.best_params is not None
    assert isinstance(res.best_params, dict)
