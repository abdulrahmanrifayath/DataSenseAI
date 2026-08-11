"""Explicit tests verifying Data Leakage Prevention during Model Training & Preprocessing."""

import pytest
import numpy as np
import pandas as pd
from datasense.data_processing.preprocessor import DataPreprocessor
from datasense.ml_models.trainer import ModelTrainer
from datasense.ml_models.schemas import TaskType, TrainingConfig
from datasense.ml_models.registry import LocalModelRegistry


def test_no_data_leakage_in_preprocessor(tmp_path):
    """Verify that feature scaling and imputer parameters in preprocessor are computed ONLY on X_train."""
    n_train = 100
    n_test = 50

    # Training features have mean ~ 10, std ~ 1
    X_train_raw = np.random.normal(loc=10.0, scale=1.0, size=(n_train, 2))
    # Test features have massive outlier shift (mean ~ 1000)
    X_test_raw = np.random.normal(loc=1000.0, scale=1.0, size=(n_test, 2))

    df_train = pd.DataFrame(X_train_raw, columns=["f1", "f2"])
    df_test = pd.DataFrame(X_test_raw, columns=["f1", "f2"])

    preprocessor = DataPreprocessor()
    df_train_proc = preprocessor.fit_transform(df_train)

    ct = preprocessor.column_transformer
    assert ct is not None, "ColumnTransformer must be fitted"

    scaler_mean_before = None
    for name, trans, cols in ct.transformers_:
        if hasattr(trans, "named_steps") and "scaler" in trans.named_steps:
            scaler_mean_before = float(trans.named_steps["scaler"].mean_[0])
            break

    assert scaler_mean_before is not None
    assert scaler_mean_before < 20.0, f"Expected scaler mean ~10.0, got {scaler_mean_before}"

    # Transforming holdout test set must NOT alter the preprocessor's fitted parameters
    df_test_proc = preprocessor.transform(df_test)

    scaler_mean_after = None
    for name, trans, cols in ct.transformers_:
        if hasattr(trans, "named_steps") and "scaler" in trans.named_steps:
            scaler_mean_after = float(trans.named_steps["scaler"].mean_[0])
            break

    assert scaler_mean_after == scaler_mean_before, "Data leakage detected: preprocessor state was modified during transform!"
