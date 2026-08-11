"""Unit tests for DataPreprocessor engine and custom scikit-learn transformers."""

import numpy as np
import pandas as pd
import pytest

from datasense.data_processing.preprocessor import DataPreprocessor
from datasense.data_processing.schemas import PreprocessingConfig, PreprocessingReport


@pytest.fixture
def sample_raw_dataframe():
    """Generates a representative test dataset with missing values, duplicates, outliers, date strings, constant columns, and correlated features."""
    np.random.seed(42)
    n = 100

    dates = pd.date_range("2025-01-01", periods=n, freq="D").astype(str)
    
    # Feature 1: normal numerical with missing values
    f1 = np.random.normal(50, 10, size=n)
    f1[::10] = np.nan

    # Feature 2: numerical with extreme outliers
    f2 = np.random.uniform(10, 20, size=n)
    f2[5] = 999.0
    f2[15] = -999.0

    # Feature 3: highly correlated with Feature 1 (r ~ 0.99)
    f3 = f1 * 2.0 + np.random.normal(0, 0.1, size=n)

    # Feature 4: constant column
    f4 = ["A"] * n

    # Feature 5: near-constant column (99% 'X')
    f5 = ["X"] * n
    f5[0] = "Y"

    # Feature 6: categorical with missing values
    f6 = ["cat", "dog", "bird", np.nan] * 25

    # Feature 7: identifier column
    user_ids = [f"USR_{i:04d}" for i in range(n)]

    # Target column
    target = np.random.choice([0, 1], size=n)

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "created_at": dates,
            "feature_normal": f1,
            "feature_outlier": f2,
            "feature_corr": f3,
            "const_col": f4,
            "near_const_col": f5,
            "category_col": f6,
            "target": target,
        }
    )

    # Append 3 duplicate rows
    duplicates = df.iloc[:3].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    return df


def test_duplicate_removal(sample_raw_dataframe):
    config = PreprocessingConfig(remove_duplicates=True)
    preprocessor = DataPreprocessor(config=config)
    transformed_df = preprocessor.fit_transform(sample_raw_dataframe)

    report = preprocessor.get_report()
    assert report.duplicates_removed == 3
    assert report.final_shape[0] == 100


def test_missing_values_imputation(sample_raw_dataframe):
    config = PreprocessingConfig(
        numerical_impute_strategy="median",
        categorical_impute_strategy="most_frequent",
        drop_constant=False,
        drop_near_constant=False,
        drop_high_correlation=False,
    )
    preprocessor = DataPreprocessor(config=config)
    transformed_df = preprocessor.fit_transform(sample_raw_dataframe)

    report = preprocessor.get_report()
    assert report.missing_values_fixed > 0
    # Transformed feature matrix should contain 0 null values
    assert transformed_df.isnull().sum().sum() == 0


def test_outlier_detection_and_clipping(sample_raw_dataframe):
    config = PreprocessingConfig(
        outlier_method="iqr",
        outlier_threshold=1.5,
        outlier_action="clip",
        drop_constant=False,
        drop_near_constant=False,
        drop_high_correlation=False,
    )
    preprocessor = DataPreprocessor(config=config)
    transformed_df = preprocessor.fit_transform(sample_raw_dataframe)

    report = preprocessor.get_report()
    assert report.outliers_detected >= 2


def test_constant_and_near_constant_filtering(sample_raw_dataframe):
    config = PreprocessingConfig(
        drop_constant=True,
        drop_near_constant=True,
        near_constant_threshold=0.98,
        drop_high_correlation=False,
    )
    preprocessor = DataPreprocessor(config=config)
    transformed_df = preprocessor.fit_transform(sample_raw_dataframe)

    report = preprocessor.get_report()
    assert "const_col" in report.columns_removed
    assert "near_const_col" in report.columns_removed
    assert "Constant column" in report.columns_removed["const_col"]
    assert "Near-constant column" in report.columns_removed["near_const_col"]


def test_high_correlation_filtering(sample_raw_dataframe):
    config = PreprocessingConfig(
        drop_high_correlation=True,
        correlation_threshold=0.95,
    )
    preprocessor = DataPreprocessor(config=config)
    transformed_df = preprocessor.fit_transform(sample_raw_dataframe)

    report = preprocessor.get_report()
    # Either feature_normal or feature_corr should be dropped due to high correlation
    assert "feature_corr" in report.columns_removed or "feature_normal" in report.columns_removed


def test_datetime_feature_extraction(sample_raw_dataframe):
    config = PreprocessingConfig(
        convert_datetimes=True,
        datetime_extract_features=True,
    )
    preprocessor = DataPreprocessor(config=config)
    transformed_df = preprocessor.fit_transform(sample_raw_dataframe)

    report = preprocessor.get_report()
    assert "created_at" in report.columns_removed
    assert "created_at_year" in transformed_df.columns or any("created_at" in c for c in transformed_df.columns)


def test_pipeline_transform_reuse(sample_raw_dataframe):
    config = PreprocessingConfig(
        numerical_scaling="standard",
        categorical_encoding="onehot",
    )
    preprocessor = DataPreprocessor(config=config)
    train_transformed = preprocessor.fit_transform(sample_raw_dataframe.iloc[:80])

    # Transform new holdout data without refitting
    test_transformed = preprocessor.transform(sample_raw_dataframe.iloc[80:])
    assert list(train_transformed.columns) == list(test_transformed.columns)
    assert test_transformed.shape[1] == train_transformed.shape[1]
