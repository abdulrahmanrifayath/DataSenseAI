"""Unit tests for DataProfiler and DataValidator engine."""

import pytest
import pandas as pd
import numpy as np
from datasense.data_processing.validator import DataProfiler, DataValidator
from datasense.data_processing.schemas import ValidationReport


def test_validator_basic_dataframe():
    """Verify validator correctly computes basic metrics on a clean dataframe."""
    df = pd.DataFrame(
        {
            "customer_id": [101, 102, 103, 104, 105],
            "age": [25, 30, 35, 40, 45],
            "signup_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
            "city": ["NY", "LA", "NY", "SF", "LA"],
        }
    )

    validator = DataValidator(df)
    report: ValidationReport = validator.validate()

    assert report.row_count == 5
    assert report.column_count == 4
    assert report.duplicate_rows_count == 0
    assert report.total_missing_cells == 0
    assert report.quality_score == 100.0
    assert "customer_id" in report.potential_id_columns


def test_validator_missing_values_and_duplicates():
    """Verify validator flags missing values, duplicate rows, and constant columns."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 2, 4, 5],
            "value": [10.0, None, None, 40.0, 50.0],
            "constant_col": ["fixed", "fixed", "fixed", "fixed", "fixed"],
        }
    )

    validator = DataValidator(df)
    report: ValidationReport = validator.validate()

    assert report.row_count == 5
    assert report.duplicate_rows_count == 1  # Row 1 and 2 are identical
    assert report.total_missing_cells == 2
    assert "constant_col" in report.constant_columns

    warning_codes = [w.code for w in report.warnings]
    assert "DUPLICATE_ROWS" in warning_codes
    assert "CONSTANT_COLUMN" in warning_codes
    assert report.quality_score < 100.0


def test_validator_empty_dataframe():
    """Verify validator handles empty dataframes gracefully without raising exceptions."""
    df = pd.DataFrame()

    validator = DataValidator(df)
    report: ValidationReport = validator.validate()

    assert report.row_count == 0
    assert report.column_count == 0
    assert report.quality_score == 0.0
    assert any(w.code == "EMPTY_DATASET" for w in report.warnings)


def test_validator_high_cardinality():
    """Verify validator detects high cardinality categorical columns."""
    unique_cities = [f"City_{i}" for i in range(60)]
    df = pd.DataFrame(
        {
            "user_index": list(range(60)),
            "location": unique_cities,
        }
    )

    validator = DataValidator(df)
    report: ValidationReport = validator.validate()

    warning_codes = [w.code for w in report.warnings if w.column == "location"]
    assert "HIGH_CARDINALITY" in warning_codes
