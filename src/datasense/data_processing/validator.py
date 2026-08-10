"""DataProfiler and DataValidator service for complete dataset validation and quality scoring."""

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from datasense.data_processing.schemas import (
    ColumnProfile,
    QualityWarning,
    ValidationReport,
)
from datasense.utilities.logger import get_logger

logger = get_logger("data_processing.validator")


class DataProfiler:
    """Comprehensive Data Validation and Profiling engine for Pandas DataFrames."""

    def __init__(self, df: pd.DataFrame):
        self.df = df if df is not None else pd.DataFrame()

    def _infer_column_type(self, col: str, series: pd.Series) -> str:
        """Infers high-level business data type for a column."""
        if pd.api.types.is_numeric_dtype(series):
            return "numerical"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        elif pd.api.types.is_bool_dtype(series):
            return "boolean"
        else:
            # Check if object string column can be parsed to datetime
            non_null = series.dropna()
            if not non_null.empty and len(non_null) > 0:
                sample = non_null.iloc[:50]
                try:
                    parsed = pd.to_datetime(sample, errors="coerce", format="ISO8601")
                    if parsed.notnull().sum() / len(sample) > 0.8:
                        return "datetime"
                except Exception:
                    pass
            return "categorical"

    def _check_potential_id(self, col: str, series: pd.Series, inferred_type: str, total_rows: int) -> bool:
        """Determines if a column is a primary ID candidate."""
        if total_rows <= 1:
            return False
        col_lower = col.lower()
        non_null_count = series.count()
        unique_count = series.nunique()

        # Direct name match heuristic (e.g., id, customer_id, user_id, uuid, index)
        is_name_match = any(token in col_lower for token in ["id", "uuid", "guid", "code", "index", "key"])

        if non_null_count == total_rows and unique_count == total_rows:
            if is_name_match or inferred_type in ["categorical", "numerical"]:
                return True

        return False

    def calculate_quality_score(
        self,
        total_cells: int,
        total_missing: int,
        duplicate_percentage: float,
        constant_columns_count: int,
        total_columns: int,
        warnings: List[QualityWarning],
    ) -> float:
        """Calculates a deterministic data quality score from 0.0 to 100.0."""
        if total_cells == 0 or total_columns == 0:
            return 0.0

        score = 100.0

        # Penalty for missing cells (up to 40 points)
        missing_ratio = total_missing / total_cells
        score -= min(40.0, missing_ratio * 100.0 * 1.5)

        # Penalty for duplicate rows (up to 25 points)
        score -= min(25.0, duplicate_percentage * 1.0)

        # Penalty for constant columns (up to 15 points)
        constant_ratio = constant_columns_count / total_columns
        score -= min(15.0, constant_ratio * 50.0)

        # Penalty for critical & warning issues
        critical_count = sum(1 for w in warnings if w.severity == "CRITICAL")
        warning_count = sum(1 for w in warnings if w.severity == "WARNING")
        score -= critical_count * 10.0
        score -= warning_count * 3.0

        return max(0.0, round(score, 1))

    def generate_report(self) -> ValidationReport:
        """Executes full statistical profiling, anomaly check, warning audit, and quality score calculation."""
        logger.info(f"Generating validation report for DataFrame shape {self.df.shape}")

        if self.df.empty:
            return ValidationReport(
                row_count=0,
                column_count=0,
                duplicate_rows_count=0,
                duplicate_rows_percentage=0.0,
                total_missing_cells=0,
                total_cells=0,
                missing_cell_percentage=0.0,
                column_profiles={},
                warnings=[
                    QualityWarning(
                        severity="CRITICAL",
                        code="EMPTY_DATASET",
                        message="Uploaded dataset is completely empty.",
                    )
                ],
                quality_score=0.0,
            )

        row_count, col_count = self.df.shape
        total_cells = row_count * col_count

        # Duplicate row metrics
        duplicate_rows_count = int(self.df.duplicated().sum())
        duplicate_rows_pct = round((duplicate_rows_count / row_count) * 100.0, 2) if row_count > 0 else 0.0

        column_profiles: Dict[str, ColumnProfile] = {}
        warnings: List[QualityWarning] = []

        numerical_cols: List[str] = []
        categorical_cols: List[str] = []
        datetime_cols: List[str] = []
        potential_id_cols: List[str] = []
        constant_cols: List[str] = []

        total_missing_cells = 0

        # Duplicate rows warning
        if duplicate_rows_count > 0:
            warnings.append(
                QualityWarning(
                    severity="WARNING" if duplicate_rows_pct < 10 else "CRITICAL",
                    code="DUPLICATE_ROWS",
                    message=f"Dataset contains {duplicate_rows_count} duplicate rows ({duplicate_rows_pct}% of total dataset).",
                )
            )

        for col in self.df.columns:
            series = self.df[col]
            missing_count = int(series.isnull().sum())
            total_missing_cells += missing_count
            missing_pct = round((missing_count / row_count) * 100.0, 2) if row_count > 0 else 0.0

            unique_count = int(series.nunique(dropna=True))
            unique_ratio = round(unique_count / row_count, 4) if row_count > 0 else 0.0

            is_constant = unique_count <= 1
            if is_constant:
                constant_cols.append(str(col))
                warnings.append(
                    QualityWarning(
                        severity="WARNING",
                        code="CONSTANT_COLUMN",
                        column=str(col),
                        message=f"Column '{col}' is constant with only {unique_count} distinct non-null value.",
                    )
                )

            inferred_type = self._infer_column_type(str(col), series)
            is_id = self._check_potential_id(str(col), series, inferred_type, row_count)

            if is_id:
                potential_id_cols.append(str(col))
                inferred_type = "id"

            if inferred_type == "numerical":
                numerical_cols.append(str(col))
            elif inferred_type == "datetime":
                datetime_cols.append(str(col))
            elif inferred_type in ["categorical", "boolean"]:
                categorical_cols.append(str(col))

            # Missingness warning per column
            if missing_pct >= 20.0:
                warnings.append(
                    QualityWarning(
                        severity="CRITICAL" if missing_pct >= 50.0 else "WARNING",
                        code="HIGH_MISSINGNESS",
                        column=str(col),
                        message=f"Column '{col}' has high missingness ({missing_pct}% missing values).",
                    )
                )

            # High cardinality categorical warning
            if inferred_type == "categorical" and unique_count > 50 and not is_id:
                warnings.append(
                    QualityWarning(
                        severity="INFO",
                        code="HIGH_CARDINALITY",
                        column=str(col),
                        message=f"Categorical column '{col}' has high cardinality ({unique_count} distinct values).",
                    )
                )

            # Sample values
            sample_vals = series.dropna().unique()[:5].tolist()
            # Convert non-serializable objects to string
            sample_vals_clean = [v if isinstance(v, (int, float, str, bool)) else str(v) for v in sample_vals]

            column_profiles[str(col)] = ColumnProfile(
                name=str(col),
                data_type=str(series.dtype),
                inferred_type=inferred_type,
                missing_count=missing_count,
                missing_percentage=missing_pct,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                is_constant=is_constant,
                is_potential_id=is_id,
                sample_values=sample_vals_clean,
            )

        missing_cell_percentage = round((total_missing_cells / total_cells) * 100.0, 2) if total_cells > 0 else 0.0

        quality_score = self.calculate_quality_score(
            total_cells=total_cells,
            total_missing=total_missing_cells,
            duplicate_percentage=duplicate_rows_pct,
            constant_columns_count=len(constant_cols),
            total_columns=col_count,
            warnings=warnings,
        )

        return ValidationReport(
            row_count=row_count,
            column_count=col_count,
            duplicate_rows_count=duplicate_rows_count,
            duplicate_rows_percentage=duplicate_rows_pct,
            total_missing_cells=total_missing_cells,
            total_cells=total_cells,
            missing_cell_percentage=missing_cell_percentage,
            numerical_columns=numerical_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            potential_id_columns=potential_id_cols,
            constant_columns=constant_cols,
            column_profiles=column_profiles,
            warnings=warnings,
            quality_score=quality_score,
        )


class DataValidator:
    """Wrapper entry point for validation profiling."""

    def __init__(self, df: pd.DataFrame):
        self.profiler = DataProfiler(df)

    def validate(self) -> ValidationReport:
        """Executes dataset profiling and validation."""
        return self.profiler.generate_report()
