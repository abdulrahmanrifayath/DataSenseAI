"""Dataset preprocessing, cleaning, scaling, and feature engineering engine using Scikit-Learn pipelines."""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler

from datasense.data_processing.schemas import (
    PreprocessingConfig,
    TransformationRecord,
    PreprocessingReport,
)
from datasense.data_processing.validator import DataProfiler
from datasense.utilities.logger import get_logger

logger = get_logger("data_processing.preprocessor")


class DataPreprocessor:
    """Automated and configurable Data Cleaning & Preprocessing Pipeline Engine.
    
    Supports missing value imputation, duplicate removal, type coercion, datetime feature engineering,
    outlier detection/clipping, constant & near-constant feature removal, high correlation filtering,
    scikit-learn ColumnTransformer scaling and encoding, and metadata reporting.
    """

    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        config_or_df: Optional[Union[PreprocessingConfig, pd.DataFrame]] = None,
    ):
        if config is not None:
            self.config = config
            self._initial_df = None
        elif isinstance(config_or_df, pd.DataFrame):
            self.config = PreprocessingConfig()
            self._initial_df = config_or_df
        elif isinstance(config_or_df, PreprocessingConfig):
            self.config = config_or_df
            self._initial_df = None
        else:
            self.config = PreprocessingConfig()
            self._initial_df = None


        self.is_fitted = False
        
        # Internal state tracking
        self.transformations: List[TransformationRecord] = []
        self.columns_removed: Dict[str, str] = {}
        self.columns_affected: set = set()
        self.missing_fixed: int = 0
        self.duplicates_removed: int = 0
        self.outliers_detected: int = 0
        self.column_types: Dict[str, str] = {}
        
        # Column classification state (saved across fit/transform)
        self.target_col: Optional[str] = self.config.target_column
        self.id_cols: List[str] = list(self.config.identifier_columns or [])
        self.datetime_cols_: List[str] = []
        self.original_datetime_cols_: List[str] = []
        self.numerical_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.columns_to_drop_: List[str] = []

        
        # Scikit-learn Pipeline
        self.column_transformer: Optional[ColumnTransformer] = None
        self.feature_names_out: List[str] = []
        self.report_: Optional[PreprocessingReport] = None

    def clean_missing_values(self, strategy: str = "auto") -> pd.DataFrame:
        """Backwards-compatible helper method to handle missing values."""
        if self._initial_df is not None:
            if strategy == "drop":
                self.config.numerical_impute_strategy = "drop"
                self.config.categorical_impute_strategy = "drop"
            return self.fit_transform(self._initial_df)
        raise ValueError("No DataFrame available for clean_missing_values. Pass DataFrame to fit_transform.")

    def _classify_columns(self, df: pd.DataFrame) -> None:
        """Classifies columns into target, id, datetime, numerical, and categorical."""
        profiler = DataProfiler(df)
        report = profiler.generate_report()
        
        self.datetime_cols_ = []
        self.numerical_cols_ = []
        self.categorical_cols_ = []
        
        exclude_set = set()
        if self.target_col and self.target_col in df.columns:
            exclude_set.add(self.target_col)
            self.column_types[self.target_col] = "target"
            
        inferred_ids = set(report.potential_id_columns)
        for col in self.id_cols:
            if col in df.columns:
                inferred_ids.add(col)
                
        for id_col in inferred_ids:
            if id_col in df.columns and id_col != self.target_col:
                exclude_set.add(id_col)
                if id_col not in self.id_cols:
                    self.id_cols.append(id_col)
                self.column_types[id_col] = "id"

        for col in df.columns:
            if col in exclude_set:
                continue
                
            series = df[col]
            dtype_str = str(series.dtype)
            
            if pd.api.types.is_datetime64_any_dtype(series):
                self.datetime_cols_.append(col)
                self.column_types[col] = "datetime"
            elif self.config.convert_datetimes and dtype_str in ["object", "string", "category"]:
                sample = series.dropna().iloc[:50]
                is_dt = False
                if not sample.empty:
                    try:
                        parsed = pd.to_datetime(sample, errors="coerce", format="ISO8601")
                        if parsed.notnull().sum() / len(sample) > 0.8:
                            is_dt = True
                    except Exception:
                        pass
                if is_dt:
                    self.datetime_cols_.append(col)
                    self.column_types[col] = "datetime"
                else:
                    self.categorical_cols_.append(col)
                    self.column_types[col] = "categorical"
            elif pd.api.types.is_numeric_dtype(series):
                self.numerical_cols_.append(col)
                self.column_types[col] = "numerical"
            else:
                self.categorical_cols_.append(col)
                self.column_types[col] = "categorical"

        self.original_datetime_cols_ = list(self.datetime_cols_)

        logger.info(
            f"Classified features - Numerical: {len(self.numerical_cols_)}, Categorical: {len(self.categorical_cols_)}, "
            f"Datetime: {len(self.datetime_cols_)}, ID: {len(self.id_cols)}, Target: '{self.target_col}'"
        )

    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detects and removes duplicate rows."""
        if not self.config.remove_duplicates or df.empty:
            return df
            
        dups = int(df.duplicated().sum())
        if dups > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            self.duplicates_removed += dups
            self.transformations.append(
                TransformationRecord(
                    step_name="remove_duplicates",
                    action="remove_duplicates",
                    columns_affected=list(df.columns),
                    details={"duplicates_removed": dups, "remaining_rows": len(df)},
                )
            )
            logger.info(f"Removed {dups} duplicate rows.")
        return df

    def _coerce_types_and_datetimes(self, df: pd.DataFrame, is_fitting: bool = True) -> pd.DataFrame:
        """Coerces numeric strings and handles date/datetime parsing & feature extraction."""
        df = df.copy()
        
        # 1. Type Coercion for Numerical
        if self.config.coerce_types and is_fitting:
            for col in list(self.categorical_cols_):
                series = df[col]
                if series.dtype in ["object", "string"]:
                    coerced = pd.to_numeric(series, errors="coerce")
                    non_null_orig = series.dropna().count()
                    non_null_coerced = coerced.dropna().count()
                    if non_null_orig > 0 and non_null_coerced / non_null_orig > 0.9:
                        df[col] = coerced
                        self.categorical_cols_.remove(col)
                        self.numerical_cols_.append(col)
                        self.column_types[col] = "numerical"
                        self.columns_affected.add(col)

        # 2. Datetime Parsing and Sub-feature extraction
        if self.config.convert_datetimes:
            extracted_cols = []
            dt_cols_to_process = list(self.datetime_cols_) if is_fitting else list(self.original_datetime_cols_)

            for col in dt_cols_to_process:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors="coerce", format="ISO8601")
                        self.columns_affected.add(col)
                        
                        if self.config.datetime_extract_features:
                            dt_s = df[col].dt
                            df[f"{col}_year"] = dt_s.year
                            df[f"{col}_month"] = dt_s.month
                            df[f"{col}_day"] = dt_s.day
                            df[f"{col}_dayofweek"] = dt_s.dayofweek
                            df[f"{col}_is_weekend"] = (dt_s.dayofweek >= 5).astype(int)
                            
                            new_num_cols = [
                                f"{col}_year",
                                f"{col}_month",
                                f"{col}_day",
                                f"{col}_dayofweek",
                                f"{col}_is_weekend",
                            ]
                            for n_col in new_num_cols:
                                if is_fitting and n_col not in self.numerical_cols_:
                                    self.numerical_cols_.append(n_col)
                                    self.column_types[n_col] = "numerical"
                                extracted_cols.append(n_col)

                            df = df.drop(columns=[col])
                            if is_fitting and col in self.datetime_cols_:
                                self.datetime_cols_.remove(col)
                                self.columns_removed[col] = "Extracted into sub-datetime features (year, month, day, etc.)"
                                self.columns_to_drop_.append(col)
                    except Exception as e:
                        logger.warning(f"Failed to parse datetime column '{col}': {e}")

            if extracted_cols and is_fitting:
                self.transformations.append(
                    TransformationRecord(
                        step_name="datetime_feature_extraction",
                        action="datetime_extract",
                        columns_affected=extracted_cols,
                        details={"extracted_count": len(extracted_cols)},
                    )
                )

        return df

    def _filter_constant_and_near_constant(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detects and drops constant (variance=0) and near-constant columns with recorded decisions."""
        df = df.copy()
        n_rows = len(df)
        if n_rows == 0:
            return df
            
        protected_set = set(self.id_cols)
        if self.target_col:
            protected_set.add(self.target_col)
            
        dropped_cols = []
        
        for col in list(df.columns):
            if col in protected_set:
                continue
                
            series = df[col]
            non_null = series.dropna()
            if non_null.empty:
                continue
                
            n_unique = non_null.nunique()
            
            # Check 1: Constant column
            if self.config.drop_constant and n_unique <= 1:
                df = df.drop(columns=[col])
                reason = "Constant column (variance=0, single distinct value)"
                self.columns_removed[col] = reason
                self.columns_to_drop_.append(col)
                dropped_cols.append(col)
                self._remove_from_feature_lists(col)
                continue
                
            # Check 2: Near-constant column
            if self.config.drop_near_constant and n_rows > 10:
                top_val_count = non_null.value_counts().iloc[0]
                dominant_ratio = top_val_count / len(non_null)
                if dominant_ratio >= self.config.near_constant_threshold:
                    df = df.drop(columns=[col])
                    reason = f"Near-constant column (dominant value ratio {dominant_ratio:.2%} >= threshold {self.config.near_constant_threshold:.2%})"
                    self.columns_removed[col] = reason
                    self.columns_to_drop_.append(col)
                    dropped_cols.append(col)
                    self._remove_from_feature_lists(col)

        if dropped_cols:
            self.transformations.append(
                TransformationRecord(
                    step_name="filter_constant_columns",
                    action="drop_column",
                    columns_affected=dropped_cols,
                    details={"dropped_count": len(dropped_cols)},
                )
            )
            logger.info(f"Dropped {len(dropped_cols)} constant/near-constant columns.")
            
        return df

    def _filter_high_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detects and drops highly correlated numerical features while recording decisions."""
        if not self.config.drop_high_correlation or len(self.numerical_cols_) <= 1:
            return df
            
        df = df.copy()
        valid_num_cols = [c for c in self.numerical_cols_ if c in df.columns and c != self.target_col and c not in self.id_cols]
        if len(valid_num_cols) <= 1:
            return df

        corr_matrix = df[valid_num_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        cols_to_drop = set()
        drop_details = {}

        for col in upper_tri.columns:
            high_corr_matches = upper_tri.index[upper_tri[col] > self.config.correlation_threshold].tolist()
            if high_corr_matches:
                cols_to_drop.add(col)
                correlated_with = high_corr_matches[0]
                r_val = float(upper_tri.loc[correlated_with, col])
                reason = f"High correlation (r={r_val:.3f}) with feature '{correlated_with}' exceeding threshold ({self.config.correlation_threshold})"
                drop_details[col] = reason

        if cols_to_drop:
            dropped_list = list(cols_to_drop)
            df = df.drop(columns=dropped_list)
            for col in dropped_list:
                self.columns_removed[col] = drop_details[col]
                self.columns_to_drop_.append(col)
                self._remove_from_feature_lists(col)
                
            self.transformations.append(
                TransformationRecord(
                    step_name="filter_high_correlation",
                    action="drop_column",
                    columns_affected=dropped_list,
                    details={"dropped_count": len(dropped_list), "reasons": drop_details},
                )
            )
            logger.info(f"Dropped {len(dropped_list)} highly correlated numerical columns.")

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detects outliers using IQR or Z-score and performs clipping/imputation/dropping."""
        if self.config.outlier_method == "none" or self.config.outlier_action == "none":
            return df

        df = df.copy()
        valid_num_cols = [c for c in self.numerical_cols_ if c in df.columns and c != self.target_col and c not in self.id_cols]
        if not valid_num_cols:
            return df

        total_outliers = 0
        affected_cols = []
        rows_to_drop = set()

        for col in valid_num_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            if self.config.outlier_method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower_bound = q1 - (self.config.outlier_threshold * iqr)
                upper_bound = q3 + (self.config.outlier_threshold * iqr)
            elif self.config.outlier_method == "zscore":
                mean = series.mean()
                std = series.std()
                if std == 0 or np.isnan(std):
                    continue
                lower_bound = mean - (self.config.outlier_threshold * std)
                upper_bound = mean + (self.config.outlier_threshold * std)
            else:
                continue

            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            col_outliers = int(outlier_mask.sum())

            if col_outliers > 0:
                total_outliers += col_outliers
                affected_cols.append(col)
                self.columns_affected.add(col)

                if self.config.outlier_action == "clip":
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                elif self.config.outlier_action == "impute":
                    median_val = series.median()
                    df.loc[outlier_mask, col] = median_val
                elif self.config.outlier_action == "drop_rows":
                    outlier_indices = df.index[outlier_mask].tolist()
                    rows_to_drop.update(outlier_indices)

        self.outliers_detected += total_outliers

        if self.config.outlier_action == "drop_rows" and rows_to_drop:
            df = df.drop(index=list(rows_to_drop)).reset_index(drop=True)
            logger.info(f"Dropped {len(rows_to_drop)} rows containing outliers.")

        if total_outliers > 0:
            self.transformations.append(
                TransformationRecord(
                    step_name="handle_outliers",
                    action=f"outlier_{self.config.outlier_action}",
                    columns_affected=affected_cols,
                    details={
                        "method": self.config.outlier_method,
                        "threshold": self.config.outlier_threshold,
                        "total_outliers_detected": total_outliers,
                        "action": self.config.outlier_action,
                    },
                )
            )

        return df

    def _remove_from_feature_lists(self, col: str) -> None:
        """Removes column from internal numerical/categorical tracking lists."""
        if col in self.numerical_cols_:
            self.numerical_cols_.remove(col)
        if col in self.categorical_cols_:
            self.categorical_cols_.remove(col)
        if col in self.datetime_cols_:
            self.datetime_cols_.remove(col)

    def _build_column_transformer(self) -> ColumnTransformer:
        """Builds Scikit-Learn ColumnTransformer for numerical scaling/imputation and categorical encoding/imputation."""
        num_transformers = []
        if self.config.numerical_impute_strategy == "constant":
            fill_val = self.config.numerical_impute_value if self.config.numerical_impute_value is not None else 0.0
            num_transformers.append(("imputer", SimpleImputer(strategy="constant", fill_value=fill_val)))
        else:
            num_transformers.append(("imputer", SimpleImputer(strategy=self.config.numerical_impute_strategy)))

        if self.config.numerical_scaling == "standard":
            num_transformers.append(("scaler", StandardScaler()))
        elif self.config.numerical_scaling == "minmax":
            num_transformers.append(("scaler", MinMaxScaler()))
        elif self.config.numerical_scaling == "robust":
            num_transformers.append(("scaler", RobustScaler()))

        num_pipeline = Pipeline(num_transformers)

        cat_transformers = []
        if self.config.categorical_impute_strategy == "constant":
            cat_transformers.append(("imputer", SimpleImputer(strategy="constant", fill_value=self.config.categorical_impute_value)))
        else:
            cat_transformers.append(("imputer", SimpleImputer(strategy="most_frequent")))

        if self.config.categorical_encoding == "onehot":
            cat_transformers.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        elif self.config.categorical_encoding == "ordinal":
            cat_transformers.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))

        cat_pipeline = Pipeline(cat_transformers)

        transformers = []
        if self.numerical_cols_:
            transformers.append(("num", num_pipeline, self.numerical_cols_))
        if self.categorical_cols_:
            transformers.append(("cat", cat_pipeline, self.categorical_cols_))

        return ColumnTransformer(transformers=transformers, remainder="passthrough", verbose_feature_names_out=False)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes full preprocessing pipeline: fit, clean, transform, and report generation."""
        logger.info(f"Starting DataPreprocessor fit_transform on DataFrame shape {df.shape}")
        initial_shape = list(df.shape)
        self.missing_fixed = int(df.isnull().sum().sum())
        
        # 1. Deduplication
        df_clean = self._handle_duplicates(df)
        
        # 2. Classify columns
        self._classify_columns(df_clean)
        
        # 3. Type Coercion & Datetime feature extraction
        df_clean = self._coerce_types_and_datetimes(df_clean, is_fitting=True)
        
        # 4. Constant & Near-constant filtering
        df_clean = self._filter_constant_and_near_constant(df_clean)
        
        # 5. High correlation filtering
        df_clean = self._filter_high_correlation(df_clean)
        
        # 6. Outlier detection & action
        df_clean = self._handle_outliers(df_clean)
        
        # Filter numerical & categorical columns to existing columns
        self.numerical_cols_ = [c for c in self.numerical_cols_ if c in df_clean.columns]
        self.categorical_cols_ = [c for c in self.categorical_cols_ if c in df_clean.columns]

        # 7. Scikit-learn Pipeline (Imputation, Scaling, Encoding)
        if self.numerical_cols_ or self.categorical_cols_:
            self.column_transformer = self._build_column_transformer()
            
            self.columns_affected.update(self.numerical_cols_)
            self.columns_affected.update(self.categorical_cols_)
            
            arr_transformed = self.column_transformer.fit_transform(df_clean)
            
            try:
                feature_names = list(self.column_transformer.get_feature_names_out())
            except Exception:
                feature_names = [f"feature_{i}" for i in range(arr_transformed.shape[1])]
                
            transformed_df = pd.DataFrame(arr_transformed, columns=feature_names, index=df_clean.index)
            self.feature_names_out = feature_names
            
            self.transformations.append(
                TransformationRecord(
                    step_name="impute_scale_encode",
                    action="scikit_learn_column_transformer",
                    columns_affected=self.numerical_cols_ + self.categorical_cols_,
                    details={
                        "numerical_impute": self.config.numerical_impute_strategy,
                        "numerical_scale": self.config.numerical_scaling,
                        "categorical_impute": self.config.categorical_impute_strategy,
                        "categorical_encode": self.config.categorical_encoding,
                    },
                )
            )
        else:
            transformed_df = df_clean.copy()
            self.feature_names_out = list(transformed_df.columns)

        self.is_fitted = True
        final_shape = list(transformed_df.shape)

        self.report_ = PreprocessingReport(
            transformations=self.transformations,
            columns_affected=list(self.columns_affected),
            missing_values_fixed=self.missing_fixed,
            duplicates_removed=self.duplicates_removed,
            outliers_detected=self.outliers_detected,
            columns_removed=self.columns_removed,
            initial_shape=initial_shape,
            final_shape=final_shape,
            column_types=self.column_types,
            feature_names_out=self.feature_names_out,
        )

        logger.info(f"Preprocessing completed. Initial shape: {initial_shape} -> Final shape: {final_shape}")
        return transformed_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies fitted preprocessing pipeline on new dataset without re-fitting parameters."""
        if not self.is_fitted or self.column_transformer is None:
            raise RuntimeError("DataPreprocessor must be fitted using fit_transform before calling transform.")
            
        logger.info(f"Applying fitted DataPreprocessor to new DataFrame shape {df.shape}")
        df_clean = self._handle_duplicates(df)
        df_clean = self._coerce_types_and_datetimes(df_clean, is_fitting=False)
        
        # Drop columns removed during fit phase
        cols_to_drop = [c for c in self.columns_to_drop_ if c in df_clean.columns]
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)

        arr_transformed = self.column_transformer.transform(df_clean)
        return pd.DataFrame(arr_transformed, columns=self.feature_names_out, index=df_clean.index)

    def get_report(self) -> PreprocessingReport:
        """Returns generated preprocessing report."""
        if not self.report_:
            raise RuntimeError("Preprocessing report is unavailable. Run fit_transform first.")
        return self.report_
