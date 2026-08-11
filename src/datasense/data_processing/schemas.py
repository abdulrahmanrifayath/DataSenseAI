"""Pydantic schemas for data validation and profiling reports."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Detected pandas data type")
    inferred_type: str = Field(..., description="Inferred business type: numerical, categorical, datetime, id, boolean")
    missing_count: int = Field(..., description="Total missing values")
    missing_percentage: float = Field(..., description="Missing values percentage (0-100)")
    unique_count: int = Field(..., description="Total unique values count")
    unique_ratio: float = Field(..., description="Ratio of unique values to total rows (0-1)")
    is_constant: bool = Field(default=False, description="True if column has only 1 distinct value")
    is_potential_id: bool = Field(default=False, description="True if column appears to be a unique primary ID key")
    sample_values: List[Any] = Field(default_factory=list, description="Sample non-null values")


class QualityWarning(BaseModel):
    severity: str = Field(..., description="Warning level: CRITICAL, WARNING, INFO")
    code: str = Field(..., description="Warning category code (e.g. HIGH_MISSINGNESS, DUPLICATE_ROWS)")
    column: Optional[str] = Field(default=None, description="Affected column name if applicable")
    message: str = Field(..., description="Human readable description of the data quality issue")


class ValidationReport(BaseModel):
    row_count: int = Field(..., description="Total number of rows in dataset")
    column_count: int = Field(..., description="Total number of columns in dataset")
    duplicate_rows_count: int = Field(..., description="Number of duplicate rows")
    duplicate_rows_percentage: float = Field(..., description="Percentage of duplicate rows (0-100)")
    total_missing_cells: int = Field(..., description="Sum of missing values across all cells")
    total_cells: int = Field(..., description="Total cells (rows * columns)")
    missing_cell_percentage: float = Field(..., description="Percentage of missing cells overall")
    
    numerical_columns: List[str] = Field(default_factory=list, description="List of numerical column names")
    categorical_columns: List[str] = Field(default_factory=list, description="List of categorical column names")
    datetime_columns: List[str] = Field(default_factory=list, description="List of datetime column names")
    potential_id_columns: List[str] = Field(default_factory=list, description="List of primary ID candidate columns")
    constant_columns: List[str] = Field(default_factory=list, description="List of single-value constant columns")
    
    column_profiles: Dict[str, ColumnProfile] = Field(default_factory=dict, description="Detailed profile per column")
    warnings: List[QualityWarning] = Field(default_factory=list, description="List of data quality warnings")
    quality_score: float = Field(..., description="Overall data quality score from 0 to 100")


class DatasetMetadataResponse(BaseModel):
    id: int
    name: str
    filename: str
    file_type: str
    file_size_bytes: int
    row_count: int
    column_count: int
    quality_score: float
    created_at: str
    preview_data: Optional[Dict[str, Any]] = None
    validation_report: Optional[ValidationReport] = None


class PreprocessingConfig(BaseModel):
    target_column: Optional[str] = Field(default=None, description="Target column name (excluded from feature transforms)")
    identifier_columns: List[str] = Field(default_factory=list, description="Identifier columns to preserve without scaling/encoding")
    
    numerical_impute_strategy: str = Field(default="median", description="Strategy for missing numerical values: median, mean, most_frequent, constant")
    numerical_impute_value: Optional[float] = Field(default=None, description="Fill value if numerical_impute_strategy is constant")
    
    categorical_impute_strategy: str = Field(default="most_frequent", description="Strategy for missing categorical values: most_frequent, constant")
    categorical_impute_value: str = Field(default="Unknown", description="Fill value if categorical_impute_strategy is constant")
    
    remove_duplicates: bool = Field(default=True, description="Whether to drop duplicate rows")
    coerce_types: bool = Field(default=True, description="Whether to coerce numeric/boolean types")
    convert_datetimes: bool = Field(default=True, description="Whether to parse date/datetime columns")
    datetime_extract_features: bool = Field(default=True, description="Whether to extract sub-features (year, month, day, dayofweek, etc.)")
    
    outlier_method: str = Field(default="iqr", description="Outlier detection method: iqr, zscore, none")
    outlier_threshold: float = Field(default=1.5, description="IQR multiplier (e.g. 1.5) or Z-score threshold (e.g. 3.0)")
    outlier_action: str = Field(default="clip", description="Outlier action: clip, impute, drop_rows, none")
    
    categorical_encoding: str = Field(default="onehot", description="Encoding method: onehot, ordinal, none")
    numerical_scaling: str = Field(default="standard", description="Scaling method: standard, minmax, robust, none")
    
    drop_constant: bool = Field(default=True, description="Whether to drop constant variance=0 columns")
    drop_near_constant: bool = Field(default=True, description="Whether to drop near-constant columns")
    near_constant_threshold: float = Field(default=0.98, description="Max allowed ratio for dominant single value (e.g. 0.98)")
    
    drop_high_correlation: bool = Field(default=True, description="Whether to drop highly correlated numerical features")
    correlation_threshold: float = Field(default=0.95, description="Absolute correlation threshold to flag/drop feature")


class TransformationRecord(BaseModel):
    step_name: str = Field(..., description="Name of the transformation step")
    action: str = Field(..., description="Action performed: impute, remove_duplicates, clip_outliers, drop_column, encode, scale, datetime_extract")
    columns_affected: List[str] = Field(default_factory=list, description="Columns modified or processed in this step")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed parameters and statistics of transformation")


class PreprocessingReport(BaseModel):
    transformations: List[TransformationRecord] = Field(default_factory=list, description="List of all transformation steps executed")
    columns_affected: List[str] = Field(default_factory=list, description="List of all unique columns modified")
    missing_values_fixed: int = Field(default=0, description="Total missing values filled or handled")
    duplicates_removed: int = Field(default=0, description="Total duplicate rows deleted")
    outliers_detected: int = Field(default=0, description="Total outliers identified across columns")
    columns_removed: Dict[str, str] = Field(default_factory=dict, description="Map of removed column name to reason for removal")
    initial_shape: List[int] = Field(..., description="Initial dataset shape [rows, cols]")
    final_shape: List[int] = Field(..., description="Final dataset shape [rows, cols]")
    column_types: Dict[str, str] = Field(default_factory=dict, description="Map of column name to feature role: numerical, categorical, datetime, id, target")
    feature_names_out: List[str] = Field(default_factory=list, description="Final feature column names after encoding/transforms")


class PreprocessingRequest(BaseModel):
    config: Optional[PreprocessingConfig] = Field(default_factory=PreprocessingConfig)


class PreprocessingResponse(BaseModel):
    id: int
    dataset_id: int
    created_at: str
    config: PreprocessingConfig
    report: PreprocessingReport
    preview_data: Optional[Dict[str, Any]] = None

