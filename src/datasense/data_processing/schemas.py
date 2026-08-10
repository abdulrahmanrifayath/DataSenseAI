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
