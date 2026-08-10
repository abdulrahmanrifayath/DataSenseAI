"""Dataset preprocessing and cleaning module."""

import pandas as pd
from datasense.utilities.logger import get_logger

logger = get_logger("data_processing.preprocessor")


class DataPreprocessor:
    """Handles dataset cleaning, imputation, scaling, and feature transformation."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def clean_missing_values(self, strategy: str = "auto") -> pd.DataFrame:
        """Imputes or handles missing values in dataframe."""
        logger.info(f"Cleaning missing values using strategy: {strategy}")
        if strategy == "drop":
            self.df = self.df.dropna()
        elif strategy == "auto":
            for col in self.df.select_dtypes(include=["number"]).columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            for col in self.df.select_dtypes(include=["object", "category"]).columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown")
        return self.df
