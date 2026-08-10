"""Exploratory Data Analysis (EDA) module."""

from typing import Dict, Any
import pandas as pd
from datasense.utilities.logger import get_logger

logger = get_logger("eda")


class EDAEngine:
    """Performs automated exploratory data analysis and summary statistics."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculates descriptive summary statistics."""
        logger.info("Computing summary statistics for dataset.")
        return {
            "describe": self.df.describe(include="all").to_dict(),
            "correlations": self.df.select_dtypes(include=["number"]).corr().to_dict(),
        }
