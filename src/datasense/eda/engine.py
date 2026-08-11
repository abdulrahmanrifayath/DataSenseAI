"""Automated Exploratory Data Analysis (EDA) Engine using Pandas, NumPy, and Plotly."""

from typing import Dict, List, Any, Optional, Tuple
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from datasense.eda.schemas import (
    InsightItem,
    EDASummary,
    NumericalStats,
    CategoricalStats,
    OutlierStats,
    EDAReport,
)
from datasense.data_processing.validator import DataProfiler
from datasense.utilities.logger import get_logger

logger = get_logger("eda.engine")


class EDAEngine:
    """Automated Exploratory Data Analysis Engine.
    
    Dynamically profiles any dataset schema, computes distribution statistics, correlation matrices,
    outliers, missingness, time-series trends, target variable relationships, interactive Plotly charts,
    and rule-based automatic insights.
    """

    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.target_column = target_column if target_column and target_column in self.df.columns else None
        
        # Categorize columns dynamically
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.datetime_cols: List[str] = []
        self.id_cols: List[str] = []
        
        self._classify_columns()

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Backwards-compatible helper returning basic summary statistics and correlations."""
        report = self.generate_report()
        return {
            "describe": self.df.describe(include="all").to_dict(),
            "correlations": report.correlation_matrix,
        }

    def _classify_columns(self) -> None:

        """Classifies DataFrame columns into numerical, categorical, datetime, and id."""
        if self.df.empty:
            return
            
        profiler = DataProfiler(self.df)
        report = profiler.generate_report()
        
        self.id_cols = list(report.potential_id_columns)
        
        for col in self.df.columns:
            series = self.df[col]
            dtype_str = str(series.dtype)
            
            if pd.api.types.is_datetime64_any_dtype(series):
                self.datetime_cols.append(str(col))
            elif dtype_str in ["object", "string", "category"]:
                # Soft datetime test
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
                    self.datetime_cols.append(str(col))
                    self.df[col] = pd.to_datetime(self.df[col], errors="coerce", format="ISO8601")
                else:
                    self.categorical_cols.append(str(col))
            elif pd.api.types.is_numeric_dtype(series):
                self.numerical_cols.append(str(col))
            else:
                self.categorical_cols.append(str(col))

    def _compute_summary(self) -> EDASummary:
        """Computes dataset shape, memory usage, missing cell ratio, and duplicate counts."""
        if self.df.empty:
            return EDASummary(
                row_count=0,
                column_count=0,
                total_cells=0,
                total_missing_cells=0,
                missing_percentage=0.0,
                duplicate_rows=0,
                memory_usage_bytes=0,
                feature_counts={},
            )

        rows, cols = self.df.shape
        total_cells = rows * cols
        total_missing = int(self.df.isnull().sum().sum())
        missing_pct = round((total_missing / total_cells) * 100.0, 2) if total_cells > 0 else 0.0
        dups = int(self.df.duplicated().sum())
        mem_bytes = int(self.df.memory_usage(deep=True).sum())

        return EDASummary(
            row_count=rows,
            column_count=cols,
            total_cells=total_cells,
            total_missing_cells=total_missing,
            missing_percentage=missing_pct,
            duplicate_rows=dups,
            memory_usage_bytes=mem_bytes,
            feature_counts={
                "numerical": len(self.numerical_cols),
                "categorical": len(self.categorical_cols),
                "datetime": len(self.datetime_cols),
                "identifier": len(self.id_cols),
            },
        )

    def _compute_numerical_stats(self) -> Dict[str, NumericalStats]:
        """Computes descriptive summary metrics (mean, std, min, quantiles, max, skewness, kurtosis)."""
        stats_dict = {}
        for col in self.numerical_cols:
            series = self.df[col].dropna()
            if series.empty:
                continue

            try:
                mean_val = float(series.mean())
                std_val = float(series.std()) if len(series) > 1 else 0.0
                min_val = float(series.min())
                q25_val = float(series.quantile(0.25))
                median_val = float(series.median())
                q75_val = float(series.quantile(0.75))
                max_val = float(series.max())
                
                # Skewness and Kurtosis
                skew_val = float(series.skew()) if len(series) > 2 else 0.0
                kurt_val = float(series.kurtosis()) if len(series) > 3 else 0.0
                
                if np.isnan(skew_val):
                    skew_val = 0.0
                if np.isnan(kurt_val):
                    kurt_val = 0.0

                missing_cnt = int(self.df[col].isnull().sum())
                zero_cnt = int((series == 0).sum())

                stats_dict[col] = NumericalStats(
                    mean=round(mean_val, 4),
                    std=round(std_val, 4),
                    min=round(min_val, 4),
                    q25=round(q25_val, 4),
                    median=round(median_val, 4),
                    q75=round(q75_val, 4),
                    max=round(max_val, 4),
                    skewness=round(skew_val, 4),
                    kurtosis=round(kurt_val, 4),
                    missing_count=missing_cnt,
                    zero_count=zero_cnt,
                )
            except Exception as e:
                logger.warning(f"Error computing numerical stats for column '{col}': {e}")

        return stats_dict

    def _compute_categorical_stats(self) -> Dict[str, CategoricalStats]:
        """Computes cardinality, top value, top frequency, and missing counts for categorical features."""
        stats_dict = {}
        n_rows = len(self.df)
        for col in self.categorical_cols:
            series = self.df[col].dropna()
            missing_cnt = int(self.df[col].isnull().sum())
            if series.empty:
                stats_dict[col] = CategoricalStats(
                    count=0,
                    unique_count=0,
                    top_value=None,
                    top_freq=0,
                    top_ratio=0.0,
                    missing_count=missing_cnt,
                )
                continue

            unique_cnt = int(series.nunique())
            val_counts = series.value_counts()
            top_val = str(val_counts.index[0])
            top_freq = int(val_counts.iloc[0])
            top_ratio = round((top_freq / len(series)), 4) if len(series) > 0 else 0.0

            stats_dict[col] = CategoricalStats(
                count=len(series),
                unique_count=unique_cnt,
                top_value=top_val,
                top_freq=top_freq,
                top_ratio=top_ratio,
                missing_count=missing_cnt,
            )

        return stats_dict

    def _compute_missing_analysis(self) -> Dict[str, Any]:
        """Computes per-column missing value metrics and missingness patterns."""
        missing_by_col = {}
        n_rows = len(self.df)
        for col in self.df.columns:
            missing_cnt = int(self.df[col].isnull().sum())
            missing_pct = round((missing_cnt / n_rows) * 100.0, 2) if n_rows > 0 else 0.0
            if missing_cnt > 0:
                missing_by_col[col] = {"missing_count": missing_cnt, "missing_percentage": missing_pct}

        return {
            "columns_with_missing": missing_by_col,
            "total_missing_columns": len(missing_by_col),
        }

    def _compute_correlations(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
        """Computes Pearson and Spearman correlation matrices and identifies top correlated pairs."""
        if len(self.numerical_cols) < 2:
            return {}, {}, []

        num_df = self.df[self.numerical_cols].dropna()
        if num_df.empty or len(num_df) < 3:
            num_df = self.df[self.numerical_cols].fillna(0)

        try:
            pearson_df = num_df.corr(method="pearson").round(4)
            spearman_df = num_df.corr(method="spearman").round(4)

            pearson_dict = pearson_df.to_dict()
            spearman_dict = spearman_df.to_dict()

            # Find top pairwise correlations (excluding self-correlations)
            pairs = []
            upper_tri = pearson_df.where(np.triu(np.ones(pearson_df.shape), k=1).astype(bool))
            for c1 in upper_tri.columns:
                for c2 in upper_tri.index:
                    val = upper_tri.loc[c2, c1]
                    if not np.isnan(val):
                        spearman_val = float(spearman_df.loc[c2, c1]) if c2 in spearman_df.index and c1 in spearman_df.columns else float(val)
                        pairs.append({
                            "feature1": str(c2),
                            "feature2": str(c1),
                            "pearson": float(val),
                            "spearman": round(spearman_val, 4),
                            "abs_pearson": round(abs(float(val)), 4),
                        })

            pairs.sort(key=lambda x: x["abs_pearson"], reverse=True)
            return pearson_dict, spearman_dict, pairs[:15]
        except Exception as e:
            logger.warning(f"Error computing correlation matrix: {e}")
            return {}, {}, []

    def _compute_outlier_analysis(self) -> Dict[str, OutlierStats]:
        """Computes IQR and Z-score outlier statistics and bounds per numerical column."""
        outlier_dict = {}
        for col in self.numerical_cols:
            series = self.df[col].dropna()
            if series.empty or len(series) < 4:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            iqr_lower = q1 - (1.5 * iqr)
            iqr_upper = q3 + (1.5 * iqr)

            iqr_count = int(((series < iqr_lower) | (series > iqr_upper)).sum())

            mean_val = float(series.mean())
            std_val = float(series.std())
            if std_val > 0:
                z_scores = np.abs((series - mean_val) / std_val)
                z_count = int((z_scores > 3.0).sum())
                z_lower = mean_val - (3.0 * std_val)
                z_upper = mean_val + (3.0 * std_val)
            else:
                z_count = 0
                z_lower, z_upper = mean_val, mean_val

            outlier_dict[col] = OutlierStats(
                iqr_outliers=iqr_count,
                zscore_outliers=z_count,
                iqr_lower_bound=round(iqr_lower, 4),
                iqr_upper_bound=round(iqr_upper, 4),
                zscore_lower_bound=round(z_lower, 4),
                zscore_upper_bound=round(z_upper, 4),
            )

        return outlier_dict

    def _compute_category_frequencies(self) -> Dict[str, Dict[str, int]]:
        """Computes top category frequency value counts for categorical features."""
        freq_dict = {}
        for col in self.categorical_cols:
            series = self.df[col].dropna()
            if series.empty:
                continue
            top_counts = series.value_counts().head(10).to_dict()
            freq_dict[col] = {str(k): int(v) for k, v in top_counts.items()}
        return freq_dict

    def _compute_time_trends(self) -> Dict[str, Any]:
        """Computes time-series trend metrics when datetime features are present."""
        if not self.datetime_cols or not self.numerical_cols:
            return {}

        dt_col = self.datetime_cols[0]
        temp_df = self.df[[dt_col] + self.numerical_cols[:3]].dropna(subset=[dt_col])
        if temp_df.empty or len(temp_df) < 5:
            return {}

        try:
            temp_df = temp_df.sort_values(dt_col)
            temp_df = temp_df.set_index(dt_col)
            resampled = temp_df.resample("D").mean().dropna()
            if len(resampled) < 3:
                resampled = temp_df.resample("ME").mean().dropna()

            trends = {}
            for col in resampled.columns:
                series = resampled[col]
                if len(series) > 2:
                    x = np.arange(len(series))
                    slope, intercept, r_val, p_val, std_err = stats.linregress(x, series.values)
                    direction = "UPWARD" if slope > 0.01 else ("DOWNWARD" if slope < -0.01 else "STABLE")
                    trends[col] = {
                        "direction": direction,
                        "slope": round(float(slope), 4),
                        "r_squared": round(float(r_val**2), 4),
                        "min_date": str(series.index.min().strftime("%Y-%m-%d")),
                        "max_date": str(series.index.max().strftime("%Y-%m-%d")),
                    }

            return {
                "datetime_column": dt_col,
                "resampled_points": len(resampled),
                "trends": trends,
            }
        except Exception as e:
            logger.warning(f"Error computing time trends: {e}")
            return {}

    def _compute_target_analysis(self) -> Dict[str, Any]:
        """Computes target variable distribution and feature-target relationships when a target is specified."""
        if not self.target_column or self.target_column not in self.df.columns:
            return {}

        target_s = self.df[self.target_column].dropna()
        if target_s.empty:
            return {}

        is_target_numeric = pd.api.types.is_numeric_dtype(target_s) and target_s.nunique() > 10
        analysis = {
            "target_column": self.target_column,
            "target_type": "numerical" if is_target_numeric else "categorical",
            "feature_relationships": [],
        }

        if is_target_numeric:
            # Correlation of numerical features with target
            num_features = [c for c in self.numerical_cols if c != self.target_column]
            if num_features:
                corrs = self.df[num_features].apply(lambda s: s.corr(self.df[self.target_column])).round(4).to_dict()
                analysis["feature_correlations"] = {str(k): float(v) for k, v in corrs.items() if not np.isnan(v)}
        else:
            # Group distributions by target class
            class_counts = target_s.value_counts().to_dict()
            analysis["class_distribution"] = {str(k): int(v) for k, v in class_counts.items()}

        return analysis

    def _generate_plotly_charts(self) -> Dict[str, str]:
        """Generates interactive Plotly visualizations and serializes them as JSON strings."""
        charts = {}
        if self.df.empty:
            return charts

        try:
            # 1. Missing Values Bar Chart
            missing_series = self.df.isnull().sum()
            missing_series = missing_series[missing_series > 0]
            if not missing_series.empty:
                fig_missing = px.bar(
                    x=missing_series.index,
                    y=missing_series.values,
                    labels={"x": "Column Name", "y": "Missing Count"},
                    title="Missing Values Breakdown per Column",
                    color=missing_series.values,
                    color_continuous_scale="Reds",
                    template="plotly_white",
                )
                charts["missing_bar"] = fig_missing.to_json()

            # 2. Pearson Correlation Matrix Heatmap
            if len(self.numerical_cols) >= 2:
                corr_df = self.df[self.numerical_cols].corr().round(3)
                fig_corr = px.imshow(
                    corr_df,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title="Pearson Feature Correlation Matrix",
                    template="plotly_white",
                    aspect="auto",
                )
                charts["corr_heatmap"] = fig_corr.to_json()

            # 3. Numerical Feature Histograms
            if self.numerical_cols:
                top_num = self.numerical_cols[0]
                fig_hist = px.histogram(
                    self.df,
                    x=top_num,
                    marginal="box",
                    title=f"Distribution Profile: {top_num}",
                    template="plotly_white",
                )
                charts[f"dist_{top_num}"] = fig_hist.to_json()

            # 4. Categorical Frequency Bar Chart
            if self.categorical_cols:
                top_cat = self.categorical_cols[0]
                cat_counts = self.df[top_cat].value_counts().head(10)
                fig_cat = px.bar(
                    x=cat_counts.index.astype(str),
                    y=cat_counts.values,
                    labels={"x": top_cat, "y": "Frequency"},
                    title=f"Category Frequency Breakdown: {top_cat}",
                    color=cat_counts.values,
                    color_continuous_scale="Blues",
                    template="plotly_white",
                )
                charts[f"freq_{top_cat}"] = fig_cat.to_json()

            # 5. Time-Series Line Chart
            if self.datetime_cols and self.numerical_cols:
                dt_col = self.datetime_cols[0]
                num_col = self.numerical_cols[0]
                temp = self.df[[dt_col, num_col]].dropna().sort_values(dt_col)
                if not temp.empty:
                    fig_time = px.line(
                        temp,
                        x=dt_col,
                        y=num_col,
                        title=f"Temporal Trend: {num_col} over {dt_col}",
                        template="plotly_white",
                    )
                    charts["time_series_line"] = fig_time.to_json()

            # 6. Target Variable Analysis Chart
            if self.target_column and self.numerical_cols:
                num_feat = [c for c in self.numerical_cols if c != self.target_column]
                if num_feat:
                    feat = num_feat[0]
                    if pd.api.types.is_numeric_dtype(self.df[self.target_column]):
                        fig_target = px.scatter(
                            self.df,
                            x=feat,
                            y=self.target_column,
                            trendline="ols",
                            title=f"Target Relationship: {self.target_column} vs {feat}",
                            template="plotly_white",
                        )
                    else:
                        fig_target = px.box(
                            self.df,
                            x=self.target_column,
                            y=feat,
                            title=f"Feature Distribution across Target Classes: {feat} by {self.target_column}",
                            template="plotly_white",
                        )
                    charts["target_relationship"] = fig_target.to_json()

        except Exception as e:
            logger.warning(f"Error generating Plotly charts: {e}")

        return charts

    def _detect_insights(
        self,
        summary: EDASummary,
        num_stats: Dict[str, NumericalStats],
        cat_stats: Dict[str, CategoricalStats],
        corr_pairs: List[Dict[str, Any]],
        outliers: Dict[str, OutlierStats],
        time_trends: Dict[str, Any],
        target_analysis: Dict[str, Any],
    ) -> List[InsightItem]:
        """Executes automated rule engine to detect actionable data patterns, anomalies, and insights."""
        insights = []

        # 1. High Missingness Pattern
        for col, stat in num_stats.items():
            missing_pct = (stat.missing_count / summary.row_count) * 100.0 if summary.row_count > 0 else 0
            if missing_pct >= 20.0:
                insights.append(
                    InsightItem(
                        category="MISSINGNESS",
                        severity="HIGH" if missing_pct >= 50.0 else "MEDIUM",
                        title=f"High Missingness in Column '{col}'",
                        description=f"Column '{col}' is missing {missing_pct:.1f}% of its values ({stat.missing_count} cells). Imputation or deletion recommended.",
                        affected_columns=[col],
                        metrics={"missing_percentage": round(missing_pct, 2), "missing_count": stat.missing_count},
                    )
                )

        # 2. Strongest Correlations
        for pair in corr_pairs:
            if pair["abs_pearson"] >= 0.75:
                insights.append(
                    InsightItem(
                        category="CORRELATION",
                        severity="HIGH" if pair["abs_pearson"] >= 0.90 else "MEDIUM",
                        title=f"Strong Correlation between '{pair['feature1']}' and '{pair['feature2']}'",
                        description=f"Features '{pair['feature1']}' and '{pair['feature2']}' demonstrate a strong linear correlation (r = {pair['pearson']:.3f}). Consider feature selection to prevent multicollinearity.",
                        affected_columns=[pair["feature1"], pair["feature2"]],
                        metrics={"pearson": pair["pearson"], "spearman": pair["spearman"]},
                    )
                )

        # 3. Dominant Categorical Values
        for col, cstat in cat_stats.items():
            if cstat.top_ratio >= 0.70 and cstat.unique_count > 1:
                insights.append(
                    InsightItem(
                        category="CATEGORICAL",
                        severity="MEDIUM" if cstat.top_ratio >= 0.85 else "INFO",
                        title=f"Dominant Category in Column '{col}'",
                        description=f"Top value '{cstat.top_value}' dominates column '{col}' representing {cstat.top_ratio:.1%} of all rows.",
                        affected_columns=[col],
                        metrics={"top_value": cstat.top_value, "top_ratio": cstat.top_ratio},
                    )
                )

        # 4. Unusual Distributions (High Skewness)
        for col, nstat in num_stats.items():
            if abs(nstat.skewness) >= 1.5:
                direction = "Right (Positive)" if nstat.skewness > 0 else "Left (Negative)"
                insights.append(
                    InsightItem(
                        category="DISTRIBUTION",
                        severity="MEDIUM",
                        title=f"Highly Skewed Distribution in '{col}'",
                        description=f"Numerical feature '{col}' exhibits significant {direction} skewness (skew = {nstat.skewness:.2f}). Logarithmic or Yeo-Johnson transformation is recommended.",
                        affected_columns=[col],
                        metrics={"skewness": nstat.skewness, "kurtosis": nstat.kurtosis},
                    )
                )

        # 5. Potential Outliers
        for col, ostat in outliers.items():
            outlier_pct = (ostat.iqr_outliers / summary.row_count) * 100.0 if summary.row_count > 0 else 0
            if outlier_pct >= 5.0:
                insights.append(
                    InsightItem(
                        category="OUTLIER",
                        severity="HIGH" if outlier_pct >= 15.0 else "MEDIUM",
                        title=f"Significant Outlier Concentration in '{col}'",
                        description=f"Column '{col}' contains {ostat.iqr_outliers} IQR outliers ({outlier_pct:.1f}% of total rows). Review clipping or winsorization strategies.",
                        affected_columns=[col],
                        metrics={"outlier_count": ostat.iqr_outliers, "outlier_percentage": round(outlier_pct, 2)},
                    )
                )

        # 6. Temporal Trends
        if "trends" in time_trends:
            for col, tinfo in time_trends["trends"].items():
                if tinfo["direction"] in ["UPWARD", "DOWNWARD"] and tinfo["r_squared"] >= 0.40:
                    insights.append(
                        InsightItem(
                            category="TEMPORAL",
                            severity="HIGH" if tinfo["r_squared"] >= 0.70 else "MEDIUM",
                            title=f"Significant {tinfo['direction']} Trend in '{col}'",
                            description=f"Feature '{col}' exhibits a clear {tinfo['direction'].lower()} temporal trend over time (R² = {tinfo['r_squared']:.2f}, slope = {tinfo['slope']:.4f}).",
                            affected_columns=[col],
                            metrics=tinfo,
                        )
                    )

        # 7. Target Relationships
        if "feature_correlations" in target_analysis:
            top_target_corrs = sorted(
                target_analysis["feature_correlations"].items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:3]
            for col, corr_val in top_target_corrs:
                if abs(corr_val) >= 0.40:
                    insights.append(
                        InsightItem(
                            category="TARGET",
                            severity="HIGH",
                            title=f"Strong Target Association: '{col}' vs Target '{self.target_column}'",
                            description=f"Feature '{col}' is strongly associated with target '{self.target_column}' (r = {corr_val:.3f}). Excellent predictive candidate.",
                            affected_columns=[col, self.target_column],
                            metrics={"target_correlation": corr_val},
                        )
                    )

        return insights

    def generate_report(self) -> EDAReport:
        """Executes full automated profiling and builds complete EDAReport object."""
        logger.info(f"Generating full EDA report for DataFrame shape {self.df.shape}")

        summary = self._compute_summary()
        num_stats = self._compute_numerical_stats()
        cat_stats = self._compute_categorical_stats()
        missing_analysis = self._compute_missing_analysis()
        pearson_matrix, spearman_matrix, top_pairs = self._compute_correlations()
        outliers = self._compute_outlier_analysis()
        cat_freqs = self._compute_category_frequencies()
        time_trends = self._compute_time_trends()
        target_analysis = self._compute_target_analysis()
        charts_json = self._generate_plotly_charts()

        insights = self._detect_insights(
            summary=summary,
            num_stats=num_stats,
            cat_stats=cat_stats,
            corr_pairs=top_pairs,
            outliers=outliers,
            time_trends=time_trends,
            target_analysis=target_analysis,
        )

        return EDAReport(
            summary=summary,
            numerical_stats=num_stats,
            categorical_stats=cat_stats,
            missing_analysis=missing_analysis,
            correlation_matrix=pearson_matrix,
            spearman_correlation_matrix=spearman_matrix,
            top_correlation_pairs=top_pairs,
            outlier_analysis=outliers,
            category_frequencies=cat_freqs,
            time_trends=time_trends,
            target_analysis=target_analysis,
            charts_plotly_json=charts_json,
            insights=insights,
        )
