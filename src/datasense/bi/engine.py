"""Business Intelligence Engine for Customer Analytics, RFM Segmentation, KPI Calculation, CLV, and Churn Analysis."""

import uuid
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

from datasense.bi.schemas import (
    ColumnMappingConfig,
    BusinessKPIs,
    RFMSegmentSummary,
    ClusterEvaluationMetrics,
    CustomerSegmentDetail,
    ChurnRiskSummary,
    CLVSummary,
    BIAnalysisReport,
)
from datasense.utilities.logger import get_logger

logger = get_logger("bi.engine")


class BIEngine:
    """Business Intelligence & Customer Analytics Engine."""

    ALIAS_MAP = {
        "customer_id": ["customer_id", "customer", "client_id", "user_id", "customer_name", "account_id", "cust_id"],
        "order_date": ["order_date", "date", "transaction_date", "timestamp", "created_at", "invoice_date"],
        "revenue": ["revenue", "sales", "amount", "total_price", "order_value", "monetary", "total_amount", "price"],
        "profit": ["profit", "margin", "net_profit", "gain"],
        "quantity": ["quantity", "qty", "units", "items_count"],
        "churn": ["churn", "churned", "is_churn", "is_churned"],
    }

    def resolve_column_mapping(
        self,
        df: pd.DataFrame,
        manual_mapping: Optional[ColumnMappingConfig] = None,
    ) -> ColumnMappingConfig:
        """Resolves configurable column mappings using manual overrides or fuzzy alias matching."""
        cols_lower = {col.lower(): col for col in df.columns}
        resolved = ColumnMappingConfig()

        # Check manual overrides first
        if manual_mapping:
            if manual_mapping.customer_id_col and manual_mapping.customer_id_col in df.columns:
                resolved.customer_id_col = manual_mapping.customer_id_col
            if manual_mapping.order_date_col and manual_mapping.order_date_col in df.columns:
                resolved.order_date_col = manual_mapping.order_date_col
            if manual_mapping.revenue_col and manual_mapping.revenue_col in df.columns:
                resolved.revenue_col = manual_mapping.revenue_col
            if manual_mapping.profit_col and manual_mapping.profit_col in df.columns:
                resolved.profit_col = manual_mapping.profit_col
            if manual_mapping.quantity_col and manual_mapping.quantity_col in df.columns:
                resolved.quantity_col = manual_mapping.quantity_col
            if manual_mapping.churn_col and manual_mapping.churn_col in df.columns:
                resolved.churn_col = manual_mapping.churn_col

        # Auto alias fuzzy matching for unassigned fields
        for field, aliases in self.ALIAS_MAP.items():
            attr_name = f"{field}_col"
            if getattr(resolved, attr_name) is None:
                for alias in aliases:
                    if alias in cols_lower:
                        setattr(resolved, attr_name, cols_lower[alias])
                        break
                    # Partial match fallback
                    matching = [col for lower, col in cols_lower.items() if alias in lower]
                    if matching:
                        setattr(resolved, attr_name, matching[0])
                        break

        logger.info(f"Resolved Column Mapping: {resolved.model_dump()}")
        return resolved

    def calculate_business_kpis(self, df: pd.DataFrame, mapping: ColumnMappingConfig) -> BusinessKPIs:
        """Calculates executive business KPIs."""
        rev_col = mapping.revenue_col
        profit_col = mapping.profit_col
        cust_col = mapping.customer_id_col

        total_rev = float(df[rev_col].sum()) if rev_col and rev_col in df.columns else 0.0
        total_profit = float(df[profit_col].sum()) if profit_col and profit_col in df.columns else None

        if cust_col and cust_col in df.columns:
            total_cust = int(df[cust_col].nunique())
            tx_per_cust = df.groupby(cust_col).size()
            repeat_count = int((tx_per_cust > 1).sum())
            repeat_rate = float(round((repeat_count / max(1, total_cust)) * 100.0, 2))
        else:
            total_cust = len(df)
            repeat_rate = 0.0

        aov = float(round(total_rev / max(1, len(df)), 2))
        margin_pct = float(round((total_profit / total_rev) * 100.0, 2)) if total_profit is not None and total_rev > 0 else None

        return BusinessKPIs(
            total_revenue=round(total_rev, 2),
            total_profit=round(total_profit, 2) if total_profit is not None else None,
            average_order_value=aov,
            total_customers=total_cust,
            repeat_purchase_rate=repeat_rate,
            profit_margin_pct=margin_pct,
        )

    def analyze(
        self,
        df: pd.DataFrame,
        manual_mapping: Optional[ColumnMappingConfig] = None,
        clustering_algorithm: str = "kmeans",
        dataset_id: Optional[int] = None,
    ) -> BIAnalysisReport:
        """Executes full Business Intelligence pipeline."""
        if df.empty:
            raise ValueError("Input DataFrame for Business Intelligence analysis is empty.")

        run_id = f"bi_{uuid.uuid4().hex[:10]}"
        mapping = self.resolve_column_mapping(df, manual_mapping)
        kpis = self.calculate_business_kpis(df, mapping)

        cust_col = mapping.customer_id_col
        date_col = mapping.order_date_col
        rev_col = mapping.revenue_col

        rfm_df: Optional[pd.DataFrame] = None
        rfm_summaries: List[RFMSegmentSummary] = []
        cluster_eval: Optional[ClusterEvaluationMetrics] = None
        customer_segments: List[CustomerSegmentDetail] = []
        churn_summary: Optional[ChurnRiskSummary] = None
        clv_summary: Optional[CLVSummary] = None
        charts: Dict[str, str] = {}

        # If customer ID is available, calculate RFM & Segmentation
        if cust_col and cust_col in df.columns:
            rfm_df = self._compute_rfm(df, mapping)
            if not rfm_df.empty:
                rfm_summaries = self._summarize_rfm_segments(rfm_df, kpis.total_revenue)
                cluster_eval, customer_segments, rfm_df = self._segment_customers(rfm_df, algorithm=clustering_algorithm)
                churn_summary = self._evaluate_churn_risk(rfm_df, df, mapping)
                clv_summary = self._estimate_clv(rfm_df, kpis.average_order_value)

        # Insights
        insights = self._generate_insights(kpis, rfm_summaries, churn_summary, clv_summary)

        # Plotly Charts
        charts = self._build_charts(df, rfm_df, mapping, kpis, customer_segments)

        report = BIAnalysisReport(
            run_id=run_id,
            dataset_id=dataset_id,
            resolved_mapping=mapping,
            business_kpis=kpis,
            rfm_segments=rfm_summaries,
            cluster_evaluation=cluster_eval,
            customer_segments=customer_segments,
            churn_summary=churn_summary,
            clv_summary=clv_summary,
            business_insights=insights,
            charts_plotly_json=charts,
        )

        logger.info(f"BI Analysis completed run_id '{run_id}'. Total Revenue: ${kpis.total_revenue:,.2f}, Total Customers: {kpis.total_customers}")
        return report

    def _compute_rfm(self, df: pd.DataFrame, mapping: ColumnMappingConfig) -> pd.DataFrame:
        """Computes Recency, Frequency, Monetary metrics per customer."""
        cust_col = mapping.customer_id_col
        date_col = mapping.order_date_col
        rev_col = mapping.revenue_col

        df_work = df.copy()

        # Parse date if available
        if date_col and date_col in df_work.columns:
            df_work[date_col] = pd.to_datetime(df_work[date_col], errors="coerce")
            max_date = df_work[date_col].max()
            if pd.isnull(max_date):
                max_date = pd.Timestamp.now()
        else:
            date_col = None
            max_date = pd.Timestamp.now()

        # Parse revenue
        if rev_col and rev_col in df_work.columns:
            df_work[rev_col] = pd.to_numeric(df_work[rev_col], errors="coerce").fillna(0.0)
        else:
            df_work["_dummy_rev"] = 1.0
            rev_col = "_dummy_rev"

        # Group by customer
        if date_col:
            rfm = df_work.groupby(cust_col).agg(
                recency=(date_col, lambda x: (max_date - x.max()).days),
                frequency=(cust_col, "count"),
                monetary=(rev_col, "sum"),
            ).reset_index()
        else:
            rfm = df_work.groupby(cust_col).agg(
                recency=(rev_col, lambda x: 30),  # Default 30 days recency if date unavailable
                frequency=(cust_col, "count"),
                monetary=(rev_col, "sum"),
            ).reset_index()

        # Ensure positive non-zero values for quantile binning
        rfm["recency"] = rfm["recency"].clip(lower=0)
        rfm["frequency"] = rfm["frequency"].clip(lower=1)
        rfm["monetary"] = rfm["monetary"].clip(lower=0.01)

        # Compute Quantile RFM Scores (1-5)
        try:
            rfm["r_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
        except Exception:
            rfm["r_score"] = 3

        try:
            rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
        except Exception:
            rfm["f_score"] = 3

        try:
            rfm["m_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
        except Exception:
            rfm["m_score"] = 3

        # Assign Persona Labels
        def assign_segment(row):
            r, f, m = row["r_score"], row["f_score"], row["m_score"]
            if r >= 4 and f >= 4 and m >= 4:
                return "Champions"
            elif f >= 4 and m >= 3:
                return "Loyal Customers"
            elif r >= 3 and f >= 2 and m >= 2:
                return "Potential Loyalists"
            elif r <= 2 and f >= 3 and m >= 3:
                return "At Risk"
            elif r <= 2 and f <= 2 and m <= 2:
                return "Hibernating"
            elif r == 1 and f == 1:
                return "Lost"
            return "Promising"

        rfm["segment_name"] = rfm.apply(assign_segment, axis=1)
        return rfm

    def _summarize_rfm_segments(self, rfm_df: pd.DataFrame, total_revenue: float) -> List[RFMSegmentSummary]:
        """Summarizes RFM metrics per persona segment."""
        summaries = []
        tot_rev = max(1.0, total_revenue)

        grouped = rfm_df.groupby("segment_name")
        for seg_name, grp in grouped:
            c_cnt = len(grp)
            m_rev = float(grp["monetary"].sum())
            rev_pct = float(round((m_rev / tot_rev) * 100.0, 2))

            summaries.append(
                RFMSegmentSummary(
                    segment_name=seg_name,
                    customer_count=c_cnt,
                    avg_recency_days=float(round(grp["recency"].mean(), 1)),
                    avg_frequency=float(round(grp["frequency"].mean(), 1)),
                    avg_monetary_value=float(round(grp["monetary"].mean(), 2)),
                    total_revenue_share_pct=rev_pct,
                )
            )
        return sorted(summaries, key=lambda s: s.total_revenue_share_pct, reverse=True)

    def _segment_customers(
        self,
        rfm_df: pd.DataFrame,
        algorithm: str = "kmeans",
    ) -> Tuple[ClusterEvaluationMetrics, List[CustomerSegmentDetail], pd.DataFrame]:
        """Performs K-Means or Hierarchical Clustering on RFM features with Silhouette & Davies-Bouldin evaluation."""
        X = rfm_df[["recency", "frequency", "monetary"]].to_numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_samples = len(X_scaled)
        max_k = min(6, n_samples - 1)

        if max_k < 2:
            rfm_df["cluster"] = 0
            detail = [
                CustomerSegmentDetail(
                    cluster_id=0,
                    segment_label="Core Customers",
                    customer_count=n_samples,
                    mean_recency=float(round(rfm_df["recency"].mean(), 1)),
                    mean_frequency=float(round(rfm_df["frequency"].mean(), 1)),
                    mean_monetary=float(round(rfm_df["monetary"].mean(), 2)),
                    revenue_share_pct=100.0,
                )
            ]
            eval_m = ClusterEvaluationMetrics(
                algorithm=algorithm,
                optimal_k=1,
                silhouette_score=1.0,
                davies_bouldin_index=0.0,
            )
            return eval_m, detail, rfm_df

        best_k = 3
        best_sil = -1.0
        best_db = 999.0
        best_labels = None

        # Grid search over K in [2, max_k]
        for k in range(2, max_k + 1):
            if algorithm == "hierarchical":
                model = AgglomerativeClustering(n_clusters=k)
            else:
                model = KMeans(n_clusters=k, random_state=42, n_init=10)

            labels = model.fit_predict(X_scaled)
            sil = float(silhouette_score(X_scaled, labels))
            db = float(davies_bouldin_score(X_scaled, labels))

            if sil > best_sil:
                best_sil = sil
                best_db = db
                best_k = k
                best_labels = labels

        rfm_df["cluster"] = best_labels
        tot_rev = max(1.0, rfm_df["monetary"].sum())

        cluster_details = []
        for cid in range(best_k):
            sub = rfm_df[rfm_df["cluster"] == cid]
            c_cnt = len(sub)
            m_rev = float(sub["monetary"].sum())
            rev_share = float(round((m_rev / tot_rev) * 100.0, 2))

            cluster_details.append(
                CustomerSegmentDetail(
                    cluster_id=cid,
                    segment_label=f"Cluster {cid + 1}",
                    customer_count=c_cnt,
                    mean_recency=float(round(sub["recency"].mean(), 1)),
                    mean_frequency=float(round(sub["frequency"].mean(), 1)),
                    mean_monetary=float(round(sub["monetary"].mean(), 2)),
                    revenue_share_pct=rev_share,
                )
            )

        cluster_eval = ClusterEvaluationMetrics(
            algorithm="K-Means" if algorithm == "kmeans" else "Hierarchical Agglomerative",
            optimal_k=best_k,
            silhouette_score=round(best_sil, 4),
            davies_bouldin_index=round(best_db, 4),
        )

        return cluster_eval, cluster_details, rfm_df

    def _evaluate_churn_risk(
        self,
        rfm_df: pd.DataFrame,
        df_raw: pd.DataFrame,
        mapping: ColumnMappingConfig,
    ) -> ChurnRiskSummary:
        """Evaluates customer churn risk using explicit or derived inactivity target."""
        rfm_work = rfm_df.copy()

        # Check explicit churn column
        churn_col = mapping.churn_col
        if churn_col and churn_col in df_raw.columns:
            # Map churn to binary 0/1
            cust_churn = df_raw.groupby(mapping.customer_id_col)[churn_col].max().astype(int).reset_index()
            rfm_work = rfm_work.merge(cust_churn, on=mapping.customer_id_col, how="left").fillna(0)
            target = rfm_work[churn_col].to_numpy()
        else:
            # Implicit churn: Recency > 90 days or > 2 * median recency
            rec_thresh = max(60, rfm_work["recency"].median() * 2)
            rfm_work["derived_churn"] = (rfm_work["recency"] > rec_thresh).astype(int)
            target = rfm_work["derived_churn"].to_numpy()

        overall_churn_rate = float(round((np.sum(target) / max(1, len(target))) * 100.0, 2))

        X = rfm_work[["recency", "frequency", "monetary"]].to_numpy()
        if len(np.unique(target)) > 1 and len(target) >= 10:
            rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
            rf.fit(X, target)
            probs = rf.predict_proba(X)[:, 1]
            high_risk_count = int(np.sum(probs >= 0.70))
            importances = rf.feature_importances_
            feat_dict = {
                "Recency": float(round(importances[0], 4)),
                "Frequency": float(round(importances[1], 4)),
                "Monetary Value": float(round(importances[2], 4)),
            }
        else:
            high_risk_count = int(np.sum(target == 1))
            feat_dict = {"Recency": 1.0, "Frequency": 0.0, "Monetary Value": 0.0}

        return ChurnRiskSummary(
            overall_churn_rate_pct=overall_churn_rate,
            high_risk_customer_count=high_risk_count,
            top_churn_drivers=feat_dict,
        )

    def _estimate_clv(self, rfm_df: pd.DataFrame, aov: float) -> CLVSummary:
        """Estimates Historical and Projected 12-Month Customer Lifetime Value (CLV)."""
        avg_hist_clv = float(round(rfm_df["monetary"].mean(), 2))

        # Projected 12m CLV = Average Order Value * Annual Frequency * Retention Rate (80%)
        retention_rate = 0.80
        rfm_df["proj_clv"] = rfm_df["frequency"] * aov * retention_rate
        avg_proj_clv = float(round(rfm_df["proj_clv"].mean(), 2))

        top_seg = rfm_df.groupby("segment_name")["proj_clv"].mean().idxmax()

        return CLVSummary(
            average_historical_clv=avg_hist_clv,
            average_projected_12m_clv=avg_proj_clv,
            top_clv_segment=str(top_seg),
        )

    def _generate_insights(
        self,
        kpis: BusinessKPIs,
        rfm_summaries: List[RFMSegmentSummary],
        churn_summary: Optional[ChurnRiskSummary],
        clv_summary: Optional[CLVSummary],
    ) -> List[str]:
        """Generates natural language actionable business insights."""
        insights = []

        insights.append(f"💰 Total Revenue: **${kpis.total_revenue:,.2f}** across **{kpis.total_customers:,}** unique customers (Average Order Value: **${kpis.average_order_value:,.2f}**).")

        if rfm_summaries:
            champions = next((s for s in rfm_summaries if s.segment_name == "Champions"), None)
            if champions:
                insights.append(
                    f"🏆 **Champions Segment**: Represents **{champions.customer_count}** top customers contributing **{champions.total_revenue_share_pct}%** of total sales revenue."
                )

            at_risk = next((s for s in rfm_summaries if s.segment_name == "At Risk"), None)
            if at_risk:
                insights.append(
                    f"⚠️ **At Risk Customers**: **{at_risk.customer_count}** high-value customers show declining purchase frequency. Target with win-back email campaigns."
                )

        if churn_summary:
            insights.append(
                f"🚨 **Churn Risk Analysis**: Overall customer churn rate is **{churn_summary.overall_churn_rate_pct}%**, with **{churn_summary.high_risk_customer_count}** customers at critical risk."
            )

        if clv_summary:
            insights.append(
                f"📈 **Lifetime Value Projection**: Projected average 12-month CLV is **${clv_summary.average_projected_12m_clv:,.2f}**, driven primarily by the **{clv_summary.top_clv_segment}** segment."
            )

        return insights

    def _build_charts(
        self,
        df_raw: pd.DataFrame,
        rfm_df: Optional[pd.DataFrame],
        mapping: ColumnMappingConfig,
        kpis: BusinessKPIs,
        customer_segments: List[CustomerSegmentDetail],
    ) -> Dict[str, str]:
        """Builds 5 interactive Plotly charts serialized as JSON."""
        charts = {}

        if rfm_df is not None and not rfm_df.empty:
            # 1. Customer Segments 2D Scatter Plot
            fig_scatter = px.scatter(
                rfm_df,
                x="recency",
                y="monetary",
                size="frequency",
                color="segment_name",
                hover_data=[mapping.customer_id_col],
                title="Customer RFM Persona Segmentation Scatter Plot",
                labels={"recency": "Recency (Days Ago)", "monetary": "Monetary Value ($)", "segment_name": "Segment"},
                template="plotly_white",
            )
            fig_scatter.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            charts["customer_segments_scatter"] = fig_scatter.to_json()

            # 2. RFM Distribution Bar Chart
            rfm_counts = rfm_df["segment_name"].value_counts().reset_index()
            rfm_counts.columns = ["Segment", "Count"]
            fig_rfm = px.bar(
                rfm_counts,
                x="Segment",
                y="Count",
                color="Segment",
                title="Customer Distribution across RFM Persona Segments",
                template="plotly_white",
            )
            fig_rfm.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            charts["rfm_distribution_bar"] = fig_rfm.to_json()

            # 3. Revenue by Segment Bar Chart
            rev_by_seg = rfm_df.groupby("segment_name")["monetary"].sum().reset_index()
            rev_by_seg.columns = ["Segment", "Revenue"]
            fig_rev = px.pie(
                rev_by_seg,
                names="Segment",
                values="Revenue",
                title="Total Revenue Contribution by Customer Segment",
                template="plotly_white",
                hole=0.4,
            )
            fig_rev.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            charts["revenue_by_segment_pie"] = fig_rev.to_json()

        # 4. Profit by Segment or KPI Summary
        if mapping.profit_col and mapping.profit_col in df_raw.columns and mapping.customer_id_col and rfm_df is not None:
            prof_df = df_raw.groupby(mapping.customer_id_col)[mapping.profit_col].sum().reset_index()
            rfm_prof = rfm_df.merge(prof_df, on=mapping.customer_id_col, how="left").fillna(0)
            prof_by_seg = rfm_prof.groupby("segment_name")[mapping.profit_col].sum().reset_index()
            prof_by_seg.columns = ["Segment", "Profit"]
            fig_prof = px.bar(
                prof_by_seg,
                x="Segment",
                y="Profit",
                color="Segment",
                title="Total Profit by Customer Segment",
                template="plotly_white",
            )
            fig_prof.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            charts["profit_by_segment_bar"] = fig_prof.to_json()

        # 5. Customer Trends Line Chart over Order Date
        date_col = mapping.order_date_col
        rev_col = mapping.revenue_col
        if date_col and date_col in df_raw.columns and rev_col and rev_col in df_raw.columns:
            df_trend = df_raw.copy()
            df_trend[date_col] = pd.to_datetime(df_trend[date_col], errors="coerce")
            df_trend = df_trend.dropna(subset=[date_col]).sort_values(date_col)
            if len(df_trend) > 0:
                trend_daily = df_trend.set_index(date_col).resample("D")[rev_col].sum().reset_index()
                fig_trend = px.line(
                    trend_daily,
                    x=date_col,
                    y=rev_col,
                    title="Daily Revenue Trend",
                    labels={date_col: "Date", rev_col: "Revenue ($)"},
                    template="plotly_white",
                )
                fig_trend.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                charts["customer_trends_line"] = fig_trend.to_json()

        return charts
