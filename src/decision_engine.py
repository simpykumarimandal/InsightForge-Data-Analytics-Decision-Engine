from __future__ import annotations

import pandas as pd


def market_research_summary(transactions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = transactions.copy()
    out: dict[str, pd.DataFrame] = {}

    if "total_spend" in df.columns and "category" in df.columns:
        out["revenue_by_category"] = (
            df.groupby("category", as_index=False)["total_spend"].sum().sort_values("total_spend", ascending=False)
        )
    if "total_spend" in df.columns and "product" in df.columns:
        out["revenue_by_product"] = (
            df.groupby("product", as_index=False)["total_spend"].sum().sort_values("total_spend", ascending=False)
        )
    if "total_spend" in df.columns and "stock_code" in df.columns:
        out["revenue_by_stock_code"] = (
            df.groupby("stock_code", as_index=False)["total_spend"]
            .sum()
            .sort_values("total_spend", ascending=False)
        )
    if "total_spend" in df.columns and "location" in df.columns:
        out["revenue_by_location"] = (
            df.groupby("location", as_index=False)["total_spend"].sum().sort_values("total_spend", ascending=False)
        )
    if "quantity" in df.columns and "product" in df.columns:
        out["quantity_by_product"] = (
            df.groupby("product", as_index=False)["quantity"].sum().sort_values("quantity", ascending=False)
        )

    for col in ("channel", "device", "location", "payment_method"):
        if col in df.columns:
            out[f"orders_by_{col}"] = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="orders")

    if "order_date" in df.columns and "total_spend" in df.columns:
        from .market_research import revenue_over_time

        out["revenue_over_time_month"] = revenue_over_time(df, freq="ME")

    return out


def recommend_actions(
    customers: pd.DataFrame,
    *,
    churn_high_threshold: float = 0.7,
    recency_threshold_days: float = 90,
    high_value_quantile: float = 0.75,
    protect_risk_threshold: float = 0.4,
) -> pd.DataFrame:
    df = customers.copy()

    if "churn_risk" not in df.columns:
        df["churn_risk"] = 0.0
    if "revenue_total" not in df.columns:
        df["revenue_total"] = 0.0
    if "recency_days" not in df.columns:
        df["recency_days"] = 0.0

    revenue_q = float(df["revenue_total"].quantile(high_value_quantile)) if len(df) else 0.0

    def _action(row: pd.Series) -> str:
        risk = float(row.get("churn_risk", 0.0))
        rev = float(row.get("revenue_total", 0.0))
        rec = float(row.get("recency_days", 0.0))

        if risk >= churn_high_threshold or rec >= recency_threshold_days:
            return "Retention: win-back offer + reactivation outreach"
        if rev >= revenue_q and risk >= protect_risk_threshold:
            return "Protect high-value: VIP support + targeted incentive"
        if rev >= revenue_q:
            return "Upsell: premium bundle recommendation"
        if risk <= 0.2 and rec <= 30:
            return "Engage: loyalty points + referral prompt"
        return "Nurture: personalized recommendations + light discount"

    df["recommended_action"] = df.apply(_action, axis=1)
    df["decision_score"] = (1 - df["churn_risk"]) * 0.5 + (df["revenue_total"].rank(pct=True) * 0.5)
    return df
