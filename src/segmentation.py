from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SegmentationResult:
    customers: pd.DataFrame
    segment_profiles: pd.DataFrame
    model: KMeans
    scaler: StandardScaler


def segment_customers(
    customer_features: pd.DataFrame,
    *,
    customer_col: str = "customer_id",
    n_clusters: int = 4,
    random_state: int = 42,
) -> SegmentationResult:
    if customer_col not in customer_features.columns:
        raise ValueError(f"Missing column: {customer_col}")

    numeric = customer_features.select_dtypes("number").copy()
    scaler = StandardScaler()
    x = scaler.fit_transform(numeric)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(x)

    customers = customer_features.copy()
    customers["segment_id"] = labels

    profiles = (
        customers.groupby("segment_id", as_index=False)[numeric.columns]
        .mean(numeric_only=True)
        .sort_values("segment_id")
    )

    return SegmentationResult(customers=customers, segment_profiles=profiles, model=model, scaler=scaler)


def name_segments(customers_with_segments: pd.DataFrame) -> pd.DataFrame:
    df = customers_with_segments.copy()
    if "segment_id" not in df.columns:
        raise ValueError("Expected column: segment_id")

    required = [c for c in ("revenue_total", "orders", "recency_days") if c in df.columns]
    if not required:
        df["segment_name"] = df["segment_id"].astype(str)
        return df

    summary = (
        df.groupby("segment_id", as_index=False)[required]
        .median(numeric_only=True)
        .fillna(0.0)
    )

    spend_rank = summary["revenue_total"].rank(method="dense") if "revenue_total" in summary.columns else 1
    recency_rank = (
        (summary["recency_days"].rank(method="dense")).max() - summary["recency_days"].rank(method="dense") + 1
        if "recency_days" in summary.columns
        else 1
    )
    freq_rank = summary["orders"].rank(method="dense") if "orders" in summary.columns else 1

    score = spend_rank * 0.5 + freq_rank * 0.3 + recency_rank * 0.2
    q75, q50, q25 = score.quantile(0.75), score.quantile(0.5), score.quantile(0.25)
    summary["segment_name"] = score.apply(
        lambda s: "Champions" if s >= q75 else "Loyal" if s >= q50 else "Potential" if s >= q25 else "At-Risk"
    )

    return df.merge(summary[["segment_id", "segment_name"]], on="segment_id", how="left")


def silhouette_scores(
    customer_features: pd.DataFrame,
    *,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute silhouette score across k to help justify the segmentation choice.
    """
    numeric = customer_features.select_dtypes("number").copy()
    if numeric.empty or len(numeric) < 3:
        return pd.DataFrame(columns=["k", "silhouette"])

    scaler = StandardScaler()
    x = scaler.fit_transform(numeric)

    rows: list[dict[str, float]] = []
    for k in range(int(k_min), int(k_max) + 1):
        if k <= 1 or k >= len(numeric):
            continue
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(x)
        score = silhouette_score(x, labels)
        rows.append({"k": float(k), "silhouette": float(score)})

    return pd.DataFrame(rows)
