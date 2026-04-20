from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


NUMERIC_DEFAULTS: dict[str, float] = {
    "visited": 0.0,
    "added_to_cart": 0.0,
    "purchased": 0.0,
    "price": 0.0,
    "quantity": 0.0,
    "discount": 0.0,
    "total_spend": 0.0,
    "recency_days": np.nan,
}


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace(0, np.nan)
    out = numer / denom
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_customer_features(
    transactions: pd.DataFrame,
    *,
    customer_col: str = "customer_id",
    order_col: str = "order_id",
    max_category_levels: int = 30,
) -> pd.DataFrame:
    df = transactions.copy()
    df["_event_count"] = 1.0

    for col, default in NUMERIC_DEFAULTS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    if customer_col not in df.columns:
        raise ValueError(f"Missing column: {customer_col}")

    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    grouped = df.groupby(customer_col, as_index=True)

    has_orders = order_col in df.columns
    if has_orders:
        orders = grouped[order_col].nunique(dropna=True).astype(float)
    else:
        orders = grouped["_event_count"].sum().astype(float)

    if "visited" in df.columns:
        visits = grouped["visited"].sum().astype(float)
    else:
        visits = orders.copy()

    if "added_to_cart" in df.columns:
        added_to_cart = grouped["added_to_cart"].sum().astype(float)
    else:
        added_to_cart = pd.Series(0.0, index=orders.index)

    if "purchased" in df.columns:
        purchased = pd.to_numeric(df["purchased"], errors="coerce").fillna(0).astype(int)
        if has_orders:
            purchases = df.loc[purchased == 1].groupby(customer_col)[order_col].nunique(dropna=True).astype(float)
        else:
            purchases = df.loc[purchased == 1].groupby(customer_col).size().astype(float)
        purchases = purchases.reindex(orders.index).fillna(0.0)
    else:
        purchases = orders.copy()

    if "total_spend" in df.columns:
        revenue_total = grouped["total_spend"].sum().astype(float)
    elif "price" in df.columns and "quantity" in df.columns:
        revenue_total = (pd.to_numeric(df["price"], errors="coerce").fillna(0) * pd.to_numeric(df["quantity"], errors="coerce").fillna(0)).groupby(df[customer_col]).sum().astype(float)
    else:
        revenue_total = pd.Series(0.0, index=orders.index)

    avg_order_value = (revenue_total / orders.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "discount" in df.columns:
        avg_discount = grouped["discount"].mean().astype(float)
    else:
        avg_discount = pd.Series(0.0, index=orders.index)

    if "recency_days" in df.columns:
        recency_days = grouped["recency_days"].min().astype(float)
    elif "order_date" in df.columns and df["order_date"].notna().any():
        last_purchase = grouped["order_date"].max()
        max_time = pd.to_datetime(df["order_date"], errors="coerce").max()
        recency_days = (max_time - last_purchase).dt.days.astype(float)
    else:
        recency_days = pd.Series(np.nan, index=orders.index, dtype=float)

    base = pd.DataFrame(
        {
            "orders": orders,
            "visits": visits,
            "added_to_cart": added_to_cart,
            "purchases": purchases,
            "revenue_total": revenue_total,
            "avg_order_value": avg_order_value,
            "avg_discount": avg_discount,
            "recency_days": recency_days,
        }
    )

    base["cart_rate"] = _safe_div(base["added_to_cart"], base["visits"])
    base["purchase_rate"] = _safe_div(base["purchases"], base["visits"])
    base["cart_to_purchase_rate"] = _safe_div(base["purchases"], base["added_to_cart"])

    for col in ("channel", "device", "location", "payment_method", "category"):
        if col not in df.columns:
            continue
        if int(df[col].nunique(dropna=True)) > int(max_category_levels):
            continue

        counts = (
            df.pivot_table(
                index=customer_col,
                columns=col,
                values="_event_count",
                aggfunc="sum",
                fill_value=0,
            )
            .astype(float)
            .sort_index(axis=1)
        )
        shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        shares.columns = [f"{col}_share__{c}" for c in shares.columns]
        base = base.join(shares, how="left")

    base = base.fillna(0.0)
    base["revenue_total_log1p"] = np.log1p(base["revenue_total"])
    base["avg_order_value_log1p"] = np.log1p(base["avg_order_value"])

    return base.reset_index()


def build_customer_features_asof(
    transactions: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    lookback_days: int | None = 180,
    customer_col: str = "customer_id",
    max_category_levels: int = 30,
) -> pd.DataFrame:
    """
    Build customer features using only history up to `as_of`.

    This is used for forward-looking churn prediction to avoid label leakage.
    """
    df = transactions.copy()
    if "order_date" not in df.columns:
        raise ValueError("Missing column: order_date")

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df[df["order_date"].notna()].copy()

    df = df[df["order_date"] <= as_of].copy()

    if lookback_days is not None:
        cutoff = as_of - pd.Timedelta(days=int(lookback_days))
    df = df[df["order_date"] >= cutoff].copy()

    if df.empty:
        return pd.DataFrame(columns=[customer_col])

    df["_event_count"] = 1.0
    has_orders = "order_id" in df.columns
    grouped = df.groupby(customer_col, as_index=True)

    # Last purchase date per customer (from observed history only).
    if "purchased" in df.columns:
        purchased_rows = df[pd.to_numeric(df["purchased"], errors="coerce").fillna(0).astype(int) == 1]
    else:
        purchased_rows = df

    last_purchase = purchased_rows.groupby(customer_col)["order_date"].max()

    # Base aggregates
    if has_orders:
        orders = grouped["order_id"].nunique(dropna=True).astype(float)
    else:
        orders = grouped["_event_count"].sum().astype(float)

    if "visited" in df.columns:
        visits = grouped["visited"].sum().astype(float)
    else:
        visits = orders.copy()

    if "added_to_cart" in df.columns:
        added_to_cart = grouped["added_to_cart"].sum().astype(float)
    else:
        added_to_cart = pd.Series(0.0, index=orders.index)

    if "purchased" in df.columns:
        p = pd.to_numeric(df["purchased"], errors="coerce").fillna(0).astype(int)
        if has_orders:
            purchases = df.loc[p == 1].groupby(customer_col)["order_id"].nunique(dropna=True).astype(float)
        else:
            purchases = df.loc[p == 1].groupby(customer_col).size().astype(float)
        purchases = purchases.reindex(orders.index).fillna(0.0)
    else:
        purchases = orders.copy()

    base = pd.DataFrame(
        {
            "orders": orders,
            "visits": visits,
            "added_to_cart": added_to_cart,
            "purchases": purchases,
            "revenue_total": grouped["total_spend"].sum() if "total_spend" in df.columns else 0.0,
            "avg_discount": grouped["discount"].mean() if "discount" in df.columns else 0.0,
        }
    )
    base["avg_order_value"] = (base["revenue_total"] / base["orders"].replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    recency = (as_of - last_purchase).dt.days
    if lookback_days is None:
        fallback = int((as_of - df["order_date"].min()).days) + 1
    else:
        fallback = int(lookback_days) + 1
    base["recency_days"] = recency.fillna(fallback).astype(float)

    base["cart_rate"] = _safe_div(base["added_to_cart"], base["visits"])
    base["purchase_rate"] = _safe_div(base["purchases"], base["visits"])
    base["cart_to_purchase_rate"] = _safe_div(base["purchases"], base["added_to_cart"])

    for col in ("channel", "device", "location", "payment_method", "category"):
        if col not in df.columns:
            continue
        if int(df[col].nunique(dropna=True)) > int(max_category_levels):
            continue

        counts = (
            df.pivot_table(
                index=customer_col,
                columns=col,
                values="_event_count",
                aggfunc="sum",
                fill_value=0,
            )
            .astype(float)
            .sort_index(axis=1)
        )
        shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        shares.columns = [f"{col}_share__{c}" for c in shares.columns]
        base = base.join(shares, how="left")

    base = base.fillna(0.0)
    base["revenue_total_log1p"] = np.log1p(base["revenue_total"])
    base["avg_order_value_log1p"] = np.log1p(base["avg_order_value"])

    return base.reset_index()


def select_feature_columns(
    customer_features: pd.DataFrame,
    *,
    exclude: Iterable[str] = ("customer_id",),
) -> list[str]:
    exclude_set = set(exclude)
    cols: list[str] = []
    for col in customer_features.columns:
        if col in exclude_set:
            continue
        if pd.api.types.is_numeric_dtype(customer_features[col]):
            cols.append(col)
    return cols
