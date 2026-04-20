from __future__ import annotations

from itertools import combinations

import pandas as pd


def revenue_over_time(
    transactions: pd.DataFrame,
    *,
    time_col: str = "order_date",
    value_col: str = "total_spend",
    freq: str = "ME",
) -> pd.DataFrame:
    df = transactions.copy()
    if time_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=[time_col, "revenue"])

    # Pandas >= 2.2 deprecated/removed some legacy aliases like "M" in favor of "ME".
    if freq == "M":
        freq = "ME"

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    df = df[df[time_col].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=[time_col, "revenue"])

    out = (
        df.set_index(time_col)[value_col]
        .resample(freq)
        .sum()
        .rename("revenue")
        .reset_index()
        .sort_values(time_col)
    )
    return out


def top_entities(
    transactions: pd.DataFrame,
    *,
    entity_col: str,
    value_col: str = "total_spend",
    top_n: int = 20,
) -> pd.DataFrame:
    df = transactions.copy()
    if entity_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=[entity_col, value_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    out = (
        df.groupby(entity_col, as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
        .head(int(top_n))
    )
    return out


def basket_pairs(
    transactions: pd.DataFrame,
    *,
    order_col: str = "order_id",
    item_col: str = "product",
    min_pair_count: int = 10,
    max_orders: int = 50_000,
    max_items_per_order: int = 50,
) -> pd.DataFrame:
    """
    Simple market-basket insight: most common item pairs bought together.

    Notes:
    - Uses co-occurrence counts only (no association-rule metrics).
    - Includes pragmatic caps for large datasets.
    """
    df = transactions.copy()
    if order_col not in df.columns or item_col not in df.columns:
        return pd.DataFrame(columns=["item_a", "item_b", "pair_count"])

    df = df[[order_col, item_col]].dropna().copy()
    if df.empty:
        return pd.DataFrame(columns=["item_a", "item_b", "pair_count"])

    # Cap for interactive usage (Streamlit) to keep runtime predictable.
    df = df.drop_duplicates()
    if df[order_col].nunique(dropna=True) > int(max_orders):
        keep_orders = df[order_col].drop_duplicates().head(int(max_orders))
        df = df[df[order_col].isin(keep_orders)]

    counts: dict[tuple[str, str], int] = {}
    for _, items in df.groupby(order_col)[item_col]:
        unique_items = pd.unique(items.astype(str))
        if unique_items.size < 2:
            continue
        if unique_items.size > int(max_items_per_order):
            unique_items = unique_items[: int(max_items_per_order)]
        for a, b in combinations(sorted(unique_items), 2):
            counts[(a, b)] = counts.get((a, b), 0) + 1

    if not counts:
        return pd.DataFrame(columns=["item_a", "item_b", "pair_count"])

    out = (
        pd.DataFrame(
            [{"item_a": a, "item_b": b, "pair_count": c} for (a, b), c in counts.items()]
        )
        .sort_values("pair_count", ascending=False)
    )
    out = out[out["pair_count"] >= int(min_pair_count)].reset_index(drop=True)
    return out
