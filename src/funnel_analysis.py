from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


FUNNEL_COLS = ("visited", "added_to_cart", "purchased")


def compute_funnel(
    transactions: pd.DataFrame,
    *,
    groupby: str | None = None,
    funnel_cols: Iterable[str] = FUNNEL_COLS,
    unit_col: str | None = None,
) -> pd.DataFrame:
    df = transactions.copy()
    for col in funnel_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0

    # Optional: compute funnel at a unit level (e.g. per order_id) so line-items don't inflate counts.
    if unit_col and unit_col in df.columns:
        keys: list[str] = [unit_col]
        if groupby and groupby in df.columns:
            keys.append(groupby)
        cols = list(funnel_cols)
        df = df[keys + cols].groupby(keys, as_index=False)[cols].max()

    if groupby and groupby in df.columns:
        grouped = df.groupby(groupby, as_index=False)
        out = grouped[list(funnel_cols)].sum(numeric_only=True)
        out.rename(columns={c: f"{c}_count" for c in funnel_cols}, inplace=True)
    else:
        sums = df[list(funnel_cols)].sum(numeric_only=True).to_dict()
        out = pd.DataFrame([{f"{k}_count": v for k, v in sums.items()}])

    visited = out["visited_count"].replace(0, np.nan) if "visited_count" in out.columns else np.nan
    added = out["added_to_cart_count"].replace(0, np.nan) if "added_to_cart_count" in out.columns else np.nan

    out["visit_to_cart_rate"] = (out["added_to_cart_count"] / visited).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["cart_to_purchase_rate"] = (out["purchased_count"] / added).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["visit_to_purchase_rate"] = (out["purchased_count"] / visited).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out


def bottleneck_stage(funnel_df: pd.DataFrame) -> str:
    if funnel_df.empty:
        return "unknown"
    overall = funnel_df.iloc[0]
    v2c = float(overall.get("visit_to_cart_rate", 0.0))
    c2p = float(overall.get("cart_to_purchase_rate", 0.0))
    return "visit_to_cart" if v2c < c2p else "cart_to_purchase"
