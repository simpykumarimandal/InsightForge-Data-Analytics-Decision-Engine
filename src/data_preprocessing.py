from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal, overload

import numpy as np
import pandas as pd


DATE_COLUMNS = ("order_date", "last_purchase_date")
ONLINE_RETAIL_REQUIRED_COLS = (
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
)


@dataclass(frozen=True)
class DataPaths:
    root: Path
    csv_path: Path
    online_retail_xlsx_path: Path


def default_paths() -> DataPaths:
    root = Path(__file__).resolve().parents[1]
    return DataPaths(
        root=root,
        csv_path=root / "dataset" / "retail.csv",
        online_retail_xlsx_path=root / "dataset" / "online_retail_II.xlsx",
    )


def _ensure_recency_columns(
    df: pd.DataFrame,
    *,
    customer_col: str = "customer_id",
    time_col: str = "order_date",
    purchase_col: str = "purchased",
) -> pd.DataFrame:
    if customer_col not in df.columns or time_col not in df.columns:
        return df

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out[out[time_col].notna()].copy()
    if out.empty:
        return out

    max_time = out[time_col].max()
    if purchase_col in out.columns:
        out[purchase_col] = pd.to_numeric(out[purchase_col], errors="coerce").fillna(0).astype(int)
        purchase_rows = out[out[purchase_col] == 1].copy()
        if purchase_rows.empty:
            purchase_rows = out
    else:
        purchase_rows = out

    last_purchase = purchase_rows.groupby(customer_col, as_index=True)[time_col].max()
    recency_days = (max_time - last_purchase).dt.days.astype(float)

    if "last_purchase_date" not in out.columns:
        out = out.merge(
            last_purchase.rename("last_purchase_date").reset_index(),
            on=customer_col,
            how="left",
        )
    if "recency_days" not in out.columns:
        out = out.merge(
            recency_days.rename("recency_days").reset_index(),
            on=customer_col,
            how="left",
        )

    return out


def add_churn_label(
    transactions: pd.DataFrame,
    *,
    churn_threshold_days: int = 90,
    customer_col: str = "customer_id",
    recency_col: str = "recency_days",
) -> pd.DataFrame:
    """
    Adds/overwrites a simple churn label based on recency.

    churn = 1 if recency_days >= churn_threshold_days else 0
    """
    df = transactions.copy()
    if recency_col not in df.columns:
        df = _ensure_recency_columns(df, customer_col=customer_col)
    if recency_col not in df.columns:
        return df

    df[recency_col] = pd.to_numeric(df[recency_col], errors="coerce")
    df["churn"] = (df[recency_col].fillna(0) >= float(churn_threshold_days)).astype(int)
    return df


def is_online_retail_frame(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in ONLINE_RETAIL_REQUIRED_COLS)


def normalize_online_retail(
    df: pd.DataFrame,
    *,
    include_returns: bool = False,
    add_churn: bool = False,
    churn_threshold_days: int = 90,
) -> pd.DataFrame:
    """
    Normalize the Online Retail II schema into the project's standard transaction schema.
    """
    raw = df.copy()

    out = pd.DataFrame(
        {
            "order_id": raw["Invoice"].astype(str),
            "order_date": pd.to_datetime(raw["InvoiceDate"], errors="coerce"),
            "customer_id": raw["Customer ID"].astype("Int64").astype(str),
            "stock_code": raw["StockCode"].astype(str),
            "product": raw["Description"].astype(str),
            "quantity": pd.to_numeric(raw["Quantity"], errors="coerce"),
            "price": pd.to_numeric(raw["Price"], errors="coerce"),
            "location": raw["Country"].astype(str),
        }
    )

    out = out[out["order_date"].notna()].copy()
    out = out[out["customer_id"].notna() & (out["customer_id"] != "<NA>")].copy()

    out["total_spend"] = (out["quantity"].fillna(0) * out["price"].fillna(0)).astype(float)

    if not include_returns:
        out = out[(out["quantity"].fillna(0) > 0) & (out["price"].fillna(0) > 0)].copy()

    out["visited"] = 0
    out["added_to_cart"] = 0
    out["purchased"] = 1
    out["discount"] = 0.0

    out = _ensure_recency_columns(out)
    if add_churn:
        out = add_churn_label(out, churn_threshold_days=churn_threshold_days)
    return out


def load_transactions(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    df = pd.read_csv(path)

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ("visited", "added_to_cart", "purchased", "churn"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "recency_days" in df.columns:
        df["recency_days"] = pd.to_numeric(df["recency_days"], errors="coerce")

    if "total_spend" in df.columns:
        df["total_spend"] = pd.to_numeric(df["total_spend"], errors="coerce")

    return coerce_transactions_frame(df)


def coerce_transactions_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in DATE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    for col in ("visited", "added_to_cart", "purchased", "churn"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    for col in ("recency_days", "total_spend", "price", "quantity", "discount"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = _ensure_recency_columns(out)
    return out


@overload
def load_online_retail_excel(xlsx: str | Path, *, sheet_name: str | int = 0) -> pd.DataFrame: ...


@overload
def load_online_retail_excel(xlsx: IO[bytes], *, sheet_name: str | int = 0) -> pd.DataFrame: ...


def load_online_retail_excel(xlsx: str | Path | IO[bytes], *, sheet_name: str | int = 0) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name=sheet_name)
    return normalize_online_retail(df)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """
    Load either the project's synthetic CSV schema or Online Retail II (.xlsx).
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return load_online_retail_excel(p)
    return load_transactions(p)


def generate_synthetic_funnel_columns(
    transactions: pd.DataFrame,
    *,
    unit_col: str = "order_id",
    seed: int = 42,
    avg_visits_per_unit: float = 6.0,
    cart_rate: float = 0.35,
) -> pd.DataFrame:
    """
    Generate synthetic `visited` and `added_to_cart` signals for datasets that only contain purchases.

    Notes:
    - This is intended for demos/hackathons to enable funnel visualizations.
    - Generated counts are unit-level (e.g., per `order_id`) and are repeated on all rows for that unit.
    - Ensures: visited >= added_to_cart >= 1 for each unit (so purchase funnels don't break).
    """
    df = transactions.copy()
    if unit_col not in df.columns:
        return df

    cart_rate = float(np.clip(cart_rate, 0.0, 1.0))
    avg_visits_per_unit = float(max(avg_visits_per_unit, 1.0))

    units = df[[unit_col]].drop_duplicates().reset_index(drop=True)
    n = int(len(units))
    if n == 0:
        return df

    rng = np.random.default_rng(int(seed))

    # Poisson around the average, but at least 1 visit.
    lam = max(avg_visits_per_unit - 1.0, 0.0)
    visited = 1 + rng.poisson(lam=lam, size=n)
    visited = visited.astype(int)

    # Carts are a subset of visits, but at least 1 per unit (since a purchase exists).
    carts = rng.binomial(n=visited, p=cart_rate).astype(int)
    carts = np.maximum(carts, 1)
    visited = np.maximum(visited, carts)

    units["visited"] = visited
    units["added_to_cart"] = carts

    df = df.merge(units, on=unit_col, how="left", suffixes=("", "__synthetic"))
    if "visited__synthetic" in df.columns:
        df["visited"] = df["visited__synthetic"]
    if "added_to_cart__synthetic" in df.columns:
        df["added_to_cart"] = df["added_to_cart__synthetic"]
    df["visited"] = pd.to_numeric(df["visited"], errors="coerce").fillna(0).astype(int)
    df["added_to_cart"] = pd.to_numeric(df["added_to_cart"], errors="coerce").fillna(0).astype(int)
    df.drop(columns=[c for c in ("visited__synthetic", "added_to_cart__synthetic") if c in df.columns], inplace=True)

    if "purchased" not in df.columns:
        df["purchased"] = 1

    return df


def latest_customer_label(
    transactions: pd.DataFrame,
    label_col: str = "churn",
    customer_col: str = "customer_id",
    time_col: str = "order_date",
) -> pd.Series:
    if customer_col not in transactions.columns:
        raise ValueError(f"Missing column: {customer_col}")
    if label_col not in transactions.columns:
        raise ValueError(f"Missing column: {label_col}")

    if time_col in transactions.columns:
        ordered = transactions.sort_values(time_col)
        latest = ordered.groupby(customer_col, as_index=False).tail(1)
    else:
        latest = transactions.groupby(customer_col, as_index=False).tail(1)

    return latest.set_index(customer_col)[label_col]


def filter_time_window(
    transactions: pd.DataFrame,
    *,
    window: Literal["all", "last_30d", "last_90d", "last_180d"] = "all",
    time_col: str = "order_date",
) -> pd.DataFrame:
    if window == "all" or time_col not in transactions.columns:
        return transactions

    max_time = pd.to_datetime(transactions[time_col], errors="coerce").max()
    if pd.isna(max_time):
        return transactions

    days = {"last_30d": 30, "last_90d": 90, "last_180d": 180}[window]
    cutoff = max_time - pd.Timedelta(days=days)
    return transactions[transactions[time_col] >= cutoff].copy()
