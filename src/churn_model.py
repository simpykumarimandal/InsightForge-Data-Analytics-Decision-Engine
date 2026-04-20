from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ChurnModelResult:
    model: LogisticRegression
    scaler: StandardScaler
    feature_columns: list[str]
    metrics: dict[str, float]
    metadata: dict[str, object] | None = None


def align_features(customer_features: pd.DataFrame, *, feature_columns: list[str]) -> pd.DataFrame:
    """
    Ensure `customer_features` contains all required feature columns.

    Missing columns are filled with 0. Extra columns are ignored.
    """
    out = customer_features.copy()
    for col in feature_columns:
        if col not in out.columns:
            out[col] = 0.0
    return out


def train_churn_model(
    customer_features: pd.DataFrame,
    churn_labels: pd.Series,
    *,
    feature_columns: list[str],
    random_state: int = 42,
) -> ChurnModelResult:
    features = align_features(customer_features, feature_columns=feature_columns)
    x = features[feature_columns].astype(float)
    y = churn_labels.astype(int).reindex(customer_features["customer_id"]).fillna(0).astype(int).values

    strat = y if len(set(y)) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=random_state, stratify=strat
    )

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    model.fit(x_train_s, y_train)

    proba = model.predict_proba(x_test_s)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }
    if len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))

    return ChurnModelResult(model=model, scaler=scaler, feature_columns=feature_columns, metrics=metrics)


def predict_churn_risk(result: ChurnModelResult, customer_features: pd.DataFrame) -> pd.Series:
    features = align_features(customer_features, feature_columns=result.feature_columns)
    x = features[result.feature_columns].astype(float)
    x_s = result.scaler.transform(x)
    proba = result.model.predict_proba(x_s)[:, 1]
    return pd.Series(proba, index=customer_features["customer_id"], name="churn_risk")


def forward_churn_labels(
    transactions: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    horizon_days: int = 90,
    customer_ids: pd.Index,
    customer_col: str = "customer_id",
    time_col: str = "order_date",
    purchase_col: str = "purchased",
) -> pd.Series:
    """
    Label churn forward-looking: 1 if NO purchase happens in (as_of, as_of+horizon], else 0.
    """
    if time_col not in transactions.columns:
        raise ValueError(f"Missing column: {time_col}")

    df = transactions.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[df[time_col].notna()].copy()

    window_end = as_of + pd.Timedelta(days=int(horizon_days))
    future = df[(df[time_col] > as_of) & (df[time_col] <= window_end)].copy()

    if purchase_col in future.columns:
        future[purchase_col] = pd.to_numeric(future[purchase_col], errors="coerce").fillna(0).astype(int)
        future = future[future[purchase_col] == 1]

    purchasers = (
        future[customer_col].dropna().astype(str).unique().tolist()
        if customer_col in future.columns
        else []
    )

    labels = pd.Series(1, index=customer_ids, name="churn")
    if purchasers:
        labels.loc[pd.Index(purchasers).intersection(customer_ids)] = 0
    return labels.astype(int)


def train_churn_model_time_split(
    *,
    train_features: pd.DataFrame,
    train_labels: pd.Series | np.ndarray,
    test_features: pd.DataFrame,
    test_labels: pd.Series | np.ndarray,
    feature_columns: list[str],
) -> ChurnModelResult:
    train_aligned = align_features(train_features, feature_columns=feature_columns)
    test_aligned = align_features(test_features, feature_columns=feature_columns)
    x_train = train_aligned[feature_columns].astype(float)
    x_test = test_aligned[feature_columns].astype(float)

    def _to_y(features: pd.DataFrame, labels: pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(labels, pd.Series):
            # If labels are row-aligned, use directly (supports repeated customer_ids).
            if len(labels) == len(features) and labels.index.equals(features.index):
                return labels.astype(int).values
            # Otherwise, treat labels as keyed by customer_id.
            if labels.index.is_unique and "customer_id" in features.columns:
                return labels.astype(int).reindex(features["customer_id"]).fillna(1).astype(int).values
            return labels.astype(int).to_numpy()
        arr = np.asarray(labels)
        if arr.shape[0] != len(features):
            raise ValueError("Labels length does not match features.")
        return arr.astype(int)

    y_train = _to_y(train_features, train_labels)
    y_test = _to_y(test_features, test_labels)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    model.fit(x_train_s, y_train)

    proba = model.predict_proba(x_test_s)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }
    if len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))

    return ChurnModelResult(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        metrics=metrics,
        metadata={"test_size": int(len(y_test))},
    )


def choose_as_of_for_forward_churn(
    transactions: pd.DataFrame,
    *,
    horizon_days: int = 90,
    time_col: str = "order_date",
) -> pd.Timestamp:
    if time_col not in transactions.columns:
        raise ValueError(f"Missing column: {time_col}")
    times = pd.to_datetime(transactions[time_col], errors="coerce").dropna()
    if times.empty:
        raise ValueError("No valid timestamps found.")
    max_time = times.max()
    return max_time - pd.Timedelta(days=int(horizon_days))
