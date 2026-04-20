from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.churn_model import (  # noqa: E402
    choose_as_of_for_forward_churn,
    forward_churn_labels,
    predict_churn_risk,
    train_churn_model,
)
from src.data_preprocessing import (  # noqa: E402
    coerce_transactions_frame,
    default_paths,
    filter_time_window,
    generate_synthetic_funnel_columns,
    load_online_retail_excel,
    load_transactions,
)
from src.decision_engine import market_research_summary, recommend_actions  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    build_customer_features,
    build_customer_features_asof,
    select_feature_columns,
)
from src.funnel_analysis import bottleneck_stage, compute_funnel  # noqa: E402
from src.market_research import basket_pairs  # noqa: E402
from src.segmentation import name_segments, segment_customers  # noqa: E402


st.set_page_config(page_title="Data Analytics & Insights", layout="wide")

# Hide ONLY Deploy button
st.markdown("""
    <style>
    .stAppDeployButton {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)
st.title("Data Analytics & Insights Dashboard")
st.caption("Customer segmentation • Funnel analytics • Churn prediction • Market research • Decision engine")

paths = default_paths()


@st.cache_data(show_spinner=False)
def _load_data_from_path(path: str) -> pd.DataFrame:
    return load_transactions(path)


with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV/XLSX (optional)", type=["csv", "xlsx", "xls"])
    xlsx_sheet = None
    if uploaded is not None and uploaded.name.lower().endswith((".xlsx", ".xls")):
        try:
            xls = pd.ExcelFile(uploaded)
            xlsx_sheet = st.selectbox("XLSX sheet", xls.sheet_names, index=0)
        except Exception:
            xlsx_sheet = 0
    time_window = st.selectbox("Time window", ["all", "last_30d", "last_90d", "last_180d"], index=0)
    n_clusters = st.slider("Segments (k)", min_value=2, max_value=8, value=4)
    churn_horizon_days = st.slider("Churn horizon (days)", min_value=30, max_value=180, value=90, step=15)
    st.subheader("Funnel (Optional)")
    simulate_funnel = st.checkbox("Generate synthetic visits/cart (demo)", value=False)
    synthetic_avg_visits = st.slider("Avg visits per order", min_value=1, max_value=20, value=6)
    synthetic_cart_rate = st.slider("Cart rate", min_value=5, max_value=95, value=35) / 100.0
    synthetic_seed = st.number_input("Synthetic seed", min_value=0, max_value=1_000_000, value=42, step=1)

if uploaded is not None:
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        transactions = load_online_retail_excel(uploaded, sheet_name=xlsx_sheet or 0)
    else:
        transactions = coerce_transactions_frame(pd.read_csv(uploaded))
else:
    if paths.csv_path.exists():
        transactions = _load_data_from_path(str(paths.csv_path))
    elif paths.online_retail_xlsx_path.exists():
        transactions = load_online_retail_excel(str(paths.online_retail_xlsx_path), sheet_name=0)
    else:
        st.error("No default dataset found. Upload a CSV/XLSX to continue.")
        st.stop()

transactions = filter_time_window(transactions, window=time_window)

if transactions.empty:
    st.error("No rows available for the selected window.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rows", f"{len(transactions):,}")
with col2:
    st.metric("Customers", f"{transactions['customer_id'].nunique():,}" if "customer_id" in transactions.columns else "—")
with col3:
    if "order_id" in transactions.columns:
        st.metric("Orders", f"{transactions['order_id'].nunique():,}")
    elif "purchased" in transactions.columns:
        st.metric("Purchases", f"{int(transactions['purchased'].sum()):,}")
    else:
        st.metric("Orders", "—")
with col4:
    st.metric("Revenue", f"{transactions['total_spend'].sum():,.0f}" if "total_spend" in transactions.columns else "—")

tabs = st.tabs(["Market Research", "Funnel", "Segmentation", "Churn", "Decisions", "Data"])

with tabs[0]:
    st.subheader("Market research snapshot")
    summary = market_research_summary(transactions)

    left, right = st.columns(2)
    with left:
        if "revenue_by_category" in summary and not summary["revenue_by_category"].empty:
            st.plotly_chart(
                px.bar(summary["revenue_by_category"], x="category", y="total_spend", title="Revenue by category"),
                use_container_width=True,
            )
        if "revenue_over_time_month" in summary and not summary["revenue_over_time_month"].empty:
            st.plotly_chart(
                px.line(summary["revenue_over_time_month"], x="order_date", y="revenue", title="Revenue over time (monthly)"),
                use_container_width=True,
            )
    with right:
        if "revenue_by_product" in summary and not summary["revenue_by_product"].empty:
            st.plotly_chart(
                px.bar(summary["revenue_by_product"].head(10), x="product", y="total_spend", title="Top products (revenue)"),
                use_container_width=True,
            )
        elif "revenue_by_stock_code" in summary and not summary["revenue_by_stock_code"].empty:
            st.plotly_chart(
                px.bar(
                    summary["revenue_by_stock_code"].head(10),
                    x="stock_code",
                    y="total_spend",
                    title="Top products (revenue, stock code)",
                ),
                use_container_width=True,
            )

    cols = st.columns(4)
    for i, key in enumerate(["orders_by_channel", "orders_by_device", "orders_by_location", "orders_by_payment_method"]):
        if key in summary and not summary[key].empty:
            with cols[i]:
                st.plotly_chart(
               #     px.pie(summary[key], names=summary[key].columns[0], values="orders", title=key.replace("_", " ").title()),
                #    use_container_width=True,
                fig = px.pie(
    summary[key],
    names=summary[key].columns[0],
    values="orders",
)

# Clean labels (remove clutter)
fig.update_traces(
    textinfo="percent+label",   # remove small noisy numbers
    textposition="inside",      # keep labels inside slices
    insidetextorientation="radial"
)

# Center + fix layout
fig.update_layout(
    margin=dict(t=40, b=40, l=40, r=40),
    showlegend=True,
    legend=dict(
        orientation="v",
        x=1.05,   # push legend right
        y=0.5
    )
)

st.plotly_chart(fig, use_container_width=True)
                )
                shown = summary[key].sort_values("orders", ascending=False)
                st.dataframe(shown, use_container_width=True, hide_index=True)

    st.subheader("Frequently bought together (pairs)")
    pairs = basket_pairs(transactions, min_pair_count=10).head(30)
    if pairs.empty:
        st.info("Not enough orders/items to compute stable pairs in this window.")
    else:
        st.dataframe(pairs, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Funnel analytics")
    unit_col = "order_id" if "order_id" in transactions.columns else None
    funnel_source = transactions
    if simulate_funnel and unit_col is not None:
        funnel_source = generate_synthetic_funnel_columns(
            transactions,
            unit_col=unit_col,
            seed=int(synthetic_seed),
            avg_visits_per_unit=float(synthetic_avg_visits),
            cart_rate=float(synthetic_cart_rate),
        )

    funnel_overall = compute_funnel(funnel_source, unit_col=unit_col)
    st.dataframe(funnel_overall, use_container_width=True, hide_index=True)

    visited_ct = float(funnel_overall.get("visited_count", pd.Series([0])).iloc[0]) if not funnel_overall.empty else 0.0
    cart_ct = (
        float(funnel_overall.get("added_to_cart_count", pd.Series([0])).iloc[0]) if not funnel_overall.empty else 0.0
    )
    purchased_ct = (
        float(funnel_overall.get("purchased_count", pd.Series([0])).iloc[0]) if not funnel_overall.empty else 0.0
    )

    if not simulate_funnel and visited_ct == 0 and cart_ct == 0 and purchased_ct > 0:
        st.info(
            "This dataset appears to be purchase-only (no `visited` / `added_to_cart` signals). "
            "Funnel rates are not meaningful unless those columns exist."
        )
    else:
        stage = bottleneck_stage(funnel_overall)
        st.info(f"Primary bottleneck: `{stage}`")

    by_channel = (
        compute_funnel(funnel_source, groupby="channel", unit_col=unit_col) if "channel" in funnel_source.columns else pd.DataFrame()
    )
    by_device = (
        compute_funnel(funnel_source, groupby="device", unit_col=unit_col) if "device" in funnel_source.columns else pd.DataFrame()
    )

    c1, c2 = st.columns(2)
    with c1:
        if not by_channel.empty:
            st.plotly_chart(
                px.bar(by_channel, x="channel", y="visit_to_purchase_rate", title="Visit -> Purchase rate by channel"),
                use_container_width=True,
            )
    with c2:
        if not by_device.empty:
            st.plotly_chart(
                px.bar(by_device, x="device", y="visit_to_purchase_rate", title="Visit -> Purchase rate by device"),
                use_container_width=True,
            )

with tabs[2]:
    st.subheader("Customer segmentation")
    customer_features = build_customer_features(transactions)
    seg = segment_customers(customer_features, n_clusters=int(n_clusters))
    seg_named = name_segments(seg.customers)

    st.plotly_chart(
        px.scatter(
            seg_named,
            x="recency_days",
            y="revenue_total",
            color="segment_name" if "segment_name" in seg_named.columns else "segment_id",
            hover_data=["customer_id", "orders", "purchase_rate"],
            title="Segments (recency vs revenue)",
        ),
        use_container_width=True,
    )

    st.dataframe(seg.segment_profiles, use_container_width=True)

with tabs[3]:
    st.subheader("Churn prediction")
    if "churn" in transactions.columns:
        customer_features = build_customer_features(transactions)
        if "order_date" in transactions.columns:
            churn_labels = (
                transactions.sort_values("order_date")
                .groupby("customer_id", as_index=False)
                .tail(1)
                .set_index("customer_id")["churn"]
            )
        else:
            churn_labels = transactions.groupby("customer_id", as_index=False).tail(1).set_index("customer_id")["churn"]

        feature_cols = select_feature_columns(customer_features)
        if churn_labels.nunique() < 2:
            st.warning("Churn labels have a single class in this window; model training is skipped.")
            st.stop()

        model_result = train_churn_model(customer_features, churn_labels, feature_columns=feature_cols)
        churn_risk = predict_churn_risk(model_result, customer_features)
    else:
        if "order_date" not in transactions.columns:
            st.warning("No `churn` label and no `order_date` column; churn modeling is unavailable.")
            st.stop()

        max_time = pd.to_datetime(transactions["order_date"], errors="coerce").max()
        if pd.isna(max_time):
            st.warning("No valid `order_date` values; churn modeling is unavailable.")
            st.stop()

        as_of_train = choose_as_of_for_forward_churn(transactions, horizon_days=int(churn_horizon_days))
        train_features = build_customer_features_asof(transactions, as_of=as_of_train, lookback_days=180)
        if train_features.empty:
            st.warning("Not enough history to build churn features for training.")
            st.stop()

        labels = forward_churn_labels(
            transactions,
            as_of=as_of_train,
            horizon_days=int(churn_horizon_days),
            customer_ids=pd.Index(train_features["customer_id"].astype(str)),
        )
        feature_cols = select_feature_columns(train_features)

        if labels.nunique() < 2:
            st.warning("Forward churn labels have a single class; try adjusting the time window/horizon.")
            st.stop()

        model_result = train_churn_model(train_features, labels, feature_columns=feature_cols)
        customer_features = build_customer_features_asof(transactions, as_of=max_time, lookback_days=180)
        churn_risk = predict_churn_risk(model_result, customer_features)

    metric_cols = st.columns(min(5, len(model_result.metrics)))
    for i, (k, v) in enumerate(model_result.metrics.items()):
        metric_cols[i].metric(k.upper(), f"{v:.3f}")

    scored = (
        customer_features.merge(churn_risk.reset_index(), on="customer_id", how="left")
        .sort_values("churn_risk", ascending=False)
    )
    st.plotly_chart(px.histogram(scored, x="churn_risk", nbins=20, title="Churn risk distribution"), use_container_width=True)
    st.dataframe(scored.head(50), use_container_width=True)

with tabs[4]:
    st.subheader("Decision engine (actions)")
    customer_features = build_customer_features(transactions)
    seg_named = name_segments(segment_customers(customer_features, n_clusters=int(n_clusters)).customers)
    if "churn" in transactions.columns:
        if "order_date" in transactions.columns:
            churn_labels = (
                transactions.sort_values("order_date")
                .groupby("customer_id", as_index=False)
                .tail(1)
                .set_index("customer_id")["churn"]
            )
        else:
            churn_labels = transactions.groupby("customer_id", as_index=False).tail(1).set_index("customer_id")["churn"]

        feature_cols = select_feature_columns(customer_features)
        if churn_labels.nunique() >= 2:
            model_result = train_churn_model(customer_features, churn_labels, feature_columns=feature_cols)
            churn_risk = predict_churn_risk(model_result, customer_features)
            customers_scored = seg_named.merge(churn_risk.reset_index(), on="customer_id", how="left")
        else:
            customers_scored = seg_named.copy()
            customers_scored["churn_risk"] = 0.0
    else:
        if "order_date" in transactions.columns:
            max_time = pd.to_datetime(transactions["order_date"], errors="coerce").max()
        else:
            max_time = pd.NaT

        if pd.isna(max_time):
            customers_scored = seg_named.copy()
            customers_scored["churn_risk"] = 0.0
        else:
            as_of_train = choose_as_of_for_forward_churn(transactions, horizon_days=int(churn_horizon_days))
            train_features = build_customer_features_asof(transactions, as_of=as_of_train, lookback_days=180)
            labels = forward_churn_labels(
                transactions,
                as_of=as_of_train,
                horizon_days=int(churn_horizon_days),
                customer_ids=pd.Index(train_features["customer_id"].astype(str)),
            )
            feature_cols = select_feature_columns(train_features)

            if labels.nunique() >= 2 and not train_features.empty:
                model_result = train_churn_model(train_features, labels, feature_columns=feature_cols)
                current_features = build_customer_features_asof(transactions, as_of=max_time, lookback_days=180)
                churn_risk = predict_churn_risk(model_result, current_features)
                customers_scored = seg_named.merge(churn_risk.reset_index(), on="customer_id", how="left")
            else:
                customers_scored = seg_named.copy()
                customers_scored["churn_risk"] = 0.0

    decisions = recommend_actions(customers_scored).sort_values("decision_score", ascending=False)
    st.dataframe(
        decisions[["customer_id", "segment_name", "revenue_total", "recency_days", "churn_risk", "recommended_action"]],
        use_container_width=True,
    )

    top_actions = decisions["recommended_action"].value_counts().reset_index()
    top_actions.columns = ["action", "customers"]
    st.plotly_chart(px.bar(top_actions, x="action", y="customers", title="Recommended actions (count)"), use_container_width=True)

with tabs[5]:
    st.subheader("Raw data")
    st.dataframe(transactions.head(2000), use_container_width=True)
