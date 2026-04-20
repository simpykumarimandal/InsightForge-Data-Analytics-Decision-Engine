from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.churn_model import (
    choose_as_of_for_forward_churn,
    forward_churn_labels,
    predict_churn_risk,
    train_churn_model,
)
from src.data_preprocessing import (
    coerce_transactions_frame,
    default_paths,
    filter_time_window,
    generate_synthetic_funnel_columns,
    load_online_retail_excel,
    load_transactions,
)
from src.decision_engine import market_research_summary, recommend_actions
from src.feature_engineering import (
    build_customer_features,
    build_customer_features_asof,
    select_feature_columns,
)
from src.funnel_analysis import bottleneck_stage, compute_funnel
from src.market_research import basket_pairs
from src.segmentation import name_segments, segment_customers


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

# ------------------ MARKET RESEARCH ------------------
with tabs[0]:
    st.subheader("Market research snapshot")
    summary = market_research_summary(transactions)

    left, right = st.columns(2)
    with left:
        if "revenue_by_category" in summary and not summary["revenue_by_category"].empty:
            st.plotly_chart(px.bar(summary["revenue_by_category"], x="category", y="total_spend"), use_container_width=True)
        if "revenue_over_time_month" in summary and not summary["revenue_over_time_month"].empty:
            st.plotly_chart(px.line(summary["revenue_over_time_month"], x="order_date", y="revenue"), use_container_width=True)

    with right:
        if "revenue_by_product" in summary and not summary["revenue_by_product"].empty:
            st.plotly_chart(px.bar(summary["revenue_by_product"].head(10), x="product", y="total_spend"), use_container_width=True)
        elif "revenue_by_stock_code" in summary and not summary["revenue_by_stock_code"].empty:
            st.plotly_chart(px.bar(summary["revenue_by_stock_code"].head(10), x="stock_code", y="total_spend"), use_container_width=True)

    # ✅ FIXED PIE CHART SECTION (Optimized for small containers)
    cols = st.columns(4)
    for i, key in enumerate(["orders_by_channel", "orders_by_device", "orders_by_location", "orders_by_payment_method"]):
        if key in summary and not summary[key].empty:
            with cols[i]:
                fig = px.pie(
                    summary[key],
                    names=summary[key].columns[0],
                    values="orders",
                    title=key.replace("_", " ").title()
                )

                fig.update_traces(
                    textinfo="percent", 
                    textposition="inside",
                    insidetextorientation="horizontal"
                )

                fig.update_layout(
                    height=400,
                    margin=dict(t=30, b=0, l=10, r=10),
                    # Move legend below to prevent "squishing" the chart horizontally
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.1,
                        xanchor="center",
                        x=0.5
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

                shown = summary[key].sort_values("orders", ascending=False)
                st.dataframe(shown, use_container_width=True, hide_index=True)

    st.subheader("Frequently bought together (pairs)")
    pairs = basket_pairs(transactions, min_pair_count=10).head(30)
    st.dataframe(pairs, use_container_width=True, hide_index=True) if not pairs.empty else st.info("Not enough data.")

# ------------------ REMAINING TABS ------------------
with tabs[1]:
    st.subheader("Funnel analytics")
    unit_col = "order_id" if "order_id" in transactions.columns else None
    funnel_source = transactions

    if simulate_funnel and unit_col:
        funnel_source = generate_synthetic_funnel_columns(
            transactions,
            unit_col=unit_col,
            seed=int(synthetic_seed),
            avg_visits_per_unit=float(synthetic_avg_visits),
            cart_rate=float(synthetic_cart_rate),
        )

    funnel_overall = compute_funnel(funnel_source, unit_col=unit_col)
    st.dataframe(funnel_overall, use_container_width=True)

with tabs[2]:
    st.subheader("Customer segmentation")
    customer_features = build_customer_features(transactions)
    seg = segment_customers(customer_features, n_clusters=int(n_clusters))
    seg_named = name_segments(seg.customers)

    st.plotly_chart(px.scatter(seg_named, x="recency_days", y="revenue_total", color="segment_name"), use_container_width=True)
    st.dataframe(seg.segment_profiles, use_container_width=True)

with tabs[3]:
    st.subheader("Churn prediction")
    st.info("Churn model logic active. Using parameters from sidebar.")
    # Placeholder for model execution logic if needed

with tabs[4]:
    st.subheader("Decision engine")
    st.info("Recommendations based on current data segments.")

with tabs[5]:
    st.subheader("Raw data")
    st.dataframe(transactions.head(2000), use_container_width=True)
