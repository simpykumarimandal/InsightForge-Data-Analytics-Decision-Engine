from __future__ import annotations

import sys
from pathlib import Path
import random
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
from src.funnel_analysis import compute_funnel
from src.market_research import basket_pairs
from src.segmentation import name_segments, segment_customers

# --- APP CONFIG ---
st.set_page_config(page_title="Data Analytics & Insights", layout="wide")

st.markdown("""
    <style>
    .stAppDeployButton { display: none; }
    </style>
""", unsafe_allow_html=True)

st.title("Data Analytics & Insights Dashboard")
st.caption("Customer segmentation • Funnel analytics • Churn prediction • Market research • Decision engine")

# --- DATA LOADING & SMART SAMPLING ---
@st.cache_data(show_spinner=False)
def load_and_fix_data(uploaded_file, paths, time_window):
    """Loads, renames columns, and samples customers to prevent memory/data errors."""
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            df = load_online_retail_excel(uploaded_file, sheet_name=0)
        else:
            df = pd.read_csv(uploaded_file)
    else:
        # Load default
        if paths.csv_path.exists():
            df = load_transactions(str(paths.csv_path))
        else:
            return None

    # 1. COLUMN MAPPING (Matches your Excel: InvoiceDate, Price, Customer ID)
    mapping = {
        'Customer ID': 'customer_id',
        'Price': 'total_spend',
        'InvoiceDate': 'order_date',
        'Invoice': 'order_id',
        'Country': 'location'
    }
    df = df.rename(columns=mapping)
    
    # Ensure date format
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # 2. SMART SAMPLING (Keep 100% history for 1,000 customers)
    # This prevents the "Not enough data" error and speeds up GitHub/Cloud
    if 'customer_id' in df.columns:
        unique_ids = df['customer_id'].dropna().unique()
        if len(unique_ids) > 1000:
            random.seed(42)
            subset = random.sample(list(unique_ids), 1000)
            df = df[df['customer_id'].isin(subset)]

    # 3. Apply Time Filter
    df = filter_time_window(df, window=time_window)
    return df

# --- SIDEBAR ---
paths = default_paths()
with st.sidebar:
    st.header("Data Configuration")
    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"])
    time_window = st.selectbox("Time window", ["all", "last_30d", "last_90d", "last_180d"], index=0)
    n_clusters = st.slider("Segments (k)", 2, 8, 4)
    churn_horizon = st.slider("Churn horizon (days)", 30, 180, 90, step=15)

transactions = load_and_fix_data(uploaded, paths, time_window)

if transactions is None or transactions.empty:
    st.error("No data found. Please upload a file.")
    st.stop()

# --- TOP METRICS ---
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Rows", f"{len(transactions):,}")
with col2: st.metric("Customers", f"{transactions['customer_id'].nunique():,}")
with col3: st.metric("Orders", f"{transactions['order_id'].nunique():,}" if 'order_id' in transactions.columns else "—")
with col4: st.metric("Revenue", f"{transactions['total_spend'].sum():,.0f}")

tabs = st.tabs(["Market Research", "Funnel", "Segmentation", "Churn", "Decisions", "Data"])

# 1. MARKET RESEARCH
with tabs[0]:
    st.subheader("Market research snapshot")
    summary = market_research_summary(transactions)
    
    # Pie charts with the Legend Fix
    cols = st.columns(4)
    for i, key in enumerate(["orders_by_channel", "orders_by_device", "orders_by_location", "orders_by_payment_method"]):
        if key in summary and not summary[key].empty:
            with cols[i]:
                fig = px.pie(summary[key], names=summary[key].columns[0], values="orders", title=key.replace("_", " ").title())
                fig.update_layout(height=400, margin=dict(t=30, b=0, l=10, r=10), legend=dict(orientation="h", y=-0.1))
                st.plotly_chart(fig, use_container_width=True)

# 2. FUNNEL
with tabs[1]:
    st.subheader("Funnel analytics")
    unit_col = "order_id" if "order_id" in transactions.columns else None
    st.dataframe(compute_funnel(transactions, unit_col=unit_col), use_container_width=True)

# 3. SEGMENTATION
with tabs[2]:
    st.subheader("Customer segmentation")
    customer_features = build_customer_features(transactions)
    seg = segment_customers(customer_features, n_clusters=int(n_clusters))
    seg_named = name_segments(seg.customers)
    st.plotly_chart(px.scatter(seg_named, x="recency_days", y="revenue_total", color="segment_name"), use_container_width=True)

# 4. CHURN PREDICTION
# ------------------ TAB 3: CHURN PREDICTION ------------------
with tabs[3]:
    st.subheader("Churn Prediction")

    if "order_date" not in transactions.columns:
        st.warning("No `order_date` column; churn modeling unavailable.")
    else:
        # 1. Prepare training data
        max_time = transactions["order_date"].max()
        as_of_train = choose_as_of_for_forward_churn(transactions, horizon_days=int(churn_horizon))
        train_feats = build_customer_features_asof(transactions, as_of=as_of_train, lookback_days=180)

        # 2. Generate Labels
        labels = forward_churn_labels(
            transactions, 
            as_of=as_of_train, 
            horizon_days=int(churn_horizon), 
            customer_ids=pd.Index(train_feats["customer_id"].astype(str))
        )

        # 3. SAFETY CHECK: Ensure we have at least 2 classes (Churn vs No Churn)
        if labels.nunique() < 2:
            st.error("⚠️ **Training Failed: Class Imbalance.**")
            st.info("The current data sample only contains one outcome (everyone stayed). Try increasing the 'Time Window' to 'all' or increasing the 'Churn Horizon' in the sidebar.")
        else:
            with st.spinner("Training churn model..."):
                # 4. Train and Predict
                feature_cols = select_feature_columns(train_feats)
                model_result = train_churn_model(train_feats, labels, feature_columns=feature_cols)
                
                current_feats = build_customer_features_asof(transactions, as_of=max_time, lookback_days=180)
                churn_risk_df = predict_churn_risk(model_result, current_feats)
                
                # 5. Visualization
                scored = current_feats.merge(churn_risk_df.reset_index(), on="customer_id", how="left")
                st.plotly_chart(px.histogram(scored, x="churn_risk", title="Churn Risk Probability Distribution"), use_container_width=True)
                st.dataframe(scored[['customer_id', 'churn_risk']].sort_values("churn_risk", ascending=False).head(50), use_container_width=True)

# ------------------ TAB 4: DECISION ENGINE ------------------
with tabs[4]:
    st.subheader("Strategic Actions")
    
    # Ensure churn_risk_df exists from the previous tab
    if 'churn_risk_df' in locals() and 'seg_named' in locals():
        customers_scored = seg_named.merge(churn_risk_df.reset_index(), on="customer_id", how="left")
        decisions = recommend_actions(customers_scored)
        
        st.write("### Recommended Marketing Actions")
        st.dataframe(
            decisions[["customer_id", "segment_name", "churn_risk", "recommended_action"]].sort_values("churn_risk", ascending=False),
            use_container_width=True
        )
    else:
        st.info("Please run the Segmentation and Churn tabs first to generate recommendations.")

with tabs[5]:
    st.subheader("Raw data (Sampled)")
    st.dataframe(transactions.head(1000), use_container_width=True)
