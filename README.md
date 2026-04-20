## Data Analytics & Insights (Hackathon)

This project builds engines that turn raw e-commerce data into decisions:

- **Customer Segmentation** (K-Means clustering on customer-level features)
- **Funnel Analytics** (Visited -> Added to Cart -> Purchased, with breakdowns)
- **Churn Prediction** (Logistic regression churn-risk scoring)
- **Market Research** (category/product/channel/device/location/payment insights)
- **Decision Models** (rule-based recommendations combining segment + churn risk)

### Run locally

1. Create/activate a virtualenv, then install deps:

```bash
pip install -r requirements.txt
```

2. Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

If will load `dataset/online_retail_II.xlsx`.
You can also upload your own CSV/XLSX in the sidebar.

### Deploy (get a public URL)

You'll get a URL only after deploying to a hosting provider.

**Option A: Streamlit Community Cloud (fastest)**

1. Push this repo to GitHub.
2. Go to Streamlit Community Cloud and create a new app.
3. Set:
   - **Main file path**: `streamlit_app.py`
   - **Python deps**: `requirements.txt`
4. Deploy -> Streamlit will give you a URL like:
   - `https://<your-app-name>.streamlit.app`

**Option B: Render**

1. Push to GitHub.
2. Create a new **Web Service** (Python).
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
5. Deploy -> Render provides the URL.
