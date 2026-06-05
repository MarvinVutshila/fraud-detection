#this is app.py

import time
import warnings
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Suppress harmless deprecation warnings from scikit-learn (if any)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fraud Detection System | Bank Grade",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="auto",
)

# ============================================================
# CUSTOM CSS (theme‑aware, no emojis, professional)
# ============================================================

st.markdown("""
<style>
/* Base – let Streamlit theme decide background */
.main {
    background-color: transparent;
}

/* Risk level colours (visible in both themes) */
.risk-low {
    color: #16a34a;
    font-weight: 600;
}
.risk-medium {
    color: #ea580c;
    font-weight: 600;
}
.risk-high {
    color: #dc2626;
    font-weight: 600;
}
.risk-critical {
    color: #7c3aed;
    font-weight: 600;
}

/* Info and alert boxes – no emojis, clean borders */
.info-box, .alert-box {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    border-left: 4px solid;
}
.info-box {
    background-color: rgba(2, 136, 199, 0.1);
    border-left-color: #0284c7;
}
.alert-box {
    background-color: rgba(220, 38, 38, 0.1);
    border-left-color: #dc2626;
}

/* Data table styling */
.dataframe-container {
    border-radius: 0.5rem;
    overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIG
# ============================================================

API_URL = st.sidebar.text_input(
    "API Endpoint",
    value="http://localhost:8000"
)

API_KEY = st.sidebar.text_input(
    "API Key",
    type="password"
)

HEADERS = {"X-API-Key": API_KEY}

MAX_ALLOWED_AMOUNT = 25691.16   # from training data

# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_results" not in st.session_state:
    st.session_state.last_results = None

# ============================================================
# API HELPERS
# ============================================================

def batch_predict(payload):
    response = requests.post(
        f"{API_URL}/predict/batch",
        json={"transactions": payload},
        headers=HEADERS,
        timeout=60
    )
    response.raise_for_status()
    return response.json()

def get_model_info():
    response = requests.get(
        f"{API_URL}/model/info",
        headers=HEADERS,
        timeout=30
    )
    response.raise_for_status()
    return response.json()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Fraud Detection System")
page = st.sidebar.radio(
    "Navigation",
    ["Batch Analysis", "Analytics Dashboard", "Model Overview", "History"]
)

# ============================================================
# BATCH ANALYSIS PAGE
# ============================================================

if page == "Batch Analysis":

    st.header("Batch Transaction Analysis")

    # Info boxes (no emojis)
    st.markdown(
        f"""
        <div class="info-box">
        <strong>Workflow</strong><br>
        Upload a CSV file containing transaction data. The system validates amounts,
        sends the data to the fraud detection API, and returns a detailed risk report.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="info-box">
        <strong>CSV format</strong><br>
        Required columns: <code>Time, Amount, V1, V2, ..., V28</code><br>
        Optional: <code>transaction_id</code><br>
        Maximum amount per transaction: <strong>${MAX_ALLOWED_AMOUNT:,.2f}</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Select a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview (first 10 rows)")
        st.dataframe(df.head(10), width='stretch')

        col_info = list(df.columns)
        st.caption(f"Detected columns: {', '.join(col_info)}")

        if st.button("Run Analysis", type="primary"):
            try:
                transactions = df.to_dict(orient="records")

                # --- Validation ---
                required = ["Time", "Amount"]
                missing = [c for c in required if c not in df.columns]
                if missing:
                    st.error(f"Missing required columns: {', '.join(missing)}")
                    st.stop()

                invalid_rows = []
                for idx, tx in enumerate(transactions):
                    amt = tx.get("Amount", 0)
                    if pd.isna(amt):
                        amt = 0
                    if amt > MAX_ALLOWED_AMOUNT:
                        invalid_rows.append((idx, amt))

                if invalid_rows:
                    rows_str = ", ".join([f"row {idx} (${amt:,.2f})" for idx, amt in invalid_rows])
                    st.markdown(
                        f"""
                        <div class="alert-box">
                        <strong>Batch rejected</strong><br>
                        The following transactions exceed the maximum allowed amount (${MAX_ALLOWED_AMOUNT:,.2f}):
                        {rows_str}<br>
                        Please correct these amounts and re‑upload.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.stop()

                # Fill missing V1..V28 with 0.0
                for tx in transactions:
                    for i in range(1, 29):
                        col = f"V{i}"
                        if col not in tx or pd.isna(tx[col]):
                            tx[col] = 0.0

                # API call
                with st.spinner("Scoring transactions..."):
                    result = batch_predict(transactions)

                results_df = pd.DataFrame(result["results"])
                st.session_state.last_results = results_df.copy()

                # Add to history
                for idx, row in results_df.iterrows():
                    tx_id = row.get("transaction_id")
                    if pd.isna(tx_id) or tx_id is None:
                        tx_id = f"batch_{idx}_{int(time.time())}"
                    amt = transactions[idx].get("Amount", 0.0)
                    prob = row["fraud_probability"]
                    risk = row["risk_level"]
                    is_fraud = row["is_fraud"]
                    decision = "BLOCK" if is_fraud else "APPROVE"

                    st.session_state.history.append({
                        "transaction_id": tx_id,
                        "amount": amt,
                        "probability": prob,
                        "risk": risk,
                        "decision": decision,
                        "timestamp": datetime.now(),
                        "source": "batch"
                    })

                # --- Display results ---
                st.success("Analysis completed successfully.")

                total = len(results_df)
                frauds = results_df["is_fraud"].sum()
                fraud_rate = frauds / total * 100
                total_amount = sum(transactions[i].get("Amount", 0) for i in range(total))

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Transactions", total)
                col2.metric("Fraudulent", int(frauds), delta=f"{fraud_rate:.1f}%")
                col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                col4.metric("Total Amount", f"${total_amount:,.2f}")

                # Interactive results table
                st.subheader("Detailed Risk Assessment")
                display_cols = ["transaction_id", "fraud_probability", "risk_level", "is_fraud"]
                available_cols = [c for c in display_cols if c in results_df.columns]

                # Add a filter
                show_only_fraud = st.checkbox("Show only fraudulent transactions", value=False)
                if show_only_fraud:
                    filtered_df = results_df[results_df["is_fraud"] == True]
                else:
                    filtered_df = results_df

                # Sort by probability descending
                filtered_df = filtered_df.sort_values("fraud_probability", ascending=False)

                st.dataframe(
                    filtered_df[available_cols],
                    width='stretch',
                    column_config={
                        "fraud_probability": st.column_config.ProgressColumn(
                            "Fraud Probability",
                            format="%.4f",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                        "risk_level": st.column_config.TextColumn("Risk Level"),
                        "is_fraud": st.column_config.CheckboxColumn("Fraud"),
                    }
                )

                # Export filtered results
                csv_export = filtered_df[available_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results (CSV)",
                    csv_export,
                    f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )

                # Visualisations
                st.subheader("Risk Distribution")
                fig_hist = px.histogram(
                    results_df,
                    x="fraud_probability",
                    nbins=30,
                    color="is_fraud",
                    title="Fraud Probability Distribution",
                    labels={"fraud_probability": "Probability", "count": "Number of transactions"}
                )
                fig_hist.update_layout(bargap=0.05)
                st.plotly_chart(fig_hist, width='stretch')

                st.subheader("Risk Level Breakdown")
                risk_counts = results_df["risk_level"].value_counts().reset_index()
                risk_counts.columns = ["Risk Level", "Count"]
                fig_pie = px.pie(risk_counts, names="Risk Level", values="Count", title="Transactions by Risk Level")
                st.plotly_chart(fig_pie, width='stretch')

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400:
                    detail = e.response.json().get("detail", str(e))
                    st.markdown(
                        f"""
                        <div class="alert-box">
                        <strong>API rejected batch</strong><br>
                        {detail}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.error(f"API error ({e.response.status_code}): {e}")
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# ============================================================
# ANALYTICS DASHBOARD (from history)
# ============================================================

elif page == "Analytics Dashboard":

    st.header("Historical Fraud Analytics")

    if not st.session_state.history:
        st.info("No data available. Upload a batch file first.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)

        # Overall KPIs
        total_scored = len(df_hist)
        total_frauds = df_hist[df_hist["decision"] == "BLOCK"].shape[0]
        avg_prob = df_hist["probability"].mean()
        total_amount = df_hist["amount"].sum()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Scored", total_scored)
        k2.metric("Blocked Transactions", total_frauds, delta=f"{total_frauds/total_scored*100:.1f}%")
        k3.metric("Average Risk Score", f"{avg_prob:.4f}")
        k4.metric("Total Amount", f"${total_amount:,.2f}")

        st.subheader("Risk Score Over Time")
        if "timestamp" in df_hist.columns:
            df_ts = df_hist.sort_values("timestamp")
            fig_time = px.line(
                df_ts,
                x="timestamp",
                y="probability",
                color="source" if "source" in df_hist.columns else None,
                title="Fraud Probability Evolution",
                labels={"probability": "Fraud Probability", "timestamp": "Analysis Time"}
            )
            st.plotly_chart(fig_time, width='stretch')

        st.subheader("Risk Level Composition")
        risk_dist = df_hist["risk"].value_counts().reset_index()
        risk_dist.columns = ["Risk Level", "Count"]
        fig_risk = px.pie(risk_dist, names="Risk Level", values="Count", title="Historical Risk Distribution")
        st.plotly_chart(fig_risk, width='stretch')

        st.subheader("Decision Breakdown")
        dec_dist = df_hist["decision"].value_counts().reset_index()
        dec_dist.columns = ["Decision", "Count"]
        fig_dec = px.bar(dec_dist, x="Decision", y="Count", color="Decision", title="Approved vs Blocked")
        st.plotly_chart(fig_dec, width='stretch')

        if "amount" in df_hist.columns:
            st.subheader("Amount Distribution by Decision")
            fig_box = px.box(df_hist, x="decision", y="amount", log_y=True, title="Transaction Amounts (log scale)")
            st.plotly_chart(fig_box, width='stretch')

# ============================================================
# MODEL OVERVIEW
# ============================================================

elif page == "Model Overview":

    st.header("Model Information")

    try:
        info = get_model_info()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Algorithm", info["model_type"])
        c2.metric("Features", info["n_features"])
        c3.metric("Decision Threshold", f"{info['threshold']:.4f}")
        c4.metric("Max Amount Limit", f"${info.get('max_allowed_amount', MAX_ALLOWED_AMOUNT):,.2f}")

        with st.expander("Feature List (click to expand)"):
            st.json(info["feature_names"])

        with st.expander("Full Model Metadata"):
            st.json(info)

    except Exception as e:
        st.error(f"Unable to retrieve model information: {str(e)}")

# ============================================================
# HISTORY
# ============================================================

elif page == "History":

    st.header("Prediction History")

    if not st.session_state.history:
        st.info("No history available. Perform a batch analysis first.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)

        # Search / filter
        search = st.text_input("Search by Transaction ID (partial match)")
        if search:
            mask = df_hist["transaction_id"].str.contains(search, case=False, na=False)
            df_hist = df_hist[mask]

        st.dataframe(
            df_hist.sort_values("timestamp", ascending=False),
            width='stretch',
            column_config={
                "probability": st.column_config.NumberColumn("Fraud Probability", format="%.6f"),
                "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
                "timestamp": st.column_config.DatetimeColumn("Analysis Time"),
            }
        )

        csv_all = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download full history (CSV)",
            csv_all,
            f"fraud_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()