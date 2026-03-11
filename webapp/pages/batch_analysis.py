"""Batch analysis page — upload CSV, run bulk predictions, download results."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import components.header as header
from api.client import ApiError, post_predict_batch
from api.models import BatchPredictResponse
from components import charts
from config import ALL_FEATURES, BATCH_MAX_ROWS, V_FEATURES


def render() -> None:
    """Render the batch analysis page."""
    header.render(
        "📦 Batch Analysis",
        f"Upload a CSV with V1–V28 + Amount columns (max {BATCH_MAX_ROWS:,} rows)",
    )

    tab_upload, tab_generate = st.tabs(["📁 Upload CSV", "🎲 Generate Random Data"])

    with tab_upload:
        _tab_upload()

    with tab_generate:
        _tab_generate()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def _tab_upload() -> None:
    """CSV upload tab."""
    uploaded = st.file_uploader(
        "Drop your CSV here",
        type=["csv"],
        help=f"Required columns: V1–V28, Amount. Max {BATCH_MAX_ROWS:,} rows.",
    )
    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not parse CSV: {exc}")
        return

    st.markdown(f"**{len(df):,} rows loaded** — preview:")
    st.dataframe(df.head(5), use_container_width=True)
    _validate_and_run(df)


def _tab_generate() -> None:
    """Random data generator tab for demo purposes."""
    import numpy as np

    n_rows = st.slider("Number of random transactions", 10, 500, 50)
    fraud_pct = st.slider("Injected fraud % (approximate)", 0, 50, 5)

    if st.button("🎲 Generate"):
        rng = np.random.default_rng(42)
        df_random = pd.DataFrame(
            rng.standard_normal((n_rows, len(V_FEATURES))),
            columns=V_FEATURES,
        )
        df_random["Amount"] = rng.uniform(1.0, 5000.0, n_rows).round(2)

        # Inject synthetic fraud-like rows (extreme V1, V14 values)
        fraud_mask = rng.random(n_rows) < (fraud_pct / 100)
        df_random.loc[fraud_mask, "V1"] = rng.uniform(-5, -3, fraud_mask.sum())
        df_random.loc[fraud_mask, "V14"] = rng.uniform(-5, -3, fraud_mask.sum())
        df_random.loc[fraud_mask, "Amount"] = rng.uniform(1.0, 5.0, fraud_mask.sum())

        st.session_state["batch_generated"] = df_random
        st.rerun()

    if "batch_generated" in st.session_state:
        df = st.session_state["batch_generated"]
        st.markdown(f"**{len(df):,} rows generated** — preview:")
        st.dataframe(df.head(5), use_container_width=True)
        _validate_and_run(df)


# ---------------------------------------------------------------------------
# Shared logic
# ---------------------------------------------------------------------------
def _validate_and_run(df: pd.DataFrame) -> None:
    """Validate columns, run predictions, display results."""
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    df_clean = df[ALL_FEATURES].head(BATCH_MAX_ROWS)

    if len(df) > BATCH_MAX_ROWS:
        st.warning(
            f"Dataset has {len(df):,} rows — only the first {BATCH_MAX_ROWS:,} will be sent.",
            icon="⚠️",
        )

    if not st.button("🚀 Run Batch Prediction", key=f"run_{id(df)}"):
        return

    with st.spinner(f"Analyzing {len(df_clean):,} transactions…"):
        try:
            result = post_predict_batch(df_clean.to_dict(orient="records"))
        except ApiError as exc:
            if exc.status_code == 503:
                st.warning("No model loaded. Run the Airflow DAG first.", icon="⚠️")
            else:
                st.error(f"API error: {exc}")
            return

    _render_results(df_clean, result)


def _render_results(df_input: pd.DataFrame, result: BatchPredictResponse) -> None:
    """Display summary KPIs, charts, detail table, and download button."""
    st.markdown("---")
    st.markdown("### Results")

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Transactions", f"{result.total:,}")
    with k2:
        st.metric("Fraud Detected", f"{result.fraud_count:,}")
    with k3:
        st.metric("Normal", f"{result.total - result.fraud_count:,}")
    with k4:
        st.metric("Fraud Rate", f"{result.fraud_rate:.2%}")

    # Charts row
    risk_counts: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "UNKNOWN": 0}
    probabilities: list[float] = []
    for p in result.predictions:
        risk_counts[p.risk_level] = risk_counts.get(p.risk_level, 0) + 1
        if p.fraud_probability is not None:
            probabilities.append(p.fraud_probability)

    normal_count = result.total - result.fraud_count

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            charts.fraud_donut(result.fraud_count, normal_count),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(charts.risk_bar(risk_counts), use_container_width=True)

    if probabilities:
        st.plotly_chart(
            charts.probability_histogram(probabilities), use_container_width=True
        )

    # Detail table
    st.markdown("### Transaction-level Results")
    df_out = df_input[["Amount"]].copy()
    df_out["is_fraud"] = [p.is_fraud for p in result.predictions]
    df_out["fraud_probability"] = [p.fraud_probability for p in result.predictions]
    df_out["risk_level"] = [p.risk_level for p in result.predictions]
    df_out["label"] = [p.prediction_label for p in result.predictions]

    st.dataframe(df_out, use_container_width=True)

    # Download
    csv_bytes = df_out.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Results CSV",
        data=csv_bytes,
        file_name="fraudguard_batch_results.csv",
        mime="text/csv",
    )
