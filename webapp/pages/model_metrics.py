"""Model metrics page — production model evaluation and comparison charts."""

from __future__ import annotations

import streamlit as st

import components.header as header
from api.client import ApiError, get_model_metrics
from api.models import ModelMetricsResponse
from components import charts


def render() -> None:
    """Render the model metrics page."""
    header.render("📊 Model Metrics", "Evaluation metrics for the production model")

    try:
        metrics = get_model_metrics()
    except ApiError as exc:
        _render_no_metrics(exc)
        return

    _render_model_header(metrics)
    st.markdown("<br>", unsafe_allow_html=True)
    _render_kpi_row(metrics)
    st.markdown("<br>", unsafe_allow_html=True)
    _render_charts(metrics)
    st.markdown("---")
    _render_metric_explainer()


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _render_no_metrics(exc: ApiError) -> None:
    """Shown when metrics cannot be fetched."""
    if exc.status_code == 503:
        st.warning(
            "No model in Production stage yet. "
            "Trigger the Airflow `fraud_detection_pipeline` DAG and wait for it to complete.",
            icon="⚠️",
        )
    else:
        st.error(f"Could not fetch metrics: {exc}")

    st.markdown(
        "Once the pipeline completes, revisit this page to see:\n"
        "- Precision, Recall, F1, ROC-AUC, AUC-PR\n"
        "- Radar and bar charts\n\n"
        "**Links:** "
        "[Airflow →](http://localhost:8080) &nbsp; "
        "[MLflow →](http://localhost:5000)"
    )


def _render_model_header(m: ModelMetricsResponse) -> None:
    """Model name, version, and stage tag."""
    st.markdown(
        f"Metrics for **{m.model_name}** (v{m.model_version}) — "
        "stage: <span style='color:#22c55e;font-weight:700;'>Production</span>",
        unsafe_allow_html=True,
    )


def _render_kpi_row(m: ModelMetricsResponse) -> None:
    """Five metric cards in a single row."""
    cols = st.columns(5)
    pairs = [
        ("Precision", m.precision, "TP / (TP + FP)"),
        ("Recall", m.recall, "TP / (TP + FN)"),
        ("F1 Score", m.f1, "Harmonic mean of Precision & Recall"),
        ("ROC-AUC", m.roc_auc, "Area under the ROC curve"),
        ("AUC-PR ★", m.average_precision_score, "Primary metric — AUC-PR score"),
    ]
    for col, (label, value, tip) in zip(cols, pairs):
        with col:
            st.metric(label, f"{value:.4f}" if value is not None else "—", help=tip)


def _render_charts(m: ModelMetricsResponse) -> None:
    """Radar + bar chart side by side."""
    metric_dict = {
        "Precision": m.precision or 0.0,
        "Recall": m.recall or 0.0,
        "F1": m.f1 or 0.0,
        "ROC-AUC": m.roc_auc or 0.0,
        "AUC-PR": m.average_precision_score or 0.0,
    }

    col_radar, col_bar = st.columns(2)
    with col_radar:
        st.plotly_chart(charts.metrics_radar(metric_dict), use_container_width=True)
    with col_bar:
        st.plotly_chart(charts.metrics_bar(metric_dict), use_container_width=True)


def _render_metric_explainer() -> None:
    """Educational note on why AUC-PR is the primary metric."""
    st.markdown("### Why AUC-PR is the Primary Metric")
    st.info(
        "Credit card fraud datasets are **extremely imbalanced** (~0.17 % fraud rate). "
        "A naive classifier that labels every transaction as *normal* achieves 99.83 % accuracy "
        "and a high ROC-AUC — but catches zero fraud.\n\n"
        "**AUC-PR** (area under the Precision-Recall curve) focuses entirely on the minority "
        "class. It penalises both missed frauds (low recall) and false alarms (low precision), "
        "making it the correct optimisation target for this problem.",
        icon="ℹ️",
    )
