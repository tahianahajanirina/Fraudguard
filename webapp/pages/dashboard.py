"""Dashboard page — service overview, pipeline architecture, quick links."""

from __future__ import annotations

import streamlit as st

import components.header as header
from api.client import ApiError, get_health, get_model_metrics, get_root


def render() -> None:
    """Render the main dashboard overview page."""
    header.render("🏠 Dashboard", "Service overview and pipeline status")

    _render_status_row()
    st.markdown("<br>", unsafe_allow_html=True)
    _render_architecture_and_links()
    st.markdown("<br>", unsafe_allow_html=True)
    _render_metrics_preview()


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _render_status_row() -> None:
    """Four KPI cards: API, model, scaler, stage."""
    try:
        health = get_health()
        root = get_root()
        api_ok = True
    except ApiError:
        health = None
        root = None
        api_ok = False

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("API Status", "Online ✓" if api_ok else "Offline ✗")
    with col2:
        model = (health.model_name or "—") if health else "—"
        display = model.replace("_fraud", "").replace("_", " ").title()
        st.metric("Active Model", display)
    with col3:
        loaded = health.scaler_loaded if health else False
        st.metric("Scaler", "Loaded ✓" if loaded else "Missing ✗")
    with col4:
        stage = (health.model_stage or "—") if health else "—"
        st.metric("Model Stage", stage)

    if not api_ok:
        st.warning(
            "The API is unreachable. Make sure the Docker stack is running: "
            "`docker-compose up --build -d`",
            icon="⚠️",
        )
    elif not (root and root.model_loaded):
        st.info(
            "No model is loaded yet. Open **Airflow** at http://localhost:8080, "
            "trigger the `fraud_detection_pipeline` DAG, and wait ~5 minutes.",
            icon="ℹ️",
        )


def _render_architecture_and_links() -> None:
    """Pipeline diagram and quick-links table side by side."""
    col_diag, col_links = st.columns([3, 2])

    with col_diag:
        st.markdown("### Pipeline Architecture")
        st.code(
            """\
creditcard.csv
     │
     ▼
[Airflow DAG]
  ├─ ingest_and_preprocess
  ├─ train_isolation_forest ──► [MLflow Registry]
  └─ train_lightgbm ──────────► [MLflow Registry]
       └─ register_best_model  (AUC-PR winner → Production)
                    │
[FastAPI /predict] ◄┘ loads Production model
                    │
       [FraudGuard Dashboard] ◄── you are here""",
            language=None,
        )

    with col_links:
        st.markdown("### Quick Links")
        st.markdown(
            "| Service | URL |\n"
            "|---------|-----|\n"
            "| 🛡️ **This App** | [localhost:8501](http://localhost:8501) |\n"
            "| ⚙️ **API Docs** | [localhost:8000/docs](http://localhost:8000/docs) |\n"
            "| 🔬 **MLflow** | [localhost:5000](http://localhost:5000) |\n"
            "| 🌀 **Airflow** | [localhost:8080](http://localhost:8080) |"
        )
        st.markdown("### ML Strategy")
        st.markdown(
            "Two models compete on **AUC-PR** — the correct metric for the "
            "highly imbalanced fraud dataset (~0.17 % fraud rate):\n\n"
            "- **IsolationForest** — unsupervised anomaly detection\n"
            "- **LightGBM** — supervised gradient boosting\n\n"
            "The winner is promoted to **Production** in the MLflow registry "
            "and serves all live prediction requests."
        )


def _render_metrics_preview() -> None:
    """Show production model metrics if available."""
    try:
        metrics = get_model_metrics()
    except ApiError:
        return  # silently skip — covered by status row warning

    st.markdown("### Production Model Performance")
    cols = st.columns(5)
    pairs = [
        ("Precision", metrics.precision),
        ("Recall", metrics.recall),
        ("F1 Score", metrics.f1),
        ("ROC-AUC", metrics.roc_auc),
        ("AUC-PR ★", metrics.average_precision_score),
    ]
    for col, (label, value) in zip(cols, pairs):
        with col:
            display = f"{value:.4f}" if value is not None else "—"
            st.metric(label, display)
