"""Single transaction prediction page."""

from __future__ import annotations

import streamlit as st

import components.header as header
from api.client import ApiError, post_predict
from api.models import PredictResponse
from components import charts, status_badge, transaction_form
from config import ALL_FEATURES

_RESULT_KEY = "single_pred_result"
_AMOUNT_KEY = "single_pred_amount"


def render() -> None:
    """Render the single-transaction prediction page."""
    header.render("🔍 Single Prediction", "Analyze one transaction in real time")

    # Form — always rendered at the top
    payload = transaction_form.render()

    # Submit → call API, persist result in session_state
    if payload is not None:
        _render_payload_inspector(payload)
        errors = _validate_payload(payload)
        if not errors:
            _call_api(payload)

    # Result — rendered from session_state so it survives reruns
    if _RESULT_KEY in st.session_state:
        st.markdown("---")
        _render_result(
            st.session_state[_RESULT_KEY],
            st.session_state.get(_AMOUNT_KEY, 0.0),
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_payload(payload: dict) -> list[str]:
    """Return validation errors; empty list means payload is valid."""
    errors = []
    missing = [f for f in ALL_FEATURES if f not in payload]
    if missing:
        errors.append(f"Missing fields: {missing}")
    wrong_type = [k for k, v in payload.items() if not isinstance(v, (int, float))]
    if wrong_type:
        errors.append(f"Non-numeric values for: {wrong_type}")
    return errors


def _render_payload_inspector(payload: dict) -> None:
    """Collapsible expander showing the exact JSON that will be sent."""
    errors = _validate_payload(payload)
    label = (
        f"🔴 Payload — {len(errors)} validation error(s)"
        if errors
        else f"✅ Payload — {len(payload)} fields ready to send"
    )
    with st.expander(label, expanded=bool(errors)):
        if errors:
            for err in errors:
                st.error(err)
        st.json(payload)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------
def _call_api(payload: dict) -> None:
    """Call the prediction API and store the result in session_state."""
    with st.spinner("Analyzing transaction…"):
        try:
            result = post_predict(payload)
        except ApiError as exc:
            if exc.status_code == 503:
                st.warning(
                    "No model loaded — run the Airflow `fraud_detection_pipeline` DAG first.",
                    icon="⚠️",
                )
            elif exc.status_code == 422:
                st.error(
                    f"API rejected the payload (422): {exc} — check the inspector above.",
                    icon="🚨",
                )
            else:
                st.error(f"API error ({exc.status_code}): {exc}", icon="🚨")
            return

    st.session_state[_RESULT_KEY] = result
    st.session_state[_AMOUNT_KEY] = payload.get("Amount", 0.0)


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------
def _render_result(result: PredictResponse, amount: float) -> None:
    """Display the full prediction outcome."""
    is_fraud = result.is_fraud
    proba = result.fraud_probability

    # Top row: verdict box + gauge
    box_col, gauge_col = st.columns([2, 1])

    with box_col:
        if is_fraud:
            st.markdown(
                '<div class="fg-fraud-box">'
                '<div class="fg-result-title">🚨 FRAUD DETECTED</div>'
                '<div class="fg-result-sub">This transaction has been flagged as fraudulent.</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="fg-safe-box">'
                '<div class="fg-result-title">✅ TRANSACTION SAFE</div>'
                '<div class="fg-result-sub">No fraud indicators detected.</div>'
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Detail metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown("**Risk Level**")
            status_badge.render_risk_badge(result.risk_level)
        with m2:
            st.markdown("**Verdict**")
            status_badge.render_prediction_label(result.prediction_label)
        with m3:
            st.metric("Amount", f"${amount:,.2f}")
        with m4:
            prob_display = f"{proba:.1%}" if proba is not None else "N/A"
            st.metric("Fraud Probability", prob_display)

        st.metric(
            "Model",
            result.model_used.replace("_fraud", "").replace("_", " ").title(),
        )

    with gauge_col:
        if proba is not None:
            st.plotly_chart(charts.fraud_gauge(proba), use_container_width=True)
        else:
            st.info(
                "Probability score not available for IsolationForest — "
                "the model returns a binary decision only.",
                icon="ℹ️",
            )
