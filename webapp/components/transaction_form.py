"""Transaction input form component.

Renders the 29-field form (V1–V28 + Amount), manages example presets
via st.session_state, and returns the submitted payload dict.

Widget keys are versioned so that loading a preset fully resets all
inputs to the new values (Streamlit caches widget state by key).
"""

from __future__ import annotations

import streamlit as st

from config import EXAMPLE_FRAUD, EXAMPLE_MEDIUM_RISK, EXAMPLE_NORMAL, V_FEATURES

_DEFAULTS_KEY = "tx_form_defaults"
_VERSION_KEY = "tx_form_version"
_COLS_PER_ROW = 4


def render() -> dict[str, float] | None:
    """Render the transaction form.

    Returns the submitted field values as a dict, or None if the form
    has not been submitted in this Streamlit run.
    """
    _render_preset_buttons()
    _render_card_preview()
    return _render_form()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_defaults() -> dict[str, float]:
    # Default to the fraud example so testing is immediate
    return st.session_state.get(_DEFAULTS_KEY, EXAMPLE_FRAUD)


def _get_version() -> int:
    return st.session_state.get(_VERSION_KEY, 0)


def _load_preset(preset: dict[str, float]) -> None:
    """Store new preset defaults and bump the version to invalidate widget cache."""
    st.session_state[_DEFAULTS_KEY] = preset.copy()
    st.session_state[_VERSION_KEY] = _get_version() + 1
    st.rerun()


def _render_preset_buttons() -> None:
    """Quick-fill buttons that reload the form with preset values."""
    col_fraud, col_medium, col_norm, _ = st.columns([1, 1, 1, 3])
    with col_fraud:
        if st.button("🚨 Fraud", type="primary", help="P(fraud) ≈ 100% — confirmed fraud"):
            _load_preset(EXAMPLE_FRAUD)
    with col_medium:
        if st.button("⚠️ Suspicious", help="P(fraud) ≈ 51% — borderline transaction"):
            _load_preset(EXAMPLE_MEDIUM_RISK)
    with col_norm:
        if st.button("✅ Normal", help="P(fraud) ≈ 0% — clearly normal"):
            _load_preset(EXAMPLE_NORMAL)


def _render_card_preview() -> None:
    """Show a decorative credit-card widget reflecting the current Amount default."""
    amount = _get_defaults().get("Amount", 0.0)
    st.markdown(
        f"""
<div class="fg-card">
  <div class="fg-card-chip"></div>
  <div class="fg-card-number">•••• &nbsp; •••• &nbsp; •••• &nbsp; ••••</div>
  <div style="margin-bottom:10px;">
    <span class="fg-card-amount">${amount:,.2f}</span>
    <span style="font-size:0.8rem;color:#64748b;margin-left:10px;">transaction amount</span>
  </div>
  <div class="fg-card-footer">
    <span>PCA FEATURES V1–V28 (anonymised)</span>
    <span>VISA ●●</span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)


def _render_form() -> dict[str, float] | None:
    """Render the input fields and return the payload dict on submission."""
    defaults = _get_defaults()
    ver = _get_version()

    # Version suffix forces Streamlit to create fresh widgets when a preset loads,
    # ensuring the new default values are actually applied.
    with st.form(f"transaction_form_v{ver}"):
        st.markdown("##### Transaction Features")

        amount = st.number_input(
            "Amount ($)",
            value=float(defaults.get("Amount", 1.0)),
            min_value=0.0,
            format="%.2f",
            help="Transaction amount in USD",
            key=f"amount_v{ver}",
        )

        st.markdown("##### PCA Components (V1 – V28)")
        v_values: dict[str, float] = {}

        feature_rows = [
            V_FEATURES[i : i + _COLS_PER_ROW]
            for i in range(0, len(V_FEATURES), _COLS_PER_ROW)
        ]
        for row_keys in feature_rows:
            cols = st.columns(len(row_keys))
            for col, key in zip(cols, row_keys):
                with col:
                    v_values[key] = st.number_input(
                        key,
                        value=float(defaults.get(key, 0.0)),
                        format="%.4f",
                        key=f"{key}_v{ver}",
                    )

        submitted = st.form_submit_button(
            "🔍 Analyze Transaction", use_container_width=True
        )

    if submitted:
        return {**v_values, "Amount": amount}
    return None
