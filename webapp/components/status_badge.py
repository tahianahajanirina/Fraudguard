"""Reusable HTML badge components for risk levels and prediction labels."""

from __future__ import annotations

import streamlit as st

from config import RISK_COLOURS


def render_risk_badge(risk_level: str) -> None:
    """Render a coloured pill badge for the given risk level."""
    level = risk_level.upper()
    bg, text, border = RISK_COLOURS.get(level, RISK_COLOURS["UNKNOWN"])
    st.markdown(
        f'<span style="background:{bg};color:{text};border:1px solid {border};'
        f'border-radius:8px;padding:3px 14px;font-weight:700;display:inline-block;">'
        f"{level}</span>",
        unsafe_allow_html=True,
    )


def render_prediction_label(label: str) -> None:
    """Render FRAUD (red) or NORMAL (green) label."""
    is_fraud = label.upper() == "FRAUD"
    colour = "#fca5a5" if is_fraud else "#86efac"
    st.markdown(
        f'<span style="font-size:1.1rem;font-weight:700;color:{colour};">{label}</span>',
        unsafe_allow_html=True,
    )
