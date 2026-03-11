"""Sidebar navigation and live API status indicator."""

from __future__ import annotations

import streamlit as st

from api.client import ApiError, get_health
from config import APP_LOGO_PATH, APP_SUBTITLE, APP_TITLE, PAGES, SCHOOL_LOGO_PATH


def render() -> str:
    """Render the sidebar and return the selected page key (e.g. 'dashboard')."""
    with st.sidebar:
        _render_logos()
        _render_brand()
        _hr()

        page_label = st.radio(
            "Navigation",
            list(PAGES.keys()),
            label_visibility="collapsed",
        )

        _hr()
        _render_status()

    return PAGES[page_label]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _hr() -> None:
    st.markdown('<hr style="border-color:#1e2d3d;margin:8px 0 12px 0;">', unsafe_allow_html=True)


def _render_logos() -> None:
    """Display app logo and school logo side by side."""
    col_app, col_school = st.columns(2)

    with col_app:
        if APP_LOGO_PATH.exists():
            st.image(str(APP_LOGO_PATH), width=88)
        else:
            st.markdown(
                '<div class="fg-logo-placeholder">App<br>Logo</div>',
                unsafe_allow_html=True,
            )

    with col_school:
        if SCHOOL_LOGO_PATH.exists():
            st.image(str(SCHOOL_LOGO_PATH), width=88)
        else:
            st.markdown(
                '<div class="fg-logo-placeholder">School<br>Logo</div>',
                unsafe_allow_html=True,
            )


def _render_brand() -> None:
    """Render the app title and school name."""
    st.markdown(
        f"<h1 style='font-size:1.35rem;margin:14px 0 2px 0;'>🛡️ {APP_TITLE}</h1>"
        f"<p style='color:#64748b;font-size:0.72rem;margin-bottom:4px;'>{APP_SUBTITLE}</p>",
        unsafe_allow_html=True,
    )


def _render_status() -> None:
    """Show a live API status indicator at the bottom of the sidebar."""
    try:
        health = get_health()
        dot = '<span class="fg-dot-green"></span>'
        model = health.model_name or "—"
        st.markdown(
            f"{dot} **API Online**\n\n"
            f"<span style='font-size:0.72rem;color:#64748b;'>Model: `{model}`</span>",
            unsafe_allow_html=True,
        )
    except ApiError:
        dot = '<span class="fg-dot-red"></span>'
        st.markdown(
            f"{dot} **API Offline**\n\n"
            "<span style='font-size:0.72rem;color:#64748b;'>Start the Docker stack first</span>",
            unsafe_allow_html=True,
        )
