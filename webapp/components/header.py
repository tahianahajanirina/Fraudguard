"""Page-level header component."""

from __future__ import annotations

import streamlit as st


def render(title: str, subtitle: str = "") -> None:
    """Render a styled page header with title and optional subtitle."""
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(
            f'<p style="color:#64748b;margin-top:-12px;margin-bottom:16px;">{subtitle}</p>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<hr style="border-color:#1e2d3d;margin-bottom:24px;">',
        unsafe_allow_html=True,
    )
