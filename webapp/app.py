"""FraudGuard — Streamlit webapp entry point.

Responsibilities:
  1. Configure the Streamlit page (must be the very first st call).
  2. Inject the CSS theme once.
  3. Render the sidebar and obtain the selected page key.
  4. Dispatch to the correct page module.

No business logic lives here.
"""

from __future__ import annotations

import streamlit as st

from config import APP_SUBTITLE, APP_TITLE
from styles.theme import inject_css

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_TITLE} · {APP_SUBTITLE}",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
    <style>
        /* Hide the default sidebar nav (page links) */
        [data-testid="stSidebarNav"] {
            display: none;
        }
        
        /* Hide the sidebar top decoration/logos area */
        [data-testid="stSidebarHeader"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)
# ---------------------------------------------------------------------------
# Inject theme CSS once per session
# ---------------------------------------------------------------------------
inject_css()

# ---------------------------------------------------------------------------
# Lazy page imports (avoids loading all modules on every rerun)
# ---------------------------------------------------------------------------
from components import sidebar  # noqa: E402  (after st.set_page_config)
from pages import (  # noqa: E402
    batch_analysis,
    dashboard,
    model_metrics,
    single_prediction,
)

_PAGE_MAP = {
    "dashboard": dashboard,
    "single_prediction": single_prediction,
    "batch_analysis": batch_analysis,
    "model_metrics": model_metrics,
}

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
page_key = sidebar.render()
_PAGE_MAP[page_key].render()
