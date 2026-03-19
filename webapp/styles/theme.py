"""Visual theme for the FraudGuard webapp.

Single source of truth for:
  - CSS injected via st.markdown (dark navy + gold finance theme)
  - PLOTLY_LAYOUT base dict applied to every chart figure
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
COLOUR_BG_PAGE = "#0A0F1E"
COLOUR_BG_CARD = "#111827"
COLOUR_BG_CARD_ALT = "#0d1220"
COLOUR_BORDER = "#1e2d3d"
COLOUR_GOLD = "#C9A84C"
COLOUR_GOLD_LIGHT = "#f8c94c"
COLOUR_BLUE = "#2563eb"
COLOUR_TEXT_PRIMARY = "#e2e8f0"
COLOUR_TEXT_MUTED = "#94a3b8"
COLOUR_TEXT_FAINT = "#475569"

COLOUR_FRAUD_BG = "#450a0a"
COLOUR_FRAUD_BORDER = "#ef4444"
COLOUR_SAFE_BG = "#052e16"
COLOUR_SAFE_BORDER = "#22c55e"

# ---------------------------------------------------------------------------
# Plotly base layout (applied to every chart via fig.update_layout(**PLOTLY_LAYOUT))
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT: dict = {
    "paper_bgcolor": COLOUR_BG_PAGE,
    "plot_bgcolor": COLOUR_BG_PAGE,
    "font": {"color": COLOUR_TEXT_PRIMARY, "family": "Inter, Segoe UI, sans-serif"},
    "xaxis": {"gridcolor": COLOUR_BORDER, "color": COLOUR_TEXT_MUTED},
    "yaxis": {"gridcolor": COLOUR_BORDER, "color": COLOUR_TEXT_MUTED},
    "margin": {"l": 20, "r": 20, "t": 48, "b": 20},
}

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Color+Emoji&family=Inter:wght@400;500;600;700&display=swap');

/* ── Base ─────────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {COLOUR_BG_PAGE};
    color: {COLOUR_TEXT_PRIMARY};
    font-family: 'Inter', 'Segoe UI', 'Noto Color Emoji', sans-serif;
}}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {COLOUR_BG_CARD_ALT} 0%, #111827 100%);
    border-right: 1px solid {COLOUR_BORDER};
}}
[data-testid="stSidebar"] * {{ color: #cbd5e1 !important; }}

/* ── Headings ──────────────────────────────────────────────────────────── */
h1 {{ color: {COLOUR_GOLD_LIGHT} !important; letter-spacing: -0.5px; }}
h2 {{ color: #93c5fd !important; }}
h3 {{ color: {COLOUR_TEXT_PRIMARY} !important; }}

/* ── Metric cards ──────────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: {COLOUR_BG_CARD};
    border: 1px solid {COLOUR_BORDER};
    border-radius: 12px;
    padding: 16px 20px;
}}
[data-testid="stMetricLabel"] {{ color: {COLOUR_TEXT_MUTED} !important; font-size: 0.78rem !important; }}
[data-testid="stMetricValue"] {{ color: {COLOUR_GOLD_LIGHT} !important; font-size: 1.5rem !important; }}

/* ── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button {{
    background: linear-gradient(135deg, #1e40af 0%, {COLOUR_BLUE} 100%);
    color: #fff !important;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-weight: 600;
    letter-spacing: 0.3px;
    transition: all 0.2s;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, {COLOUR_BLUE} 0%, #3b82f6 100%);
    box-shadow: 0 4px 15px rgba(37,99,235,0.4);
    transform: translateY(-1px);
}}

/* ── Inputs ────────────────────────────────────────────────────────────── */
.stNumberInput input, .stTextInput input {{
    background: {COLOUR_BG_CARD} !important;
    color: {COLOUR_TEXT_PRIMARY} !important;
    border: 1px solid {COLOUR_BORDER} !important;
    border-radius: 6px !important;
}}

/* ── Tabs ──────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tab"] {{
    color: #64748b !important;
    border-bottom: 2px solid transparent;
    font-weight: 500;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: {COLOUR_GOLD_LIGHT} !important;
    border-bottom-color: {COLOUR_GOLD_LIGHT};
}}

/* ── Result boxes ─────────────────────────────────────────────────────── */
.fg-fraud-box {{
    background: linear-gradient(135deg, {COLOUR_FRAUD_BG}, #7f1d1d);
    border: 1px solid {COLOUR_FRAUD_BORDER};
    border-radius: 12px;
    padding: 22px 26px;
    text-align: center;
}}
.fg-safe-box {{
    background: linear-gradient(135deg, {COLOUR_SAFE_BG}, #14532d);
    border: 1px solid {COLOUR_SAFE_BORDER};
    border-radius: 12px;
    padding: 22px 26px;
    text-align: center;
}}
.fg-result-title {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 6px; }}
.fg-result-sub   {{ font-size: 0.9rem; color: {COLOUR_TEXT_MUTED}; }}

/* ── Credit-card widget ───────────────────────────────────────────────── */
.fg-card {{
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2442 50%, #162035 100%);
    border: 1px solid #2d4a6e;
    border-radius: 18px;
    padding: 26px 28px;
    font-family: 'Courier New', monospace;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    min-height: 170px;
}}
.fg-card-chip {{
    width: 42px; height: 32px;
    background: linear-gradient(135deg, #d4af37, #f5d060);
    border-radius: 5px;
    margin-bottom: 16px;
    display: inline-block;
}}
.fg-card-number {{ font-size: 1.15rem; letter-spacing: 4px; color: {COLOUR_TEXT_PRIMARY}; margin-bottom: 12px; }}
.fg-card-footer {{ display: flex; justify-content: space-between; font-size: 0.78rem; color: {COLOUR_TEXT_MUTED}; }}
.fg-card-amount {{ font-size: 1.4rem; font-weight: 700; color: {COLOUR_GOLD_LIGHT}; }}

/* ── Risk pills ───────────────────────────────────────────────────────── */
.fg-risk-HIGH    {{ background:#450a0a; color:#fca5a5; border:1px solid #ef4444; border-radius:8px; padding:3px 14px; font-weight:700; display:inline-block; }}
.fg-risk-MEDIUM  {{ background:#431407; color:#fdba74; border:1px solid #f97316; border-radius:8px; padding:3px 14px; font-weight:700; display:inline-block; }}
.fg-risk-LOW     {{ background:#052e16; color:#86efac; border:1px solid #22c55e;  border-radius:8px; padding:3px 14px; font-weight:700; display:inline-block; }}
.fg-risk-UNKNOWN {{ background:#1e1b4b; color:#a5b4fc; border:1px solid #6366f1;  border-radius:8px; padding:3px 14px; font-weight:700; display:inline-block; }}

/* ── Status dots ─────────────────────────────────────────────────────── */
.fg-dot-green {{ display:inline-block; width:9px; height:9px; background:#22c55e; border-radius:50%; margin-right:6px; box-shadow: 0 0 6px #22c55e; }}
.fg-dot-red   {{ display:inline-block; width:9px; height:9px; background:#ef4444; border-radius:50%; margin-right:6px; }}
.fg-dot-amber {{ display:inline-block; width:9px; height:9px; background:#f59e0b; border-radius:50%; margin-right:6px; }}

/* ── Dividers ────────────────────────────────────────────────────────── */
hr {{ border-color: {COLOUR_BORDER} !important; }}

/* ── Logo placeholder ────────────────────────────────────────────────── */
.fg-logo-placeholder {{
    width: 78px; height: 48px;
    background: {COLOUR_BG_CARD};
    border: 1px dashed #334155;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: {COLOUR_TEXT_FAINT};
    font-size: 0.6rem;
    text-align: center;
}}
</style>
"""


def inject_css() -> None:
    """Inject the FraudGuard theme CSS into the Streamlit app (call once at startup)."""
    st.markdown(_CSS, unsafe_allow_html=True)
