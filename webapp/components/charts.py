"""Reusable Plotly figure factories.

All functions return a go.Figure — they never call st.plotly_chart themselves,
so callers control layout, sizing, and column placement.
All figures are styled with the FraudGuard dark finance theme via PLOTLY_LAYOUT.
"""

from __future__ import annotations

import plotly.graph_objects as go

from styles.theme import COLOUR_GOLD_LIGHT, PLOTLY_LAYOUT

# ---------------------------------------------------------------------------
# Colour palette used across charts
# ---------------------------------------------------------------------------
_FRAUD_RED = "#ef4444"
_SAFE_GREEN = "#22c55e"
_AMBER = "#f97316"
_BLUE = "#3b82f6"
_PURPLE = "#8b5cf6"
_CYAN = "#06b6d4"
_TEAL = "#10b981"
_INDIGO = "#6366f1"


def fraud_gauge(probability: float | None) -> go.Figure:
    """Semicircular gauge showing fraud probability (0–100 %).

    A gold threshold line marks the 50 % decision boundary.
    """
    value = (probability * 100) if probability is not None else 0
    needle_colour = (
        _FRAUD_RED if value > 70 else (_AMBER if value > 30 else _SAFE_GREEN)
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 34, "color": needle_colour}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#475569",
                    "tickfont": {"color": "#94a3b8", "size": 10},
                },
                "bar": {"color": needle_colour, "thickness": 0.22},
                "bgcolor": "#111827",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#052e16"},
                    {"range": [30, 70], "color": "#431407"},
                    {"range": [70, 100], "color": "#450a0a"},
                ],
                "threshold": {
                    "line": {"color": COLOUR_GOLD_LIGHT, "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            title={"text": "Fraud Probability", "font": {"color": "#94a3b8", "size": 13}},
        )
    )
    fig.update_layout(**{**PLOTLY_LAYOUT, "height": 240, "margin": dict(l=20, r=20, t=40, b=10)})
    return fig


def metrics_radar(metrics: dict[str, float]) -> go.Figure:
    """Radar / spider chart for model evaluation metrics."""
    labels = list(metrics.keys())
    values = list(metrics.values())

    fig = go.Figure(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(37,99,235,0.18)",
            line=dict(color=_BLUE, width=2),
            marker=dict(color=COLOUR_GOLD_LIGHT, size=7),
        )
    )
    fig.update_layout(
        **{
            **PLOTLY_LAYOUT,
            "polar": {
                "bgcolor": "#111827",
                "radialaxis": {
                    "visible": True,
                    "range": [0, 1],
                    "tickfont": {"color": "#64748b", "size": 9},
                    "gridcolor": "#1e2d3d",
                },
                "angularaxis": {
                    "tickfont": {"color": "#94a3b8", "size": 11},
                    "gridcolor": "#1e2d3d",
                },
            },
            "showlegend": False,
            "height": 360,
            "title": {"text": "Performance Radar", "font": {"color": "#94a3b8", "size": 13}},
        }
    )
    return fig


def metrics_bar(metrics: dict[str, float]) -> go.Figure:
    """Horizontal bar chart comparing all model evaluation metrics."""
    colours = [_BLUE, _PURPLE, _CYAN, _TEAL, COLOUR_GOLD_LIGHT]
    labels = list(metrics.keys())
    values = list(metrics.values())

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colours[: len(labels)], line=dict(width=0)),
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=11),
        )
    )
    fig.update_layout(
        **{
            **PLOTLY_LAYOUT,
            "xaxis": {**PLOTLY_LAYOUT.get("xaxis", {}), "range": [0, 1.1]},
            "height": 340,
            "title": {"text": "Metric Comparison", "font": {"color": "#94a3b8", "size": 13}},
        }
    )
    return fig


def fraud_donut(fraud_count: int, normal_count: int) -> go.Figure:
    """Donut pie chart showing fraud vs. normal split."""
    fig = go.Figure(
        go.Pie(
            labels=["Normal", "Fraud"],
            values=[normal_count, fraud_count],
            hole=0.58,
            marker=dict(colors=[_SAFE_GREEN, _FRAUD_RED], line=dict(color="#0a0e1a", width=2)),
            textfont=dict(color="#e2e8f0"),
        )
    )
    fig.update_layout(
        **{
            **PLOTLY_LAYOUT,
            "legend": {"font": {"color": "#94a3b8"}},
            "height": 290,
            "title": {"text": "Fraud Distribution", "font": {"color": "#94a3b8", "size": 13}},
        }
    )
    return fig


def risk_bar(risk_counts: dict[str, int]) -> go.Figure:
    """Bar chart of transaction counts by risk level."""
    colour_map = {
        "LOW": _SAFE_GREEN,
        "MEDIUM": _AMBER,
        "HIGH": _FRAUD_RED,
        "UNKNOWN": _INDIGO,
    }
    labels = list(risk_counts.keys())
    counts = list(risk_counts.values())
    colours = [colour_map.get(lbl, _BLUE) for lbl in labels]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=counts,
            marker=dict(color=colours, line=dict(width=0)),
            text=counts,
            textposition="outside",
            textfont=dict(color="#e2e8f0"),
        )
    )
    fig.update_layout(
        **{
            **PLOTLY_LAYOUT,
            "height": 290,
            "title": {"text": "Risk Level Distribution", "font": {"color": "#94a3b8", "size": 13}},
        }
    )
    return fig


def probability_histogram(probabilities: list[float]) -> go.Figure:
    """Histogram of fraud probabilities across a batch."""
    fig = go.Figure(
        go.Histogram(
            x=probabilities,
            nbinsx=30,
            marker=dict(
                color=_BLUE,
                line=dict(color="#0a0e1a", width=0.5),
            ),
            opacity=0.85,
        )
    )
    fig.add_vline(x=0.5, line=dict(color=COLOUR_GOLD_LIGHT, width=2, dash="dash"))
    fig.update_layout(
        **{
            **PLOTLY_LAYOUT,
            "height": 290,
            "bargap": 0.05,
            "title": {
                "text": "Fraud Probability Distribution",
                "font": {"color": "#94a3b8", "size": 13},
            },
            "xaxis": {**PLOTLY_LAYOUT.get("xaxis", {}), "title": "Fraud Probability"},
            "yaxis": {**PLOTLY_LAYOUT.get("yaxis", {}), "title": "Count"},
        }
    )
    return fig
