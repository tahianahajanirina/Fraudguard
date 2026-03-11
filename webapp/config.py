"""FraudGuard webapp — central configuration and constants."""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ASSETS_DIR: Path = Path(__file__).parent / "assets"
APP_LOGO_PATH: Path = ASSETS_DIR / "logo_app.png"
SCHOOL_LOGO_PATH: Path = ASSETS_DIR / "logo_school.png"

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_URL", "http://api:8000")
API_TIMEOUT_FAST: int = 8    # health / root checks
API_TIMEOUT_PREDICT: int = 15
API_TIMEOUT_BATCH: int = 60

# ---------------------------------------------------------------------------
# App identity
# ---------------------------------------------------------------------------
APP_TITLE: str = "FraudGuard"
APP_SUBTITLE: str = "Credit Card Fraud Detection"
SCHOOL_NAME: str = "Mastère Spécialisé IA · Télécom Paris"
APP_VERSION: str = "0.1.0"
COURSE_CODE: str = "DATA713"

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
V_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]
ALL_FEATURES: list[str] = V_FEATURES + ["Amount"]
BATCH_MAX_ROWS: int = 1_000

# ---------------------------------------------------------------------------
# Navigation pages (ordered)
# ---------------------------------------------------------------------------
PAGES: dict[str, str] = {
    "🏠 Dashboard": "dashboard",
    "🔍 Single Prediction": "single_prediction",
    "📦 Batch Analysis": "batch_analysis",
    "📊 Model Metrics": "model_metrics",
}

# ---------------------------------------------------------------------------
# Risk level colours  {level: (background, text, border)}
# ---------------------------------------------------------------------------
RISK_COLOURS: dict[str, tuple[str, str, str]] = {
    "HIGH":    ("#450a0a", "#fca5a5", "#ef4444"),
    "MEDIUM":  ("#431407", "#fdba74", "#f97316"),
    "LOW":     ("#052e16", "#86efac", "#22c55e"),
    "UNKNOWN": ("#1e1b4b", "#a5b4fc", "#6366f1"),
}

# ---------------------------------------------------------------------------
# Example transactions
# ---------------------------------------------------------------------------
EXAMPLE_NORMAL: dict[str, float] = {
    "V1": -1.3598, "V2": -0.0727, "V3": 2.5363, "V4": 1.3781,
    "V5": -0.3383, "V6": 0.4623,  "V7": 0.2395, "V8": 0.0986,
    "V9": 0.3637,  "V10": 0.0907, "V11": -0.5515, "V12": -0.6178,
    "V13": -0.9913, "V14": -0.3111, "V15": 1.4681, "V16": -0.4704,
    "V17": 0.2079, "V18": 0.0257, "V19": 0.4039, "V20": 0.2514,
    "V21": -0.0183, "V22": 0.2778, "V23": -0.1104, "V24": 0.0669,
    "V25": 0.1285, "V26": -0.1891, "V27": 0.1335, "V28": -0.0210,
    "Amount": 149.62,
}

EXAMPLE_FRAUD: dict[str, float] = {
    # Real fraud transaction from the test split (y_test == 1). P(fraud) ≈ 1.0.
    "V1": -1.2712, "V2": 2.4627,  "V3": -2.8514, "V4": 2.3245,
    "V5": -1.3722, "V6": -0.9482, "V7": -3.0652, "V8": 1.1669,
    "V9": -2.2688, "V10": -4.8811, "V11": 2.2551, "V12": -4.6864,
    "V13": 0.6524, "V14": -6.1743, "V15": 0.5944, "V16": -4.8497,
    "V17": -6.5365, "V18": -3.1191, "V19": 1.7155, "V20": 0.5605,
    "V21": 0.6529, "V22": 0.0819,  "V23": -0.2213, "V24": -0.5236,
    "V25": 0.2242, "V26": 0.7563,  "V27": 0.6328, "V28": 0.2502,
    "Amount": 0.01,
}

EXAMPLE_MEDIUM_RISK: dict[str, float] = {
    # Legitimate transaction (y=0) the model flags as borderline. P(fraud) ≈ 0.51.
    "V1": 1.2374,  "V2": 0.7056,  "V3": -0.3506, "V4": 1.3542,
    "V5": 0.0599,  "V6": -1.2510, "V7": 0.2935,  "V8": -0.2299,
    "V9": -0.0494, "V10": -0.6735, "V11": 0.0779, "V12": -0.2616,
    "V13": -0.3342, "V14": -1.2815, "V15": 1.1195, "V16": 0.6091,
    "V17": 0.9102, "V18": 0.6261,  "V19": -0.6237, "V20": -0.1156,
    "V21": -0.0709, "V22": -0.1444, "V23": -0.1202, "V24": 0.2446,
    "V25": 0.6818, "V26": -0.3265, "V27": 0.0335,  "V28": 0.0569,
    "Amount": 1.00,
}
