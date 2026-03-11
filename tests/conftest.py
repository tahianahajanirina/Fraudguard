"""Shared fixtures for FraudGuard test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_transaction() -> dict:
    """A valid transaction payload with V1-V28 + Amount."""
    return {
        "V1": -1.3598, "V2": -0.0727, "V3": 2.5363, "V4": 1.3781,
        "V5": -0.3383, "V6": 0.4623, "V7": 0.2395, "V8": 0.0986,
        "V9": 0.3637, "V10": 0.0907, "V11": -0.5515, "V12": -0.6178,
        "V13": -0.9913, "V14": -0.3111, "V15": 1.4681, "V16": -0.4704,
        "V17": 0.2079, "V18": 0.0257, "V19": 0.4039, "V20": 0.2514,
        "V21": -0.0183, "V22": 0.2778, "V23": -0.1104, "V24": 0.0669,
        "V25": 0.1285, "V26": -0.1891, "V27": 0.1335, "V28": -0.0210,
        "Amount": 149.62,
    }


@pytest.fixture()
def sample_dataframe() -> pd.DataFrame:
    """Small DataFrame mimicking creditcard.csv (100 rows, 2 frauds)."""
    rng = np.random.RandomState(42)
    n = 100
    data = {f"V{i}": rng.randn(n) for i in range(1, 29)}
    data["Time"] = np.arange(n, dtype=float)
    data["Amount"] = rng.exponential(50, size=n)
    data["Class"] = np.zeros(n, dtype=int)
    data["Class"][0] = 1
    data["Class"][50] = 1
    return pd.DataFrame(data)
