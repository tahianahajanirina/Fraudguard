"""Tests for the FraudGuard API prediction logic."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from main import _predict_single


# ===================================================================
# _predict_single — the core prediction logic
# ===================================================================
class TestPredictSingle:
    """Verify that _predict_single correctly interprets model outputs."""

    @pytest.fixture()
    def row_df(self):
        return pd.DataFrame([{f"V{i}": 0.0 for i in range(1, 29)} | {"Amount": 0.0}])

    def test_lightgbm_fraud(self, row_df):
        """High fraud probability → is_fraud=1, risk=HIGH."""
        model = MagicMock()
        model.predict.return_value = np.array([[0.1, 0.9]])
        is_fraud, proba, risk = _predict_single(model, "lightgbm_fraud", row_df)
        assert is_fraud == 1
        assert proba == pytest.approx(0.9)
        assert risk == "HIGH"

    def test_lightgbm_normal(self, row_df):
        """Low fraud probability → is_fraud=0, risk=LOW."""
        model = MagicMock()
        model.predict.return_value = np.array([[0.95, 0.05]])
        is_fraud, proba, risk = _predict_single(model, "lightgbm_fraud", row_df)
        assert is_fraud == 0
        assert proba == pytest.approx(0.05)
        assert risk == "LOW"

    def test_lightgbm_medium_risk(self, row_df):
        """Probability between 0.3 and 0.7 → risk=MEDIUM."""
        model = MagicMock()
        model.predict.return_value = np.array([[0.5, 0.5]])
        _, proba, risk = _predict_single(model, "lightgbm_fraud", row_df)
        assert proba == pytest.approx(0.5)
        assert risk == "MEDIUM"

    def test_isolation_forest_anomaly(self, row_df):
        """IF returns -1 (anomaly) → is_fraud=1, proba=None."""
        model = MagicMock()
        model.predict.return_value = np.array([-1])
        is_fraud, proba, risk = _predict_single(model, "isolation_forest_fraud", row_df)
        assert is_fraud == 1
        assert proba is None
        assert risk == "UNKNOWN"

    def test_isolation_forest_normal(self, row_df):
        """IF returns 1 (normal) → is_fraud=0."""
        model = MagicMock()
        model.predict.return_value = np.array([1])
        is_fraud, proba, risk = _predict_single(model, "isolation_forest_fraud", row_df)
        assert is_fraud == 0
        assert proba is None

    def test_lightgbm_threshold_boundary(self, row_df):
        """Exactly 0.5 probability → is_fraud=1 (threshold is >=0.5)."""
        model = MagicMock()
        model.predict.return_value = np.array([[0.5, 0.5]])
        is_fraud, _, _ = _predict_single(model, "lightgbm_fraud", row_df)
        assert is_fraud == 1

    def test_lightgbm_dataframe_output(self, row_df):
        """pyfunc may return a DataFrame — verify iloc branch works."""
        model = MagicMock()
        model.predict.return_value = pd.DataFrame([[0.1, 0.9]])
        is_fraud, proba, risk = _predict_single(model, "lightgbm_fraud", row_df)
        assert is_fraud == 1
        assert proba == pytest.approx(0.9)
        assert risk == "HIGH"


# ===================================================================
# Endpoint integration tests
# ===================================================================
class TestPredictEndpoint:
    """Test the /predict endpoint via FastAPI TestClient."""

    @pytest.fixture(autouse=True)
    def _setup_client(self, sample_transaction):
        from fastapi.testclient import TestClient

        from main import app

        self.app = app
        self.client = TestClient(app, raise_server_exceptions=False)
        self.sample_transaction = sample_transaction

    def test_predict_no_model_returns_503(self):
        """POST /predict without a loaded model → 503."""
        self.app.state.model = None
        resp = self.client.post("/predict", json=self.sample_transaction)
        assert resp.status_code == 503

    def test_predict_with_model_returns_fraud_result(self):
        """POST /predict with mocked model+scaler → 200 with expected keys."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.2, 0.8]])

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[0.0]])

        self.app.state.model = mock_model
        self.app.state.scaler = mock_scaler
        self.app.state.model_name = "lightgbm_fraud"

        resp = self.client.post("/predict", json=self.sample_transaction)
        assert resp.status_code == 200
        data = resp.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "risk_level" in data
        assert "prediction_label" in data
        assert data["is_fraud"] is True
        assert data["fraud_probability"] == pytest.approx(0.8, abs=0.01)
