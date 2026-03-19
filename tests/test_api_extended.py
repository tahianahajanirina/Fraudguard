"""Extended tests for the FraudGuard API — covering all endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestRootEndpoint:
    """GET / — project metadata."""

    @pytest.fixture(autouse=True)
    def _setup(self, sample_transaction):
        from fastapi.testclient import TestClient
        from main import app
        self.client = TestClient(app)
        self.app = app

    def test_root_returns_200(self):
        resp = self.client.get("/")
        assert resp.status_code == 200

    def test_root_contains_required_keys(self):
        resp = self.client.get("/")
        data = resp.json()
        assert "name" in data
        assert "status" in data
        assert "model_loaded" in data
        assert data["name"] == "FraudGuard"
        assert data["status"] == "running"

    def test_root_model_loaded_false_without_model(self):
        self.app.state.model = None
        resp = self.client.get("/")
        assert resp.json()["model_loaded"] is False


class TestHealthEndpoint:
    """GET /health — service health check."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from fastapi.testclient import TestClient
        from main import app
        self.client = TestClient(app)
        self.app = app

    def test_health_returns_200(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200

    def test_health_degraded_without_model(self):
        self.app.state.model = None
        resp = self.client.get("/health")
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False

    def test_health_healthy_with_model(self):
        mock_model = MagicMock()
        mock_model._model_impl.lgb_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        self.app.state.model = mock_model
        self.app.state.model_name = "lightgbm_fraud"
        self.app.state.model_version = "1"
        self.app.state.model_score = 0.85
        self.app.state.scaler = MagicMock()
        resp = self.client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_name"] == "lightgbm_fraud"
        assert data["model_version"] == "1"
        assert data["model_score"] == pytest.approx(0.85)

    def test_health_contains_scaler_status(self):
        self.app.state.model = MagicMock()
        self.app.state.scaler = MagicMock()
        resp = self.client.get("/health")
        assert "scaler_loaded" in resp.json()


class TestMetricsEndpoint:
    """GET /metrics — Prometheus metrics exposition."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from fastapi.testclient import TestClient
        from main import app
        self.client = TestClient(app)

    def test_metrics_returns_200(self):
        resp = self.client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_content_type_is_prometheus(self):
        resp = self.client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]

    def test_metrics_contains_api_counters(self):
        resp = self.client.get("/metrics")
        body = resp.text
        assert "api_requests_total" in body
        assert "model_predictions_total" in body


class TestPredictBatchEndpoint:
    """POST /predict_batch — batch fraud prediction."""

    @pytest.fixture(autouse=True)
    def _setup(self, sample_transaction):
        from fastapi.testclient import TestClient
        from main import app
        self.app = app
        self.client = TestClient(app, raise_server_exceptions=False)
        self.sample_transaction = sample_transaction

    def _make_batch(self, n: int) -> dict:
        return {"transactions": [self.sample_transaction] * n}

    def test_batch_503_without_model(self):
        self.app.state.model = None
        resp = self.client.post("/predict_batch", json=self._make_batch(3))
        assert resp.status_code == 503

    def test_batch_returns_correct_structure(self):
        mock_model = MagicMock()
        mock_model._model_impl.lgb_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_scaler = MagicMock()
        mock_scaler.transform.side_effect = lambda x: np.zeros((len(x), 1))
        self.app.state.model = mock_model
        self.app.state.scaler = mock_scaler
        self.app.state.model_name = "lightgbm_fraud"

        resp = self.client.post("/predict_batch", json=self._make_batch(3))
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert data["total"] == 3
        assert "fraud_count" in data
        assert "fraud_rate" in data
        assert len(data["predictions"]) == 3

    def test_batch_fraud_count_correct(self):
        mock_model = MagicMock()
        # All predictions → fraud (proba=0.9)
        mock_model._model_impl.lgb_model.predict_proba.return_value = np.array([[0.1, 0.9]])
        mock_scaler = MagicMock()
        mock_scaler.transform.side_effect = lambda x: np.zeros((len(x), 1))
        self.app.state.model = mock_model
        self.app.state.scaler = mock_scaler
        self.app.state.model_name = "lightgbm_fraud"

        resp = self.client.post("/predict_batch", json=self._make_batch(5))
        data = resp.json()
        assert data["fraud_count"] == 5
        assert data["fraud_rate"] == pytest.approx(1.0)

    def test_batch_size_limit_1000(self):
        """Batch > 1000 transactions → 422 validation error."""
        resp = self.client.post("/predict_batch", json=self._make_batch(1001))
        assert resp.status_code == 422

    def test_batch_size_exactly_1000_is_ok(self):
        mock_model = MagicMock()
        mock_model._model_impl.lgb_model.predict_proba.return_value = np.array([[0.9, 0.1]])
        mock_scaler = MagicMock()
        mock_scaler.transform.side_effect = lambda x: np.zeros((len(x), 1))
        self.app.state.model = mock_model
        self.app.state.scaler = mock_scaler
        self.app.state.model_name = "lightgbm_fraud"

        resp = self.client.post("/predict_batch", json=self._make_batch(1000))
        assert resp.status_code == 200
        assert resp.json()["total"] == 1000

    def test_batch_empty_transactions_returns_zero_fraud_rate(self):
        mock_model = MagicMock()
        mock_scaler = MagicMock()
        mock_scaler.transform.side_effect = lambda x: np.zeros((len(x), 1))
        self.app.state.model = mock_model
        self.app.state.scaler = mock_scaler
        self.app.state.model_name = "lightgbm_fraud"

        resp = self.client.post("/predict_batch", json={"transactions": []})
        assert resp.status_code in (200, 422)
        if resp.status_code == 200:
            assert resp.json()["fraud_rate"] == 0.0


class TestModelMetricsEndpoint:
    """GET /model_metrics — MLflow metrics retrieval."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from fastapi.testclient import TestClient
        from main import app
        self.app = app
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_model_metrics_503_without_model(self):
        self.app.state.model = None
        resp = self.client.get("/model_metrics")
        assert resp.status_code == 503

    def test_model_metrics_returns_correct_keys(self):
        self.app.state.model = MagicMock()
        self.app.state.model_name = "lightgbm_fraud"
        self.app.state.model_version = "1"

        mock_version = MagicMock()
        mock_version.run_id = "run_123"
        mock_version.current_stage = "Production"

        mock_run = MagicMock()
        mock_run.data.metrics = {
            "precision_score": 0.9,
            "recall_score": 0.85,
            "f1_score": 0.87,
            "roc_auc_score": 0.95,
            "average_precision_score": 0.88,
        }

        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = [mock_version]
        mock_client.get_run.return_value = mock_run

        with patch("main.MlflowClient", return_value=mock_client):
            resp = self.client.get("/model_metrics")

        assert resp.status_code == 200
        data = resp.json()
        assert data["precision"] == pytest.approx(0.9)
        assert data["recall"] == pytest.approx(0.85)
        assert data["f1"] == pytest.approx(0.87)
        assert data["roc_auc"] == pytest.approx(0.95)
        assert data["average_precision_score"] == pytest.approx(0.88)

    def test_model_metrics_503_when_no_production_version(self):
        self.app.state.model = MagicMock()
        self.app.state.model_name = "lightgbm_fraud"

        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = []  # no versions

        with patch("main.MlflowClient", return_value=mock_client):
            resp = self.client.get("/model_metrics")

        assert resp.status_code == 503

    def test_model_metrics_503_on_mlflow_error(self):
        self.app.state.model = MagicMock()
        self.app.state.model_name = "lightgbm_fraud"

        mock_client = MagicMock()
        mock_client.search_model_versions.side_effect = Exception("MLflow unreachable")

        with patch("main.MlflowClient", return_value=mock_client):
            resp = self.client.get("/model_metrics")

        assert resp.status_code == 503


class TestInputValidation:
    """Validate Transaction schema enforcement."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from fastapi.testclient import TestClient
        from main import app
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_missing_field_returns_422(self):
        """Payload without V1 → 422 Unprocessable Entity."""
        payload = {f"V{i}": 0.0 for i in range(2, 29)}
        payload["Amount"] = 100.0
        resp = self.client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_invalid_type_returns_422(self):
        """String in numeric field → 422."""
        payload = {f"V{i}": 0.0 for i in range(1, 29)}
        payload["Amount"] = "not-a-number"
        resp = self.client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_missing_amount_returns_422(self):
        """Payload without Amount → 422."""
        payload = {f"V{i}": 0.0 for i in range(1, 29)}
        resp = self.client.post("/predict", json=payload)
        assert resp.status_code == 422
