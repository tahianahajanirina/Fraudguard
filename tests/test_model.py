"""Tests for the Continuous Training DAG logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException


# ===================================================================
# decide_retraining — branching logic
# ===================================================================
class TestDecideRetraining:
    """The CT DAG must trigger retraining when perf OR drift is bad."""

    def _call(self, perf_degraded: bool, drift_detected: bool) -> str:
        from fraud_retraining_ct import decide_retraining

        ti = MagicMock()
        ti.xcom_pull.side_effect = lambda task_ids: {
            "check_model_performance": {
                "model_name": "lgbm",
                "current_auc_pr": 0.5 if perf_degraded else 0.9,
                "perf_degraded": perf_degraded,
            },
            "check_data_drift": {
                "anomaly_rate": 0.1 if drift_detected else 0.001,
                "expected_rate": 0.002,
                "drift_detected": drift_detected,
            },
        }[task_ids]
        return decide_retraining(ti=ti)

    def test_trigger_on_perf_degraded(self):
        assert self._call(perf_degraded=True, drift_detected=False) == "trigger_retraining"

    def test_trigger_on_drift(self):
        assert self._call(perf_degraded=False, drift_detected=True) == "trigger_retraining"

    def test_trigger_on_both(self):
        assert self._call(perf_degraded=True, drift_detected=True) == "trigger_retraining"

    def test_skip_when_all_ok(self):
        assert self._call(perf_degraded=False, drift_detected=False) == "skip_retraining"


# ===================================================================
# check_model_performance — MLflow interaction
# ===================================================================
class TestCheckPerformance:
    """Verify performance check correctly flags degradation."""

    def _run(self, auc_pr: float, mlflow_error: bool = False) -> dict:
        from fraud_retraining_ct import check_model_performance

        mock_path = MagicMock()
        mock_path.read_text.return_value = "lightgbm_fraud\n1\n0.85\n"

        client = MagicMock()
        if mlflow_error:
            client.get_latest_versions.side_effect = MlflowException("not found")
        else:
            mv = MagicMock()
            mv.run_id = "run_123"
            client.get_latest_versions.return_value = [mv]
            run = MagicMock()
            run.data.metrics = {"average_precision_score": auc_pr}
            client.get_run.return_value = run

        with (
            patch("fraud_retraining_ct.BEST_MODEL_PATH", mock_path),
            patch("mlflow.tracking.MlflowClient", return_value=client),
        ):
            return check_model_performance()

    def test_good_performance(self):
        result = self._run(auc_pr=0.85)
        assert result["perf_degraded"] is False

    def test_degraded_performance(self):
        result = self._run(auc_pr=0.5)
        assert result["perf_degraded"] is True

    def test_model_missing_triggers_retraining(self):
        result = self._run(auc_pr=0.0, mlflow_error=True)
        assert result["perf_degraded"] is True

    def test_no_production_version(self):
        """get_latest_versions returns [] (model exists, not in Production) → perf_degraded."""
        from fraud_retraining_ct import check_model_performance

        mock_path = MagicMock()
        mock_path.read_text.return_value = "lightgbm_fraud\n1\n0.85\n"

        client = MagicMock()
        client.get_latest_versions.return_value = []  # empty, no exception

        with (
            patch("fraud_retraining_ct.BEST_MODEL_PATH", mock_path),
            patch("mlflow.tracking.MlflowClient", return_value=client),
        ):
            result = check_model_performance()

        assert result["perf_degraded"] is True
        assert result["current_auc_pr"] == 0.0


# ===================================================================
# check_data_drift — drift detection logic
# ===================================================================
class TestCheckDrift:
    """Verify drift detection flags abnormal anomaly rates."""

    def _run(self, anomaly_preds: np.ndarray, n_fraud: int, n_total: int) -> dict:
        from fraud_retraining_ct import check_data_drift

        X_test = pd.DataFrame({f"V{i}": np.zeros(n_total) for i in range(1, 29)})
        X_test["Amount"] = 0.0
        y_test = pd.DataFrame({"Class": [0] * (n_total - n_fraud) + [1] * n_fraud})

        mock_splits = MagicMock()
        mock_splits.__truediv__ = lambda self, x: f"/fake/{x}"

        mock_if = MagicMock()
        mock_if.predict.return_value = anomaly_preds

        client = MagicMock()
        mv = MagicMock()
        mv.version = "1"
        client.get_latest_versions.return_value = [mv]

        with (
            patch("fraud_retraining_ct.SPLITS_DIR", mock_splits),
            patch("pandas.read_parquet", side_effect=[X_test, y_test]),
            patch("mlflow.tracking.MlflowClient", return_value=client),
            patch("mlflow.sklearn.load_model", return_value=mock_if),
        ):
            return check_data_drift()

    def test_drift_detected(self):
        """50% anomaly rate vs 2% expected → drift."""
        preds = np.array([-1] * 50 + [1] * 50)
        result = self._run(anomaly_preds=preds, n_fraud=2, n_total=100)
        assert result["drift_detected"] is True

    def test_no_drift(self):
        """0.2% anomaly rate vs 0.2% expected → no drift."""
        preds = np.array([1] * 998 + [-1] * 2)
        result = self._run(anomaly_preds=preds, n_fraud=2, n_total=1000)
        assert result["drift_detected"] is False

    def test_model_missing_no_crash(self):
        """Missing IF model → drift_detected=False, no crash."""
        X_test = pd.DataFrame({f"V{i}": np.zeros(10) for i in range(1, 29)})
        X_test["Amount"] = 0.0

        client = MagicMock()
        client.get_latest_versions.side_effect = MlflowException("not found")

        with (
            patch("pandas.read_parquet", return_value=X_test),
            patch("mlflow.tracking.MlflowClient", return_value=client),
        ):
            from fraud_retraining_ct import check_data_drift

            result = check_data_drift()

        assert result["drift_detected"] is False

    def test_empty_versions_no_crash(self):
        """IF model exists but has no versions → drift_detected=False."""
        X_test = pd.DataFrame({f"V{i}": np.zeros(10) for i in range(1, 29)})
        X_test["Amount"] = 0.0

        client = MagicMock()
        client.get_latest_versions.return_value = []  # empty, no exception

        with (
            patch("pandas.read_parquet", return_value=X_test),
            patch("mlflow.tracking.MlflowClient", return_value=client),
        ):
            from fraud_retraining_ct import check_data_drift

            result = check_data_drift()

        assert result["drift_detected"] is False
        assert result["anomaly_rate"] == 0.0
