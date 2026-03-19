"""Tests for skip_retraining_fn and edge cases in the retraining DAG."""

from __future__ import annotations

from unittest.mock import MagicMock


class TestSkipRetrainingFn:
    """skip_retraining_fn — should log and exit cleanly."""

    def test_skip_retraining_does_not_raise(self):
        from fraud_retraining_ct import skip_retraining_fn
        ti = MagicMock()
        # Should not raise any exception
        skip_retraining_fn(ti=ti)

    def test_skip_retraining_accepts_context(self):
        from fraud_retraining_ct import skip_retraining_fn
        # Called with full Airflow context dict
        context = {"ti": MagicMock(), "dag": MagicMock(), "run_id": "test_run"}
        skip_retraining_fn(**context)  # Should not raise


class TestDecideRetrainingEdgeCases:
    """Additional edge cases for decide_retraining."""

    def _call(self, perf_degraded: bool, drift_detected: bool) -> str:
        from fraud_retraining_ct import decide_retraining
        ti = MagicMock()
        ti.xcom_pull.side_effect = lambda task_ids: {
            "check_model_performance": {
                "perf_degraded": perf_degraded,
                "current_auc_pr": 0.5 if perf_degraded else 0.9,
                "model_name": "lightgbm_fraud",
            },
            "check_data_drift": {
                "drift_detected": drift_detected,
                "anomaly_rate": 0.1 if drift_detected else 0.001,
                "expected_rate": 0.002,
            },
        }[task_ids]
        return decide_retraining(ti=ti)

    def test_returns_string(self):
        result = self._call(perf_degraded=False, drift_detected=False)
        assert isinstance(result, str)

    def test_output_is_valid_branch_id(self):
        """Return value must be one of the two valid branch IDs."""
        valid = {"trigger_retraining", "skip_retraining"}
        assert self._call(True, False) in valid
        assert self._call(False, True) in valid
        assert self._call(False, False) in valid
        assert self._call(True, True) in valid
