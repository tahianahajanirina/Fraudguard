"""Tests for register_best_model — DAG 1 model comparison and promotion."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest


class TestRegisterBestModel:
    """Verify model comparison, stage transitions, and best_model.txt output."""

    def _make_client(self, models: dict[str, dict]) -> MagicMock:
        """Build a mock MlflowClient with search_model_versions and get_run."""
        client = MagicMock()

        def _search(filter_string: str):
            for name, info in models.items():
                if name in filter_string:
                    if info.get("empty"):
                        return []
                    mv = MagicMock()
                    mv.version = info["version"]
                    mv.run_id = info["run_id"]
                    return [mv]
            return []

        client.search_model_versions.side_effect = _search

        def _get_run(run_id: str):
            for info in models.values():
                if info["run_id"] == run_id:
                    run = MagicMock()
                    run.data.metrics = info["metrics"]
                    return run
            raise ValueError(f"Unknown run_id: {run_id}")

        client.get_run.side_effect = _get_run
        return client

    def test_winner_is_highest_auc_pr(self, tmp_path):
        """LightGBM (higher AUC-PR) should be promoted to Production."""
        from fraud_pipeline import register_best_model

        models = {
            "isolation_forest_fraud": {
                "version": "1",
                "run_id": "run_if",
                "metrics": {"average_precision_score": 0.20, "precision_score": 0.5},
            },
            "lightgbm_fraud": {
                "version": "2",
                "run_id": "run_lgbm",
                "metrics": {"average_precision_score": 0.85, "precision_score": 0.9},
            },
        }
        client = self._make_client(models)

        with (
            patch("mlflow.tracking.MlflowClient", return_value=client),
            patch("fraud_pipeline.ARTIFACTS_DIR", tmp_path),
            patch("fraud_pipeline.BEST_MODEL_PATH", tmp_path / "best_model.txt"),
        ):
            register_best_model()

        # Winner (lightgbm) → Production, loser (IF) → Staging
        transition_calls = client.transition_model_version_stage.call_args_list
        assert call(name="lightgbm_fraud", version="2", stage="Production") in transition_calls
        assert call(name="isolation_forest_fraud", version="1", stage="Staging") in transition_calls

    def test_best_model_txt_written(self, tmp_path):
        """best_model.txt must contain winner name, version, and AUC-PR."""
        from fraud_pipeline import register_best_model

        models = {
            "isolation_forest_fraud": {
                "version": "1",
                "run_id": "run_if",
                "metrics": {"average_precision_score": 0.20},
            },
            "lightgbm_fraud": {
                "version": "3",
                "run_id": "run_lgbm",
                "metrics": {"average_precision_score": 0.90},
            },
        }
        client = self._make_client(models)
        best_model_path = tmp_path / "best_model.txt"

        with (
            patch("mlflow.tracking.MlflowClient", return_value=client),
            patch("fraud_pipeline.ARTIFACTS_DIR", tmp_path),
            patch("fraud_pipeline.BEST_MODEL_PATH", best_model_path),
        ):
            register_best_model()

        content = best_model_path.read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "lightgbm_fraud"
        assert lines[1] == "3"
        assert float(lines[2]) == pytest.approx(0.90)

    def test_no_versions_raises(self, tmp_path):
        """If a model has no versions in registry → RuntimeError."""
        from fraud_pipeline import register_best_model

        models = {
            "isolation_forest_fraud": {
                "version": "1",
                "run_id": "run_if",
                "metrics": {"average_precision_score": 0.20},
            },
            "lightgbm_fraud": {"empty": True},
        }
        client = self._make_client(models)

        with (
            patch("mlflow.tracking.MlflowClient", return_value=client),
            patch("fraud_pipeline.ARTIFACTS_DIR", tmp_path),
            patch("fraud_pipeline.BEST_MODEL_PATH", tmp_path / "best_model.txt"),
            pytest.raises(RuntimeError, match="No versions found"),
        ):
            register_best_model()
