"""Tests for train_isolation_forest and train_lightgbm pipeline functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_splits(tmp_path: Path, n: int = 200, fraud_rate: float = 0.02) -> Path:
    """Write train/test parquet splits to tmp_path/splits/."""
    splits = tmp_path / "splits"
    splits.mkdir()
    rng = np.random.RandomState(0)

    n_train, n_test = int(n * 0.8), int(n * 0.2)
    n_fraud_train = max(2, int(n_train * fraud_rate))
    n_fraud_test = max(1, int(n_test * fraud_rate))

    def _make_df(rows, n_fraud):
        X = pd.DataFrame({f"V{i}": rng.randn(rows) for i in range(1, 29)})
        X["Amount"] = rng.exponential(50, size=rows)
        y = pd.DataFrame({"Class": [0] * (rows - n_fraud) + [1] * n_fraud})
        return X, y

    X_train, y_train = _make_df(n_train, n_fraud_train)
    X_test, y_test = _make_df(n_test, n_fraud_test)

    X_train.to_parquet(splits / "X_train.parquet", index=False)
    X_test.to_parquet(splits / "X_test.parquet", index=False)
    y_train.to_parquet(splits / "y_train.parquet", index=False)
    y_test.to_parquet(splits / "y_test.parquet", index=False)

    return splits


class TestTrainIsolationForest:
    """train_isolation_forest — MLflow logging and model registration."""

    def test_logs_all_metrics(self, tmp_path):
        splits = _make_splits(tmp_path)

        logged_metrics = {}

        def fake_log_metric(key, value):
            logged_metrics[key] = value

        mock_run = MagicMock()
        mock_run.__enter__ = lambda s: s
        mock_run.__exit__ = MagicMock(return_value=False)

        with (
            patch("fraud_pipeline.SPLITS_DIR", splits),
            patch("mlflow.set_experiment"),
            patch("mlflow.start_run", return_value=mock_run),
            patch("mlflow.log_metric", side_effect=fake_log_metric),
            patch("mlflow.log_param"),
            patch("mlflow.log_figure"),
            patch("mlflow.sklearn.log_model"),
        ):
            from fraud_pipeline import train_isolation_forest
            train_isolation_forest()

        assert "precision_score" in logged_metrics
        assert "recall_score" in logged_metrics
        assert "f1_score" in logged_metrics
        assert "roc_auc_score" in logged_metrics
        assert "average_precision_score" in logged_metrics

    def test_metrics_are_valid_floats(self, tmp_path):
        splits = _make_splits(tmp_path)
        logged_metrics = {}

        def fake_log_metric(key, value):
            logged_metrics[key] = value

        mock_run = MagicMock()
        mock_run.__enter__ = lambda s: s
        mock_run.__exit__ = MagicMock(return_value=False)

        with (
            patch("fraud_pipeline.SPLITS_DIR", splits),
            patch("mlflow.set_experiment"),
            patch("mlflow.start_run", return_value=mock_run),
            patch("mlflow.log_metric", side_effect=fake_log_metric),
            patch("mlflow.log_param"),
            patch("mlflow.log_figure"),
            patch("mlflow.sklearn.log_model"),
        ):
            from fraud_pipeline import train_isolation_forest
            train_isolation_forest()

        for key, val in logged_metrics.items():
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def test_registers_model(self, tmp_path):
        splits = _make_splits(tmp_path)
        logged_models = []

        def fake_log_model(model, artifact_path, registered_model_name):
            logged_models.append(registered_model_name)

        mock_run = MagicMock()
        mock_run.__enter__ = lambda s: s
        mock_run.__exit__ = MagicMock(return_value=False)

        with (
            patch("fraud_pipeline.SPLITS_DIR", splits),
            patch("mlflow.set_experiment"),
            patch("mlflow.start_run", return_value=mock_run),
            patch("mlflow.log_metric"),
            patch("mlflow.log_param"),
            patch("mlflow.log_figure"),
            patch("mlflow.sklearn.log_model", side_effect=fake_log_model),
        ):
            from fraud_pipeline import train_isolation_forest
            train_isolation_forest()

        assert "isolation_forest_fraud" in logged_models


class TestTrainLightGBM:
    """train_lightgbm — MLflow logging and model registration."""

    def test_logs_all_metrics(self, tmp_path):
        splits = _make_splits(tmp_path)
        logged_metrics = {}

        def fake_log_metric(key, value):
            logged_metrics[key] = value

        mock_run = MagicMock()
        mock_run.__enter__ = lambda s: s
        mock_run.__exit__ = MagicMock(return_value=False)

        with (
            patch("fraud_pipeline.SPLITS_DIR", splits),
            patch("mlflow.set_experiment"),
            patch("mlflow.start_run", return_value=mock_run),
            patch("mlflow.log_metric", side_effect=fake_log_metric),
            patch("mlflow.log_param"),
            patch("mlflow.log_figure"),
            patch("mlflow.lightgbm.log_model"),
        ):
            from fraud_pipeline import train_lightgbm
            train_lightgbm()

        assert "precision_score" in logged_metrics
        assert "recall_score" in logged_metrics
        assert "f1_score" in logged_metrics
        assert "roc_auc_score" in logged_metrics
        assert "average_precision_score" in logged_metrics

    def test_metrics_are_valid_floats(self, tmp_path):
        splits = _make_splits(tmp_path)
        logged_metrics = {}

        def fake_log_metric(key, value):
            logged_metrics[key] = value

        mock_run = MagicMock()
        mock_run.__enter__ = lambda s: s
        mock_run.__exit__ = MagicMock(return_value=False)

        with (
            patch("fraud_pipeline.SPLITS_DIR", splits),
            patch("mlflow.set_experiment"),
            patch("mlflow.start_run", return_value=mock_run),
            patch("mlflow.log_metric", side_effect=fake_log_metric),
            patch("mlflow.log_param"),
            patch("mlflow.log_figure"),
            patch("mlflow.lightgbm.log_model"),
        ):
            from fraud_pipeline import train_lightgbm
            train_lightgbm()

        for key, val in logged_metrics.items():
            assert isinstance(val, float), f"{key} should be float"
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def test_registers_model(self, tmp_path):
        splits = _make_splits(tmp_path)
        logged_models = []

        def fake_log_model(model, artifact_path, registered_model_name):
            logged_models.append(registered_model_name)

        mock_run = MagicMock()
        mock_run.__enter__ = lambda s: s
        mock_run.__exit__ = MagicMock(return_value=False)

        with (
            patch("fraud_pipeline.SPLITS_DIR", splits),
            patch("mlflow.set_experiment"),
            patch("mlflow.start_run", return_value=mock_run),
            patch("mlflow.log_metric"),
            patch("mlflow.log_param"),
            patch("mlflow.log_figure"),
            patch("mlflow.lightgbm.log_model", side_effect=fake_log_model),
        ):
            from fraud_pipeline import train_lightgbm
            train_lightgbm()

        assert "lightgbm_fraud" in logged_models

    def test_uses_scale_pos_weight_for_imbalance(self, tmp_path):
        """scale_pos_weight must be logged as a hyperparameter."""
        splits = _make_splits(tmp_path)
        logged_params = {}

        def fake_log_param(key, value):
            logged_params[key] = value

        mock_run = MagicMock()
        mock_run.__enter__ = lambda s: s
        mock_run.__exit__ = MagicMock(return_value=False)

        with (
            patch("fraud_pipeline.SPLITS_DIR", splits),
            patch("mlflow.set_experiment"),
            patch("mlflow.start_run", return_value=mock_run),
            patch("mlflow.log_metric"),
            patch("mlflow.log_param", side_effect=fake_log_param),
            patch("mlflow.log_figure"),
            patch("mlflow.lightgbm.log_model"),
        ):
            from fraud_pipeline import train_lightgbm
            train_lightgbm()

        assert "scale_pos_weight" in logged_params
        assert float(logged_params["scale_pos_weight"]) > 1.0  # imbalanced dataset
