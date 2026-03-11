"""Tests for the data ingestion and preprocessing pipeline."""

from __future__ import annotations

from unittest.mock import patch

import joblib
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler


@pytest.fixture()
def preprocessed(tmp_path, sample_dataframe):
    """Run ingest_and_preprocess on a sample CSV, return results."""
    csv_path = tmp_path / "creditcard.csv"
    sample_dataframe.to_csv(csv_path, index=False)

    splits_dir = tmp_path / "splits"
    scaler_path = tmp_path / "scaler.pkl"

    with (
        patch("fraud_pipeline.CSV_PATH", csv_path),
        patch("fraud_pipeline.SPLITS_DIR", splits_dir),
        patch("fraud_pipeline.SCALER_PATH", scaler_path),
    ):
        from fraud_pipeline import ingest_and_preprocess

        ingest_and_preprocess()

    return {
        "splits_dir": splits_dir,
        "scaler_path": scaler_path,
        "X_train": pd.read_parquet(splits_dir / "X_train.parquet"),
        "X_test": pd.read_parquet(splits_dir / "X_test.parquet"),
        "y_train": pd.read_parquet(splits_dir / "y_train.parquet")["Class"],
        "y_test": pd.read_parquet(splits_dir / "y_test.parquet")["Class"],
    }


class TestPreprocessing:
    def test_time_dropped_and_columns_correct(self, preprocessed):
        """Time must be dropped, features must be V1-V28 + Amount."""
        expected = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        assert list(preprocessed["X_train"].columns) == expected
        assert "Time" not in preprocessed["X_train"].columns

    def test_train_test_split_ratio(self, preprocessed):
        """Split should be approximately 80/20."""
        total = len(preprocessed["X_train"]) + len(preprocessed["X_test"])
        train_ratio = len(preprocessed["X_train"]) / total
        assert train_ratio == pytest.approx(0.8, abs=0.05)

    def test_stratification_preserves_fraud_ratio(self, preprocessed):
        """Fraud ratio should be similar in train and test sets."""
        train_fraud = preprocessed["y_train"].mean()
        test_fraud = preprocessed["y_test"].mean()
        assert train_fraud == pytest.approx(test_fraud, abs=0.05)

    def test_amount_is_standardized(self, preprocessed):
        """Amount in training set should have mean~0 and std~1 after scaling."""
        amount = preprocessed["X_train"]["Amount"]
        assert amount.mean() == pytest.approx(0.0, abs=0.2)
        assert amount.std() == pytest.approx(1.0, abs=0.3)

    def test_scaler_is_valid(self, preprocessed):
        """Scaler must be saved and be a StandardScaler instance."""
        scaler = joblib.load(preprocessed["scaler_path"])
        assert isinstance(scaler, StandardScaler)
