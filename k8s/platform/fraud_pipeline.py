"""Airflow DAG: end-to-end fraud detection pipeline from raw CSV to registered model."""

from __future__ import annotations

import logging
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
CSV_PATH = Path("/opt/airflow/data/creditcard.csv")
ARTIFACTS_DIR = Path("/opt/airflow/artifacts")
SPLITS_DIR = Path("/opt/airflow/artifacts/splits")
SCALER_PATH = Path("/opt/airflow/artifacts/scaler.pkl")
BEST_MODEL_PATH = Path("/opt/airflow/artifacts/best_model.txt")
EXPERIMENT_NAME = "fraud-detection"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task 1 — Ingest and preprocess
# ---------------------------------------------------------------------------
def ingest_and_preprocess() -> None:
    """Load the raw CSV, scale Amount, split into train/test parquet files."""
    import hashlib
    import joblib
    import os

    import boto3
    import pandas as pd
    from botocore.exceptions import ClientError
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # If the CSV is not available locally, download it from S3 (data bucket).
    if not CSV_PATH.exists():
        log.info("CSV not found locally — downloading from S3 data bucket")
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        s3_dl = boto3.client(
            "s3",
            endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name=os.environ["AWS_DEFAULT_REGION"],
        )
        s3_dl.download_file("data", "creditcard.csv", str(CSV_PATH))
        log.info("Downloaded creditcard.csv from s3://data/creditcard.csv")

    # Upload raw dataset to S3 only when file content changes.
    sha256 = hashlib.sha256()
    with CSV_PATH.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    dataset_sha256 = sha256.hexdigest()

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
    )

    bucket = os.environ.get("MLFLOW_S3_BUCKET", "mlflow")
    bucket_owner = os.environ.get("S3_EXPECTED_BUCKET_OWNER", "000000000000")
    dataset_key = "datasets/creditcard.csv"
    hash_key = "datasets/creditcard.csv.sha256"

    existing_sha256 = None
    try:
        hash_obj = s3.get_object(Bucket=bucket, Key=hash_key, ExpectedBucketOwner=bucket_owner)
        existing_sha256 = hash_obj["Body"].read().decode("utf-8").strip()
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code not in {"NoSuchKey", "404"}:
            raise

    already_uploaded = existing_sha256 == dataset_sha256

    if already_uploaded:
        print("Dataset unchanged (same SHA256), skipping S3 upload.")
    else:
        s3.upload_file(
            str(CSV_PATH),
            bucket,
            dataset_key,
            ExtraArgs={"ExpectedBucketOwner": bucket_owner},
        )
        s3.put_object(
            Bucket=bucket,
            Key=hash_key,
            Body=dataset_sha256.encode("utf-8"),
            ExpectedBucketOwner=bucket_owner,
        )
        print("Dataset changed, uploaded new version to S3 bucket.")

    # Load dataset
    df = pd.read_csv(CSV_PATH)
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Class'].value_counts()}")

    # Drop Time column
    df = df.drop(columns=["Time"])

    # Separate features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Train / test split (80 / 20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale only the Amount column — fit on train, transform both
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])

    # Persist splits as parquet
    X_train.to_parquet(SPLITS_DIR / "X_train.parquet", index=True)
    X_test.to_parquet(SPLITS_DIR / "X_test.parquet", index=True)
    y_train.to_frame().to_parquet(SPLITS_DIR / "y_train.parquet", index=True)
    y_test.to_frame().to_parquet(SPLITS_DIR / "y_test.parquet", index=True)

    # Persist scaler
    joblib.dump(scaler, SCALER_PATH)

    fraud_count = y.sum()
    fraud_ratio = fraud_count / len(y) * 100
    print(f"Total rows: {len(df):,}")
    print(f"Fraud count: {fraud_count:,}")
    print(f"Fraud ratio: {fraud_ratio:.4f}%")
    print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")


# ---------------------------------------------------------------------------
# Task 2 — Train Isolation Forest
# ---------------------------------------------------------------------------
def train_isolation_forest() -> None:
    """Train an IsolationForest baseline, log metrics and model to MLflow."""
    import matplotlib.pyplot as plt
    import mlflow
    import mlflow.sklearn
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load splits
    X_train = pd.read_parquet(SPLITS_DIR / "X_train.parquet")
    X_test = pd.read_parquet(SPLITS_DIR / "X_test.parquet")
    y_train = pd.read_parquet(SPLITS_DIR / "y_train.parquet")["Class"]
    y_test = pd.read_parquet(SPLITS_DIR / "y_test.parquet")["Class"]

    # Contamination from full dataset class distribution
    y_full = pd.concat([y_train, y_test])
    contamination = float(len(y_full[y_full == 1]) / len(y_full))
    print(f"Contamination rate: {contamination:.6f}")

    hyperparams = {
        "n_estimators": 200,
        "contamination": contamination,
        "max_samples": "auto",
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="isolation_forest"):
        # Train
        model = IsolationForest(**hyperparams)
        model.fit(X_train)

        # Predict — remap: -1 (anomaly) → 1 (fraud), 1 (normal) → 0
        raw_preds = model.predict(X_test)
        y_pred = (raw_preds == -1).astype(int)

        # Metrics
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_pred)
        avg_precision = average_precision_score(y_test, y_pred)

        print(f"precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")
        print(f"roc_auc={roc_auc:.4f}  avg_precision={avg_precision:.4f}")

        # Log metrics
        mlflow.log_metric("precision_score", precision)
        mlflow.log_metric("recall_score", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc_score", roc_auc)
        mlflow.log_metric("average_precision_score", avg_precision)

        # Log hyperparams
        mlflow.log_param("n_estimators", hyperparams["n_estimators"])
        mlflow.log_param("contamination", hyperparams["contamination"])
        mlflow.log_param("max_samples", hyperparams["max_samples"])
        mlflow.log_param("random_state", hyperparams["random_state"])

        # Confusion matrix figure
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        ax.set_title("Isolation Forest — Confusion Matrix")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # Register model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="isolation_forest_fraud",
        )


# ---------------------------------------------------------------------------
# Task 3 — Train LightGBM
# ---------------------------------------------------------------------------
def train_lightgbm() -> None:
    """Train a LightGBM classifier, log metrics, plots, and model to MLflow."""
    import matplotlib.pyplot as plt
    import mlflow
    import mlflow.lightgbm
    import pandas as pd
    from lightgbm import LGBMClassifier
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load splits
    X_train = pd.read_parquet(SPLITS_DIR / "X_train.parquet")
    X_test = pd.read_parquet(SPLITS_DIR / "X_test.parquet")
    y_train = pd.read_parquet(SPLITS_DIR / "y_train.parquet")["Class"]
    y_test = pd.read_parquet(SPLITS_DIR / "y_test.parquet")["Class"]

    # Class-imbalance weight
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    hyperparams = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_estimators": 1000,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    with mlflow.start_run(run_name="lightgbm"):
        # Train
        model = LGBMClassifier(**hyperparams)
        model.fit(X_train, y_train)

        # Predict
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        # Metrics
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, proba)
        avg_precision = average_precision_score(y_test, proba)

        print(f"precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")
        print(f"roc_auc={roc_auc:.4f}  avg_precision={avg_precision:.4f}")

        # Log metrics
        mlflow.log_metric("precision_score", precision)
        mlflow.log_metric("recall_score", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc_score", roc_auc)
        mlflow.log_metric("average_precision_score", avg_precision)

        # Log hyperparams
        for param_name, param_value in hyperparams.items():
            mlflow.log_param(param_name, param_value)

        # Feature importance — top 15
        import numpy as np

        feature_names = X_train.columns.tolist()
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        top_features = [feature_names[i] for i in top_idx]
        top_importances = importances[top_idx]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top_features[::-1], top_importances[::-1])
        ax.set_title("LightGBM — Top 15 Features by Importance")
        ax.set_xlabel("Importance")
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        ax.set_title("LightGBM — Confusion Matrix")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # Register model
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name="lightgbm_fraud",
        )


# ---------------------------------------------------------------------------
# Task 4 — Register best model
# ---------------------------------------------------------------------------
def register_best_model() -> None:
    """Compare both models by AUC-PR and promote the winner to Production."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    model_names = ["isolation_forest_fraud", "lightgbm_fraud"]
    model_info: dict[str, dict] = {}

    for name in model_names:
        versions = client.search_model_versions(f"name='{name}'")
        if not versions:
            raise RuntimeError(f"No versions found for model '{name}'")
        latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        run_id = latest.run_id
        version = latest.version

        run_data = client.get_run(run_id).data.metrics
        model_info[name] = {
            "version": version,
            "run_id": run_id,
            "precision_score": run_data.get("precision_score", float("nan")),
            "recall_score": run_data.get("recall_score", float("nan")),
            "f1_score": run_data.get("f1_score", float("nan")),
            "roc_auc_score": run_data.get("roc_auc_score", float("nan")),
            "average_precision_score": run_data.get("average_precision_score", float("nan")),
        }

    # Print comparison table
    header = f"{'Model':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'AUC-PR':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, info in model_info.items():
        print(
            f"{name:<30} "
            f"{info['precision_score']:>10.4f} "
            f"{info['recall_score']:>10.4f} "
            f"{info['f1_score']:>10.4f} "
            f"{info['roc_auc_score']:>10.4f} "
            f"{info['average_precision_score']:>10.4f}"
        )
    print("=" * len(header) + "\n")

    # Pick winner by highest AUC-PR
    winner_name = max(
        model_names,
        key=lambda n: model_info[n]["average_precision_score"],
    )
    loser_name = [n for n in model_names if n != winner_name][0]
    winner = model_info[winner_name]
    loser = model_info[loser_name]

    print(f"Winner: {winner_name} (AUC-PR={winner['average_precision_score']:.4f})")
    print(f"Loser:  {loser_name}  (AUC-PR={loser['average_precision_score']:.4f})")

    # Transition stages
    client.transition_model_version_stage(
        name=winner_name,
        version=winner["version"],
        stage="Production",
    )
    client.transition_model_version_stage(
        name=loser_name,
        version=loser["version"],
        stage="Staging",
    )
    print(f"Transitioned {winner_name} v{winner['version']} → Production")
    print(f"Transitioned {loser_name} v{loser['version']} → Staging")

    # Write best_model.txt
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_PATH.write_text(
        f"{winner_name}\n{winner['version']}\n{winner['average_precision_score']}\n"
    )
    print(f"Written best model info to {BEST_MODEL_PATH}")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="fraud_detection_pipeline",
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "fraud", "telecom-paris"],
    description="end-to-end fraud detection pipeline from raw CSV to registered model",
) as dag:
    task_ingest = PythonOperator(
        task_id="ingest_and_preprocess",
        python_callable=ingest_and_preprocess,
    )

    task_iso_forest = PythonOperator(
        task_id="train_isolation_forest",
        python_callable=train_isolation_forest,
    )

    task_lgbm = PythonOperator(
        task_id="train_lightgbm",
        python_callable=train_lightgbm,
    )

    task_register = PythonOperator(
        task_id="register_best_model",
        python_callable=register_best_model,
    )

    # Dependency graph:
    # ingest_and_preprocess → [train_isolation_forest, train_lightgbm] → register_best_model
    task_ingest >> [task_iso_forest, task_lgbm] >> task_register
