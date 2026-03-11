"""Airflow DAG: daily continuous training — check model performance & data drift."""

from __future__ import annotations

import logging
from pathlib import Path

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago

# ---------------------------------------------------------------------------
# Path constants (same as fraud_pipeline.py)
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path("/opt/airflow/artifacts")
SPLITS_DIR = Path("/opt/airflow/artifacts/splits")
BEST_MODEL_PATH = Path("/opt/airflow/artifacts/best_model.txt")
EXPERIMENT_NAME = "fraud-detection"

# Thresholds
AUC_PR_THRESHOLD = 0.7
DRIFT_MULTIPLIER = 5  # drift if anomaly_rate > DRIFT_MULTIPLIER * expected_rate

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task 1 — Check model performance
# ---------------------------------------------------------------------------
def check_model_performance(**context) -> dict:
    """Query MLflow registry for the Production model and verify its AUC-PR."""
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    # Read model name from best_model.txt
    lines = BEST_MODEL_PATH.read_text().strip().split("\n")
    model_name = lines[0]

    # Find the Production version directly from MLflow registry
    client = MlflowClient()
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    except MlflowException:
        log.warning("Registered model '%s' not found in registry", model_name)
        return {"model_name": model_name, "current_auc_pr": 0.0, "perf_degraded": True}

    if not prod_versions:
        log.warning("No Production version found for '%s'", model_name)
        return {"model_name": model_name, "current_auc_pr": 0.0, "perf_degraded": True}

    mv = prod_versions[0]
    run = client.get_run(mv.run_id)
    current_auc_pr = run.data.metrics.get("average_precision_score", 0.0)

    perf_degraded = current_auc_pr < AUC_PR_THRESHOLD

    result = {
        "model_name": model_name,
        "current_auc_pr": current_auc_pr,
        "perf_degraded": perf_degraded,
    }
    log.info("Performance check: %s", result)
    return result


# ---------------------------------------------------------------------------
# Task 2 — Check data drift
# ---------------------------------------------------------------------------
def check_data_drift(**context) -> dict:
    """Load Isolation Forest from registry, predict on X_test, measure anomaly rate."""
    import mlflow.sklearn
    import pandas as pd
    from mlflow.tracking import MlflowClient

    X_test = pd.read_parquet(SPLITS_DIR / "X_test.parquet")

    # Load the latest Isolation Forest model from registry
    from mlflow.exceptions import MlflowException

    client = MlflowClient()
    try:
        versions = client.get_latest_versions("isolation_forest_fraud")
    except MlflowException:
        log.warning("Isolation Forest model not found in registry — skipping drift check")
        return {"anomaly_rate": 0.0, "expected_rate": 0.0, "drift_detected": False}
    if not versions:
        log.warning("No Isolation Forest version found — skipping drift check")
        return {"anomaly_rate": 0.0, "expected_rate": 0.0, "drift_detected": False}
    mv = versions[0]
    model = mlflow.sklearn.load_model(f"models:/isolation_forest_fraud/{mv.version}")

    raw_preds = model.predict(X_test)
    anomaly_rate = float((raw_preds == -1).sum()) / len(raw_preds)

    # Expected contamination rate (~0.17%)
    y_test = pd.read_parquet(SPLITS_DIR / "y_test.parquet")["Class"]
    expected_rate = float(y_test.sum()) / len(y_test)

    drift_detected = anomaly_rate > DRIFT_MULTIPLIER * expected_rate

    result = {
        "anomaly_rate": anomaly_rate,
        "expected_rate": expected_rate,
        "drift_detected": drift_detected,
    }
    log.info("Drift check: %s", result)
    return result


# ---------------------------------------------------------------------------
# Task 3 — Decide retraining (branch)
# ---------------------------------------------------------------------------
def decide_retraining(**context) -> str:
    """Branch: trigger retraining if performance degraded or drift detected."""
    ti = context["ti"]
    perf = ti.xcom_pull(task_ids="check_model_performance")
    drift = ti.xcom_pull(task_ids="check_data_drift")

    if perf["perf_degraded"] or drift["drift_detected"]:
        log.info("Retraining needed — perf_degraded=%s, drift=%s", perf, drift)
        return "trigger_retraining"

    log.info("No retraining needed — perf_degraded=%s, drift=%s", perf, drift)
    return "skip_retraining"


# ---------------------------------------------------------------------------
# Task 4b — Skip retraining
# ---------------------------------------------------------------------------
def skip_retraining_fn(**context) -> None:
    """Log that no retraining is needed."""
    log.info("No retraining needed — model performance OK, no drift detected")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="fraud_retraining_ct",
    schedule="@daily",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "fraud", "ct", "telecom-paris"],
    description="Daily check: model performance + data drift → trigger retraining if needed",
) as dag:
    task_check_perf = PythonOperator(
        task_id="check_model_performance",
        python_callable=check_model_performance,
    )

    task_check_drift = PythonOperator(
        task_id="check_data_drift",
        python_callable=check_data_drift,
    )

    task_decide = BranchPythonOperator(
        task_id="decide_retraining",
        python_callable=decide_retraining,
    )

    task_trigger = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="fraud_detection_pipeline",
        wait_for_completion=False,
    )

    task_skip = PythonOperator(
        task_id="skip_retraining",
        python_callable=skip_retraining_fn,
    )

    # Dependency graph
    [task_check_perf, task_check_drift] >> task_decide >> [task_trigger, task_skip]
