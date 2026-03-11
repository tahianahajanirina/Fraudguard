"""FraudGuard — FastAPI prediction service for credit card fraud detection."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import joblib
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, field_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BEST_MODEL_PATH = Path("/artifacts/best_model.txt")
SCALER_PATH = Path("/artifacts/scaler.pkl")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — load model and scaler on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the production model and scaler on startup."""
    app.state.model = None
    app.state.scaler = None
    app.state.model_name = None
    app.state.model_version = None
    app.state.model_score = None

    if BEST_MODEL_PATH.exists():
        try:
            lines = BEST_MODEL_PATH.read_text().strip().splitlines()
            model_name = lines[0]
            model_version = lines[1]
            model_score = float(lines[2])

            log.info(f"Loading model '{model_name}' v{model_version} from MLflow registry")
            model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
            log.info("Model loaded successfully")

            scaler = joblib.load(SCALER_PATH)
            log.info("Scaler loaded successfully")

            app.state.model = model
            app.state.scaler = scaler
            app.state.model_name = model_name
            app.state.model_version = model_version
            app.state.model_score = model_score
        except Exception as exc:
            log.warning(f"Failed to load model or scaler: {exc}")
    else:
        log.warning(
            f"{BEST_MODEL_PATH} not found — API starts without a model. "
            "Run the Airflow pipeline first."
        )

    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FraudGuard",
    description="Credit card fraud detection API — Mastère Spécialisé IA, Télécom Paris",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------
class Transaction(BaseModel):
    """Single credit card transaction with PCA features V1-V28 and Amount."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "V1": -1.3598,
                "V2": -0.0727,
                "V3": 2.5363,
                "V4": 1.3781,
                "V5": -0.3383,
                "V6": 0.4623,
                "V7": 0.2395,
                "V8": 0.0986,
                "V9": 0.3637,
                "V10": 0.0907,
                "V11": -0.5515,
                "V12": -0.6178,
                "V13": -0.9913,
                "V14": -0.3111,
                "V15": 1.4681,
                "V16": -0.4704,
                "V17": 0.2079,
                "V18": 0.0257,
                "V19": 0.4039,
                "V20": 0.2514,
                "V21": -0.0183,
                "V22": 0.2778,
                "V23": -0.1104,
                "V24": 0.0669,
                "V25": 0.1285,
                "V26": -0.1891,
                "V27": 0.1335,
                "V28": -0.0210,
                "Amount": 149.62,
            }
        }
    )

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class BatchRequest(BaseModel):
    """Batch of transactions for bulk prediction."""

    transactions: list[Transaction]

    @field_validator("transactions")
    @classmethod
    def limit_batch_size(cls, v: list[Transaction]) -> list[Transaction]:
        """Enforce a maximum of 1000 transactions per batch."""
        if len(v) > 1000:
            raise ValueError("Batch size must not exceed 1000 transactions")
        return v


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def _predict_single(
    model: mlflow.pyfunc.PyFuncModel,
    model_name: str,
    row_df: pd.DataFrame,
) -> tuple[int, float | None, str]:
    """Run model inference and return (is_fraud, probability, risk_level)."""
    if "lightgbm" in model_name.lower():
        # The pyfunc wrapper's predict() returns class labels (0/1), not probabilities.
        # Access the native LGBMClassifier via _model_impl.lgb_model and call
        # predict_proba to get the actual fraud probability.
        lgb = model._model_impl.lgb_model
        proba = float(lgb.predict_proba(row_df)[:, 1][0])
        is_fraud = int(proba >= 0.5)
    else:
        # IsolationForest pyfunc returns 1 (normal) or -1 (anomaly)
        raw = model.predict(row_df)
        raw_val = int(raw[0]) if hasattr(raw, "__len__") else int(raw)
        is_fraud = 1 if raw_val == -1 else 0
        proba = None

    if proba is None:
        risk_level = "UNKNOWN"
    elif proba > 0.7:
        risk_level = "HIGH"
    elif proba >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return is_fraud, proba, risk_level


def _transaction_to_df(tx: Transaction) -> pd.DataFrame:
    """Convert a Transaction Pydantic model to a single-row DataFrame."""
    return pd.DataFrame([tx.model_dump()])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root() -> dict:
    """Project metadata."""
    return {
        "name": "FraudGuard",
        "description": "Credit card fraud detection — MLOps pipeline",
        "course": "Mastère Spécialisé IA",
        "institute": "Télécom Paris, Institut Polytechnique de Paris",
        "status": "running",
        "model_loaded": app.state.model is not None,
        "model_name": app.state.model_name,
    }


@app.get("/health")
def health() -> dict:
    """Service health check."""
    import os

    return {
        "status": "ok",
        "model_name": app.state.model_name,
        "model_stage": "Production",
        "mlflow_uri": os.getenv("MLFLOW_TRACKING_URI", "not set"),
        "scaler_loaded": app.state.scaler is not None,
    }


@app.post("/predict")
def predict(transaction: Transaction) -> dict:
    """Predict fraud for a single transaction."""
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="No model is loaded. Run the Airflow pipeline first.",
        )

    try:
        row_df = _transaction_to_df(transaction)
        row_df["Amount"] = app.state.scaler.transform(row_df[["Amount"]])

        is_fraud, proba, risk_level = _predict_single(
            app.state.model, app.state.model_name, row_df
        )

        return {
            "is_fraud": bool(is_fraud),
            "fraud_probability": proba,
            "risk_level": risk_level,
            "model_used": app.state.model_name,
            "prediction_label": "FRAUD" if is_fraud else "NORMAL",
        }
    except Exception as exc:
        log.error(f"Prediction error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal prediction error") from exc


@app.post("/predict_batch")
def predict_batch(body: BatchRequest) -> dict:
    """Predict fraud for a batch of transactions (max 1000)."""
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="No model is loaded. Run the Airflow pipeline first.",
        )

    try:
        df = pd.DataFrame([tx.model_dump() for tx in body.transactions])
        df["Amount"] = app.state.scaler.transform(df[["Amount"]])

        predictions = []
        for _, row in df.iterrows():
            row_df = pd.DataFrame([row])
            is_fraud, proba, risk_level = _predict_single(
                app.state.model, app.state.model_name, row_df
            )
            predictions.append(
                {
                    "is_fraud": bool(is_fraud),
                    "fraud_probability": proba,
                    "risk_level": risk_level,
                    "prediction_label": "FRAUD" if is_fraud else "NORMAL",
                }
            )

        fraud_count = sum(1 for p in predictions if p["is_fraud"])
        return {
            "predictions": predictions,
            "total": len(predictions),
            "fraud_count": fraud_count,
            "fraud_rate": fraud_count / len(predictions) if predictions else 0.0,
        }
    except Exception as exc:
        log.error(f"Batch prediction error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal prediction error") from exc


@app.get("/model_metrics")
def model_metrics() -> dict:
    """Fetch the production model's evaluation metrics from MLflow."""
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="No model is loaded. Run the Airflow pipeline first.",
        )

    try:
        client = MlflowClient()
        versions = client.search_model_versions(
            f"name='{app.state.model_name}'"
        )
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        if not prod_versions:
            raise HTTPException(
                status_code=503,
                detail=f"No Production version found for model '{app.state.model_name}'",
            )

        run_id = prod_versions[0].run_id
        metrics = client.get_run(run_id).data.metrics

        return {
            "model_name": app.state.model_name,
            "model_version": app.state.model_version,
            "precision": metrics.get("precision_score"),
            "recall": metrics.get("recall_score"),
            "f1": metrics.get("f1_score"),
            "roc_auc": metrics.get("roc_auc_score"),
            "average_precision_score": metrics.get("average_precision_score"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.error(f"MLflow unreachable or error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"MLflow unavailable: {exc}",
        ) from exc
