"""HTTP client for the FraudGuard FastAPI backend.

All network calls are centralised here. Every function returns a typed
dataclass (defined in api.models) or raises ApiError — callers never
handle requests exceptions directly.
"""

from __future__ import annotations

import requests

from api.models import (
    BatchPrediction,
    BatchPredictResponse,
    HealthResponse,
    ModelMetricsResponse,
    PredictResponse,
    RootResponse,
)
from config import (
    API_BASE_URL,
    API_TIMEOUT_BATCH,
    API_TIMEOUT_FAST,
    API_TIMEOUT_PREDICT,
)


class ApiError(Exception):
    """Raised when the API returns an error or is unreachable."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get(path: str, timeout: int) -> dict:
    """Perform a GET and return the parsed JSON body."""
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(f"Cannot reach the API at {API_BASE_URL}") from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError("API request timed out") from exc
    except requests.exceptions.HTTPError as exc:
        raise ApiError(str(exc), status_code=exc.response.status_code) from exc


def _post(path: str, payload: dict, timeout: int) -> dict:
    """Perform a POST with JSON body and return the parsed JSON response."""
    try:
        response = requests.post(
            f"{API_BASE_URL}{path}", json=payload, timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(f"Cannot reach the API at {API_BASE_URL}") from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError("API request timed out") from exc
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.json().get("detail", str(exc))
        raise ApiError(detail, status_code=exc.response.status_code) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_health() -> HealthResponse:
    """Fetch the /health endpoint."""
    data = _get("/health", timeout=API_TIMEOUT_FAST)
    return HealthResponse(
        status=data.get("status", "unknown"),
        model_name=data.get("model_name"),
        model_stage=data.get("model_stage"),
        mlflow_uri=data.get("mlflow_uri"),
        scaler_loaded=data.get("scaler_loaded", False),
    )


def get_root() -> RootResponse:
    """Fetch the / endpoint for project metadata."""
    data = _get("/", timeout=API_TIMEOUT_FAST)
    return RootResponse(
        name=data.get("name", "FraudGuard"),
        description=data.get("description", ""),
        course=data.get("course", ""),
        institute=data.get("institute", ""),
        status=data.get("status", "unknown"),
        model_loaded=data.get("model_loaded", False),
        model_name=data.get("model_name"),
    )


def get_model_metrics() -> ModelMetricsResponse:
    """Fetch the /model_metrics endpoint."""
    data = _get("/model_metrics", timeout=API_TIMEOUT_FAST)
    return ModelMetricsResponse(
        model_name=data.get("model_name", "—"),
        model_version=data.get("model_version", "—"),
        precision=data.get("precision"),
        recall=data.get("recall"),
        f1=data.get("f1"),
        roc_auc=data.get("roc_auc"),
        average_precision_score=data.get("average_precision_score"),
    )


def post_predict(payload: dict) -> PredictResponse:
    """Send a single-transaction prediction request."""
    data = _post("/predict", payload, timeout=API_TIMEOUT_PREDICT)
    return PredictResponse(
        is_fraud=data.get("is_fraud", False),
        fraud_probability=data.get("fraud_probability"),
        risk_level=data.get("risk_level", "UNKNOWN"),
        model_used=data.get("model_used", "—"),
        prediction_label=data.get("prediction_label", "—"),
    )


def post_predict_batch(rows: list[dict]) -> BatchPredictResponse:
    """Send a batch prediction request (max 1 000 rows)."""
    data = _post(
        "/predict_batch",
        {"transactions": rows},
        timeout=API_TIMEOUT_BATCH,
    )
    predictions = [
        BatchPrediction(
            is_fraud=p.get("is_fraud", False),
            fraud_probability=p.get("fraud_probability"),
            risk_level=p.get("risk_level", "UNKNOWN"),
            prediction_label=p.get("prediction_label", "—"),
        )
        for p in data.get("predictions", [])
    ]
    return BatchPredictResponse(
        predictions=predictions,
        total=data.get("total", len(predictions)),
        fraud_count=data.get("fraud_count", 0),
        fraud_rate=data.get("fraud_rate", 0.0),
    )
