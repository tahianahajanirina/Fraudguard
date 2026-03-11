"""Typed response shapes for the FraudGuard API."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HealthResponse:
    status: str
    model_name: str | None
    model_stage: str | None
    mlflow_uri: str | None
    scaler_loaded: bool


@dataclass
class RootResponse:
    name: str
    description: str
    course: str
    institute: str
    status: str
    model_loaded: bool
    model_name: str | None


@dataclass
class PredictResponse:
    is_fraud: bool
    fraud_probability: float | None
    risk_level: str
    model_used: str
    prediction_label: str


@dataclass
class BatchPrediction:
    is_fraud: bool
    fraud_probability: float | None
    risk_level: str
    prediction_label: str


@dataclass
class BatchPredictResponse:
    predictions: list[BatchPrediction] = field(default_factory=list)
    total: int = 0
    fraud_count: int = 0
    fraud_rate: float = 0.0


@dataclass
class ModelMetricsResponse:
    model_name: str
    model_version: str
    precision: float | None
    recall: float | None
    f1: float | None
    roc_auc: float | None
    average_precision_score: float | None
