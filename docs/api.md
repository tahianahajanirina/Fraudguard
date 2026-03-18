# 📡 API Documentation

## Overview

The FraudGuard API is a FastAPI service that serves real-time fraud predictions using the production model from MLflow. It supports single and batch predictions with risk scoring.

**Base URL**: `http://localhost:8000`

**Interactive docs**: `http://localhost:8000/docs` (Swagger UI)

## Endpoints

### `GET /` — Project Metadata

Returns general information about the project.

```bash
curl http://localhost:8000/
```

**Response** `200 OK`:
```json
{
  "name": "FraudGuard",
  "description": "Credit Card Fraud Detection API",
  "course": "DATA713",
  "institute": "Mastère Spécialisé IA · Télécom Paris",
  "status": "running",
  "model_loaded": true,
  "model_name": "lightgbm_fraud"
}
```

---

### `GET /health` — Health Check

Returns service health, model status, and scaler state.

```bash
curl http://localhost:8000/health
```

**Response** `200 OK`:
```json
{
  "status": "healthy",
  "model_name": "lightgbm_fraud",
  "model_stage": "Production",
  "mlflow_uri": "http://mlflow:5000",
  "scaler_loaded": true
}
```

---

### `POST /predict` — Single Prediction

Predict whether a single transaction is fraudulent.

**Request body**: JSON object with 28 PCA features (V1–V28) and Amount.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.3598, "V2": -0.0727, "V3": 2.5363, "V4": 1.3781,
    "V5": -0.3383, "V6": 0.4623, "V7": 0.2395, "V8": 0.0986,
    "V9": 0.3637, "V10": 0.0907, "V11": -0.5515, "V12": -0.6178,
    "V13": -0.9913, "V14": -0.3111, "V15": 1.4681, "V16": -0.4704,
    "V17": 0.2079, "V18": 0.0257, "V19": 0.4039, "V20": 0.2514,
    "V21": -0.0183, "V22": 0.2778, "V23": -0.1104, "V24": 0.0669,
    "V25": 0.1285, "V26": -0.1891, "V27": 0.1335, "V28": -0.0210,
    "Amount": 149.62
  }'
```

**Response** `200 OK`:
```json
{
  "is_fraud": false,
  "fraud_probability": 0.003,
  "risk_level": "LOW",
  "model_used": "lightgbm_fraud",
  "prediction_label": "NORMAL"
}
```

**Risk levels**:
| Level | Probability Range | Description |
|-------|------------------|-------------|
| `HIGH` | > 0.7 | High likelihood of fraud |
| `MEDIUM` | 0.3 – 0.7 | Requires manual review |
| `LOW` | < 0.3 | Likely legitimate |
| `UNKNOWN` | — | Probability unavailable (IsolationForest) |

**Error responses**:
- `422 Unprocessable Entity` — Invalid input (missing/wrong fields)
- `503 Service Unavailable` — No model loaded

---

### `POST /predict_batch` — Batch Predictions

Predict fraud for multiple transactions (max 1000).

```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"V1": -1.35, "V2": -0.07, ..., "V28": -0.02, "Amount": 149.62},
      {"V1": 1.19, "V2": 0.26, ..., "V28": 0.01, "Amount": 2.69}
    ]
  }'
```

**Response** `200 OK`:
```json
{
  "predictions": [
    {
      "is_fraud": false,
      "fraud_probability": 0.003,
      "risk_level": "LOW",
      "model_used": "lightgbm_fraud",
      "prediction_label": "NORMAL"
    },
    {
      "is_fraud": true,
      "fraud_probability": 0.92,
      "risk_level": "HIGH",
      "model_used": "lightgbm_fraud",
      "prediction_label": "FRAUD"
    }
  ],
  "total": 2,
  "fraud_count": 1,
  "fraud_rate": 0.5
}
```

---

### `GET /model_metrics` — Model Evaluation Metrics

Returns evaluation metrics for the current production model.

```bash
curl http://localhost:8000/model_metrics
```

**Response** `200 OK`:
```json
{
  "model_name": "lightgbm_fraud",
  "model_version": "1",
  "precision": 0.95,
  "recall": 0.82,
  "f1": 0.88,
  "roc_auc": 0.97,
  "average_precision_score": 0.84
}
```

---

## Input Schema

The `Transaction` model requires exactly 29 fields:

| Field | Type | Description |
|-------|------|-------------|
| V1 – V28 | `float` | PCA-transformed features from the original dataset |
| Amount | `float` | Transaction amount (will be scaled by the API) |

> **Note**: The `Time` feature from the original dataset is not used. The API applies the fitted StandardScaler to the `Amount` field before inference.

## Model Loading

At startup, the API:

1. Reads `/artifacts/best_model.txt` to identify the winning model
2. Loads the model from MLflow: `models:/{model_name}/Production`
3. Loads the scaler from `/artifacts/scaler.pkl`
4. If no model is available, the API starts in degraded mode (health endpoint works, predictions return 503)

## Error Handling

| Status Code | Meaning |
|-------------|---------|
| `200` | Success |
| `422` | Validation error (check input fields) |
| `500` | Internal server error |
| `503` | No model loaded (run the pipeline first) |
