# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FraudGuard is an MLOps pipeline for credit card fraud detection (Telecom Paris DATA713 project). It trains two models (IsolationForest and LightGBM), compares them on AUC-PR, promotes the winner to production via MLflow, and serves predictions through a FastAPI endpoint.

## Architecture

Four Docker services orchestrated via `docker-compose.yml`:
- **Postgres** (port 5432): Shared metadata store for Airflow and MLflow
- **MLflow** (port 5000): Experiment tracking and model registry, backed by Postgres + file-based artifact store
- **Airflow** (port 8080): DAG orchestration with LocalExecutor (admin:admin)
- **FastAPI API** (port 8000): Prediction service loading the production model from MLflow registry

Pipeline flow (defined in `airflow/dags/fraud_pipeline.py`):
```
ingest_and_preprocess → [train_isolation_forest, train_lightgbm] → register_best_model
```

The API (`api/main.py`) loads the production model and scaler at startup via a lifespan context manager, then serves `/predict` and `/predict_batch` endpoints.

## Common Commands

### Start/stop the full stack
```bash
docker-compose up --build       # Build and start all services
docker-compose down -v          # Stop and remove volumes
```

### Linting (root pyproject.toml configures ruff)
```bash
ruff check .                    # Lint all Python files
ruff format .                   # Format all Python files
```

Ruff config: line-length=100, target=py311, rules: E, F, I.

### Running the pipeline
Trigger the `fraud_detection_pipeline` DAG from the Airflow UI at http://localhost:8080 (schedule=None, must be triggered manually).

### API usage
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"V1": -1.35, ..., "V28": -0.02, "Amount": 149.62}'
```

## Key Technical Details

- **Data**: Expects `creditcard.csv` in parent directory (`../creditcard.csv` relative to project root). Features V1-V28 are PCA-transformed; Time is dropped; Amount is standardized via StandardScaler.
- **Model comparison metric**: AUC-PR (average_precision_score) chosen due to extreme class imbalance (~0.17% fraud). The winner gets promoted to "Production" stage, loser to "Staging".
- **Artifacts**: Stored in `/mlflow/artifacts` Docker volume, shared between Airflow and the API. Includes parquet splits, scaler (joblib), and `best_model.txt`.
- **Package management**: Uses `uv` for dependency installation in both Dockerfiles. Each sub-project (`airflow/`, `api/`) has its own `pyproject.toml` and `uv.lock`.
- **Python 3.11** across all services.
- **No tests currently exist** despite pytest being listed as a dev dependency.

## Two Sub-Projects

| Path | Purpose | Key deps |
|------|---------|----------|
| `airflow/` | DAG + ML training | airflow 2.8, lightgbm, scikit-learn, mlflow, imbalanced-learn |
| `api/` | Prediction service | fastapi, uvicorn, mlflow, pydantic 2.6 |

Both use hatchling as build backend with `packages = []` (no installable package, just dependency management).
