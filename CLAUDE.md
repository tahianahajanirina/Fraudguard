# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FraudGuard is an MLOps pipeline for credit card fraud detection (Telecom Paris DATA713 / Mastère Spécialisé IA). It trains two models (IsolationForest and LightGBM), compares them on AUC-PR, promotes the winner to production via MLflow, and serves predictions through a FastAPI endpoint. A Streamlit webapp provides a dashboard UI.

## Architecture

Six Docker services orchestrated via `docker-compose.yml`:

| Service      | Port | Role                                           |
|-------------|------|-------------------------------------------------|
| `postgres`   | 5432 | Shared metadata store for Airflow and MLflow    |
| `mlflow`     | 5000 | Experiment tracking + Model Registry            |
| `airflow`    | 8080 | DAG orchestration with LocalExecutor            |
| `api`        | 8000 | FastAPI prediction service                      |
| `webapp`     | 8501 | Streamlit dashboard UI                          |
| `pgadmin`    | 5051 | PostgreSQL admin UI                             |
| `localstack` | 4566 | S3-compatible bucket for MLflow artifacts       |

MLflow metadata are stored in PostgreSQL; models/artifacts are stored in an S3 bucket (`s3://mlflow`) hosted by LocalStack.

Pipeline flow (defined in `airflow/dags/fraud_pipeline.py`):
```
ingest_and_preprocess → [train_isolation_forest, train_lightgbm] → register_best_model
```

The API (`api/main.py`) loads the production model and scaler at startup via a lifespan context manager, then serves `/predict` and `/predict_batch` endpoints.

## Common Commands (via Makefile)

```bash
make up                  # docker-compose up --build
make down                # docker-compose down
make down-clean          # docker-compose down -v (removes volumes)

make test                # Run all tests (api + preprocessing + model + pipeline)
make test-api            # API tests only (inside Docker)
make test-preprocessing  # Preprocessing tests only
make test-model          # Model tests only
make test-pipeline       # Pipeline tests only

make lint                # uvx ruff check .
make format              # uvx ruff format .

make k8s-deploy-dev      # Build images + deploy dev overlay
make k8s-deploy-prod     # Build images + deploy prod overlay
make k8s-destroy-dev     # Delete fraudguard-dev namespace
make k8s-destroy-prod    # Delete fraudguard-prod namespace
```

Ruff config (root `pyproject.toml`): line-length=100, target=py311, rules: E, F, I.

### Running the pipeline

Trigger the `fraud_detection_pipeline` DAG from the Airflow UI at http://localhost:8080 (schedule=None, must be triggered manually). Default credentials: admin/admin.

### Environment setup

```bash
cp .env.example .env     # Create env file before first run
```

### API usage

```bash
curl http://localhost:8000/health                              # Health check
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"V1": -1.35, ..., "V28": -0.02, "Amount": 149.62}'    # Single prediction
```

## Key Technical Details

- **Data**: Expects `creditcard.csv` in parent directory (`../creditcard.csv`). Features V1-V28 are PCA-transformed; Time is dropped; Amount is standardized via StandardScaler.
- **Model comparison metric**: AUC-PR (average_precision_score) chosen due to extreme class imbalance (~0.17% fraud). The winner gets promoted to "Production" stage, loser to "Staging".
- **Artifacts**: Stored in S3 bucket via LocalStack, shared between Airflow and the API. Includes parquet splits, scaler (joblib), and `best_model.txt`.
- **Package management**: Uses `uv` for dependency installation in all Dockerfiles. Each sub-project has its own `pyproject.toml` and `uv.lock`.
- **Python 3.11** across all services.

## Sub-Projects

| Path | Purpose | Key deps |
|------|---------|----------|
| `airflow/` | DAG + ML training | airflow 2.8, lightgbm, scikit-learn, mlflow, imbalanced-learn |
| `api/` | Prediction service | fastapi, uvicorn, mlflow, pydantic 2.6 |
| `webapp/` | Streamlit dashboard | streamlit 1.32, plotly, pandas, requests |

All use hatchling as build backend with `packages = []` (no installable package, just dependency management).

## Tests

Tests run inside Docker containers via Makefile targets (see above). Test files:

| File | Scope |
|------|-------|
| `tests/test_api.py` | FastAPI endpoints (predict, health, batch) |
| `tests/test_preprocessing.py` | Data ingestion and preprocessing |
| `tests/test_model.py` | Model training and evaluation |
| `tests/test_pipeline.py` | End-to-end DAG pipeline |
| `tests/conftest.py` | Shared pytest fixtures |

## Load Testing

`load_tests/locustfile.py` — Locust load test for the API.

## Kubernetes Deployment

- Base manifests: `k8s/api/` (Deployment, Service, ConfigMap)
- Platform services: `k8s/platform/`
- Overlays: `k8s/overlays/dev` (1 replica) and `k8s/overlays/prod` (2 replicas)
- Deploy script: `deploy-k8s.sh`

## CI/CD

GitHub Actions workflow: `.github/workflows/api-k8s-cicd.yml`
- Push to `develop` → deploy dev overlay
- Push to `main` → deploy prod overlay
- Requires a self-hosted runner with local Kubernetes access

## Other Directories

- `conception/` — Architecture diagrams (drawio, png) and design documentation
- `mlflow/` — MLflow Dockerfile and config
