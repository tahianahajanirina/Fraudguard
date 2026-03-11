# FraudGuard

> POC MLOps pipeline for credit card fraud detection Project.

**Mastère Spécialisé IA — Télécom Paris, Institut Polytechnique de Paris**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Airflow](https://img.shields.io/badge/Airflow-2.8.1-green)
![MLflow](https://img.shields.io/badge/MLflow-2.11-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

---

## Architecture

```
creditcard.csv
     │
     ▼
[Airflow DAG]──────────────────────────────────────┐
  ├─ ingest_and_preprocess                          │
  ├─ train_isolation_forest ──► [MLflow Registry]  │
  └─ train_lightgbm ──────────► [MLflow Registry]  │
       └─ register_best_model                       │
                                                    │
[FastAPI /predict] ◄────── loads Production model ◄┘
```

Services share a single network:

| Service    | Port | Role                                      |
|------------|------|-------------------------------------------|
| `postgres`  | —    | Airflow metadata + MLflow backend store   |
| `mlflow`    | 5000 | Experiment tracking + Model Registry      |
| `airflow`   | 8080 | Pipeline orchestration (LocalExecutor)    |
| `api`       | 8000 | FastAPI REST prediction service           |
| `pgadmin`   | 5051 | PostgreSQL UI                             |
| `localstack`| 4566 | S3-compatible bucket for MLflow artifacts |

MLflow metadata are stored in PostgreSQL, while models/artifacts are stored in an S3 bucket (`s3://mlflow`) hosted locally by LocalStack.

---

## ML Approach

Two models compete on the same 80/20 stratified split:

- **IsolationForest** — unsupervised anomaly detection baseline
  - `n_estimators=200`, `contamination` set to the actual fraud rate
- **LightGBM** — supervised gradient boosting
  - `num_leaves=63`, `learning_rate=0.05`, `scale_pos_weight` balances the 0.17% fraud rate

The winner is chosen by **AUC-PR (Average Precision Score)** — the correct metric for heavily imbalanced datasets. ROC-AUC and accuracy are misleading when negatives dominate.

---

## Prerequisites

- Docker and Docker Compose installed
- `creditcard.csv/creditcard.csv` present as a sibling folder to `fraudguard/`
- `anomaly-detection-lightgbm-isolation-forest.ipynb` present as a sibling

```
MLOPS_Project/
├── creditcard.csv/
│   └── creditcard.csv
├── anomaly-detection-lightgbm-isolation-forest.ipynb
└── fraudguard/          ← this directory
```

---

## Quick Start

1. **Enter the project directory**
   ```bash
   cd fraudguard
   ```

2. **Configure environment variables**
  ```bash
  cp .env.example .env
  ```

3. **Build and start all services**
   ```bash
  docker compose up --build -d
   ```

4. **Wait ~60-90 seconds** for first-time bootstrap (Postgres + LocalStack bucket + Airflow init)

5. **Open Airflow** at http://localhost:8080 — login uses values from `.env`

6. **Unpause and trigger** the `fraud_detection_pipeline` DAG

7. **Monitor** the run in the Airflow UI — takes ~5 minutes

8. **Open MLflow** at http://localhost:5000 to compare experiment runs and inspect the Model Registry

9. **Once the pipeline completes**, the API is live at http://localhost:8000/docs

10. **(Optional) Inspect bucket via LocalStack** (S3 endpoint at http://localhost:4566)

---

## Example API Call

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

Example response:
```json
{
  "is_fraud": false,
  "fraud_probability": 0.003,
  "risk_level": "LOW",
  "model_used": "lightgbm_fraud",
  "prediction_label": "NORMAL"
}
```

---

## API Kubernetes + GitHub Actions (Local Cluster)

This repository now includes:

- Kubernetes base manifests under `k8s/api`
- Environment overlays under `k8s/overlays/dev` and `k8s/overlays/prod`
- A CI/CD workflow under `.github/workflows/api-k8s-cicd.yml`

### 1) Local Kubernetes prerequisites

- Docker Desktop Kubernetes **or** Minikube running
- `kubectl` installed on the machine that will run deployment
- A GitHub self-hosted runner on that same machine (required to access local cluster)

### 2) Kubernetes manifests included

- Base (`k8s/api`): shared Deployment, Service, ConfigMap
- Dev overlay (`k8s/overlays/dev`):
  - namespace `fraudguard-dev`
  - 1 replica, lighter resources
  - default MLflow URI for local cluster
- Prod overlay (`k8s/overlays/prod`):
  - namespace `fraudguard-prod`
  - 2 replicas, stricter resources
  - image pull policy `Always`

### 3) GitHub Actions workflow behavior

Workflow: `.github/workflows/api-k8s-cicd.yml`

On push (when API/K8s files change), it:

1. Builds Docker image from `api/Dockerfile`
2. Pushes image to `ghcr.io/<owner>/fraudguard-api`
3. Uses self-hosted runner to apply Kubernetes manifests locally
4. Updates deployment image and waits for rollout

Environment routing:

- Push to `develop` -> deploy `dev` overlay (`fraudguard-dev`)
- Push to `main` -> deploy `prod` overlay (`fraudguard-prod`)
- `workflow_dispatch` -> choose `dev` or `prod` manually

### 4) Required GitHub setup

Create a self-hosted runner in your repo settings and install it on your local machine.

Set repository secrets:

- `KUBE_CONTEXT` (optional): e.g. `docker-desktop` or `minikube`
- `MLFLOW_TRACKING_URI` (optional):
  - Docker Desktop: `http://host.docker.internal:5000`
  - Minikube: `http://host.minikube.internal:5000`
- `GHCR_USERNAME` and `GHCR_PAT` (optional, only if image is private)

### 5) Access the API in Kubernetes

After deployment:

```bash
kubectl -n fraudguard get pods
kubectl -n fraudguard get svc fraudguard-api
kubectl -n fraudguard port-forward svc/fraudguard-api 8000:8000
```

Then open: http://localhost:8000/docs

To target a specific environment manually:

```bash
kubectl apply -k k8s/overlays/dev
kubectl apply -k k8s/overlays/prod
```

### 6) First deployment sanity check

```bash
kubectl -n fraudguard rollout status deployment/fraudguard-api
kubectl -n fraudguard logs deploy/fraudguard-api --tail=100
curl http://localhost:8000/health
```

---

## Stopping the Project

```bash
docker-compose down -v
```

This removes all containers and named volumes (Postgres data, MLflow artifacts, processed splits).

