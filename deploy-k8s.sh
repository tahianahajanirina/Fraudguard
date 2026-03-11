#!/usr/bin/env bash
# deploy-k8s.sh — Deploy FraudGuard to local Kubernetes (Docker Desktop)
# Usage: ./deploy-k8s.sh [dev|prod]
set -euo pipefail

ENV="${1:-dev}"
NAMESPACE="fraudguard-${ENV}"

echo "=== FraudGuard K8s Deploy — ${ENV} ==="

# 1. Verify prerequisites
echo "[1/6] Checking prerequisites..."
command -v kubectl >/dev/null 2>&1 || { echo "ERROR: kubectl not found"; exit 1; }
kubectl cluster-info >/dev/null 2>&1 || { echo "ERROR: No Kubernetes cluster available"; exit 1; }

# 2. Verify local images exist
echo "[2/6] Checking local Docker images..."
for img in fraudguard-api fraudguard-airflow fraudguard-mlflow fraudguard-webapp; do
  if ! docker image inspect "${img}:latest" >/dev/null 2>&1; then
    echo "ERROR: Image ${img}:latest not found. Run 'docker compose build' first."
    exit 1
  fi
done
echo "  All 4 images found."

# 3. Locate creditcard.csv
CSV_FILE=""
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
for candidate in \
  "${SCRIPT_DIR}/creditcard.csv" \
  "${SCRIPT_DIR}/../creditcard.csv" \
  "${SCRIPT_DIR}/../creditcard.csv/creditcard.csv"; do
  if [ -f "${candidate}" ]; then
    CSV_FILE="$(cd "$(dirname "${candidate}")" && pwd)/$(basename "${candidate}")"
    break
  fi
done
if [ -z "${CSV_FILE}" ]; then
  echo "WARNING: creditcard.csv not found — S3 upload will be skipped."
  echo "  Place it next to this script or in the parent directory."
fi

# 4. Delete old jobs (Jobs are immutable)
echo "[3/7] Cleaning up old jobs..."
kubectl -n "${NAMESPACE}" delete job localstack-init 2>/dev/null || true

# 5. Apply Kustomize overlay
echo "[4/7] Applying Kustomize overlay (${ENV})..."
kubectl apply -k "k8s/overlays/${ENV}/"

# 6. Wait for core services
echo "[5/7] Waiting for rollouts..."
kubectl -n "${NAMESPACE}" rollout status statefulset/postgres --timeout=120s
kubectl -n "${NAMESPACE}" rollout status deployment/localstack --timeout=90s

echo "  Waiting for localstack-init job..."
kubectl -n "${NAMESPACE}" wait --for=condition=complete job/localstack-init --timeout=120s

kubectl -n "${NAMESPACE}" rollout status deployment/mlflow --timeout=120s
kubectl -n "${NAMESPACE}" rollout status deployment/airflow-web --timeout=120s
kubectl -n "${NAMESPACE}" rollout status deployment/airflow-scheduler --timeout=120s
kubectl -n "${NAMESPACE}" rollout status deployment/fraudguard-api --timeout=120s
kubectl -n "${NAMESPACE}" rollout status deployment/webapp --timeout=120s

# 7. Upload CSV to LocalStack S3
echo "[6/7] Uploading creditcard.csv to S3..."
if [ -n "${CSV_FILE}" ]; then
  LOCALSTACK_POD=$(kubectl -n "${NAMESPACE}" get pod -l app=localstack -o jsonpath='{.items[0].metadata.name}')
  # Check if already uploaded
  if kubectl -n "${NAMESPACE}" exec "${LOCALSTACK_POD}" -- \
    awslocal s3 ls s3://data/creditcard.csv >/dev/null 2>&1; then
    echo "  creditcard.csv already in S3 — skipping."
  else
    echo "  Copying CSV to LocalStack pod (this may take a moment)..."
    # Use -n flag and pipe via stdin to avoid Windows drive letter colon issue
    kubectl cp -n "${NAMESPACE}" "${CSV_FILE}" "${LOCALSTACK_POD}:/tmp/creditcard.csv"
    kubectl -n "${NAMESPACE}" exec "${LOCALSTACK_POD}" -- \
      awslocal s3 cp /tmp/creditcard.csv s3://data/creditcard.csv
    kubectl -n "${NAMESPACE}" exec "${LOCALSTACK_POD}" -- rm -f /tmp/creditcard.csv
    echo "  CSV uploaded to s3://data/creditcard.csv"
  fi
else
  echo "  Skipped — creditcard.csv not found locally."
fi

# 8. Summary
echo "[7/7] Deployment complete!"
echo ""
echo "=== Pod Status ==="
kubectl -n "${NAMESPACE}" get pods
echo ""
echo "=== Services ==="
kubectl -n "${NAMESPACE}" get svc
echo ""
echo "=== Access (run these commands) ==="
echo "  kubectl -n ${NAMESPACE} port-forward svc/fraudguard-api 18000:8000   # API:      http://localhost:18000/docs"
echo "  kubectl -n ${NAMESPACE} port-forward svc/mlflow 15000:5000           # MLflow:   http://localhost:15000"
echo "  kubectl -n ${NAMESPACE} port-forward svc/airflow 18080:8080          # Airflow:  http://localhost:18080"
echo "  kubectl -n ${NAMESPACE} port-forward svc/webapp 18501:8501           # Webapp:   http://localhost:18501"
