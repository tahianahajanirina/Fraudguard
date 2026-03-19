# FraudGuard — Makefile
# Usage: make <target>

# ---------------------------------------------------------------------------
# Tests (inside Docker containers)
# ---------------------------------------------------------------------------
test-api:
	docker compose exec -e PYTHONPATH=/app api uv run --no-project pytest /app/tests/test_api.py -v

test-preprocessing:
	docker compose exec -e PYTHONPATH=/opt/airflow/dags airflow \
		python -m pytest /opt/airflow/tests/test_preprocessing.py -v

test-model:
	docker compose exec -e PYTHONPATH=/opt/airflow/dags airflow \
		python -m pytest /opt/airflow/tests/test_model.py -v

test-pipeline:
	docker compose exec -e PYTHONPATH=/opt/airflow/dags airflow \
		python -m pytest /opt/airflow/tests/test_pipeline.py -v

test: test-api test-preprocessing test-model test-pipeline

# ---------------------------------------------------------------------------
# Linting (local)
# ---------------------------------------------------------------------------
lint:
	uvx ruff check .

format:
	uvx ruff format .

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
up:
	docker compose up --build

down:
	docker compose down

down-clean:
	docker compose down -v

# ---------------------------------------------------------------------------
# Kubernetes (Docker Desktop)
# ---------------------------------------------------------------------------
k8s-build:
	docker compose build api airflow mlflow webapp

k8s-deploy-dev: k8s-build
	bash deploy-k8s.sh dev

k8s-deploy-prod: k8s-build
	bash deploy-k8s.sh prod

k8s-destroy-dev:
	kubectl delete namespace fraudguard-dev --ignore-not-found

k8s-destroy-prod:
	kubectl delete namespace fraudguard-prod --ignore-not-found

# ---------------------------------------------------------------------------
.PHONY: test test-api test-preprocessing test-model test-pipeline lint format up down down-clean k8s-build k8s-deploy-dev k8s-deploy-prod k8s-destroy-dev k8s-destroy-prod
