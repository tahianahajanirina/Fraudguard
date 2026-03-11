# FraudGuard — Makefile
# Usage: make <target>

# ---------------------------------------------------------------------------
# Tests (inside Docker containers)
# ---------------------------------------------------------------------------
test-api:
	docker-compose exec -e PYTHONPATH=/app api uv run --no-project pytest /app/tests/test_api.py -v

test-preprocessing:
	docker-compose exec -e PYTHONPATH=/opt/airflow/dags airflow \
		python -m pytest /opt/airflow/tests/test_preprocessing.py -v

test-model:
	docker-compose exec -e PYTHONPATH=/opt/airflow/dags airflow \
		python -m pytest /opt/airflow/tests/test_model.py -v

test-pipeline:
	docker-compose exec -e PYTHONPATH=/opt/airflow/dags airflow \
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
	docker-compose up --build

down:
	docker-compose down

down-clean:
	docker-compose down -v

# ---------------------------------------------------------------------------
.PHONY: test test-api test-preprocessing test-model test-pipeline lint format up down down-clean
