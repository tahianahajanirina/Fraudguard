FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv pip install --system 'mlflow>=2.11,<3' psycopg2-binary boto3

EXPOSE 5000

CMD ["sh", "-c", "mlflow server --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} --artifacts-destination ${MLFLOW_ARTIFACTS_DESTINATION} --serve-artifacts --host 0.0.0.0 --port ${MLFLOW_PORT:-5000}"]
