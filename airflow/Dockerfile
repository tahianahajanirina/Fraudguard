FROM apache/airflow:2.8.1-python3.11

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1

# Remove airflow's user-installed numpy/scipy (1.24.x) — incompatible with LightGBM
RUN find /home/airflow/.local/lib/python3.11/site-packages \
         -maxdepth 1 -name "numpy*" -o -name "scipy*" \
    | xargs rm -rf

# Install ML dependencies to system Python (visible to all users)
RUN uv pip install --system \
    "numpy>=1.26,<2" \
    "scipy>=1.13,<2" \
    "lightgbm>=4.3,<5" \
    "scikit-learn>=1.4,<2" \
    "mlflow>=2.11,<3" \
    "pandas>=2.2,<3" \
    "imbalanced-learn>=0.12,<1" \
    "joblib>=1.3,<2" \
    "pyarrow>=15,<16" \
    "psycopg2-binary>=2.9,<3" \
    "matplotlib>=3.8,<4"

USER airflow
