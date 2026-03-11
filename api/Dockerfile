FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-install-project && \
    uv pip install "pytest>=7,<8" "httpx>=0.27,<1"

COPY main.py .

CMD ["uv", "run", "--no-project", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
