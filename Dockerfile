FROM python:3.11-slim

# Install uv from official Astral release image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Environment settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install system packages required for C-extensions, FAISS (OpenMP), git, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync (locked, production non-dev dependencies)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code and project configuration
COPY . .

# Install the project package itself
RUN uv sync --frozen --no-dev

# Expose HTTP port
EXPOSE 8080

# Run uvicorn server binding to 0.0.0.0 and listening on $PORT (default 8080)
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8080}"]