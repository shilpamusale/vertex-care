# --- Stage 1: Build Stage ---
FROM python:3.11-slim as builder

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# If you need system packages for building (only if wheels fail without them)
# RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
COPY requirements.txt pyproject.toml ./
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# --- Stage 2: Final Stage ---
FROM python:3.11-slim
WORKDIR /app

COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy app code and necessary folders
COPY vertexcare/ ./vertexcare
COPY run_pipeline.py ./
COPY dashboard.py ./
COPY configs/ ./configs     # if needed
COPY scripts/ ./scripts     # if needed
# COPY data/ ./data           # if needed
# COPY models/ ./models       # if needed

RUN pip install gunicorn

# Set default port if not overridden by environment
ENV PORT=8000

EXPOSE 8000

CMD gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind "0.0.0.0:${PORT}" vertexcare.api.main:app
