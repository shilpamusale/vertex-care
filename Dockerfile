# Dockerfile for VertexCare Streamlit UI

# --- Stage 1: Build Stage ---
FROM python:3.11-slim as builder

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --upgrade pip
COPY requirements.txt pyproject.toml ./
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# --- Stage 2: Final Stage ---
FROM python:3.11-slim
WORKDIR /app

COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy only the code needed for the dashboard to run.
COPY dashboard.py ./
COPY vertexcare/ ./vertexcare
COPY scripts/ ./scripts
COPY configs/ ./configs

# Set the PORT environment variable that Cloud Run provides. Default to 8080.
ENV PORT=8080

EXPOSE 8080

CMD ["streamlit", "run", "dashboard.py", "--server.port=${PORT}", "--server.address=0.0.0.0", "--server.headless=true"]
