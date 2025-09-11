# Dockerfile for VertexCare Streamlit UI

FROM python:3.11-slim as builder

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --upgrade pip
COPY requirements.txt pyproject.toml ./
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

FROM python:3.11-slim
WORKDIR /app

COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy dashboard.py from scripts folder
COPY scripts/dashboard.py ./dashboard.py

# Copy other needed folders
COPY vertexcare/ ./vertexcare
COPY scripts/ ./scripts
COPY configs/ ./configs

ENV PORT=8080
EXPOSE 8080

CMD streamlit run dashboard.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true
