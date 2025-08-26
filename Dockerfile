# Dockerfile for VertexCare Production Deployment

# --- Stage 1: Build Stage ---
# This stage installs all dependencies into a "wheelhouse" for a clean final image.
# We use a specific, slim version of Python for a smaller, more secure image.
FROM python:3.11-slim as builder

# Set the working directory inside the container
WORKDIR /app

# Set environment variables to optimize Python's behavior in a container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build dependencies
RUN pip install --upgrade pip

# Copy the files that define our project and its dependencies
COPY requirements.txt pyproject.toml ./

# This is a key optimization: we build the Python packages into "wheels"
# in this stage. This caches the dependencies in a separate layer,
# which makes future builds much faster if the requirements haven't changed.
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt


# --- Stage 2: Final Stage ---
# This stage creates the final, lean production image by copying only what's
# necessary from the builder stage.
FROM python:3.11-slim

WORKDIR /app

# Copy the pre-built wheels from the builder stage
COPY --from=builder /app/wheels /wheels

# Install the dependencies from the wheels. This is much faster than
# downloading them from the internet again.
RUN pip install --no-cache /wheels/*

# Copy all application code, configuration, data, and trained models
COPY vertexcare/ ./vertexcare
COPY run_pipeline.py ./
COPY dashboard.py ./

# Install Gunicorn, our production-grade web server
RUN pip install gunicorn

# Tell Docker to expose port 8000, which is where our API will be listening
EXPOSE 8000

# This is the final command that runs when the container starts.
# It starts the Gunicorn server with 4 worker processes, using the Uvicorn
# worker class (which is required for FastAPI), and binds it to port 8000.

CMD gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind "0.0.0.0:$PORT" vertexcare.api.main:app
