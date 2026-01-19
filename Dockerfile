# ------------------------------------------------------------------------------
# Stage 1: builder
# - Uses a full Python image with uv to resolve/build dependencies quickly.
# - Installs build deps only here (kept out of final runtime image).
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# System deps needed to compile some Python packages (adjust as needed).
# Keep this minimal; delete libs that you don't actually need.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (https://github.com/astral-sh/uv)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./
COPY README.md ./

# Create a virtual environment and install dependencies using uv
# uv sync will create a .venv and install all dependencies
RUN uv sync --frozen --no-dev


# ------------------------------------------------------------------------------
# Stage 2: runtime
# - Tiny final image with only runtime libs + dependencies from the builder.
# - Non-root user, safer defaults, minimal attack surface.
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Add a non-root user for security (UID/GID chosen to avoid collisions).
RUN useradd -u 10001 -m -s /usr/sbin/nologin appuser

WORKDIR /app

# Minimal runtime libs (add only what you truly need at runtime).
# Added libGL and libglib for OpenCV (cv2) support
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy your application code last for better layer caching during dev.
# Adjust paths if your repo layout differs.
COPY backend ./backend
COPY libs ./libs
COPY scripts/download_dataset.py /app/

#RUN mkdir -p /app/storage/tmp /app/storage/.sync /app/storage/emb_model_cache && \
#    chown -R appuser:appuser /app/storage \

# Crea le directory necessarie e assegna la proprietà all'utente non-root
RUN mkdir -p /app/storage/emb_model_cache && \
    chown -R appuser:appuser /app/storage

RUN mkdir -p /app/data/tmp /app/data/sync && \
    chown -R appuser:appuser /app/data

# Download dataset as appuser (not root)
USER appuser
RUN python /app/download_dataset.py
USER root

# Create merged_files directory for knowledge graph builder
RUN mkdir -p /app/libs/llm_graph_builder/merged_files && \
    chmod -R 777 /app/libs/llm_graph_builder/merged_files


# Uvicorn port (matches docker-compose mapping).
EXPOSE 5000

# Default command is for the API; docker-compose overrides this for the worker.
# In compose, you already run:
#   api:    command: uvicorn backend.app:app --host 0.0.0.0 --port 5000 --reload
# Keeping a sensible default helps when running the image directly.
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "5000"]
