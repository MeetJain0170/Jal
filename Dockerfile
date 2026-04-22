# ─────────────────────────────────────────────────────────────────────────────
# JalDrishti — Production Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Build:  docker build -t jaldrishti .
# Run:    docker run -p 5500:5500 -v jaldrishti-gallery:/app/outputs/gallery jaldrishti
#
# GPU variant: replace FROM line with nvidia/cuda base and add --gpus all
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

WORKDIR /app

# ─── Environment ─────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/app/.cache/torch \
    PIP_NO_CACHE_DIR=1

# ─── System dependencies ──────────────────────────────────────────────────────
# libgl1-mesa-glx   : OpenCV headless rendering
# libglib2.0-0      : GLib for OpenCV
# libsm6 libxext6   : X11 shared memory ext (OpenCV headless still links these)
# libxrender-dev    : X render extension
# libgomp1          : OpenMP — torch parallel ops
# git               : torch.hub.load() clones repos
# wget curl         : model weight downloads (fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Pre-cache MiDaS weights inside the image (offline-safe) ─────────────────
# This runs during build so the container never needs internet at runtime.
# The TORCH_HOME env var directs torch.hub to /app/.cache/torch
RUN python -c "\
import torch; \
torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', pretrained=True, trust_repo=True); \
print('MiDaS weights cached successfully')"

# ─── Application code ─────────────────────────────────────────────────────────
COPY . .

# Ensure output directories exist (gallery persists via Docker volume)
RUN mkdir -p outputs/gallery outputs/checkpoints outputs/results outputs/logs outputs/videos

# ─── Expose & health ─────────────────────────────────────────────────────────
EXPOSE 5500

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:5500/api/status || exit 1

# ─── Entrypoint ──────────────────────────────────────────────────────────────
CMD ["python", "api.py"]
