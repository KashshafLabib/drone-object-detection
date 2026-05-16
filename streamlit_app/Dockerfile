# ─────────────────────────────────────────────────────────────
# Dockerfile — Drone Detection & Counting System (Streamlit)
# ─────────────────────────────────────────────────────────────
# Build:   docker build -t drone-detection .
# Run:     docker run -p 8501:8501 -v /path/to/weights:/app/weights drone-detection
# ─────────────────────────────────────────────────────────────

FROM python:3.10-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies required by OpenCV and general C extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ─────────────────────────────
# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install CPU-only PyTorch first (keeps image ~2.5 GB instead of ~8 GB),
# then install the rest of the requirements
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ───────────────────────────────────
COPY app.py detector.py config.py components.py styles.py ./

# ── Create weights mount point ──────────────────────────────
# Weights are mounted at runtime: -v /path/to/weights:/app/weights
# The app searches /app/weights/ for .pt files automatically
RUN mkdir -p /app/weights

# ── Streamlit configuration ─────────────────────────────────
# Disable Streamlit's usage stats and email prompt
RUN mkdir -p /root/.streamlit && \
    printf '[server]\nheadless = true\nport = 8501\naddress = "0.0.0.0"\nenableCORS = false\nenableXsrfProtection = false\nmaxUploadSize = 500\n\n[browser]\ngatherUsageStats = false\n' \
    > /root/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py"]
