# Python 3.11 slim — required by hackathon rules
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only minimal system deps needed to compile wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Step 1: Install torch CPU-only explicitly.
# CRITICAL: must be done BEFORE -r requirements.txt so pip does NOT upgrade
# torch to a CUDA version when resolving openenv-core / fastmcp dependencies.
# The --extra-index-url keeps PyPI as fallback while pytorch.org serves torch.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.0+cpu

# Step 2: Install all other dependencies.
# torch is already satisfied so pip will NOT re-download or upgrade it.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (dockerignore excludes .git, __pycache__, models/, etc.)
COPY . .

# Expose Port 7860 (required for Hugging Face Spaces)
EXPOSE 7860

# Keeps Python stdout/stderr unbuffered so logs appear in real time
ENV PYTHONUNBUFFERED=1

# Run the FastAPI server
# Rule: --workers 1 for thread safety (threading.Lock guards /reset and /step)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
