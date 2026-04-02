# CommunityPulse-Env — Dockerfile
# Runs the FastAPI environment server on port 7860 (HF Spaces default)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Environment variables (overridden at runtime via HF Space secrets)
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV HF_TOKEN=""
ENV ENV_URL="http://localhost:7860"

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
