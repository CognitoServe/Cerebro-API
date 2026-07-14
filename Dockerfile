# syntax=docker/dockerfile:1
FROM python:3.12.7-slim

# Don't write .pyc files; don't buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Add a non-root user for security
RUN useradd -m app && chown -R app:app /app
USER app

# Expose the service port
EXPOSE 8000

# Health check using a Python one-liner
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Uvicorn: 1 worker (stateful in-memory job store — scale-out needs Redis)
# Override with docker run -e AGENT_MODEL=... -e OPENAI_API_KEY=...
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
