# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Don't write .pyc files; don't buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the service port
EXPOSE 8000

# Uvicorn: 1 worker (stateful in-memory job store — scale-out needs Redis)
# Override with docker run -e AGENT_MODEL=... -e OPENAI_API_KEY=...
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
