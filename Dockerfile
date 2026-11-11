FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.python.org \
    -r requirements.txt

# Copy application source code
COPY . .

# Set Python path to ensure clean absolute imports
ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8080"]
