FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ingest.py .
# .env file will be loaded at runtime from volume or env_file

# Create directories for data persistence
RUN mkdir -p /app/chroma_db /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "ingest.py", "/app/data"]

