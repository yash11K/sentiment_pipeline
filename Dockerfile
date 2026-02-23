FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure data directories exist
RUN mkdir -p data/raw data/processed

# Expose port
EXPOSE 8000

# Run migrations then start the app
CMD ["sh", "-c", "alembic upgrade head && python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000"]
