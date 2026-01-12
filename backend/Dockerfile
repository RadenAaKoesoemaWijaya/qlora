# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .
COPY requirements-new.txt .

# Install Python dependencies
# First install base requirements, then ML requirements
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-new.txt

# Copy application code
COPY . .

# Create directories for data persistence
RUN mkdir -p datasets checkpoints

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
