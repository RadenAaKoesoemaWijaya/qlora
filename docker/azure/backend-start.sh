#!/bin/sh
# Backend startup script untuk Azure Container Apps

set -e

echo "Starting QLoRA Backend..."

# Set default values jika tidak ada
export WORKERS=${WORKERS:-4}
export PORT=${PORT:-8000}
export MAX_REQUESTS=${MAX_REQUESTS:-1000}
export MAX_REQUESTS_JITTER=${MAX_REQUESTS_JITTER:-100}

# Wait untuk dependencies jika ada
if [ -n "$DATABASE_URL" ]; then
    echo "Checking database connection..."
    # Add database connection check if needed
fi

if [ -n "$REDIS_URL" ]; then
    echo "Checking Redis connection..."
    # Add Redis connection check if needed
fi

# Start dengan gunicorn untuk production
exec gunicorn backend.main:app \
    --bind 0.0.0.0:$PORT \
    --workers $WORKERS \
    --worker-class uvicorn.workers.UvicornWorker \
    --max-requests $MAX_REQUESTS \
    --max-requests-jitter $MAX_REQUESTS_JITTER \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --timeout 120 \
    --keep-alive 2 \
    --preload
